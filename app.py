"""
Hugging Face Space for Pi0 Inference on RoboEval Tasks

This Gradio app allows users to run Pi0 model inference on bimanual robot tasks
and view the resulting execution videos.
"""

import os
import tempfile
import copy
import numpy as np
import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import gradio as gr
import subprocess
import sys

# --- Headless defaults (set BEFORE mujoco/roboeval imports) ---
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

# Note: Dependencies are installed via setup.sh before the app starts
# This keeps the app code clean and separates installation logic

# Run setup if dependencies aren't installed
def check_and_install_dependencies():
    """Check if dependencies are installed, run setup if not."""
    dependencies_ok = True
    
    try:
        import roboeval
        print("✓ roboeval imported")
    except ImportError as e:
        print(f"✗ roboeval import failed: {e}")
        dependencies_ok = False
    
    try:
        import lerobot
        print("✓ lerobot imported")
    except ImportError as e:
        print(f"✗ lerobot import failed: {e}")
        dependencies_ok = False
    
    try:
        import openpi
        print("✓ openpi imported")
    except ImportError as e:
        print(f"✗ openpi import failed: {e}")
        dependencies_ok = False
    
    # If core dependencies are missing, run setup
    if not dependencies_ok:
        print("\n" + "="*60)
        print("INSTALLING MISSING DEPENDENCIES")
        print("="*60)
        print("Running setup.sh to install roboeval, lerobot, and openpi...")
        
        import subprocess
        import os
        
        setup_path = os.path.join(os.path.dirname(__file__), "setup.sh")
        result = subprocess.run(
            ["bash", setup_path],
            cwd=os.path.dirname(__file__),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Setup script failed with return code {result.returncode}")
            raise RuntimeError("Setup script failed to install dependencies")
        
        print("\n" + "="*60)
        print("SETUP COMPLETE - Verifying installations...")
        print("="*60)
        
        # Verify installations
        try:
            import roboeval
            print("✓ roboeval installed successfully")
        except ImportError as e:
            print(f"✗ roboeval still not available: {e}")
            
        try:
            import lerobot
            print("✓ lerobot installed successfully")
        except ImportError as e:
            print(f"✗ lerobot still not available: {e}")
            
        try:
            import openpi
            print("✓ openpi installed successfully")
        except ImportError as e:
            print(f"✗ openpi still not available: {e}")
    
    return True

import datetime
print(f"===== Application Startup at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
check_and_install_dependencies()

# --- OpenPI (local inference) ---
try:
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    OPENPI_AVAILABLE = True
    print("OpenPI imported successfully")
except ImportError as e:
    print(f"Error: OpenPI import failed after installation: {e}")
    
    # All dependencies should be in requirements.txt now
    print(f"OpenPI import failed. Check that all dependencies are properly installed.")
    OPENPI_AVAILABLE = False

# --- RoboEval imports ---
from roboeval.action_modes import JointPositionActionMode
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX

# Import all environment classes
from roboeval.envs.manipulation import (
    CubeHandover, CubeHandoverOrientation, CubeHandoverPosition,
    CubeHandoverPositionAndOrientation, VerticalCubeHandover,
    StackTwoBlocks, StackTwoBlocksOrientation, StackTwoBlocksPosition,
    StackTwoBlocksPositionAndOrientation
)
from roboeval.envs.lift_pot import (
    LiftPot, LiftPotOrientation, LiftPotPosition, LiftPotPositionAndOrientation,
)
from roboeval.envs.lift_tray import (
    LiftTray, DragOverAndLiftTray, LiftTrayOrientation, LiftTrayPosition, LiftTrayPositionAndOrientation,
)
from roboeval.envs.pack_objects import (
    PackBox, PackBoxOrientation, PackBoxPosition, PackBoxPositionAndOrientation,
)
from roboeval.envs.stack_books import (
    PickSingleBookFromTable, PickSingleBookFromTableOrientation,
    PickSingleBookFromTablePosition, PickSingleBookFromTablePositionAndOrientation,
    StackSingleBookShelf, StackSingleBookShelfPosition, StackSingleBookShelfPositionAndOrientation,
)
from roboeval.envs.rotate_utility_objects import (
    RotateValve, RotateValveObstacle, RotateValvePosition, RotateValvePositionAndOrientation,
)

# --- Video ---
from moviepy.editor import VideoClip

# ---------------------- Environment Registry ----------------------
_ENV_CLASSES = {
    "CubeHandover": (CubeHandover, "handover the rod from one hand to the other hand"),
    "CubeHandoverOrientation": (CubeHandoverOrientation, "handover the rod from one hand to the other hand"),
    "CubeHandoverPosition": (CubeHandoverPosition, "handover the rod from one hand to the other hand"),
    "CubeHandoverPositionOrientation": (CubeHandoverPositionAndOrientation, "handover the rod from one hand to the other hand"),
    "CubeHandoverVertical": (VerticalCubeHandover, "handover the rod from one hand to the other hand"),

    "LiftPot": (LiftPot, "lift the pot by the handles"),
    "LiftPotOrientation": (LiftPotOrientation, "lift the pot by the handles"),
    "LiftPotPosition": (LiftPotPosition, "lift the pot by the handles"),
    "LiftPotPositionOrientation": (LiftPotPositionAndOrientation, "lift the pot by the handles"),

    "LiftTray": (LiftTray, "lift the tray"),
    "LiftTrayDrag": (DragOverAndLiftTray, "lift the tray"),
    "LiftTrayOrientation": (LiftTrayOrientation, "lift the tray"),
    "LiftTrayPosition": (LiftTrayPosition, "lift the tray"),
    "LiftTrayPositionOrientation": (LiftTrayPositionAndOrientation, "lift the tray"),

    "PackBox": (PackBox, "close the box"),
    "PackBoxOrientation": (PackBoxOrientation, "close the box"),
    "PackBoxPosition": (PackBoxPosition, "close the box"),
    "PackBoxPositionOrientation": (PackBoxPositionAndOrientation, "close the box"),

    "PickSingleBookFromTable": (PickSingleBookFromTable, "pick up the book from the table"),
    "PickSingleBookFromTableOrientation": (PickSingleBookFromTableOrientation, "pick up the book from the table"),
    "PickSingleBookFromTablePosition": (PickSingleBookFromTablePosition, "pick up the book from the table"),
    "PickSingleBookFromTablePositionOrientation": (PickSingleBookFromTablePositionAndOrientation, "pick up the book from the table"),

    "RotateValve": (RotateValve, "rotate the valve counter clockwise"),
    "RotateValveObstacle": (RotateValveObstacle, "rotate the valve counter clockwise"),
    "RotateValvePosition": (RotateValvePosition, "rotate the valve counter clockwise"),
    "RotateValvePositionOrientation": (RotateValvePositionAndOrientation, "rotate the valve counter clockwise"),

    "StackSingleBookShelf": (StackSingleBookShelf, "put the book on the table onto the shelf"),
    "StackSingleBookShelfPosition": (StackSingleBookShelfPosition, "put the book on the table onto the shelf"),
    "StackSingleBookShelfPositionOrientation": (StackSingleBookShelfPositionAndOrientation, "put the book on the table onto the shelf"),

    "StackTwoBlocks": (StackTwoBlocks, "stack the two cubes"),
    "StackTwoBlocksOrientation": (StackTwoBlocksOrientation, "stack the two cubes"),
    "StackTwoBlocksPosition": (StackTwoBlocksPosition, "stack the two cubes"),
    "StackTwoBlocksPositionOrientation": (StackTwoBlocksPositionAndOrientation, "stack the two cubes")
}

# ---------------------- Configuration ----------------------
DEFAULT_DEVICE = "cuda:0" if os.path.exists("/dev/nvidia0") else "cpu"
DEFAULT_DOWNSAMPLE_RATE = 25
DEFAULT_MAX_STEPS = 200
DEFAULT_FPS = 25

# Check GPU availability and print diagnostics
def check_gpu_status():
    """Check and print GPU availability."""
    import jax
    print("\n" + "="*60)
    print("GPU DIAGNOSTICS")
    print("="*60)
    
    # Check JAX devices
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX default backend: {jax.default_backend()}")
    
    # Check if GPU is available
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    if gpu_devices:
        print(f"✅ GPU available! Found {len(gpu_devices)} GPU device(s)")
        for i, device in enumerate(gpu_devices):
            print(f"   GPU {i}: {device}")
    else:
        print(f"❌ No GPU found. Running on: {jax.default_backend()}")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("❌ PyTorch CUDA not available")
    except Exception as e:
        print(f"⚠️  Could not check PyTorch CUDA: {e}")
    
    print("="*60 + "\n")
    return len(gpu_devices) > 0

# Run GPU check at startup
GPU_AVAILABLE = check_gpu_status()

# Global policy cache to avoid reloading
_POLICY_CACHE = {}

def clear_gpu_memory():
    """Clear GPU memory and policy cache."""
    global _POLICY_CACHE
    
    # Clear the policy cache
    _POLICY_CACHE.clear()
    
    # Force JAX to clear GPU memory
    try:
        import jax
        import gc
        
        # Clear JAX compilation caches (for JAX >= 0.4.36)
        try:
            jax.clear_caches()
        except AttributeError:
            # Fallback for older JAX versions
            try:
                jax.clear_backends()
            except AttributeError:
                pass  # Neither method available, rely on gc
        
        # Force Python garbage collection
        gc.collect()
        
        print("GPU memory cleared successfully")
    except Exception as e:
        print(f"Warning: Could not fully clear GPU memory: {e}")

# ---------------------- OpenPI Helpers ----------------------
def get_checkpoint_path(task_name: str, ckpt_path: Optional[str] = None) -> str:
    """
    Return a local path to the checkpoint for the given task. If `ckpt_path` is provided,
    it is returned verbatim. Otherwise, download only the files under:
        repo: tan7271/pi0_base_checkpoints (model repo)
        subdir: {task_name}_testing/{step}/
    We prefer step=2999 if present, else the numerically largest available step.

    This version avoids snapshot_download entirely to dodge the "0 files" issue.
    """
    if ckpt_path:
        return ckpt_path

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "tan7271/pi0_base_checkpoints"
    revision = "main"
    base_dir = f"{task_name}_testing"
    cache_dir = os.path.expanduser("~/.cache/roboeval/pi0_checkpoints")

    api = HfApi()
    try:
        all_files: List[str] = api.list_repo_files(
            repo_id=repo_id, revision=revision, repo_type="model"
        )
    except Exception as e:
        raise RuntimeError(f"Could not list files for {repo_id}@{revision}: {e}")

    # Find available numeric steps under {task}_testing/
    steps = sorted({
        int(p.split("/")[1])
        for p in all_files
        if p.startswith(base_dir + "/") and len(p.split("/")) >= 3 and p.split("/")[1].isdigit()
    })
    if not steps:
        nearby = [p for p in all_files if base_dir in p][:10]
        raise FileNotFoundError(
            f"No files found under '{base_dir}/' in {repo_id}@{revision}. "
            f"Example paths I do see: {nearby}"
        )

    chosen_step = 2999 if 2999 in steps else steps[-1]
    subdir = f"{base_dir}/{chosen_step}"

    print(
        f"Downloading checkpoint for {task_name} directly via hf_hub_download "
        f"(repo={repo_id}, subdir={subdir})..."
    )

    # We only need these parts; if you want rollouts, drop the filter below.
    needed_roots = (
        f"{subdir}/_CHECKPOINT_METADATA",
        f"{subdir}/assets/",
        f"{subdir}/params/",
        f"{subdir}/train_state/",
    )
    wanted = [
        p for p in all_files
        if p == f"{subdir}/_CHECKPOINT_METADATA"
        or any(p.startswith(root) for root in needed_roots[1:])
    ]

    # If the filtered list is empty (unexpected), grab the entire subdir.
    if not wanted:
        wanted = [p for p in all_files if p.startswith(subdir + "/")]
        if not wanted:
            raise FileNotFoundError(
                f"Repo listing shows no files under '{subdir}/'. "
                f"Steps seen: {steps}"
            )

    manual_root = os.path.join(cache_dir, "manual")
    os.makedirs(manual_root, exist_ok=True)

    # Download every file we want into a local mirror of the repo layout.
    for relpath in wanted:
        hf_hub_download(
            repo_id=repo_id,
            filename=relpath,
            revision=revision,
            repo_type="model",
            local_dir=manual_root,
            local_dir_use_symlinks=True,  # saves space on shared filesystems
        )

    manual_ckpt_dir = os.path.join(manual_root, subdir)

    # Basic sanity: ensure the directory exists and isn't empty
    def _nonempty_dir(path: str) -> bool:
        return os.path.isdir(path) and any(True for _ in os.scandir(path))

    if not _nonempty_dir(manual_ckpt_dir):
        try:
            siblings = [e.name for e in os.scandir(os.path.dirname(manual_ckpt_dir))]
        except Exception:
            siblings = []
        raise FileNotFoundError(
            f"Downloaded files, but '{manual_ckpt_dir}' is missing/empty.\n"
            f"Siblings present: {siblings}\n"
            f"(repo_id={repo_id}, subdir={subdir})"
        )

    return manual_ckpt_dir


def load_pi0_base_bimanual_droid(task_name: str, ckpt_path: str):
    """Load Pi0 policy model for the given task."""
    if not OPENPI_AVAILABLE:
        raise RuntimeError("OpenPI is not available. Cannot load Pi0 model.")
    
    # Get checkpoint path (download from HF if needed)
    checkpoint_path = get_checkpoint_path(task_name, ckpt_path)
    
    cache_key = f"{task_name}:{checkpoint_path}"
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]
    
    # Clear old policies from cache to free GPU memory for new task
    if len(_POLICY_CACHE) > 0:
        print(f"Clearing {len(_POLICY_CACHE)} cached model(s) to free GPU memory...")
        clear_gpu_memory()
    
    cfg = _config.get_config("pi0_base_bimanual_droid_finetune")
    bimanual_assets = _config.AssetsConfig(
        assets_dir=f"{checkpoint_path}/assets/",
        asset_id=f"tan7271/{task_name}",
    )
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, assets=bimanual_assets))
    policy = _policy_config.create_trained_policy(cfg, checkpoint_path)
    
    _POLICY_CACHE[cache_key] = policy
    return policy


def make_openpi_example_from_roboeval(obs_dict: dict, prompt: str) -> dict:
    """Convert RoboEval observation to OpenPI format."""
    obs = obs_dict[0] if isinstance(obs_dict, (tuple, list)) else obs_dict
    example = {"prompt": prompt}

    # Cameras (CHW→HWC)
    exterior_chw = obs["rgb_head"]
    left_wrist_chw = obs["rgb_left_wrist"]
    right_wrist_chw = obs["rgb_right_wrist"]
    example["observation/exterior_image_1_left"] = np.moveaxis(exterior_chw, 0, -1)
    example["observation/wrist_image_left"] = np.moveaxis(left_wrist_chw, 0, -1)
    example["observation/wrist_image_right"] = np.moveaxis(right_wrist_chw, 0, -1)

    # Joints and grippers
    prop = np.asarray(obs["proprioception"], dtype=np.float32).reshape(-1)
    example["observation/joint_position"] = prop

    grip = np.asarray(obs["proprioception_grippers"], dtype=np.float32).reshape(-1)[:2]
    example["observation/gripper_position"] = grip

    return example


def map_policy_action_to_env_abs(action_vec: np.ndarray, env) -> np.ndarray:
    """Map policy action to environment action."""
    a = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if a.shape[0] != 16:
        raise ValueError(f"Expected (16,), got {a.shape}.")
    return a


def _clip_to_space(env, action: np.ndarray) -> np.ndarray:
    """Safety: clip to env action space."""
    return np.clip(action, env.action_space.low, env.action_space.high)


@dataclasses.dataclass
class InferenceRequest:
    """Normalized payload for invoking model backends from the UI."""
    task_name: str
    checkpoint_path: str
    custom_instruction: Optional[str]
    max_steps: int
    fps: int
    progress: gr.Progress


@dataclasses.dataclass
class ModelDefinition:
    """Metadata and execution hook for a model option."""
    label: str
    description: str
    run_inference: Callable[[InferenceRequest], Tuple[Optional[str], str]]


# ---------------------- Video Helpers ----------------------
def save_frames_to_video(frames, output_path: str, fps: int = 25) -> str:
    """Save rollout frames to a video file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert frames to proper format
    frames = np.array(frames)
    if frames.ndim == 4 and frames.shape[1] == 3:  # (T, C, H, W)
        frames = np.moveaxis(frames, 1, -1)  # → (T, H, W, C)
    
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.write_videofile(output_path, fps=fps, codec="libx264", logger=None)
    
    return output_path


# ---------------------- RoboEval Environment Setup ----------------------
def setup_env(task_name: str, downsample_rate: int = 25):
    """Setup RoboEval environment for the given task."""
    cameras = [
        CameraConfig(name="head", rgb=True, depth=False, resolution=(256, 256)),
        CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=(256, 256)),
        CameraConfig(name="right_wrist", rgb=True, depth=False, resolution=(256, 256)),
    ]
    
    Env, prompt = _ENV_CLASSES[task_name]
    env = Env(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=False,
            block_until_reached=False,
            ee=False,
            floating_dofs=[],
        ),
        observation_config=ObservationConfig(cameras=cameras, proprioception=True),
        render_mode="rgb_array",
        robot_cls=BimanualPanda,
        control_frequency=CONTROL_FREQUENCY_MAX // downsample_rate,
    )
    
    return env, prompt


def _unpack_obs(obs_or_tuple):
    """Unpack observation if it's a tuple."""
    return obs_or_tuple[0] if isinstance(obs_or_tuple, (tuple, list)) else obs_or_tuple


# ---------------------- Inference Loop ----------------------
def run_inference_loop(
    env,
    policy,
    instruction: str,
    max_steps: int = 200,
    open_loop_horizon: int = 10,
):
    """Run inference loop for one episode."""
    obs = env.reset()
    successes = 0
    images_env = []
    chunk = None
    i_in_chunk = 0

    for step_idx in range(max_steps):
        cur_obs = _unpack_obs(obs)
        
        # Collect environment render
        images_env.append(copy.deepcopy(env.render()))
        
        # Request new action chunk when needed
        if chunk is None or i_in_chunk >= open_loop_horizon:
            example = make_openpi_example_from_roboeval(cur_obs, instruction)
            out = policy.infer(example)
            chunk = out["actions"]
            i_in_chunk = 0

        # Take next action from cached chunk
        a_vec = chunk[i_in_chunk]
        i_in_chunk += 1

        env_action = map_policy_action_to_env_abs(a_vec, env)
        env_action = _clip_to_space(env, env_action)

        obs, reward, terminated, truncated, info = env.step(env_action)
        successes += int(reward > 0)

        if terminated or truncated:
            break

    stats = {"steps": step_idx + 1, "success_signal": successes}
    return stats, images_env


# ---------------------- Main Inference Function ----------------------
def run_pi0_inference(request: InferenceRequest) -> Tuple[Optional[str], str]:
    """
    Main function to run Pi0 inference.
    
    Returns:
        Tuple of (video_path, status_message)
    """
    try:
        task_name = request.task_name
        checkpoint_path = request.checkpoint_path
        custom_instruction = request.custom_instruction
        max_steps = int(request.max_steps)
        fps = int(request.fps)
        progress = request.progress

        progress(0, desc="Loading model and environment...")
        
        # Check GPU status
        import jax
        gpu_info = ""
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            gpu_info = f"🎮 **GPU**: {len(gpu_devices)} GPU(s) detected - {gpu_devices[0]}\n"
        else:
            gpu_info = f"⚠️ **GPU**: Not detected! Running on {jax.default_backend()}\n"
        
        # Check if OpenPI is available
        if not OPENPI_AVAILABLE:
            return None, gpu_info + f"❌ **OpenPI not available**\n\nOpenPI is required for Pi0 model inference but is not installed. Please check the build logs for installation errors."
        
        # Validate task
        if task_name not in _ENV_CLASSES:
            return None, f"❌ Unknown task: {task_name}"
        
        # Load policy
        progress(0.2, desc="Loading Pi0 policy...")
        policy = load_pi0_base_bimanual_droid(task_name, checkpoint_path)
        
        # Setup environment
        progress(0.4, desc="Setting up environment...")
        env, default_prompt = setup_env(task_name, downsample_rate=DEFAULT_DOWNSAMPLE_RATE)
        instruction = custom_instruction if custom_instruction else default_prompt
        
        # Run inference
        progress(0.5, desc="Running inference...")
        stats, images_env = run_inference_loop(
            env, policy, instruction, max_steps=max_steps
        )
        
        # Save video
        progress(0.8, desc="Saving video...")
        video_path = os.path.join(tempfile.gettempdir(), f"pi0_rollout_{task_name}.mp4")
        save_frames_to_video(images_env, video_path, fps=fps)
        
        # Cleanup
        env.close()
        
        # Clear GPU memory after inference to prevent OOM on next run
        import gc
        gc.collect()
        
        progress(1.0, desc="Complete!")
        
        status = gpu_info + f"✅ **Inference Complete!**\n\n"
        status += f"- **Task**: {task_name}\n"
        status += f"- **Steps**: {stats['steps']}\n"
        status += f"- **Success Signal**: {stats['success_signal']}\n"
        status += f"- **Instruction**: {instruction}\n"
        
        return video_path, status
        
    except Exception as e:
        import traceback
        
        # Check if it's an out of memory error
        if "Out of memory" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            error_msg = f"""❌ **Out of Memory Error**

The model is too large for the current hardware configuration.

**Pi0 Model Requirements:**
- Minimum: 8 GB GPU memory
- Recommended: 16+ GB GPU memory

**Solutions:**
1. **Upgrade this Space to use a GPU** (Settings → Hardware → T4 small or better)
2. Use a smaller/quantized checkpoint
3. Contact the Space owner to enable GPU hardware

**Technical Details:**
```
{str(e)}
```
"""
        else:
            error_msg = f"❌ **Error during inference:**\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
        
        return None, error_msg


def run_openvla_inference(request: InferenceRequest) -> Tuple[Optional[str], str]:
    """
    Placeholder for OpenVLA backend integration.
    
    Currently returns a descriptive message until the OpenVLA runtime is wired up.
    """
    status = (
        "⚠️ **OpenVLA integration is not yet available in this Space.**\n\n"
        "The frontend is model-aware, so you can wire in the backend by implementing "
        "`run_openvla_inference` to load checkpoints and execute rollouts.\n\n"
        "Requested configuration:\n"
        f"- Task: {request.task_name}\n"
        f"- Checkpoint: {request.checkpoint_path or 'auto'}\n"
        f"- Steps: {request.max_steps}\n"
        f"- FPS: {request.fps}\n"
    )
    return None, status


# Registry of supported models (UI order follows this definition)
MODEL_REGISTRY: Dict[str, ModelDefinition] = {
    "pi0_openpi": ModelDefinition(
        label="Pi0 Base (OpenPI)",
        description=(
            "Runs the Pi0 bimanual policy using the OpenPI runtime. "
            "Supports all RoboEval tasks and will automatically fetch checkpoints from "
            "`tan7271/pi0_base_checkpoints` when no custom path is provided."
        ),
        run_inference=run_pi0_inference,
    ),
    "openvla": ModelDefinition(
        label="OpenVLA",
        description=(
            "Runs the OpenVLA (Open Vision-Language-Action) policy. "
            "OpenVLA is a vision-language-action model for robot manipulation tasks. "
            "Provide a checkpoint path or leave empty to use default OpenVLA checkpoints."
        ),
        run_inference=run_openvla_inference,
    ),
}


def _format_model_info(model_key: str) -> str:
    model = MODEL_REGISTRY.get(model_key)
    if not model:
        return f"❓ Unknown model selection: `{model_key}`"
    return f"**{model.label}**\n\n{model.description}"


def run_model_inference(
    model_key: str,
    task_name: str,
    checkpoint_path: str,
    custom_instruction: Optional[str],
    max_steps: float,
    fps: float,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str]:
    """Dispatch inference based on the selected model."""
    model = MODEL_REGISTRY.get(model_key)
    if not model:
        return None, f"❌ Unknown model selection: {model_key}"

    request = InferenceRequest(
        task_name=task_name,
        checkpoint_path=checkpoint_path or "",
        custom_instruction=custom_instruction or None,
        max_steps=int(max_steps),
        fps=int(fps),
        progress=progress,
    )

    return model.run_inference(request)


# Helper for reactive UI updates
def update_model_description(model_key: str) -> str:
    return _format_model_info(model_key)


# ---------------------- Gradio Interface ----------------------
def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Robot Policy Inference on RoboEval Tasks") as demo:
        gr.Markdown("""
        # 🤖 Robot Policy Inference on RoboEval Tasks
        
        Choose a supported model backend (starting with Pi0 via OpenPI) and run it on RoboEval tasks to watch the generated execution video.
        
        ⚠️ **Hardware Requirements:** This Space requires a GPU with at least 8GB memory.
        If you see "Out of Memory" errors, upgrade the Space hardware in Settings → Hardware → T4 small.
        
        **Note**: Leave the checkpoint path empty to use the model's default retrieval logic (Pi0 fetches from `tan7271/pi0_base_checkpoints`), or provide a custom local path.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                default_model_key = next(iter(MODEL_REGISTRY.keys()))
                model_dropdown = gr.Dropdown(
                    choices=[(definition.label, key) for key, definition in MODEL_REGISTRY.items()],
                    value=default_model_key,
                    label="Select Model",
                    info="Choose which policy backend to use"
                )

                model_info = gr.Markdown(_format_model_info(default_model_key))
                
                task_dropdown = gr.Dropdown(
                    choices=sorted(_ENV_CLASSES.keys()),
                    label="Select Task",
                    value="LiftPot",
                    info="Choose the robot manipulation task"
                )
                
                checkpoint_input = gr.Textbox(
                    label="Checkpoint Path (Optional)",
                    placeholder="Leave empty to auto-download from HF Hub",
                    value="",
                    info="Optional: Provide a custom checkpoint path. Leave empty to use the selected model's default."
                )
                
                instruction_input = gr.Textbox(
                    label="Custom Instruction (Optional)",
                    placeholder="Leave empty to use default task instruction",
                    value="",
                    info="Override the default task instruction"
                )
                
                with gr.Row():
                    max_steps_input = gr.Number(
                        label="Max Steps",
                        value=DEFAULT_MAX_STEPS,
                        precision=0,
                        minimum=10,
                        maximum=1000
                    )
                    
                    fps_input = gr.Number(
                        label="Video FPS",
                        value=DEFAULT_FPS,
                        precision=0,
                        minimum=1,
                        maximum=30
                    )
                
                run_button = gr.Button("🚀 Run Inference", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                status_output = gr.Markdown("*Click 'Run Inference' to start...*")
                video_output = gr.Video(label="Execution Video", interactive=False)
        
        model_dropdown.change(
            fn=update_model_description,
            inputs=model_dropdown,
            outputs=model_info,
            queue=False,
        )

        # Event handler
        run_button.click(
            fn=run_model_inference,
            inputs=[
                model_dropdown,
                task_dropdown,
                checkpoint_input,
                instruction_input,
                max_steps_input,
                fps_input
            ],
            outputs=[video_output, status_output]
        )
        
        gr.Markdown("""
        ---
        ### Available Tasks
        
        The dropdown includes all RoboEval tasks such as:
        - **Cube Handover**: Transfer a rod between robot hands
        - **Lift Pot**: Bimanually lift a pot by its handles
        - **Lift Tray**: Lift and balance a tray
        - **Pack Box**: Close a box with objects inside
        - **Rotate Valve**: Turn a valve counter-clockwise
        - **Stack Books**: Manipulate books on tables and shelves
        - And many more variations with position/orientation constraints
        
        ### GPU Usage
        
        This Space uses a T4 GPU (~$0.60/hour). It auto-sleeps after 10 minutes of inactivity to minimize costs.
        """)
    
    return demo


# ---------------------- Launch ----------------------
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()

