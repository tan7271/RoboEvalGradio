"""
OpenPI Inference Worker - Runs in openpi_env
Receives inference requests via stdin, returns results via stdout
"""
import sys
import json
import os
import tempfile
import copy
import numpy as np
import dataclasses
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set headless mode
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

# Verify critical dependencies are available
print("===== Checking OpenPI Environment Dependencies =====", file=sys.stderr, flush=True)

try:
    import roboeval
    print("✓ roboeval imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✗ roboeval import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    import lerobot
    print("✓ lerobot imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✗ lerobot import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    import openpi
    print("✓ openpi imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✗ openpi import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# Import OpenPI dependencies (only available in openpi_env)
try:
    # Import modules (same pattern as old app.py that worked)
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    print("✓ OpenPI config and policy modules imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✗ OpenPI config/policy import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    from roboeval.action_modes import JointPositionActionMode
    from roboeval.utils.observation_config import ObservationConfig, CameraConfig
    from roboeval.robots.configs.panda import BimanualPanda
    from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX
    print("✓ RoboEval core modules imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✗ RoboEval core modules import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# Import all environment classes
try:
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
    print("✓ RoboEval environment classes imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✗ RoboEval environment classes import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# Video
try:
    from moviepy.editor import VideoClip
    print("✓ moviepy imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✗ moviepy import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

print("===== All dependencies verified successfully =====", file=sys.stderr, flush=True)

# Environment registry
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

DEFAULT_DOWNSAMPLE_RATE = 25

# Global policy cache
_POLICY_CACHE = {}


def get_checkpoint_path(task_name: str, ckpt_path: Optional[str] = None) -> str:
    """
    Return a local path to the checkpoint for the given task.
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

    if not wanted:
        wanted = [p for p in all_files if p.startswith(subdir + "/")]
        if not wanted:
            raise FileNotFoundError(
                f"Repo listing shows no files under '{subdir}/'. "
                f"Steps seen: {steps}"
            )

    manual_root = os.path.join(cache_dir, "manual")
    os.makedirs(manual_root, exist_ok=True)

    for relpath in wanted:
        hf_hub_download(
            repo_id=repo_id,
            filename=relpath,
            revision=revision,
            repo_type="model",
            local_dir=manual_root,
            local_dir_use_symlinks=True,
        )

    manual_ckpt_dir = os.path.join(manual_root, subdir)

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


def clear_gpu_memory():
    """Clear JAX GPU memory and policy cache."""
    global _POLICY_CACHE
    
    # Clear the policy cache
    _POLICY_CACHE.clear()
    
    # Force JAX to clear GPU memory
    try:
        import jax
        import gc
        
        # Clear JAX compilation caches
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
        
        print("GPU memory cleared successfully", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Warning: Could not fully clear GPU memory: {e}", file=sys.stderr, flush=True)


def load_pi0_policy(task_name: str, ckpt_path: str):
    """Load Pi0 policy model for the given task."""
    checkpoint_path = get_checkpoint_path(task_name, ckpt_path)
    
    cache_key = f"{task_name}:{checkpoint_path}"
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]
    
    # Clear old policies from cache to free GPU memory for new task
    if len(_POLICY_CACHE) > 0:
        print(f"Clearing {len(_POLICY_CACHE)} cached model(s) to free GPU memory...", file=sys.stderr, flush=True)
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

    exterior_chw = obs["rgb_head"]
    left_wrist_chw = obs["rgb_left_wrist"]
    right_wrist_chw = obs["rgb_right_wrist"]
    example["observation/exterior_image_1_left"] = np.moveaxis(exterior_chw, 0, -1)
    example["observation/wrist_image_left"] = np.moveaxis(left_wrist_chw, 0, -1)
    example["observation/wrist_image_right"] = np.moveaxis(right_wrist_chw, 0, -1)

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
        
        images_env.append(copy.deepcopy(env.render()))
        
        if chunk is None or i_in_chunk >= open_loop_horizon:
            example = make_openpi_example_from_roboeval(cur_obs, instruction)
            out = policy.infer(example)
            chunk = out["actions"]
            i_in_chunk = 0

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


def save_frames_to_video(frames, output_path: str, fps: int = 25) -> str:
    """Save rollout frames to a video file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    frames = np.array(frames)
    if frames.ndim == 4 and frames.shape[1] == 3:
        frames = np.moveaxis(frames, 1, -1)
    
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.write_videofile(output_path, fps=fps, codec="libx264", logger=None)
    
    return output_path


def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run OpenPI inference based on request parameters.
    
    Args:
        request: Dictionary with keys:
            - task_name: str
            - checkpoint_path: str or None
            - max_steps: int
            - fps: int
            - custom_instruction: str or None
    
    Returns:
        Dictionary with keys:
            - success: bool
            - video_path: str or None
            - status_message: str
            - error: str or None
    """
    try:
        task_name = request["task_name"]
        checkpoint_path = request.get("checkpoint_path") or None
        max_steps = int(request["max_steps"])
        fps = int(request["fps"])
        custom_instruction = request.get("custom_instruction") or None
        
        # Validate task
        if task_name not in _ENV_CLASSES:
            return {
                "success": False,
                "video_path": None,
                "status_message": f"❌ Unknown task: {task_name}",
                "error": f"Unknown task: {task_name}"
            }
        
        # Load policy
        policy = load_pi0_policy(task_name, checkpoint_path or "")
        
        # Setup environment
        env, default_prompt = setup_env(task_name, downsample_rate=DEFAULT_DOWNSAMPLE_RATE)
        instruction = custom_instruction if custom_instruction else default_prompt
        
        # Run inference
        stats, images_env = run_inference_loop(
            env, policy, instruction, max_steps=max_steps
        )
        
        # Save video
        video_path = os.path.join(tempfile.gettempdir(), f"pi0_rollout_{task_name}_{os.getpid()}.mp4")
        save_frames_to_video(images_env, video_path, fps=fps)
        
        # Cleanup
        env.close()
        
        # Clear GPU memory after inference to prevent OOM on next run
        import gc
        gc.collect()
        # Note: We don't clear the policy cache here to allow reuse of the same model
        # The cache will be cleared when loading a different task
        
        status = f"✅ **Inference Complete!**\n\n"
        status += f"- **Task**: {task_name}\n"
        status += f"- **Steps**: {stats['steps']}\n"
        status += f"- **Success Signal**: {stats['success_signal']}\n"
        status += f"- **Instruction**: {instruction}\n"
        
        return {
            "success": True,
            "video_path": video_path,
            "status_message": status,
            "error": None
        }
        
    except Exception as e:
        import traceback
        error_msg = f"❌ **Error during inference:**\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
        return {
            "success": False,
            "video_path": None,
            "status_message": error_msg,
            "error": str(e)
        }


def main():
    """Main loop: read requests from stdin, write results to stdout"""
    # Print startup message to stderr so parent process knows we're ready
    print("===== OpenPI Worker Ready =====", file=sys.stderr, flush=True)
    print("Waiting for inference requests...", file=sys.stderr, flush=True)
    
    # Ensure stdin is in text mode and properly buffered
    if sys.stdin.isatty():
        # If stdin is a TTY, this shouldn't happen in subprocess context
        print("⚠️  Warning: stdin is a TTY, not a pipe", file=sys.stderr, flush=True)
    
    while True:
        try:
            # Read a line from stdin (this will block until data is available or EOF)
            line = sys.stdin.readline()
            if not line:
                # stdin closed (EOF) - exit gracefully
                print("===== OpenPI Worker: stdin closed (EOF), exiting =====", file=sys.stderr, flush=True)
                break
            
            # Skip empty lines
            if not line.strip():
                continue
            
            try:
                request = json.loads(line.strip())
            except json.JSONDecodeError as e:
                # Invalid JSON request - send error response
                error_result = {
                    "success": False,
                    "video_path": None,
                    "status_message": f"❌ Invalid JSON request: {str(e)}",
                    "error": f"JSON decode error: {str(e)}"
                }
                print(json.dumps(error_result), flush=True)
                continue
            
            try:
                result = run_inference(request)
                print(json.dumps(result), flush=True)
            except Exception as e:
                # Error during inference - send error response as JSON
                import traceback
                error_msg = f"Error in worker inference: {str(e)}\n{traceback.format_exc()}"
                print(error_msg, file=sys.stderr, flush=True)
                error_result = {
                    "success": False,
                    "video_path": None,
                    "status_message": f"❌ Worker error: {str(e)}",
                    "error": str(e)
                }
                print(json.dumps(error_result), flush=True)
            
        except KeyboardInterrupt:
            print("===== OpenPI Worker: interrupted =====", file=sys.stderr, flush=True)
            break
        except Exception as e:
            # Fatal error in main loop - try to send error response before exiting
            import traceback
            error_msg = f"Fatal error in worker main loop: {str(e)}\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr, flush=True)
            try:
                error_result = {
                    "success": False,
                    "video_path": None,
                    "status_message": "❌ Worker fatal error",
                    "error": str(e)
                }
                print(json.dumps(error_result), flush=True)
            except:
                pass  # If we can't send JSON, at least stderr was logged


if __name__ == "__main__":
    main()

