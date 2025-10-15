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
from typing import Optional, Tuple
import gradio as gr
import subprocess
import sys

# --- Headless defaults (set BEFORE mujoco/roboeval imports) ---
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

# --- Install RoboEval at runtime ---
def install_roboeval():
    """Install RoboEval from GitHub using the GH_TOKEN."""
    try:
        import roboeval
        print("RoboEval already installed")
        return True
    except ImportError:
        print("Installing RoboEval...")
        gh_token = os.environ.get("GH_TOKEN")
        if not gh_token:
            raise RuntimeError("GH_TOKEN environment variable not set")
        
        repo_url = f"https://{gh_token}@github.com/helen9975/RoboEval.git"
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            f"git+{repo_url}", "--no-cache-dir"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Installation failed: {result.stderr}")
            raise RuntimeError(f"Failed to install RoboEval: {result.stderr}")
        
        print("RoboEval installed successfully")
        return True

# Install RoboEval
install_roboeval()

# --- OpenPI (local inference) ---
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

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
DEFAULT_FPS = 5

# Global policy cache to avoid reloading
_POLICY_CACHE = {}

# ---------------------- OpenPI Helpers ----------------------
def load_pi0_base_bimanual_droid(task_name: str, ckpt_path: str):
    """Load Pi0 policy model for the given task."""
    cache_key = f"{task_name}:{ckpt_path}"
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]
    
    cfg = _config.get_config("pi0_base_bimanual_droid_finetune")
    bimanual_assets = _config.AssetsConfig(
        assets_dir=f"{ckpt_path}/assets/",
        asset_id=f"{task_name}",
    )
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, assets=bimanual_assets))
    policy = _policy_config.create_trained_policy(cfg, ckpt_path)
    
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
def run_pi0_inference(
    task_name: str,
    checkpoint_path: str,
    custom_instruction: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    fps: int = DEFAULT_FPS,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Main function to run Pi0 inference.
    
    Returns:
        Tuple of (video_path, status_message)
    """
    try:
        progress(0, desc="Loading model and environment...")
        
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
        
        progress(1.0, desc="Complete!")
        
        status = f"✅ **Inference Complete!**\n\n"
        status += f"- **Task**: {task_name}\n"
        status += f"- **Steps**: {stats['steps']}\n"
        status += f"- **Success Signal**: {stats['success_signal']}\n"
        status += f"- **Instruction**: {instruction}\n"
        
        return video_path, status
        
    except Exception as e:
        import traceback
        error_msg = f"❌ **Error during inference:**\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
        return None, error_msg


# ---------------------- Gradio Interface ----------------------
def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Pi0 Inference on RoboEval Tasks") as demo:
        gr.Markdown("""
        # 🤖 Pi0 Model Inference on RoboEval Tasks
        
        Run Pi0 bimanual manipulation policy on various robot tasks and view the execution video.
        
        **Note**: This Space uses a private checkpoint. Make sure you've configured access properly.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                
                task_dropdown = gr.Dropdown(
                    choices=sorted(_ENV_CLASSES.keys()),
                    label="Select Task",
                    value="LiftPot",
                    info="Choose the robot manipulation task"
                )
                
                checkpoint_input = gr.Textbox(
                    label="Checkpoint Path",
                    placeholder="/path/to/checkpoint or huggingface-model-id",
                    value="",
                    info="Path to Pi0 model checkpoint directory"
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
        
        # Event handler
        run_button.click(
            fn=run_pi0_inference,
            inputs=[
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

