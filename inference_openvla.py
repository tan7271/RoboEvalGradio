"""
OpenVLA Inference Worker - Runs in openvla_env
Receives inference requests via stdin, returns results to stdout
"""
import sys
import json
import os
import tempfile
import copy
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Print early startup message so parent knows we're starting
print("===== OpenVLA Worker: Starting up... =====", file=sys.stderr, flush=True)

# Set headless mode
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

# Verify critical dependencies are available
# Redirect stdout during imports to prevent libraries from printing to stdout
import io
from contextlib import redirect_stdout

print("===== Checking OpenVLA Environment Dependencies =====", file=sys.stderr, flush=True)

# Capture stdout during imports to prevent interference with JSON protocol
stdout_capture = io.StringIO()

try:
    try:
        with redirect_stdout(stdout_capture):
            import torch
        print(f"✓ torch imported successfully (version: {torch.__version__})", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ torch import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"✗ torch import error: {e}", file=sys.stderr, flush=True)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        with redirect_stdout(stdout_capture):
            from PIL import Image
        print("✓ pillow imported successfully", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ pillow import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        with redirect_stdout(stdout_capture):
            import transformers
        print(f"✓ transformers imported successfully (version: {transformers.__version__})", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ transformers import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        with redirect_stdout(stdout_capture):
            from transformers import (
                AutoConfig,
                AutoImageProcessor,
                AutoModelForVision2Seq,
                AutoProcessor,
            )
        print("✓ transformers components imported successfully", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ transformers components import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        with redirect_stdout(stdout_capture):
            import roboeval
        print("✓ roboeval imported successfully", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ roboeval import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Import OpenVLA dependencies (only available in openvla_env)
    try:
        with redirect_stdout(stdout_capture):
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        print("✓ OpenVLA (prismatic) modules imported successfully", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ OpenVLA (prismatic) import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        with redirect_stdout(stdout_capture):
            from roboeval.action_modes import JointPositionActionMode
            from roboeval.utils.observation_config import CameraConfig, ObservationConfig
            from roboeval.robots.configs.panda import BimanualPanda
            from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX
        print("✓ RoboEval core modules imported successfully", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ RoboEval core modules import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Import all environment classes
    try:
        with redirect_stdout(stdout_capture):
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
        with redirect_stdout(stdout_capture):
            from moviepy.editor import VideoClip
        print("✓ moviepy imported successfully", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"✗ moviepy import failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Check if anything was printed to stdout during imports
    captured_import_output = stdout_capture.getvalue()
    if captured_import_output:
        print(f"⚠️  WARNING: Libraries printed to stdout during import: {repr(captured_import_output[:500])}", file=sys.stderr, flush=True)

    # Ensure stdout is properly restored (should be automatic, but let's be explicit)
    # Clear the capture buffer to free memory
    stdout_capture.close()
    del stdout_capture

    print("===== All dependencies verified successfully =====", file=sys.stderr, flush=True)
    print("===== OpenVLA Worker: Imports complete =====", file=sys.stderr, flush=True)
except Exception as e:
    import traceback
    print(f"✗ Fatal error during dependency imports: {e}", file=sys.stderr, flush=True)
    print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
    sys.exit(1)

# Configuration constants
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_DOWNSAMPLE_RATE = 25
CAMERA_RESOLUTION = (256, 256)

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

# Global model cache
_MODEL_CACHE = {}


def get_checkpoint_path(task_name: str, ckpt_path: Optional[str] = None) -> str:
    """
    Return a local path to the OpenVLA checkpoint.
    Downloads from Hugging Face Hub if not provided.
    """
    if ckpt_path:
        return ckpt_path

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "tan7271/OpenVLA_models"
    revision = "main"
    # Hardcoded subdirectory as requested
    subdir = "openvla-7b+handover_cube_static_delta+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    cache_dir = os.path.expanduser("~/.cache/roboeval/openvla_checkpoints")

    api = HfApi()
    try:
        all_files: List[str] = api.list_repo_files(
            repo_id=repo_id, revision=revision, repo_type="model"
        )
    except Exception as e:
        raise RuntimeError(f"Could not list files for {repo_id}@{revision}: {e}")

    # Find all files under the subdirectory
    wanted = [p for p in all_files if p.startswith(subdir + "/")]

    if not wanted:
        # List some example paths for debugging
        example_paths = [p for p in all_files if "/" in p][:10]
        raise FileNotFoundError(
            f"No files found under '{subdir}/' in {repo_id}@{revision}. "
            f"Example paths I do see: {example_paths}"
        )

    manual_root = os.path.join(cache_dir, "manual")
    os.makedirs(manual_root, exist_ok=True)

    # Download all files in the subdirectory
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


def register_openvla() -> None:
    """Register OpenVLA components with the Transformers library."""
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_vla_model(ckpt_path: str, device: str = DEFAULT_DEVICE) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    """
    Load OpenVLA model and processor from checkpoint.
    """
    if ckpt_path in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt_path]
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
    
    stats_path = os.path.join(ckpt_path, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Dataset statistics file not found: {stats_path}")
    
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    with open(stats_path, "r") as f:
        model.norm_stats = json.load(f)

    _MODEL_CACHE[ckpt_path] = (processor, model)
    return processor, model


def setup_env(task_name: str, downsample_rate: int = DEFAULT_DOWNSAMPLE_RATE):
    """Setup RoboEval environment for the given task."""
    cameras = [
        CameraConfig(name="head", rgb=True, depth=False, resolution=CAMERA_RESOLUTION),
        CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=CAMERA_RESOLUTION),
        CameraConfig(name="right_wrist", rgb=True, depth=False, resolution=CAMERA_RESOLUTION),
    ]
    
    Env, prompt = _ENV_CLASSES[task_name]
    env = Env(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            block_until_reached=False,
            ee=True,
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
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    instruction: str,
    device: str = DEFAULT_DEVICE,
    max_steps: int = 200
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """
    Run the agent in a closed loop using OpenVLA predictions.
    
    Returns:
        Tuple of (stats dict, list of RGB frames)
    """
    obs = env.reset()
    images = []
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    successes = 0

    for step_idx in range(max_steps):
        # Collect observation image
        cur_obs = _unpack_obs(obs)
        images.append(copy.deepcopy(cur_obs["rgb_head"]))
        
        # Get action from model
        image = Image.fromarray(np.moveaxis(images[-1], 0, -1))
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        action = model.predict_action(**inputs, do_sample=False)

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        successes += int(reward > 0)
        
        if terminated or truncated:
            break

    stats = {"steps": step_idx + 1, "success_signal": successes}
    return stats, images


def save_frames_to_video(frames: List[np.ndarray], output_path: str, fps: int = 25) -> str:
    """Save rollout frames to a video file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    frames = np.moveaxis(np.array(frames), 1, -1)  # (T, C, H, W) → (T, H, W, C)
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.write_videofile(output_path, fps=fps, codec="libx264", logger=None)
    
    return output_path


def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run OpenVLA inference based on request parameters.
    
    Args:
        request: Dictionary with keys:
            - task_name: str
            - checkpoint_path: str (required)
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
        checkpoint_path = request.get("checkpoint_path")
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
        
        # Get checkpoint path (downloads from Hugging Face Hub if not provided)
        checkpoint_path = get_checkpoint_path(task_name, checkpoint_path)
        
        # Register OpenVLA components
        register_openvla()
        
        # Load model
        device = DEFAULT_DEVICE
        processor, model = load_vla_model(checkpoint_path, device)
        
        # Setup environment
        env, default_prompt = setup_env(task_name, downsample_rate=DEFAULT_DOWNSAMPLE_RATE)
        instruction = custom_instruction if custom_instruction else default_prompt
        
        # Run inference
        stats, images = run_inference_loop(
            env=env,
            processor=processor,
            model=model,
            instruction=instruction,
            device=device,
            max_steps=max_steps
        )
        
        # Save video
        video_path = os.path.join(tempfile.gettempdir(), f"openvla_rollout_{task_name}_{os.getpid()}.mp4")
        save_frames_to_video(images, video_path, fps=fps)
        
        # Cleanup
        env.close()
        
        status = f"✅ **OpenVLA Inference Complete!**\n\n"
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
        error_msg = f"❌ **Error during OpenVLA inference:**\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
        return {
            "success": False,
            "video_path": None,
            "status_message": error_msg,
            "error": str(e)
        }


def main():
    """Main loop: read requests from stdin, write results to stdout"""
    # Print startup message to stderr so parent process knows we're ready
    print("===== OpenVLA Worker Ready =====", file=sys.stderr, flush=True)
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
                print("===== OpenVLA Worker: stdin closed (EOF), exiting =====", file=sys.stderr, flush=True)
                break
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Debug: log what we received
            print(f"DEBUG: Received line: {repr(line[:200])}", file=sys.stderr, flush=True)
            
            try:
                request = json.loads(line.strip())
            except json.JSONDecodeError as e:
                # Invalid JSON request - send error response
                print(f"DEBUG: JSON decode error. Line content: {repr(line[:500])}", file=sys.stderr, flush=True)
                error_result = {
                    "success": False,
                    "video_path": None,
                    "status_message": f"❌ Invalid JSON request: {str(e)}",
                    "error": f"JSON decode error: {str(e)}. Received: {line[:200]}"
                }
                print(json.dumps(error_result), flush=True)
                continue
            
            try:
                task_name = request.get('task_name', 'unknown')
                print(f"DEBUG: Starting inference for task: {task_name}", file=sys.stderr, flush=True)
                
                # Redirect stdout temporarily to capture any output from the environment
                # This prevents environment output from interfering with our JSON protocol
                import io
                from contextlib import redirect_stdout
                
                # Capture any stdout from the environment during inference
                stdout_capture = io.StringIO()
                try:
                    with redirect_stdout(stdout_capture):
                        result = run_inference(request)
                finally:
                    # Check if anything was printed to stdout (this shouldn't happen, but log it if it does)
                    captured_output = stdout_capture.getvalue()
                    if captured_output:
                        print(f"⚠️  WARNING: Environment printed to stdout during inference: {repr(captured_output[:500])}", file=sys.stderr, flush=True)
                        # If we captured output, it means something printed to stdout
                        # This could interfere with our JSON protocol, so log it
                
                print(f"DEBUG: Inference completed for {task_name}, sending result", file=sys.stderr, flush=True)
                result_json = json.dumps(result)
                print(result_json, flush=True)
                print(f"DEBUG: Result sent successfully for {task_name}", file=sys.stderr, flush=True)
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
                error_json = json.dumps(error_result)
                print(error_json, flush=True)
                print(f"DEBUG: Error result sent successfully", file=sys.stderr, flush=True)
            
        except KeyboardInterrupt:
            print("===== OpenVLA Worker: interrupted =====", file=sys.stderr, flush=True)
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
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"Fatal error in OpenVLA worker: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr, flush=True)
        # Try to send error response if possible
        try:
            error_result = {
                "success": False,
                "video_path": None,
                "status_message": f"❌ Worker fatal error: {str(e)}",
                "error": str(e)
            }
            print(json.dumps(error_result), flush=True)
        except:
            pass  # If we can't send JSON, at least stderr was logged
        sys.exit(1)

