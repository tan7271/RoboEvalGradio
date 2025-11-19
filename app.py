"""
Hugging Face Space for Robot Policy Inference on RoboEval Tasks

This Gradio app allows users to run model inference (OpenPI, OpenVLA) on bimanual robot tasks
and view the resulting execution videos. Models run in isolated conda environments.
"""

import os
import json
import atexit
import dataclasses
from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple
import gradio as gr
import subprocess
import sys
import datetime

# --- Headless defaults ---
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

# Note: Model dependencies are installed in separate conda environments via setup.sh
# This app runs in the base environment and dispatches to subprocess workers

print(f"===== Application Startup at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")

# Verify environments exist on startup
def verify_environments():
    """Check that conda environments exist"""
    # Check if conda is available
    try:
        conda_check = subprocess.run(
            ["which", "conda"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if conda_check.returncode != 0:
            # Try alternative: check if conda is in common locations
            import shutil
            conda_path = shutil.which("conda")
            if not conda_path:
                print("⚠️  Warning: conda not found in PATH. Skipping environment verification.")
                print("  Assuming environments will be available when needed.")
                return True, False  # Assume OpenPI exists, OpenVLA disabled
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️  Warning: Could not check for conda. Skipping environment verification.")
        print("  Assuming environments will be available when needed.")
        return True, False  # Assume OpenPI exists, OpenVLA disabled
    
    # If conda is available, check environments
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"⚠️  Warning: conda env list failed: {result.stderr}")
            print("  Assuming environments will be available when needed.")
            return True, False
        
        has_openpi = "openpi_env" in result.stdout
        has_openvla = "openvla_env" in result.stdout
        
        print("Environment check:")
        print(f"  {'✓' if has_openpi else '✗'} openpi_env")
        print(f"  {'✓' if has_openvla else '✗'} openvla_env (optional)")
        
        if not has_openpi:
            print("⚠️  Warning: openpi_env not found in conda env list.")
            print("  Will attempt to use it anyway - check will happen when worker starts.")
        
        return has_openpi, has_openvla
        
    except subprocess.TimeoutExpired:
        print("⚠️  Warning: conda env list timed out. Skipping verification.")
        return True, False
    except Exception as e:
        print(f"⚠️  Warning: Error checking conda environments: {e}")
        print("  Assuming environments will be available when needed.")
        return True, False

HAS_OPENPI, HAS_OPENVLA = verify_environments()

# ---------------------- Environment Registry ----------------------
# Task names for UI dropdown (workers have their own registry)
_ENV_CLASSES = {
    "CubeHandover": "handover the rod from one hand to the other hand",
    "CubeHandoverOrientation": "handover the rod from one hand to the other hand",
    "CubeHandoverPosition": "handover the rod from one hand to the other hand",
    "CubeHandoverPositionOrientation": "handover the rod from one hand to the other hand",
    "CubeHandoverVertical": "handover the rod from one hand to the other hand",
    "LiftPot": "lift the pot by the handles",
    "LiftPotOrientation": "lift the pot by the handles",
    "LiftPotPosition": "lift the pot by the handles",
    "LiftPotPositionOrientation": "lift the pot by the handles",
    "LiftTray": "lift the tray",
    "LiftTrayDrag": "lift the tray",
    "LiftTrayOrientation": "lift the tray",
    "LiftTrayPosition": "lift the tray",
    "LiftTrayPositionOrientation": "lift the tray",
    "PackBox": "close the box",
    "PackBoxOrientation": "close the box",
    "PackBoxPosition": "close the box",
    "PackBoxPositionOrientation": "close the box",
    "PickSingleBookFromTable": "pick up the book from the table",
    "PickSingleBookFromTableOrientation": "pick up the book from the table",
    "PickSingleBookFromTablePosition": "pick up the book from the table",
    "PickSingleBookFromTablePositionOrientation": "pick up the book from the table",
    "RotateValve": "rotate the valve counter clockwise",
    "RotateValveObstacle": "rotate the valve counter clockwise",
    "RotateValvePosition": "rotate the valve counter clockwise",
    "RotateValvePositionOrientation": "rotate the valve counter clockwise",
    "StackSingleBookShelf": "put the book on the table onto the shelf",
    "StackSingleBookShelfPosition": "put the book on the table onto the shelf",
    "StackSingleBookShelfPositionOrientation": "put the book on the table onto the shelf",
    "StackTwoBlocks": "stack the two cubes",
    "StackTwoBlocksOrientation": "stack the two cubes",
    "StackTwoBlocksPosition": "stack the two cubes",
    "StackTwoBlocksPositionOrientation": "stack the two cubes"
}

# ---------------------- Configuration ----------------------
DEFAULT_MAX_STEPS = 200
DEFAULT_FPS = 25

# ---------------------- Subprocess Worker Management ----------------------
# Global: persistent subprocess pool
_INFERENCE_WORKERS: Dict[str, subprocess.Popen] = {
    "openpi": None,
    "openvla": None
}


def find_conda():
    """Find conda executable in common locations."""
    import shutil
    
    # Try standard PATH lookup
    conda_path = shutil.which("conda")
    if conda_path:
        return conda_path
    
    # Try common conda installation locations
    common_paths = [
        "/opt/conda/bin/conda",
        "/usr/local/conda/bin/conda",
        "/home/user/miniconda3/bin/conda",
        "/home/user/anaconda3/bin/conda",
        "/root/miniconda3/bin/conda",
        "/root/anaconda3/bin/conda",
        "/opt/conda/condabin/conda",  # Alternative conda location
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_conda_env_python(env_name: str):
    """Find Python executable in a conda environment by checking common locations."""
    # Common conda environment locations
    base_paths = [
        "/opt/conda/envs",
        "/usr/local/conda/envs",
        "/home/user/miniconda3/envs",
        "/home/user/anaconda3/envs",
        "/root/miniconda3/envs",
        "/root/anaconda3/envs",
        os.path.expanduser("~/miniconda3/envs"),
        os.path.expanduser("~/anaconda3/envs"),
    ]
    
    for base_path in base_paths:
        env_path = os.path.join(base_path, env_name)
        python_path = os.path.join(env_path, "bin", "python")
        if os.path.exists(python_path):
            return python_path
    
    return None


def get_inference_worker(model_key: str) -> subprocess.Popen:
    """
    Get or create persistent inference worker subprocess.
    
    Workers stay alive to keep models loaded in memory (fast subsequent calls).
    """
    global _INFERENCE_WORKERS
    
    env_name = f"{model_key}_env"
    script_name = f"inference_{model_key}.py"
    
    # Find conda executable
    conda_path = find_conda()
    
    if conda_path:
        # Use conda if available
        print(f"Found conda at: {conda_path}")
        
        # Check if environment exists (optional check - will fail later if it doesn't)
        try:
            result = subprocess.run(
                [conda_path, "env", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and env_name not in result.stdout:
                print(f"⚠️  Warning: {env_name} not found in conda env list. Will attempt anyway.")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify {env_name} exists: {e}. Will attempt anyway.")
        
        if _INFERENCE_WORKERS[model_key] is None or _INFERENCE_WORKERS[model_key].poll() is not None:
            print(f"Starting {model_key} worker in {env_name} using conda...")
            
            proc = subprocess.Popen(
                [conda_path, "run", "-n", env_name, "python", script_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            _INFERENCE_WORKERS[model_key] = proc
            print(f"✓ {model_key} worker started (PID: {proc.pid})")
        
        return _INFERENCE_WORKERS[model_key]
    
    else:
        # Fallback: Try to find Python in conda environment directly
        print("⚠️  conda command not found. Attempting to find environment Python directly...")
        env_python = find_conda_env_python(env_name)
        
        if env_python:
            print(f"Found Python for {env_name} at: {env_python}")
            
            if _INFERENCE_WORKERS[model_key] is None or _INFERENCE_WORKERS[model_key].poll() is not None:
                print(f"Starting {model_key} worker using environment Python directly...")
                
                proc = subprocess.Popen(
                    [env_python, script_name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                )
                
                _INFERENCE_WORKERS[model_key] = proc
                print(f"✓ {model_key} worker started (PID: {proc.pid})")
            
            return _INFERENCE_WORKERS[model_key]
        
        else:
            # Neither conda nor environment Python found
            raise RuntimeError(
                f"Could not find conda or {env_name} environment.\n\n"
                "This usually means:\n"
                "1. setup.sh has not run yet, or\n"
                "2. setup.sh failed to create the conda environment, or\n"
                "3. conda is not installed in this Space.\n\n"
                "Please check:\n"
                "- That setup.sh ran successfully during Space build\n"
                "- The build logs for any errors\n"
                "- That conda is available (HuggingFace Spaces should have it at /opt/conda/bin/conda)\n\n"
                f"Expected environment: {env_name}\n"
                "Expected locations: /opt/conda/envs/{env_name}, ~/miniconda3/envs/{env_name}, etc."
            )


def cleanup_workers():
    """Terminate worker subprocesses on shutdown"""
    for model_key, proc in _INFERENCE_WORKERS.items():
        if proc and proc.poll() is None:
            print(f"Terminating {model_key} worker...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

atexit.register(cleanup_workers)


@dataclasses.dataclass
class InferenceRequest:
    """Normalized payload for invoking model backends from the UI."""
    task_name: str
    checkpoint_path: str
    custom_instruction: Optional[str]
    max_steps: int
    fps: int
    progress: gr.Progress
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_name": self.task_name,
            "checkpoint_path": self.checkpoint_path or "",
            "custom_instruction": self.custom_instruction,
            "max_steps": self.max_steps,
            "fps": self.fps,
        }


@dataclasses.dataclass
class ModelDefinition:
    """Metadata and execution hook for a model option."""
    label: str
    description: str
    run_inference: Callable[[InferenceRequest], Tuple[Optional[str], str]]


# ---------------------- Main Inference Functions ----------------------
def run_pi0_inference(request: InferenceRequest) -> Tuple[Optional[str], str]:
    """Dispatch OpenPI inference to subprocess"""
    try:
        request.progress(0, desc="Starting OpenPI worker...")
        worker = get_inference_worker("openpi")
        
        request.progress(0.1, desc="Sending inference request...")
        # Send request
        request_dict = request.to_dict()
        request_json = json.dumps(request_dict)
        worker.stdin.write(request_json + "\n")
        worker.stdin.flush()
        
        request.progress(0.2, desc="Waiting for inference result...")
        # Read result
        result_line = worker.stdout.readline()
        if not result_line:
            return None, "❌ Worker process ended unexpectedly"
        
        result = json.loads(result_line)
        
        request.progress(1.0, desc="Complete!")
        
        if result["success"]:
            return result["video_path"], result["status_message"]
        else:
            error_msg = f"❌ OpenPI Error: {result.get('error', 'Unknown error')}\n\n{result.get('status_message', '')}"
            return None, error_msg
            
    except Exception as e:
        import traceback
        return None, f"❌ Worker communication error: {str(e)}\n\n{traceback.format_exc()}"


def run_openvla_inference(request: InferenceRequest) -> Tuple[Optional[str], str]:
    """Dispatch OpenVLA inference to subprocess"""
    try:
        request.progress(0, desc="Starting OpenVLA worker...")
        worker = get_inference_worker("openvla")
        
        request.progress(0.1, desc="Sending inference request...")
        # Send request
        request_dict = request.to_dict()
        request_json = json.dumps(request_dict)
        worker.stdin.write(request_json + "\n")
        worker.stdin.flush()
        
        request.progress(0.2, desc="Waiting for inference result...")
        # Read result
        result_line = worker.stdout.readline()
        if not result_line:
            return None, "❌ Worker process ended unexpectedly"
        
        result = json.loads(result_line)
        
        request.progress(1.0, desc="Complete!")
        
        if result["success"]:
            return result["video_path"], result["status_message"]
        else:
            error_msg = f"❌ OpenVLA Error: {result.get('error', 'Unknown error')}\n\n{result.get('status_message', '')}"
            return None, error_msg
            
    except Exception as e:
        import traceback
        return None, f"❌ Worker communication error: {str(e)}\n\n{traceback.format_exc()}"


# Registry of supported models (populated dynamically based on available environments)
MODEL_REGISTRY: Dict[str, ModelDefinition] = {
    "openpi": ModelDefinition(
        label="Pi0 Base (OpenPI)",
        description=(
            "Runs the Pi0 bimanual policy using the OpenPI runtime. "
            "Supports all RoboEval tasks and will automatically fetch checkpoints from "
            "`tan7271/pi0_base_checkpoints` when no custom path is provided."
        ),
        run_inference=run_pi0_inference,
    ),
}

# Add OpenVLA only if environment exists
if HAS_OPENVLA:
    MODEL_REGISTRY["openvla"] = ModelDefinition(
        label="OpenVLA",
        description=(
            "Runs the OpenVLA (Open Vision-Language-Action) policy. "
            "OpenVLA is a vision-language-action model for robot manipulation tasks. "
            "**Checkpoint path is required** - provide a path to an OpenVLA checkpoint directory."
        ),
        run_inference=run_openvla_inference,
    )
else:
    print("ℹ OpenVLA environment not found - OpenVLA model will not be available")


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
        
        Choose a supported model backend and run it on RoboEval tasks to watch the generated execution video.
        
        **Architecture**: Models run in isolated conda environments to avoid dependency conflicts. The first inference with each model may take 30-60 seconds to load the model, but subsequent inferences are fast.
        
        ⚠️ **Hardware Requirements:** This Space requires a GPU with at least 8GB memory.
        If you see "Out of Memory" errors, upgrade the Space hardware in Settings → Hardware → T4 small.
        
        **Checkpoint Paths**:
        - **OpenPI**: Leave empty to auto-download from `tan7271/pi0_base_checkpoints`, or provide a custom path
        - **OpenVLA**: **Required** - provide a path to an OpenVLA checkpoint directory
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
        
        ### Model Switching
        
        You can switch between OpenPI and OpenVLA models instantly. Each model runs in its own isolated environment with optimized dependencies. The first inference with each model loads it into memory (30-60s), but subsequent inferences are fast.
        
        ### GPU Usage
        
        This Space uses a T4 GPU (~$0.60/hour). It auto-sleeps after 10 minutes of inactivity to minimize costs.
        """)
    
    return demo


# ---------------------- Launch ----------------------
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()

