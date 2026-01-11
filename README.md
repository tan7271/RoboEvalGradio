---
title: Robot Policy Inference on RoboEval Tasks
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
startup_duration_timeout: 30m
pinned: false
license: mit
---

# Robot Policy Inference on RoboEval Tasks

A Hugging Face Space for running robot manipulation policy inference on various tasks from the RoboEval benchmark. Supports **OpenPI** (Pi0 bimanual policy) and **OpenVLA** (vision-language-action) backends.

[Watch the demo video](https://youtu.be/pTVMu93jPbw)

## 🚀 Features

- **Multiple Model Backends**: Switch between OpenPI and OpenVLA using environment variables
- **Interactive Gradio Interface**: Easy-to-use web interface for running inference
- **Multiple Tasks**: Support for 20+ bimanual manipulation tasks
- **Real-time Video Output**: View robot execution videos immediately after inference
- **Customizable Parameters**: Adjust max steps, FPS, and task instructions
- **GPU Acceleration**: Runs on T4 GPU for fast inference
- **Dynamic Model Detection**: Interface adapts based on which backend is installed

## 🔀 Switching Between OpenPI and OpenVLA

This Space supports both OpenPI and OpenVLA, but they **cannot run simultaneously** due to dependency conflicts. Choose your backend using the `MODEL_BACKEND` environment variable:

### Quick Setup

1. Go to **Settings → Variables** in your Space
2. Add environment variable:
   - **Name**: `MODEL_BACKEND`
   - **Value**: `openpi` (default) or `openvla`
3. Save and rebuild

See [SWITCHING_MODELS.md](./SWITCHING_MODELS.md) for detailed instructions and technical explanation.

## 📋 Available Tasks

The Space supports all RoboEval bimanual manipulation tasks:

### Manipulation Tasks
- **Cube Handover**: Transfer a rod between robot hands
- **Cube Handover (Orientation)**: Handover with orientation constraints
- **Cube Handover (Position)**: Handover with position constraints
- **Cube Handover (Position + Orientation)**: Full constraint handover
- **Vertical Cube Handover**: Vertical handover variant

### Lifting Tasks
- **Lift Pot**: Bimanually lift a pot by its handles
- **Lift Tray**: Lift and balance a tray
- **Lift Tray (Drag)**: Drag and lift tray variant

### Packing Tasks
- **Pack Box**: Close a box containing objects
- **Pack Box (Orientation)**: Packing with orientation constraints
- **Pack Box (Position)**: Packing with position constraints

### Book Manipulation
- **Pick Single Book**: Pick up a book from a table
- **Stack Single Book**: Place book on a shelf
- **Stack Two Blocks**: Stack two cubes together

### Utility Tasks
- **Rotate Valve**: Turn a valve counter-clockwise
- **Rotate Valve (Obstacle)**: Valve rotation with obstacles
- **Rotate Valve (Position)**: Valve rotation with position constraints

## 🛠️ Usage

1. **Select a Task**: Choose from the dropdown menu of available tasks
2. **Provide Checkpoint Path**: Enter the path to your Pi0 model checkpoint
3. **Customize (Optional)**:
   - Override the default task instruction
   - Adjust maximum steps (default: 200)
   - Set video FPS (default: 5)
4. **Run Inference**: Click "🚀 Run Inference" to start
5. **View Results**: Watch the execution video and see performance stats

## 🔧 Setup for Private Checkpoints

This Space is configured to work with private RoboEval repositories:

1. **Create GitHub Token**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create a token with `repo` scope
   - Copy the token

2. **Configure Space Secrets**:
   - In your HF Space: Settings → Variables and secrets
   - Add a **secret** (not a variable) named exactly: `GH_TOKEN`
   - Set the value to your GitHub personal access token
   - **Important for Docker builds**: HuggingFace Spaces should automatically pass secrets as build arguments
   - If the build still fails with "GH_TOKEN not set", check:
     - The secret is set as a **Secret** (not a Variable)
     - The name is exactly `GH_TOKEN` (case-sensitive)
     - The token has `repo` scope and is not expired
     - Try rebuilding the Space after setting the secret

3. **Checkpoint Access**:
   - Upload your Pi0 checkpoint to the Space
   - Or provide a path to a checkpoint in HF Hub
   - The checkpoint should follow the expected directory structure

## 💰 Cost Information

- **Hardware**: T4 Small GPU (~$0.60/hour)
- **Auto-sleep**: 10 minutes of inactivity
- **Estimated cost**: $5-20/month for moderate use
- **Billing**: Only charged when Space is actively running

## 🔍 Technical Details

### Model Architecture
- **Policy**: Pi0 base bimanual droid (finetuned)
- **Input**: Multi-camera RGB + proprioception
- **Output**: 16-DOF joint actions
- **Horizon**: 10-step open-loop planning

### Environment
- **Physics**: MuJoCo 3.3.3
- **Robot**: Bimanual Panda arms
- **Cameras**: Head, left wrist, right wrist (256x256)
- **Control**: 20Hz (500Hz downsampled by 25x)

### Video Output
- **Format**: MP4 (H.264)
- **Resolution**: 256x256
- **FPS**: Configurable (default: 5)
- **Storage**: Temporary files in `/tmp`

## 🐛 Troubleshooting

### Common Issues

1. **"Unknown task" error**:
   - Ensure task name matches exactly from dropdown
   - Check that RoboEval is properly installed

2. **Checkpoint loading fails**:
   - Verify checkpoint path is correct
   - Ensure checkpoint has required `assets/` directory
   - Check GitHub token has repo access

3. **GPU out of memory**:
   - Reduce max_steps parameter
   - Try CPU mode (slower but works)

4. **Video not generating**:
   - Check that inference completed successfully
   - Verify ffmpeg is installed (included in packages.txt)

### Performance Tips

- **Faster inference**: Reduce max_steps to 100-150
- **Better quality**: Increase FPS to 10-15
- **Cost savings**: Use shorter max_steps and lower FPS
- **Debugging**: Check the status output for detailed error messages

## 📚 References

- **RoboEval**: [GitHub Repository](https://github.com/your-org/RoboEval)
- **Pi0 Paper**: [OpenPI: An Open-Source Framework for Learning-Based Robot Manipulation](https://arxiv.org/abs/2024.xxxx)
- **MuJoCo**: [Official Documentation](https://mujoco.readthedocs.io/)

## 📄 License

This Space is for research and educational purposes. Please refer to the original RoboEval and Pi0 licenses for usage terms.

---

**Note**: This is a private Space that requires proper authentication setup to access private repositories and checkpoints.
