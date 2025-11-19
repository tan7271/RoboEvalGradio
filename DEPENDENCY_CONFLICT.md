# Dependency Conflict: OpenPI vs OpenVLA

## Problem

OpenPI and OpenVLA cannot coexist in the same Python environment due to incompatible CUDA/PyTorch requirements:

### OpenVLA Requirements
- `torch==2.2.0` (CUDA 12.1)
- `nvidia-cudnn-cu12==8.9.2.26`
- `transformers==4.40.1`
- `draccus==0.8.0`

### OpenPI Requirements  
- `torch>=2.7.0` (CUDA 12.8)
- `nvidia-cudnn-cu12>=9.1.1`
- `transformers==4.48.1`
- `draccus>=0.10.0`

### The Core Issue
When both are installed together:
1. OpenVLA downgrades torch from 2.9.1 → 2.2.0
2. OpenVLA downgrades cuDNN from 9.10 → 8.9
3. OpenPI's JAX components are compiled against cuDNN 9.1+ but runtime loads cuDNN 8.9/9.0
4. Result: **"Loaded runtime CuDNN library: 9.0.0 but source was compiled with: 9.1.1"**

This causes OpenPI initialization to fail with `ImportError: initialization failed`.

## Solution Options

### Option 1: Separate Containers (Recommended for Production)
Run OpenPI and OpenVLA in separate Docker containers or Hugging Face Spaces:
- **OpenPI Space**: Uses current setup with torch>=2.7.0
- **OpenVLA Space**: Separate space with torch==2.2.0

### Option 2: Conditional Installation
Install only one model at a time based on environment variable:
```bash
if [ "$MODEL" = "openvla" ]; then
    pip install git+https://github.com/openvla/openvla.git
else
    pip install git+https://github.com/tan7271/OpenPiRoboEval.git
fi
```

### Option 3: Virtual Environments (Local Development)
Use separate conda/venv environments:
```bash
# OpenPI environment
conda create -n openpi python=3.11
conda activate openpi
# ... install OpenPI deps

# OpenVLA environment  
conda create -n openvla python=3.11
conda activate openvla
# ... install OpenVLA deps
```

## Current Implementation

This repository currently supports **OpenPI only**. OpenVLA has been removed from:
- `requirements.txt` (removed peft, timm, accelerate that were added for OpenVLA)
- `setup.sh` (removed OpenVLA git installation)
- `app.py` (removed OpenVLA from MODEL_REGISTRY)

## Re-enabling OpenVLA

To create a separate OpenVLA-only deployment:

1. Revert to commit before OpenPI was added
2. Use these versions in `requirements.txt`:
```
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
transformers==4.40.1
```
3. Add back OpenVLA installation to `setup.sh`
4. Keep only OpenVLA in `MODEL_REGISTRY`

## References

- OpenPI repo: https://github.com/tan7271/OpenPiRoboEval
- OpenVLA repo: https://github.com/openvla/openvla
- Related issue: CUDA library version mismatches causing initialization failures


