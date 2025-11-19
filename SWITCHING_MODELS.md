# Switching Between OpenPI and OpenVLA

This Space supports both OpenPI and OpenVLA backends, but they **cannot run simultaneously** due to dependency conflicts. You can switch between them using an environment variable.

## How to Switch Models

### Option 1: Using Hugging Face Space Settings (Recommended)

1. Go to your Space settings: **Settings → Variables**
2. Add a new **Environment Variable**:
   - **Name**: `MODEL_BACKEND`
   - **Value**: `openpi` or `openvla`
3. Click **Save**
4. The Space will rebuild with the selected backend

### Option 2: Local Development

Set the environment variable before running:

```bash
# For OpenPI (default)
export MODEL_BACKEND=openpi
bash setup.sh
python app.py

# For OpenVLA
export MODEL_BACKEND=openvla
bash setup.sh
python app.py
```

## What Happens During Build

The `setup.sh` script checks the `MODEL_BACKEND` variable:

- **`MODEL_BACKEND=openpi`** (default):
  - Installs PyTorch 2.9+ with cuDNN 9.1+
  - Installs lerobot and OpenPI from git
  - OpenPI appears in the model dropdown

- **`MODEL_BACKEND=openvla`**:
  - Installs PyTorch 2.2.0 with cuDNN 8.9
  - Installs OpenVLA from git
  - OpenVLA appears in the model dropdown

## Dynamic Model Detection

The app automatically detects which backend is installed:

```python
# In app.py
def _populate_model_registry():
    try:
        import openpi
        # Register OpenPI
    except ImportError:
        pass
    
    try:
        import openvla
        # Register OpenVLA  
    except ImportError:
        pass
```

The Gradio interface will only show the models that are actually available.

## Why Can't Both Run Together?

**Dependency Conflict Summary:**

| Package | OpenPI Requirement | OpenVLA Requirement | Conflict |
|---------|-------------------|---------------------|----------|
| torch | >=2.7.0 | ==2.2.0 | ❌ Incompatible |
| nvidia-cudnn-cu12 | >=9.1.1 | ==8.9.2.26 | ❌ Incompatible |
| transformers | ==4.48.1 | ==4.40.1 | ❌ Incompatible |
| draccus | ==0.10.0 | ==0.8.0 | ❌ Incompatible |

When both are installed together, OpenVLA downgrades PyTorch and cuDNN, causing OpenPI's JAX components to fail with:
```
Loaded runtime CuDNN library: 9.0.0 but source was compiled with: 9.1.1
```

See [`DEPENDENCY_CONFLICT.md`](./DEPENDENCY_CONFLICT.md) for detailed technical explanation.

## Testing Locally

```bash
# Test OpenPI
export MODEL_BACKEND=openpi
bash setup.sh
python app.py

# Clean environment
pip uninstall -y openpi lerobot torch torchvision torchaudio

# Test OpenVLA
export MODEL_BACKEND=openvla  
bash setup.sh
python app.py
```

## Default Behavior

If `MODEL_BACKEND` is not set, the Space defaults to **OpenPI**.

## Troubleshooting

### Space shows "No model backends available"
- Check build logs for installation errors
- Verify `MODEL_BACKEND` is set correctly
- Ensure `GH_TOKEN` secret is configured for private repos

### Wrong model appears after switching
- Space caching may cause old builds to persist
- Try: **Settings → Factory Reboot**
- Or: Change any other setting to force a full rebuild

### Want to use both models?
Create two separate Spaces:
- `your-space-openpi` with `MODEL_BACKEND=openpi`
- `your-space-openvla` with `MODEL_BACKEND=openvla`


