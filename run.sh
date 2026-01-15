#!/bin/bash
# Launch script for Docker container
# Runs the Gradio app in the base conda environment

# Fix OMP_NUM_THREADS if it has invalid format (e.g., "7500m")
# numexpr requires it to be a valid integer
if [ -n "$OMP_NUM_THREADS" ]; then
    # Remove any non-numeric suffix (like 'm', 'k', etc.)
    OMP_NUM_THREADS_CLEAN=$(echo "$OMP_NUM_THREADS" | sed 's/[^0-9].*$//')
    if [ -n "$OMP_NUM_THREADS_CLEAN" ] && [ "$OMP_NUM_THREADS_CLEAN" -gt 0 ] 2>/dev/null; then
        export OMP_NUM_THREADS=$OMP_NUM_THREADS_CLEAN
    else
        export OMP_NUM_THREADS=1
    fi
else
    export OMP_NUM_THREADS=1
fi

# Source conda if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Install RoboEval at runtime if not already installed
# This uses GH_TOKEN environment variable (available at runtime in HuggingFace Spaces)
if [ -n "$GH_TOKEN" ]; then
    # Check if RoboEval is already installed before running installer
    if python -c "import roboeval" 2>/dev/null; then
        # Already installed, skip
        :
    else
        echo "Installing RoboEval at runtime (if not already installed)..."
        bash install_roboeval_runtime.sh || {
            echo "⚠️  Warning: RoboEval installation failed. The app will continue but some features may not work."
        }
    fi
else
    echo "⚠️  Warning: GH_TOKEN not set. Skipping RoboEval installation."
    echo "   Some features may not work. Set GH_TOKEN as a secret in Space settings."
fi

# Launch the app
conda run -n base python app.py

