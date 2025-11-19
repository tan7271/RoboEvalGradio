#!/bin/bash
# Launch script for Docker container
# Runs the Gradio app in the base conda environment

# Source conda if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Install RoboEval at runtime if not already installed
# This uses GH_TOKEN environment variable (available at runtime in HuggingFace Spaces)
if [ -n "$GH_TOKEN" ]; then
    echo "Installing RoboEval at runtime (if not already installed)..."
    bash install_roboeval_runtime.sh || {
        echo "⚠️  Warning: RoboEval installation failed. The app will continue but some features may not work."
    }
else
    echo "⚠️  Warning: GH_TOKEN not set. Skipping RoboEval installation."
    echo "   Some features may not work. Set GH_TOKEN as a secret in Space settings."
fi

# Launch the app
conda run -n base python app.py

