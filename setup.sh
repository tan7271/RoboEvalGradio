#!/bin/bash
set -e

echo "===== Installing Dependencies ====="

# Install RoboEval with submodules
echo "Installing RoboEval with submodules..."
CLONE_DIR="/tmp/roboeval_install"
rm -rf $CLONE_DIR

# Clone with submodules
echo "Cloning RoboEval repository with submodules..."
git clone --recurse-submodules https://${GH_TOKEN}@github.com/helen9975/RoboEval.git $CLONE_DIR

# Install
echo "Installing RoboEval from cloned repository..."
pip install $CLONE_DIR --no-cache-dir

# Copy thirdparty to site-packages
echo "Copying thirdparty submodules to site-packages..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r $CLONE_DIR/thirdparty $SITE_PACKAGES/
echo "Copied thirdparty to $SITE_PACKAGES/thirdparty"

echo "RoboEval installed successfully with submodules"

# Install lerobot from specific commit
echo "Installing lerobot from git (specific commit required by OpenPI)..."
pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 --no-cache-dir
echo "lerobot installed successfully"

# Upgrade safetensors to fix version conflict
echo "Upgrading safetensors to >=0.4.1..."
pip install "safetensors>=0.4.1" --upgrade --no-cache-dir
echo "safetensors upgraded successfully"

# Install OpenPI with openpi-client
echo "Installing OpenPI from tan7271/OpenPiRoboEval..."

# Install openpi-client
echo "Installing openpi-client..."
pip install git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git#subdirectory=packages/openpi-client --no-cache-dir --no-deps
echo "openpi-client installed successfully"

# Install OpenPI
pip install git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git --no-cache-dir --no-deps --force-reinstall
echo "OpenPI installed successfully"

echo "===== All dependencies installed ====="

