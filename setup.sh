#!/bin/bash
set -e

echo "===== Multi-Environment Setup ====="
echo "Building OpenPI environment (OpenVLA temporarily disabled)..."

# Check if conda is installed, install if not
if ! command -v conda &> /dev/null; then
    echo ""
    echo "===== Installing Miniconda ====="
    echo "Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    MINICONDA_INSTALLER="/tmp/miniconda.sh"
    MINICONDA_PREFIX="${HOME}/miniconda3"
    
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${MINICONDA_INSTALLER}
    bash ${MINICONDA_INSTALLER} -b -p ${MINICONDA_PREFIX}
    rm ${MINICONDA_INSTALLER}
    
    # Initialize conda
    ${MINICONDA_PREFIX}/bin/conda init bash
    source ${MINICONDA_PREFIX}/etc/profile.d/conda.sh
    
    # Add conda to PATH for this script
    export PATH="${MINICONDA_PREFIX}/bin:${PATH}"
    
    echo "✓ Miniconda installed at ${MINICONDA_PREFIX}"
else
    echo "✓ Conda already installed"
    # Initialize conda if not already done
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        source ${HOME}/miniconda3/etc/profile.d/conda.sh
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source /opt/conda/etc/profile.d/conda.sh
    fi
fi

# Install RoboEval in base (shared by both)
echo ""
echo "===== Installing RoboEval (shared) ====="
CLONE_DIR="/tmp/roboeval_install"
rm -rf $CLONE_DIR

# Clone with submodules
echo "Cloning RoboEval repository with submodules..."
git clone --recurse-submodules https://${GH_TOKEN}@github.com/helen9975/RoboEval.git $CLONE_DIR

# Install
echo "Installing RoboEval from cloned repository..."
pip install $CLONE_DIR --no-cache-dir

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r $CLONE_DIR/thirdparty $SITE_PACKAGES/
echo "RoboEval installed in base environment"

# Create OpenPI environment
echo ""
echo "===== Creating OpenPI Environment ====="
conda create -n openpi_env python=3.10 -y
conda run -n openpi_env pip install -r requirements_openpi.txt --no-cache-dir

# Install lerobot and OpenPI in openpi_env
echo "Installing lerobot in openpi_env..."
conda run -n openpi_env pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 --no-cache-dir
echo "lerobot installed successfully"

echo "Upgrading safetensors in openpi_env..."
conda run -n openpi_env pip install "safetensors>=0.4.1" --upgrade --no-cache-dir
echo "safetensors upgraded successfully"

echo "Installing openpi-client in openpi_env..."
conda run -n openpi_env pip install git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git#subdirectory=packages/openpi-client --no-cache-dir --no-deps
echo "openpi-client installed successfully"

echo "Installing OpenPI in openpi_env..."
conda run -n openpi_env pip install git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git --no-cache-dir --no-deps --force-reinstall
echo "OpenPI installed successfully"

# Copy RoboEval to openpi_env
OPENPI_SITE=$(conda run -n openpi_env python -c "import site; print(site.getsitepackages()[0])")
cp -r $SITE_PACKAGES/roboeval* $OPENPI_SITE/ || true
cp -r $SITE_PACKAGES/thirdparty $OPENPI_SITE/ || true
echo "OpenPI environment ready"

# Create OpenVLA environment (TEMPORARILY DISABLED - uncomment to enable)
# echo ""
# echo "===== Creating OpenVLA Environment ====="
# conda create -n openvla_env python=3.10 -y
# 
# # OpenVLA requires older PyTorch versions
# echo "Installing OpenVLA-compatible PyTorch in openvla_env..."
# conda run -n openvla_env pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
# echo "PyTorch installed successfully"
# 
# echo "Installing OpenVLA dependencies in openvla_env..."
# conda run -n openvla_env pip install -r requirements_openvla.txt --no-cache-dir
# echo "Dependencies installed successfully"
# 
# # Install OpenVLA from GitHub
# echo "Installing OpenVLA from openvla/openvla..."
# conda run -n openvla_env pip install git+https://github.com/openvla/openvla.git --no-cache-dir
# echo "OpenVLA installed successfully"
# 
# # Copy RoboEval to openvla_env
# OPENVLA_SITE=$(conda run -n openvla_env python -c "import site; print(site.getsitepackages()[0])")
# cp -r $SITE_PACKAGES/roboeval* $OPENVLA_SITE/ || true
# cp -r $SITE_PACKAGES/thirdparty $OPENVLA_SITE/ || true
# echo "OpenVLA environment ready"

echo ""
echo "===== Setup Complete ====="
echo "✓ Base environment: Gradio + RoboEval"
echo "✓ openpi_env: PyTorch 2.7+ + OpenPI"
echo "ℹ openvla_env: Disabled (uncomment in setup.sh to enable)"
