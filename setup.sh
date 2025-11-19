#!/bin/bash
set -e

echo "===== Multi-Environment Setup ====="
echo "Building OpenPI environment (OpenVLA temporarily disabled)..."

# Install build dependencies first (bazel for labmaze, Rust for safetensors)
echo ""
echo "===== Installing Build Dependencies ====="

# Install bazel (required for labmaze)
if ! command -v bazel &> /dev/null; then
    echo "Installing bazel..."
    # Install bazelisk (bazel wrapper) which is easier to install
    BAZELISK_VERSION="v1.19.0"
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64"
    mkdir -p ~/.local/bin
    wget -q ${BAZELISK_URL} -O ~/.local/bin/bazel
    chmod +x ~/.local/bin/bazel
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "✓ Bazel installed"
else
    echo "✓ Bazel already installed"
fi

# Install Rust (required for safetensors)
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ${HOME}/.cargo/env
    export PATH="${HOME}/.cargo/bin:${PATH}"
    echo "✓ Rust installed"
else
    echo "✓ Rust already installed"
    # Make sure cargo is in PATH
    if [ -f "${HOME}/.cargo/env" ]; then
        source ${HOME}/.cargo/env
    fi
fi

# Try to install via apt-get if available (for additional build tools)
if command -v apt-get &> /dev/null && [ "$EUID" -eq 0 ]; then
    echo "Installing additional build tools via apt-get..."
    apt-get update -qq
    apt-get install -y -qq \
        build-essential \
        g++ \
        gcc \
        make \
        cmake \
        pkg-config \
        || echo "Warning: Some build tools may not be available"
    echo "✓ Additional build tools installed"
else
    echo "⚠️  Skipping apt-get installation (not root or not available)"
fi

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

# Install RoboEval (this will build labmaze which requires bazel)
echo "Installing RoboEval from cloned repository..."
echo "Note: This may take several minutes as it builds labmaze with bazel..."
pip install $CLONE_DIR --no-cache-dir || {
    echo "⚠️  RoboEval installation had some errors, but continuing..."
    echo "Some optional dependencies may not be available"
}

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
