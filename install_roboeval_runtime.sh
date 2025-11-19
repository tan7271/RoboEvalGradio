#!/bin/bash
# Runtime installation script for RoboEval
# This runs at container startup when GH_TOKEN is available as an environment variable

set -e

echo "===== Installing RoboEval at Runtime ====="

# Source conda if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Check if RoboEval is already installed
if python -c "import roboeval" 2>/dev/null; then
    echo "✓ RoboEval is already installed, skipping installation"
    exit 0
fi

# Check if GH_TOKEN is available
if [ -z "$GH_TOKEN" ]; then
    echo "❌ ERROR: GH_TOKEN environment variable not set."
    echo "   Please set GH_TOKEN as a secret in HuggingFace Space settings."
    echo "   The app will continue but RoboEval-dependent features will not work."
    exit 1
fi

CLONE_DIR="/tmp/roboeval_install"
rm -rf $CLONE_DIR

# Clone with submodules
echo "Cloning RoboEval repository with submodules..."
git clone --recurse-submodules https://${GH_TOKEN}@github.com/helen9975/RoboEval.git $CLONE_DIR

# Set environment variables for building
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export USE_BAZEL_VERSION=7.5.0

# Check if Bazel is available (should be in /usr/local/bin if installed in Dockerfile)
if ! command -v bazel &> /dev/null; then
    if [ -f "/usr/local/bin/bazel" ]; then
        export PATH="/usr/local/bin:${PATH}"
    else
        echo "⚠️  Warning: Bazel not found. RoboEval installation may fail."
    fi
fi

# Check if Rust is available (should be in /root/.cargo if installed in Dockerfile)
if ! command -v cargo &> /dev/null; then
    if [ -f "/root/.cargo/env" ]; then
        source /root/.cargo/env
        export PATH="/root/.cargo/bin:${PATH}"
    elif [ -f "${HOME}/.cargo/env" ]; then
        source ${HOME}/.cargo/env
        export PATH="${HOME}/.cargo/bin:${PATH}"
    else
        echo "⚠️  Warning: Rust/Cargo not found. RoboEval installation may fail."
    fi
fi

# Install RoboEval
echo "Installing RoboEval from cloned repository..."
echo "Note: This may take several minutes as it builds labmaze with bazel..."

pip install $CLONE_DIR --no-cache-dir || {
    echo "⚠️  RoboEval installation had some errors, but continuing..."
    echo "Trying to install pre-built safetensors wheel..."
    pip install safetensors --only-binary :all: || pip install "safetensors>=0.4.1" --no-build-isolation || true
}

# Get site-packages location
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Copy thirdparty directory
cp -r $CLONE_DIR/thirdparty $SITE_PACKAGES/ || true

echo "✓ RoboEval installed in base environment"

# Install git-based packages in openpi_env if it exists
if conda env list | grep -q "openpi_env"; then
    echo "Installing git-based packages in openpi_env..."
    conda run -n openpi_env pip install --no-cache-dir \
        git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
        git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git#subdirectory=packages/openpi-client --no-deps \
        git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git --no-deps --force-reinstall || {
        echo "⚠️  Warning: Failed to install git packages in openpi_env"
    }
    
    # Copy RoboEval to openpi_env
    OPENPI_SITE=$(conda run -n openpi_env python -c "import site; print(site.getsitepackages()[0])")
    cp -r ${SITE_PACKAGES}/roboeval* ${OPENPI_SITE}/ 2>/dev/null || true
    cp -r ${SITE_PACKAGES}/thirdparty ${OPENPI_SITE}/ 2>/dev/null || true
    echo "✓ RoboEval copied to openpi_env"
fi

echo "✓ RoboEval installation complete"

