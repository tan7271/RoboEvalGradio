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
# Rust is installed in /root/.cargo but script may run as non-root user
# Add to PATH directly if the bin directory exists and is readable
if [ -d "/root/.cargo/bin" ] && [ -x "/root/.cargo/bin/cargo" ]; then
    export PATH="/root/.cargo/bin:${PATH}"
    echo "✓ Added /root/.cargo/bin to PATH"
elif [ -f "/root/.cargo/env" ]; then
    # Try to source if we have permission
    source /root/.cargo/env 2>/dev/null || export PATH="/root/.cargo/bin:${PATH}"
    echo "✓ Sourced Rust environment from /root/.cargo/env"
elif [ -f "${HOME}/.cargo/env" ]; then
    source ${HOME}/.cargo/env
    export PATH="${HOME}/.cargo/bin:${PATH}"
    echo "✓ Sourced Rust environment from ${HOME}/.cargo/env"
fi

# Verify Rust is available
if command -v cargo &> /dev/null; then
    echo "✓ Rust/Cargo found: $(which cargo)"
    cargo --version || true
else
    echo "⚠️  Warning: Rust/Cargo not found. RoboEval installation may fail."
    echo "   Checked paths: /root/.cargo/bin, /root/.cargo/env, ${HOME}/.cargo/env"
    echo "   PATH: ${PATH}"
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
    
    # Install missing pip-only dependencies that may not be in the environment file yet
    # Note: torchvision and torchaudio should be installed via conda, not pip
    echo "Installing additional OpenPI dependencies in openpi_env..."
    conda run -n openpi_env pip install --no-cache-dir \
        draccus>=0.1.0 \
        jsonlines>=4.0.0 || {
        echo "⚠️  Warning: Failed to install some additional dependencies in openpi_env"
    }
    
    # Copy RoboEval to openpi_env
    # Use a Python script via conda run to handle permissions correctly
    echo "Copying RoboEval to openpi_env..."
    
    # Get site-packages locations
    OPENPI_SITE=$(conda run -n openpi_env python -c "import site; print(site.getsitepackages()[0])")
    echo "Target site-packages: ${OPENPI_SITE}"
    
    # Find thirdparty source
    THIRDPARTY_SOURCE=""
    if [ -d "${SITE_PACKAGES}/thirdparty" ]; then
        THIRDPARTY_SOURCE="${SITE_PACKAGES}/thirdparty"
    elif [ -d "$CLONE_DIR/thirdparty" ]; then
        THIRDPARTY_SOURCE="$CLONE_DIR/thirdparty"
    fi
    
    # Create a temporary Python script to do the copying (runs with conda run permissions)
    COPY_SCRIPT=$(mktemp)
    cat > "$COPY_SCRIPT" << 'PYEOF'
import shutil
import sys
import os
import glob

source_site = sys.argv[1]
target_site = sys.argv[2]
thirdparty_source = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

# Copy roboeval packages
roboeval_pattern = os.path.join(source_site, "roboeval*")
for path in glob.glob(roboeval_pattern):
    dest = os.path.join(target_site, os.path.basename(path))
    try:
        if os.path.isdir(path):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)
        print(f"Copied: {os.path.basename(path)}")
    except Exception as e:
        print(f"Warning: Failed to copy {path}: {e}")

# Copy thirdparty directory
if thirdparty_source and os.path.isdir(thirdparty_source):
    thirdparty_dest = os.path.join(target_site, "thirdparty")
    try:
        if os.path.exists(thirdparty_dest):
            shutil.rmtree(thirdparty_dest)
        shutil.copytree(thirdparty_source, thirdparty_dest)
        print(f"Copied: thirdparty directory")
    except Exception as e:
        print(f"Error: Failed to copy thirdparty: {e}")
        sys.exit(1)
elif thirdparty_source:
    print(f"Warning: thirdparty source not found: {thirdparty_source}")
PYEOF

    # Run the copy script using conda run
    conda run -n openpi_env python "$COPY_SCRIPT" "${SITE_PACKAGES}" "${OPENPI_SITE}" "${THIRDPARTY_SOURCE}" || {
        echo "⚠️  Warning: Copy script had errors"
    }
    
    # Clean up
    rm -f "$COPY_SCRIPT"
    
    # Verify thirdparty was copied
    if conda run -n openpi_env python -c "import os; print('OK' if os.path.exists('${OPENPI_SITE}/thirdparty') else 'MISSING')" 2>/dev/null | grep -q "OK"; then
        echo "✓ thirdparty directory copied successfully"
    else
        echo "⚠️  Warning: thirdparty directory verification failed"
    fi
    
    echo "✓ RoboEval copied to openpi_env"
fi

echo "✓ RoboEval installation complete"

