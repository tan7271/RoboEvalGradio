#!/bin/bash
# Runtime installation script for RoboEval
# This runs at container startup when GH_TOKEN is available as an environment variable

set -e

# Verbose mode: set ROBOEVAL_VERBOSE=1 to see detailed output
VERBOSE=${ROBOEVAL_VERBOSE:-0}
log_verbose() {
    if [ "$VERBOSE" = "1" ]; then
        echo "$@"
    fi
}

if [ "$VERBOSE" = "1" ]; then
    echo "===== Installing RoboEval at Runtime ====="
fi

# Source conda if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Check if RoboEval is already installed
if python -c "import roboeval" 2>/dev/null; then
    log_verbose "✓ RoboEval is already installed, skipping installation"
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
log_verbose "Cloning RoboEval repository with submodules..."
git clone --recurse-submodules https://${GH_TOKEN}@github.com/helen9975/RoboEval.git $CLONE_DIR

# Helper function: Ensure build tools (Bazel and Rust) are in PATH
ensure_build_tools() {
    # Set environment variables for building
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    export USE_BAZEL_VERSION=7.5.0
    
    # Check if Bazel is available
    if ! command -v bazel &> /dev/null; then
        if [ -f "/usr/local/bin/bazel" ]; then
            export PATH="/usr/local/bin:${PATH}"
        else
            log_verbose "⚠️  Warning: Bazel not found. RoboEval installation may fail."
        fi
    fi
    
    # Check if Rust is available
    if [ -d "/root/.cargo/bin" ] && [ -x "/root/.cargo/bin/cargo" ]; then
        export PATH="/root/.cargo/bin:${PATH}"
        log_verbose "✓ Added /root/.cargo/bin to PATH"
    elif [ -f "/root/.cargo/env" ]; then
        source /root/.cargo/env 2>/dev/null || export PATH="/root/.cargo/bin:${PATH}"
        log_verbose "✓ Sourced Rust environment from /root/.cargo/env"
    elif [ -f "${HOME}/.cargo/env" ]; then
        source ${HOME}/.cargo/env
        export PATH="${HOME}/.cargo/bin:${PATH}"
        log_verbose "✓ Sourced Rust environment from ${HOME}/.cargo/env"
    fi
    
    # Verify Rust is available
    if command -v cargo &> /dev/null; then
        log_verbose "✓ Rust/Cargo found: $(which cargo)"
        log_verbose "$(cargo --version || true)"
    else
        log_verbose "⚠️  Warning: Rust/Cargo not found. RoboEval installation may fail."
        log_verbose "   Checked paths: /root/.cargo/bin, /root/.cargo/env, ${HOME}/.cargo/env"
        log_verbose "   PATH: ${PATH}"
    fi
}

# Helper function: Install RoboEval from cloned directory
install_roboeval_from_clone() {
    local env_name=$1
    local clone_dir=$2
    
    log_verbose "Installing RoboEval from cloned repository..."
    if [ "$VERBOSE" = "1" ]; then
        echo "Note: This may take several minutes as it builds labmaze with bazel..."
    fi
    
    if [ -z "$env_name" ] || [ "$env_name" = "base" ]; then
        # Install in base environment
        pip install "$clone_dir" --no-cache-dir || {
            log_verbose "⚠️  RoboEval installation had some errors, but continuing..."
            log_verbose "Trying to install pre-built safetensors wheel..."
            pip install safetensors --only-binary :all: || pip install "safetensors>=0.4.1" --no-build-isolation || true
        }
    else
        # Install in specified conda environment
        local clone_dir_esc=$(echo "$clone_dir" | sed 's/"/\\"/g')
        conda run -n "$env_name" bash -c "
            export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1;
            export USE_BAZEL_VERSION=7.5.0;
            export PATH=\"/usr/local/bin:\$PATH\";
            if [ -f \"/root/.cargo/env\" ]; then source /root/.cargo/env; fi;
            pip install \"${clone_dir_esc}\" --no-cache-dir || {
                echo '⚠️  RoboEval installation had some errors, but continuing...';
                pip install safetensors --only-binary :all: || pip install 'safetensors>=0.4.1' --no-build-isolation || true;
            }
        " || {
            log_verbose "⚠️  Warning: RoboEval installation into $env_name had errors"
        }
    fi
}

# Helper function: Copy thirdparty to user site-packages
copy_thirdparty_to_user_site() {
    local py_minor=$1  # e.g., "3.12" or "3.10"
    local site_packages=$2
    local clone_dir=$3
    
    log_verbose "Copying thirdparty to user's local site-packages..."
    local user_site_packages="/home/user/.local/lib/python${py_minor}/site-packages"
    mkdir -p "$user_site_packages" 2>/dev/null || true
    
    # Find thirdparty source
    local thirdparty_source=""
    if [ -d "${site_packages}/thirdparty" ]; then
        thirdparty_source="${site_packages}/thirdparty"
    elif [ -d "$clone_dir/thirdparty" ]; then
        thirdparty_source="$clone_dir/thirdparty"
    fi
    
    if [ -n "$thirdparty_source" ] && [ -d "$thirdparty_source" ]; then
        cp -r "$thirdparty_source" "$user_site_packages/" 2>/dev/null || {
            log_verbose "⚠️  Warning: Failed to copy thirdparty to user site-packages"
        }
        
        # Verify
        if [ -d "$user_site_packages/thirdparty" ]; then
            log_verbose "✓ thirdparty copied to user site-packages: $user_site_packages"
        fi
    fi
}

# Helper function: Install RoboEval into a conda environment
install_roboeval_into_env() {
    local env_name=$1
    local py_minor=$2  # Python minor version (e.g., "3.12" or "3.10")
    
    if ! conda env list | grep -q "$env_name"; then
        return 0  # Environment doesn't exist, skip
    fi
    
    log_verbose "Installing RoboEval into $env_name..."
    
    # Ensure build tools are set up
    ensure_build_tools
    
    # Install RoboEval
    install_roboeval_from_clone "$env_name" "$CLONE_DIR"
    
    # Get site-packages location for base env (for thirdparty source)
    local base_site_packages=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    
    # Copy thirdparty to user site-packages
    copy_thirdparty_to_user_site "$py_minor" "$base_site_packages" "$CLONE_DIR"
    
    log_verbose "✓ RoboEval installed in $env_name"
}

# Ensure build tools are available
ensure_build_tools

# Install RoboEval in base environment
install_roboeval_from_clone "base" "$CLONE_DIR"

# Get site-packages location
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Copy thirdparty directory to base environment
cp -r $CLONE_DIR/thirdparty $SITE_PACKAGES/ || true

log_verbose "✓ RoboEval installed in base environment"

# Install git-based packages and RoboEval in openpi_env if it exists
# Gate behind INSTALL_GIT_DEPS flag (default: install if not set, but can be disabled)
if [ "${INSTALL_GIT_DEPS:-1}" = "1" ]; then
    if conda env list | grep -q "openpi_env"; then
        log_verbose "Installing git-based packages in openpi_env..."
        conda run -n openpi_env pip install --no-cache-dir \
            git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
            git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git#subdirectory=packages/openpi-client --no-deps \
            git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git --no-deps --force-reinstall || {
            log_verbose "⚠️  Warning: Failed to install git packages in openpi_env"
        }
        
        log_verbose "Installing additional OpenPI dependencies in openpi_env..."
        conda run -n openpi_env pip install --no-cache-dir \
            draccus>=0.1.0 \
            jsonlines>=4.0.0 || {
            log_verbose "⚠️  Warning: Failed to install some additional dependencies in openpi_env"
        }
    fi
fi

# Install RoboEval into openpi_env
install_roboeval_into_env "openpi_env" "3.12"

# Install git-based packages and RoboEval in openvla_env if it exists
if [ "${INSTALL_GIT_DEPS:-1}" = "1" ]; then
    if conda env list | grep -q "openvla_env"; then
        log_verbose "Installing git-based packages in openvla_env..."
        conda run -n openvla_env pip install --no-cache-dir \
            git+https://github.com/openvla/openvla.git || {
            log_verbose "⚠️  Warning: Failed to install OpenVLA in openvla_env"
        }
    fi
fi

# Install RoboEval into openvla_env
install_roboeval_into_env "openvla_env" "3.10"

log_verbose "✓ RoboEval installation complete"
