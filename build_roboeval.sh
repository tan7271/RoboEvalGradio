#!/bin/bash
set -e

# Build script for RoboEval - runs in Docker build stage with Bazel/Rust available
# This script builds RoboEval and outputs artifacts to be copied to runtime stage

echo "===== Building RoboEval ====="

# GH_TOKEN should be passed as build arg or env var
# Check if GH_TOKEN is set (required for private repo)
# Try multiple ways to get the token (build arg, env var, etc.)
if [ -z "$GH_TOKEN" ] || [ "$GH_TOKEN" = "" ]; then
    echo "❌ ERROR: GH_TOKEN not set. Cannot clone private RoboEval repository."
    echo ""
    echo "   Troubleshooting:"
    echo "   1. Go to your HuggingFace Space Settings > Variables and secrets"
    echo "   2. Add a SECRET (not variable) named exactly: GH_TOKEN"
    echo "   3. Set the value to your GitHub personal access token"
    echo "   4. Make sure the token has 'repo' scope"
    echo "   5. Rebuild the Space"
    echo ""
    echo "   Note: For Docker builds, secrets may need to be explicitly configured."
    echo "   If the secret is set but still not working, check HuggingFace Spaces"
    echo "   documentation for Docker build secret configuration."
    exit 1
fi

CLONE_DIR="/tmp/roboeval_install"
rm -rf $CLONE_DIR

# Clone with submodules (using token for authentication)
echo "Cloning RoboEval repository with submodules..."
echo "Using GH_TOKEN for authentication..."
git clone --recurse-submodules https://${GH_TOKEN}@github.com/helen9975/RoboEval.git $CLONE_DIR

# Set environment variables for building
# PyO3 compatibility for Python 3.13 (safetensors)
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Use older bazel version for labmaze compatibility (Bazel 7.x works better)
export USE_BAZEL_VERSION=7.5.0

# Install RoboEval (this will build labmaze which requires bazel)
echo "Installing RoboEval from cloned repository..."
echo "Note: This may take several minutes as it builds labmaze with bazel..."

# Install RoboEval
pip install $CLONE_DIR --no-cache-dir || {
    echo "⚠️  RoboEval installation had some errors, but continuing..."
    echo "Some optional dependencies may not be available"
    echo "Trying to install pre-built safetensors wheel..."
    # Try to install safetensors from pre-built wheel
    pip install safetensors --only-binary :all: || pip install "safetensors>=0.4.1" --no-build-isolation || true
}

# Get site-packages location
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Copy thirdparty directory
cp -r $CLONE_DIR/thirdparty $SITE_PACKAGES/ || true

echo "✓ RoboEval built successfully"
echo "  Site packages: $SITE_PACKAGES"
echo "  RoboEval package: $SITE_PACKAGES/roboeval*"
echo "  Thirdparty: $SITE_PACKAGES/thirdparty"

