# Multi-stage Dockerfile for RoboEval Gradio Space
# Build stage: Install build tools and build RoboEval
# Runtime stage: Create conda envs and run app

# ==================== BUILD STAGE ====================
FROM continuumio/anaconda3:main AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update -qq && \
    apt-get install -y -qq \
        build-essential \
        g++ \
        gcc \
        make \
        cmake \
        pkg-config \
        wget \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install Bazel (bazelisk)
RUN BAZELISK_VERSION="v1.19.0" && \
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" && \
    mkdir -p /usr/local/bin && \
    wget -q ${BAZELISK_URL} -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel && \
    export USE_BAZEL_VERSION=7.5.0

# Install Rust (required for safetensors)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH"

# Copy build script
COPY build_roboeval.sh /build/
RUN chmod +x /build/build_roboeval.sh

# Build RoboEval (GH_TOKEN passed as build arg)
ARG GH_TOKEN
ENV GH_TOKEN=${GH_TOKEN}
ENV PATH="$HOME/.cargo/bin:$PATH"
ENV USE_BAZEL_VERSION=7.5.0
ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

RUN /build/build_roboeval.sh

# Package RoboEval artifacts for copying to runtime stage
RUN PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    SITE_PACKAGES="/opt/conda/lib/python${PYTHON_VERSION}/site-packages" && \
    mkdir -p /build/artifacts && \
    cp -r ${SITE_PACKAGES}/roboeval* /build/artifacts/ 2>/dev/null || true && \
    cp -r ${SITE_PACKAGES}/thirdparty /build/artifacts/ 2>/dev/null || true

# ==================== RUNTIME STAGE ====================
FROM continuumio/anaconda3:main

WORKDIR /code

# Install base dependencies (Gradio and minimal requirements)
RUN conda install -n base -y \
        pip \
    && conda run -n base pip install --no-cache-dir \
        "gradio>=4.0.0" \
        "numpy>=1.22.4,<2.0.0" \
        "pillow>=11.0.0" \
        "huggingface-hub>=0.20.0,<0.26.0"

# Copy RoboEval artifacts from builder stage
COPY --from=builder /build/artifacts /tmp/roboeval_artifacts
RUN PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    SITE_PACKAGES="/opt/conda/lib/python${PYTHON_VERSION}/site-packages" && \
    cp -r /tmp/roboeval_artifacts/roboeval* ${SITE_PACKAGES}/ 2>/dev/null || true && \
    cp -r /tmp/roboeval_artifacts/thirdparty ${SITE_PACKAGES}/ 2>/dev/null || true && \
    rm -rf /tmp/roboeval_artifacts

# Copy environment YAML files
COPY environment_openpi.yml /code/
COPY environment_openvla.yml /code/

# Create OpenPI environment
RUN conda env create -f /code/environment_openpi.yml

# Install git-based packages in openpi_env (requires GH_TOKEN)
ARG GH_TOKEN
ENV GH_TOKEN=${GH_TOKEN}
RUN if [ -n "$GH_TOKEN" ]; then \
        conda run -n openpi_env pip install --no-cache-dir \
            git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
            git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git#subdirectory=packages/openpi-client --no-deps \
            git+https://${GH_TOKEN}@github.com/tan7271/OpenPiRoboEval.git --no-deps --force-reinstall; \
    else \
        echo "⚠️  Warning: GH_TOKEN not set. Skipping private git installs."; \
    fi

# Install RoboEval into openpi_env
RUN PYTHON_VERSION=$(conda run -n openpi_env python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    OPENPI_SITE=$(conda run -n openpi_env python -c "import site; print(site.getsitepackages()[0])") && \
    BASE_SITE="/opt/conda/lib/python${PYTHON_VERSION}/site-packages" && \
    cp -r ${BASE_SITE}/roboeval* ${OPENPI_SITE}/ 2>/dev/null || true && \
    cp -r ${BASE_SITE}/thirdparty ${OPENPI_SITE}/ 2>/dev/null || true

# Create OpenVLA environment (currently disabled - uncomment to enable)
# RUN conda env create -f /code/environment_openvla.yml
# RUN conda run -n openvla_env pip install --no-cache-dir \
#         git+https://github.com/openvla/openvla.git
# RUN PYTHON_VERSION=$(conda run -n openvla_env python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
#     OPENVLA_SITE=$(conda run -n openvla_env python -c "import site; print(site.getsitepackages()[0])") && \
#     BASE_SITE="/opt/conda/lib/python${PYTHON_VERSION}/site-packages" && \
#     cp -r ${BASE_SITE}/roboeval* ${OPENVLA_SITE}/ 2>/dev/null || true && \
#     cp -r ${BASE_SITE}/thirdparty ${OPENVLA_SITE}/ 2>/dev/null || true

# Set up non-root user
RUN useradd -m -u 1000 user

# Switch to non-root user
USER user

# Set environment variables
ENV HOME=/home/user \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    XDG_RUNTIME_DIR=/tmp

# Set working directory
WORKDIR $HOME/app

# Copy application code
COPY --chown=user . $HOME/app

# Make run.sh executable
RUN chmod +x $HOME/app/run.sh

# Expose port
EXPOSE 7860

# Run the application
CMD ["./run.sh"]

