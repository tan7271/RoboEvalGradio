# Multi-stage Dockerfile for RoboEval Gradio Space
# Build stage: Install build tools and build RoboEval
# Runtime stage: Create conda envs and run app

# ==================== RUNTIME STAGE ====================
# Note: RoboEval will be installed at runtime using GH_TOKEN environment variable
# (HuggingFace Spaces doesn't pass secrets as Docker build args)
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

# Install build tools needed for runtime RoboEval installation
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

# Install Bazel (bazelisk) for runtime RoboEval builds
RUN BAZELISK_VERSION="v1.19.0" && \
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" && \
    mkdir -p /usr/local/bin && \
    wget -q ${BAZELISK_URL} -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Install Rust (required for safetensors) for runtime RoboEval builds
# Install for root user, make available system-wide
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    chmod -R 755 /root/.cargo && \
    chmod -R 755 /root/.rustup && \
    # Add Rust to system PATH so it's available to all users
    echo 'export PATH="/root/.cargo/bin:$PATH"' >> /etc/profile.d/rust.sh && \
    chmod +x /etc/profile.d/rust.sh

# Copy environment YAML files
COPY environment_openpi.yml /code/
COPY environment_openvla.yml /code/

# Create OpenPI environment
RUN conda env create -f /code/environment_openpi.yml

# Note: Git-based packages and RoboEval will be installed at runtime
# (GH_TOKEN is available as environment variable at runtime, not build time)

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

# Give user write access to conda environments (needed for runtime installation)
RUN chmod -R u+w /opt/conda/envs/ 2>/dev/null || true

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
    XDG_RUNTIME_DIR=/tmp \
    OMP_NUM_THREADS=1 \
    PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR $HOME/app

# Copy application code
COPY --chown=user . $HOME/app

# Copy runtime installation script
COPY --chown=user install_roboeval_runtime.sh $HOME/app/
RUN chmod +x $HOME/app/install_roboeval_runtime.sh

# Make run.sh executable
RUN chmod +x $HOME/app/run.sh

# Expose port
EXPOSE 7860

# Run the application
CMD ["./run.sh"]

