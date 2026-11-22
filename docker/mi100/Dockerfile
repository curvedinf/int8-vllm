FROM rocm/dev-ubuntu-24.04:10.2.0-full@sha256:21801c0ba1bc4aba31f81ab11fe576c43daa7f2c7f42bac7baaa4d2f75273f86

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    amdrocm-hipfile10.2=10.2.0-2 build-essential ca-certificates curl git libclang-dev libnuma-dev libssl-dev \
    pkg-config python3.12-dev python3.12-venv && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/int8-vllm
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 \
    VLLM_TARGET_DEVICE=rocm LD_LIBRARY_PATH=/opt/rocm/lib \
    PATH=/opt/int8-vllm/.venv/bin:/root/.cargo/bin:/opt/rocm/bin:$PATH
COPY rust-toolchain.toml ./
RUN curl -fsSLo /tmp/rustup.sh https://sh.rustup.rs && sh /tmp/rustup.sh -y --profile minimal --default-toolchain none && cargo --version
RUN python3.12 -m venv .venv
COPY requirements/ requirements/
COPY docker/mi100/install-deps.sh docker/mi100/
RUN --mount=type=cache,target=/root/.cache/pip docker/mi100/install-deps.sh --rocm10.2 \
    && .venv/bin/python -c 'import ctypes, torch; ctypes.CDLL("libhipfile.so.0")'
# Keep runtime launcher changes out of the native extension build cache key.
COPY CMakeLists.txt LICENSE MANIFEST.in README.md pyproject.toml setup.py ./
COPY cmake/ cmake/
COPY csrc/ csrc/
COPY rust/ rust/
COPY tools/ tools/
COPY vllm/ vllm/
COPY scripts/build_residual_g128.sh scripts/
ARG MAX_JOBS=4
ARG VLLM_VERSION_OVERRIDE=0.33.0+mi100
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/opt/rust-target \
    export CARGO_BUILD_JOBS="$MAX_JOBS" CARGO_TARGET_DIR=/opt/rust-target \
    && .venv/bin/python -m pip install --dry-run --no-build-isolation --no-deps -e . \
    && .venv/bin/python -m pip install --no-build-isolation --no-deps -e . \
    && scripts/build_residual_g128.sh && .venv/bin/python -c 'import vllm._rocm_C'
# HY4's gfx908 matrix/vector schedules require this FlyDSL compiler ABI.
RUN --mount=type=cache,target=/root/.cache/pip \
    .venv/bin/python -m pip install --dry-run 'flydsl==0.5.0' \
    && .venv/bin/python -m pip install 'flydsl==0.5.0' \
    && .venv/bin/python -c 'import flydsl.compiler'
# AITER builds its kernels at runtime from this pinned checkout.
ARG AITER_REV=bbc57fe793043748ac15ae7309a66b28a425cea4
RUN git init /opt/int8-aiter && cd /opt/int8-aiter && git remote add origin https://github.com/curvedinf/int8-aiter.git \
    && git fetch --depth=1 origin "$AITER_REV" && git checkout --detach FETCH_HEAD \
    && git submodule update --init --recursive --depth=1
# The optional per-worker CPU map uses vLLM's numactl wrapper at runtime.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends numactl \
    && rm -rf /var/lib/apt/lists/*
COPY python_startup/ python_startup/
COPY scripts/ scripts/
# Use the toolkit compiler for AITER JIT. The wheel also installs a hipcc
# shim, but its runtime-only SDK omits the device bitcode.
ENV MI100_CONTAINER=1 PATH=/opt/rocm/bin:/opt/int8-vllm/.venv/bin:/root/.cargo/bin:$PATH \
    PYTHONPATH=/opt/int8-vllm/python_startup:/opt/int8-vllm:/opt/int8-aiter \
    AITER_JIT_DIR=/cache/aiter/${AITER_REV}-gfx908-rocm10.2.0-torch2.15
ENTRYPOINT ["scripts/mi100-serve.sh"]
