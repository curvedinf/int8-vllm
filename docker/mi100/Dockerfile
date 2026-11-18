FROM rocm/dev-ubuntu-24.04:7.2.1-complete@sha256:3db551c4e1229aac1857ac44fcb6141bb749f41348eb572452f11279153c13c3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git libclang-dev libnuma-dev libssl-dev \
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
RUN --mount=type=cache,target=/root/.cache/pip docker/mi100/install-deps.sh
ARG AITER_REV=4bd8f95f3a5cd3016592fd34ed5fb2cefc9670f0
RUN git init /opt/int8-aiter && cd /opt/int8-aiter && git remote add origin https://github.com/curvedinf/int8-aiter.git \
    && git fetch --depth=1 origin "$AITER_REV" && git checkout --detach FETCH_HEAD \
    && git submodule update --init --recursive --depth=1
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
COPY python_startup/ python_startup/
COPY scripts/ scripts/
ENV PYTHONPATH=/opt/int8-vllm/python_startup:/opt/int8-vllm:/opt/int8-aiter \
    AITER_JIT_DIR=/cache/aiter/${AITER_REV}-gfx908
ENTRYPOINT ["scripts/mi100-serve.sh"]
