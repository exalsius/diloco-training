# --- Builder with CUDA toolchain ---
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel AS builder

# uv
COPY --from=ghcr.io/astral-sh/uv:0.8.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin

WORKDIR /app


# Build tools
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential git cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*


# Resolve deps into a venv (no dev)
COPY uv.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN /app/.venv/bin/python -m ensurepip --upgrade || true \
 && /app/.venv/bin/python -m pip install --upgrade pip wheel setuptools ninja packaging


RUN --mount=type=cache,target=/root/.cache/pip \
    /app/.venv/bin/python -m pip install --no-build-isolation --verbose flash-attn

RUN /app/.venv/bin/python -c "import flash_attn, torch; print('flash_attn', flash_attn.__version__, ' torch', torch.__version__)"


FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime AS final
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin

# Build tools
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/diloco_training /app/diloco_training

ENV PYTHONPATH="/app" \
    PATH="/app/.venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1

EXPOSE 29500
