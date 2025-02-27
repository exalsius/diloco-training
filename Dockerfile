FROM python:3.12-slim AS base

#### building dependencies
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

COPY uv.lock pyproject.toml /app/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-project --no-dev

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev

##### final image
FROM base

# Copy only what's needed from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/diloco_training /app/diloco_training

ENV PYTHONPATH="/app"
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 29500
