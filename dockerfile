# syntax=docker/dockerfile:1.6

########################
# Builder: deps -> wheels
########################
FROM python:3.10-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps; extend only if really needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git && \
    rm -rf /var/lib/apt/lists/*

# Use the lean runtime requirements (NOT the giant dev one)
COPY requirements.api.txt .

# Build wheels for all runtime deps (cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --wheel-dir=/wheels -r requirements.api.txt

# (Optional) if you ever need build-time code, copy it here
# COPY src ./src

########################
# Runtime: slim + only what we need
########################
FROM python:3.10-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

# Install deps from prebuilt wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# App code
COPY --chown=appuser:appuser src ./src

# Entrypoint setup
RUN mkdir -p /entrypoints
COPY --chown=appuser:appuser --chmod=755 entrypoints/start.sh /entrypoints/start.sh

EXPOSE 8000

# Single source of truth: start script
ENTRYPOINT ["/entrypoints/start.sh"]
