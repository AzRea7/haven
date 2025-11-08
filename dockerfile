# syntax=docker/dockerfile:1.6

########################
# Builder: deps -> wheels
########################
FROM python:3.10-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git && \
    rm -rf /var/lib/apt/lists/*

# Lean runtime deps file (make sure this exists in repo)
COPY requirements.api.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --wheel-dir=/wheels -r requirements.api.txt

########################
# Runtime: slim image
########################
FROM python:3.10-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

# Install deps from wheels as root into system site-packages
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Create entrypoints dir as root
RUN mkdir -p /entrypoints

# Create non-root user and give ownership of app + entrypoints
RUN useradd -ms /bin/bash appuser && \
    chown -R appuser:appuser /app /entrypoints

# Copy app code + entrypoint with correct ownership
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser --chmod=755 entrypoints/start.sh /entrypoints/start.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoints/start.sh"]
