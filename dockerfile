# --- Builder: install deps, build wheels ---
FROM python:3.10-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps if you later add geo/ML libs; keep minimal for now
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git && \
    rm -rf /var/lib/apt/lists/*

# Copy metadata first for better layer caching
COPY pyproject.toml ./pyproject.toml
COPY requirements.txt ./requirements.txt

# Build wheels for all Python deps
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# Copy source
COPY src ./src

# --- Runtime: slim image with only what we need ---
FROM python:3.10-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Non-root user (safer in prod)
RUN useradd -ms /bin/bash appuser
USER appuser

# Bring wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# App code
COPY --chown=appuser:appuser src ./src
ENV PYTHONPATH=/app/src

# Entrypoint
COPY --chown=appuser:appuser entrypoints/start.sh /entrypoints/start.sh
RUN chmod +x /entrypoints/start.sh

EXPOSE 8000
CMD ["/entrypoints/start.sh"]
