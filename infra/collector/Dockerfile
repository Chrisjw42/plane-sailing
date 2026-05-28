# 24/7 ADSB collector image.
# Build context is the plane-sailing/ repo root (see ../docker-compose.yaml).
FROM python:3.11-slim

# librtlsdr0:      runtime .so that pyrtlsdr dlopens to talk to the SDR.
# git + ca-certs:  so uv can fetch the cosmosdr git dependency over HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        librtlsdr0 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.5"

# Run as a non-root user whose uid matches the host owner (chris, 1000) so files
# written into the bind-mounted log volume are owned sensibly on the host. USB
# access is granted at runtime via the compose `group_add: ["46"]` (plugdev).
RUN useradd -m -u 1000 cosmos

WORKDIR /app
# PYTHONPATH=/app makes the `planesailing` package importable regardless of how
# python is invoked. The project's pyproject declares `py-modules=["planesailing"]`
# (single-file module) but planesailing is actually a *package* (directory), so
# `uv sync` installs nothing for it into site-packages — it only ever imported by
# accident when cwd happened to be the repo root. Running a script file
# (python /app/infra/collector/run.py) puts the *script's* dir on sys.path, not
# the cwd, so the accident no longer saves us. /app on PYTHONPATH fixes it cleanly
# without touching the application's packaging.
ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Dependency layer: cached unless pyproject/uv.lock change. --no-install-project
# keeps app-code edits below from busting this (expensive) layer. README.md is
# referenced by pyproject's `readme` field, so it must be present.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code (consumed as a library) + orchestration entrypoint.
COPY planesailing/ ./planesailing/
COPY infra/collector/run.py infra/collector/healthcheck.py ./infra/collector/

# Install the project itself into the venv.
RUN uv sync --frozen --no-dev

USER cosmos

# Use the venv python directly (it's on PATH). NOT `uv run` — at runtime uv
# re-syncs the project env, and as the non-root user it can't write the
# root-owned /app/.venv, so it rebuilds a fresh env and re-downloads every
# dependency on each start. Deps are already baked in by the uv sync steps above.
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s \
    CMD python /app/infra/collector/healthcheck.py || exit 1

ENTRYPOINT ["python", "/app/infra/collector/run.py"]
