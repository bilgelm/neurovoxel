# syntax=docker/dockerfile:1.7-labs

##############################
# Stage 1: Build dependencies
##############################
FROM python:3.13-slim-bookworm AS deps
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
WORKDIR /app

# Install uv (static binaries)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Only project metadata up front for better caching
COPY pyproject.toml uv.lock ./

# Cache uv downloads so rebuilds reuse wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

##############################
# Stage 2: Runtime
##############################
FROM python:3.13-slim-bookworm AS runtime
WORKDIR /app

# (Optional) curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Bring the prebuilt virtual environment
COPY --from=deps /app/.venv ./.venv

# Copy your runtime code (avoid tests, etc. via .dockerignore)
COPY src ./src
COPY .streamlit ./.streamlit
COPY pyproject.toml uv.lock README.md ./

# Make venv tools first on PATH and expose src/ to imports
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit from the prebuilt venv
CMD ["streamlit", "run", "src/neurovoxel/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
