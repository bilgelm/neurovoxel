# --- Base Stage: Install Dependencies ---
FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=True
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project definition files for caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# --- Final Stage: Build Application Image ---
FROM python:3.13-slim-bookworm AS runner

WORKDIR /app

# Copy the virtual environment from the base stage
COPY --from=base /app/.venv ./.venv

# Copy application code
COPY . /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the entrypoint
CMD ["/app/.venv/bin/streamlit", "run", "/app/src/neurovoxel/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
