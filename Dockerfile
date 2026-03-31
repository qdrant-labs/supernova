FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY vectorforge/ vectorforge/
COPY scripts/ scripts/
COPY configs/ configs/

# Install dependencies
RUN uv sync --frozen --no-dev

# CONFIG_PATH is the path to the YAML config file inside the container
# e.g. configs/wikipedia_openai.yaml
ENV CONFIG_PATH=configs/wikipedia_openai.yaml

ENTRYPOINT ["uv", "run", "python", "scripts/run_pipeline.py"]
CMD ["configs/wikipedia_openai.yaml"]
