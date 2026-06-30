FROM python:3.12-slim

# Switch to Tsinghua mirror for faster APT downloads and to avoid 502 errors
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# Install necessary system dependencies for audio processing and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Use Tsinghua mirror to speed up package downloads in China
ENV UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# Create virtual environment outside the workspace so it is not shadowed by host volume mounts
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Install dependencies into the virtualenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application code
COPY . .

# Install the application
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/opt/venv/bin:$PATH"

# Default command (can be overridden, e.g. for running scripts)
CMD ["/bin/bash"]
