# ─────────────────────────────────────────────────────────────
# Dockerfile
# ─────────────────────────────────────────────────────────────
# WHAT IS A DOCKERFILE?
#   A script that tells Docker how to build a container image.
#   A container is like a self-contained mini-computer with our app
#   pre-installed, so it runs the same everywhere.
#
# HF SPACES uses this to deploy our environment.
# ─────────────────────────────────────────────────────────────

# Start from an official Python 3.11 image (slim = smaller size)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first (Docker caches this layer)
COPY requirements.txt .

# Install Python packages
# --no-cache-dir  = don't store pip cache (saves disk space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Tell Docker that our app listens on port 7860
# (HF Spaces requires port 7860)
EXPOSE 7860

# Set environment variables with sensible defaults
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# The command to run when the container starts
# -   "uvicorn"           : the ASGI server (runs FastAPI)
# -   "server.app:app"    : file server/app.py, the `app` object
# -   "--host 0.0.0.0"    : accept connections from any IP
# -   "--port 7860"       : on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]