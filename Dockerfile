# ===========================================================================
# scientific-llm - Step 8b: Docker image for the FastAPI serving layer
# (Step 8a).
# ===========================================================================
# Base image note: this uses plain python:3.11-slim, NOT an nvidia/cuda
# base image, and NOT Python 3.13 (your local venv's version). Both are
# deliberate:
#   - torch's pip wheels (installed below via the same cu124 index-url
#     setup_step1.ps1 uses) bundle their own CUDA runtime libraries, and
#     bitsandbytes bundles its own precompiled CUDA kernels too - neither
#     needs the system CUDA toolkit installed inside the container, only
#     an NVIDIA driver on the HOST plus GPU passthrough at `docker run`
#     time (--gpus all - see README Step 8b). This is the same reason
#     your Windows venv never needed a separate CUDA toolkit install.
#   - Python 3.11 rather than 3.13: none of this project's dependencies
#     are pinned to an exact Python version, and 3.11 is the more
#     broadly-tested version across the ML container ecosystem right
#     now. The container is a separate, self-contained artifact from
#     your dev venv, not a copy of it - if you want exact version parity
#     instead, this is the one line to change, but it is not needed for
#     anything to work.
#
# What is copied in: only src/__init__.py, src/api/, and src/model/ -
# the actual import chain of the two endpoints this image serves
# (/health, /generate). The model cache and any trained adapter
# checkpoint are NOT baked into the image - see README Step 8b for why,
# and mount them as volumes at `docker run` time instead.
# ===========================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    ADAPTER_DIR=/app/outputs/checkpoints/step4_verify

WORKDIR /app

# curl: used by the HEALTHCHECK below (and handy for debugging inside
# the container). build-essential: defensive - most of this project's
# dependencies ship prebuilt wheels for common platforms, but if pip
# ever needs to compile a fallback sdist for a transitive dependency,
# this avoids an otherwise-confusing build failure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/__init__.py src/__init__.py
COPY src/api/ src/api/
COPY src/model/ src/model/

EXPOSE 8000

# Matches the /health endpoint's own contract (schemas.py's
# HealthResponse) - "ok" means the model actually loaded, not just that
# the process is alive, so Docker's health status reflects the thing
# that actually matters here.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health | grep -q '"status":"ok"' || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
