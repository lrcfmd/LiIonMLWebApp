# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# Stage 1: Build wheel using uv
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
COPY src/ ./src/
RUN uv build --wheel

# Stage 2: Runtime image
FROM python:${PYTHON_VERSION}-slim
WORKDIR /app

# Install torch. Default is CPU-only (small image, ~200 MB).
# For GPU support, build with:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 ...
# Common CUDA indices:
#   cu126 (CUDA 12.6), cu124 (CUDA 12.4), cu121 (CUDA 12.1)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}"

COPY --from=builder /app/dist/*.whl /app/
RUN WHEEL=$(ls /app/*.whl) && pip install --no-cache-dir "$WHEEL"

# Trained model weights and element property data are injected at runtime
# via the LMDS tool assets system (see schema/model.json → assets).
# Do not COPY them here — they are downloaded into /app/data/ when a run starts.

RUN mkdir -p /app/data/output /app/data/trained_models /app/data/element_properties /home/lmds \
    && groupadd lmds \
    && useradd -g lmds -u 1000 -m -d /home/lmds lmds \
    && chown -R 1000:1000 /app /home/lmds
ENV LIION_MODELS_PATH=/app/data/trained_models

USER 1000:1000

ENTRYPOINT ["model-run"]