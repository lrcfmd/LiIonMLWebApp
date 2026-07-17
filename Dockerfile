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

# Install the model wheel (CPU torch by default).
# For GPU support, override the base image to pytorch/pytorch:*-cuda*
# or install torch with the CUDA extra.
COPY --from=builder /app/dist/*.whl /app/
RUN WHEEL=$(ls /app/*.whl) && pip install --no-cache-dir "$WHEEL"

# Trained model weights and element property data are injected at runtime
# via the LMDS tool assets system (see schema/model.json → assets).
# Do not COPY them here — they are downloaded into /app/data/ when a run starts.

RUN mkdir -p /app/data/output /app/data/trained_models /app/data/element_properties
ENV LIION_MODELS_PATH=/app/data/trained_models

ENTRYPOINT ["model-run"]