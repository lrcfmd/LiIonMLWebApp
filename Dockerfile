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

# Bake trained model checkpoints and element property data into the image.
# These files must exist in data/ before building:
#   data/trained_models/TransferFinalModel_Reg.pth   — CrabNet regression weights
#   data/trained_models/TransferFinalModel_Clf.pth   — CrabNet classification weights
#   data/element_properties/mat2vec.csv              — element embeddings
COPY data/trained_models/ /app/data/trained_models/
COPY data/element_properties/ /app/data/element_properties/

# Alternatively, mount at runtime and remove the COPY above:
#   docker run -v /path/to/models:/app/data/trained_models \
#              -v /path/to/element_properties:/app/data/element_properties ...
RUN mkdir -p /app/data/output
ENV LIION_MODELS_PATH=/app/data/trained_models

ENTRYPOINT ["model-run"]