# Li-Ionics ML Model

A CrabNet-based model for predicting lithium solid-state electrolyte conductivity from chemical composition, deployed on the LMDS (Local Model Deployment Service) platform.

## What it does

Given a chemical composition (e.g. `LiPO3`), the model returns:

- **Classification** (int 0/1): 1 if the predicted conductivity > 10⁻⁴ S/cm, 0 otherwise
- **Regression** (float): predicted conductivity in log10(S/cm)

This is a hosted version of the deep learning model introduced in:

- *A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning*
- *The Liverpool MaterialS Discover server: A Suite of Computational Tools for the Collaborative discovery of Materials*

The classification model achieves an accuracy of 0.71; the regression model has a mean absolute error of 0.99. The model was trained on The Liverpool Ionics Dataset using a CrabNet architecture.

## Quick Start

```bash
# 1. Run instance mode locally (single composition)
./scripts/run_instance.sh LiPO3
# Output: {"query": "LiPO3", "classification": 1, "regression": -1.23}

# 2. Run dataset mode with inline compositions (API-style)
./scripts/run_api.sh "LiPO3" "NaCl" "SrTiO3"
# Output: result.json with predictions for all three

# 3. Run dataset mode with a CSV file
./scripts/run_dataset.sh data/compositions.csv
```

## Project Structure

```
├── schema/
│   ├── model.json              # Model contract (name, modes, I/O)
│   └── model.schema.json       # JSON Schema meta-schema (shared)
├── src/
│   ├── cmds/
│   │   ├── run.py              # Entrypoint — copy as-is
│   │   ├── model_types.py      # Generate TypedDicts from model.json
│   │   └── validate_model.py   # Validate model.json against schema
│   ├── infrastructure/          # Copy as-is — runner, S3, storage, logging
│   └── model/
│       ├── __init__.py         # Wire handler: handler = LiIonModel()
│       ├── model.py             # LiIonModel — process() implementation
│       ├── crab/                # Vendored CrabNet architecture
│       │   ├── kingcrab.py       # CrabNet nn.Module (transformer)
│       │   └── model.py         # Model wrapper (load_network, predict)
│       ├── utils/               # Vendored utilities
│       │   ├── composition.py    # Formula parsing
│       │   ├── utils.py          # EDM_CsvLoader, Scaler, optimizers
│       │   ├── optim.py          # SWA optimizer
│       │   └── get_compute_device.py
│       └── types.py             # Auto-generated types (optional)
├── data/                        # NOT tracked in git
│   ├── trained_models/          # CrabNet checkpoints (~111 MB)
│   │   ├── TransferFinalModel_Reg.pth
│   │   └── TransferFinalModel_Clf.pth
│   ├── element_properties/      # Element embeddings
│   │   └── mat2vec.csv
│   └── compositions.csv         # Sample dataset
├── scripts/
│   ├── run_instance.sh          # Single composition
│   ├── run_api.sh               # Inline batch (API-style)
│   └── run_dataset.sh           # CSV file batch
├── tests/
│   ├── conftest.py              # Mock model checkpoints for testing
│   └── test_model.py            # pytest tests
├── Dockerfile                   # Multi-stage build
├── pyproject.toml               # Package config
└── .env.example                 # Environment variables
```

## Data Files

The trained model weights and element property data are **not tracked in git** (they total ~111 MB). They must be present in `data/` before running or building Docker:

```
data/trained_models/TransferFinalModel_Reg.pth   (55.6 MB)
data/trained_models/TransferFinalModel_Clf.pth   (55.6 MB)
data/element_properties/mat2vec.csv               (466 KB)
```

If these files are missing, the model will raise a clear `FileNotFoundError` at startup.

## Modes

### Instance mode (immediate)

Single composition → inline JSON response.

```bash
uv run model-run --local --parameters '{"mode":"instance","inputs":{"query":{"value":"LiPO3"}},"parameters":{}}'
```

Response:

```json
{"query": "LiPO3", "classification": 1, "regression": -1.23}
```

### Dataset mode (deferred)

Batch of compositions (inline list or CSV file) → `result.json` asset.

**Inline list:**

```bash
uv run model-run --local --parameters '{"mode":"dataset","inputs":{"compositions":{"value":["LiPO3","NaCl","SrTiO3"]}},"parameters":{}}'
```

**CSV file:**

```bash
uv run model-run --local --parameters '{"mode":"dataset","inputs":{"file":{"uri":"file://data/compositions.csv","mime_type":"text/csv"}},"parameters":{}}'
```

The CSV must have a `composition` column.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `S3_ENDPOINT` | `http://minio:9000` | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET` | `lmds` | S3 bucket name |
| `LIION_MODELS_PATH` | `data/trained_models` | Path to CrabNet checkpoints |

## GPU Support

The model auto-detects CUDA via `torch.cuda.is_available()`. If a GPU is available, CrabNet runs on it automatically. To force CPU, set `CUDA_VISIBLE_DEVICES=""`.

## Docker

```bash
# Build
docker build -t liion:1.0.0 .

# Run instance mode
docker run --rm liion:1.0.0 --local --parameters '{"mode":"instance","inputs":{"query":{"value":"LiPO3"}},"parameters":{}}'

# Run dataset mode
docker run --rm liion:1.0.0 --local --parameters '{"mode":"dataset","inputs":{"compositions":{"value":["LiPO3","NaCl"]}},"parameters":{}}'
```

## Citing

Please cite the following papers if you use this tool:

- *A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning*
- *The Liverpool MaterialS Discover server: A Suite of Computational Tools for the Collaborative discovery of Materials*
