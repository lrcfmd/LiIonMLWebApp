# CONTEXT — Li-Ionics ML Migration to LMDS App-Template

## Overview

This document records the migration of the Li-Ionics ML Flask application
(`tools/liion`) to the LMDS app-template model-service format. The original
repository hosted a CrabNet-based lithium solid-state electrolyte conductivity
predictor as a Flask web app with a search bar and a REST API. The migration
replaces the Flask layer with the LMDS runner infrastructure while keeping the
CrabNet model code and trained weights intact.

## Original Application

The repository at `tools/liion` was a minimal Flask app mirroring
[https://lmds.liverpool.ac.uk/ionics_ml](https://lmds.liverpool.ac.uk/ionics_ml).

### What it did

Given a chemical composition (or a comma-separated list of up to ~2,000), it
returned two predictions per composition:

1. **Classification** (int 0/1): 1 if predicted conductivity > 10⁻⁴ S/cm, 0
   otherwise. Accuracy: 0.71.
2. **Regression** (float): predicted conductivity in log10(S/cm). MAE: 0.99.

### Architecture

- **Flask web form** (`/predict`): user enters comma-separated compositions in a
  search bar; returns an HTML table of `(composition, classification, regression)`.
- **REST API** (`/predict/API`, PUT/POST): accepts `{"compositions": "LiPO3"}` or
  `{"compositions": ["LiPO3", "NaCl"]}`; returns
  `{"Ionics ML Results": [[composition, classification_int, regression_float], ...]}`.
- **Model**: CrabNet — a transformer-based architecture for materials. Two
  pre-trained PyTorch checkpoints (55.6 MB each):
  - `TransferFinalModel_Reg.pth` (regression)
  - `TransferFinalModel_Clf.pth` (classification)
- **Feature extraction**: `ElMD(comp, metric="mat2vec").feature_vector` converts
  chemical compositions to element-descriptor vectors.
- **Data**: `mat2vec.csv` (466 KB) — element property embeddings used by the
  CrabNet `Embedder` layer.

### Source layout (before migration)

```
tools/liion/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── forms.py             # WTForms SearchForm
│   ├── routes.py            # /predict (web) + /predict/API (REST)
│   ├── templates/
│   │   ├── ionics_ml.html
│   │   └── api_info.html
│   └── trained_models/
│       ├── TransferFinalModel_Reg.pth   (55.6 MB, tracked in git)
│       └── TransferFinalModel_Clf.pth   (55.6 MB, tracked in git)
├── crab/
│   ├── kingcrab.py          # CrabNet nn.Module (transformer encoder + residual NN)
│   └── model.py             # Model wrapper: load_data, predict, load_network, fit
├── utils/
│   ├── composition.py       # Formula parsing (parse_formula, _element_composition)
│   ├── utils.py             # EDM_CsvLoader, Scaler, DummyScaler, Lamb, Lookahead, losses
│   ├── optim.py             # SWA optimizer
│   ├── get_compute_device.py # CUDA auto-detection
│   ├── get_core_count.py
│   ├── figures.py           # Plotting helpers (training only)
│   ├── estimatorselectionhelper.py  # sklearn grid search helper (training only)
│   └── modelselectionhelper.py       # sklearn model selection (training only)
├── data/
│   └── element_properties/
│       └── mat2vec.csv      (466 KB, tracked in git)
├── wsgi.py                 # Flask entrypoint
├── README.md
└── (no pyproject.toml, no Dockerfile, no tests)
```

### Inference pipeline (from `routes.py`)

Both the web form and the API followed the same path:

1. Parse compositions from input (comma-split or list).
2. Normalize formulas: `ElMD(comp).pretty_formula`.
3. Extract features: `ElMD(query, metric="mat2vec").feature_vector`.
4. Build a DataFrame `{"formula": queries, "target": np.ones(len(queries))}`
   (dummy target — inference only).
5. `model.load_data(df, train=False)` — `EDM_CsvLoader` processes the DataFrame
   into element-fraction tensors.
6. `model.predict(model.data_loader)` — CrabNet forward pass; returns
   `(act, pred, formulae, uncert)`.
7. For classification: round sigmoid output → 0/1.
8. For regression: unscaled prediction (log10 S/cm).

## Migration Approach

### Decisions

1. **Modify in place**: the migration was done directly in `tools/liion`, not in
   a new `tools/liion-model` directory. The repo already existed and the user
   requested in-place modification.

2. **Keep vendored code as-is**: `crab/` and `utils/` were moved under
   `src/model/` as vendored subpackages without stripping training-only code.
   This avoids breaking `Model.load_network()`, which instantiates `Lamb`,
   `Lookahead`, and `SWA` optimizers as part of the checkpoint loading path.
   The trade-off: `matplotlib` and `seaborn` are pulled in as dependencies
   (they're imported at module level in `utils/utils.py`), which adds ~50 MB
   to the Docker image. This was deemed acceptable for correctness and
   simplicity.

3. **Allow GPU**: `get_compute_device()` auto-detects CUDA. No changes were
   made — if a GPU is available, CrabNet uses it. To force CPU, set
   `CUDA_VISIBLE_DEVICES=""`.

4. **Both models loaded eagerly at startup**: the original Flask app loaded
   both `TransferFinalModel_Reg.pth` and `TransferFinalModel_Clf.pth` at module
   import time. The LMDS model does the same in `LiIonModel.__init__()`. This
   means 111 MB of weights are loaded once when the handler is instantiated.

5. **Extract large files from git**: the two `.pth` files (111 MB total) and
   `mat2vec.csv` (466 KB) were moved to `data/` and untracked from git via
   `git rm --cached`. A `.gitignore` was added. The files remain on disk for
   local development and Docker builds. Git history was not rewritten (the
   blobs remain in `.git` but are no longer tracked going forward).

### Steps taken

#### Step 1 — Extract large files

- Created `data/trained_models/` and moved both `.pth` files there.
- `data/element_properties/mat2vec.csv` was already in `data/` — left in place.
- Added `.gitignore` covering `data/trained_models/`, `data/element_properties/`,
  `.venv/`, `__pycache__/`, `*.pyc`, `*.egg-info/`, `dist/`, `*.whl`, `.env`.
- `git rm --cached` to stop tracking the large files (they stay on disk).
- Did not rewrite git history (per user request).

#### Step 2 — Scaffold LMDS infrastructure

Copied from `tools/app-template` as-is:

- `src/infrastructure/` — runner.py, handler.py, s3.py, storage.py, logging.py,
  `__init__.py`
- `src/cmds/` — run.py (entrypoint), model_types.py, validate_model.py,
  `__init__.py`
- `schema/model.schema.json` — shared JSON Schema meta-schema

These files are never modified. When the template's infrastructure code
changes, re-sync by copying the changed files from `app-template`.

#### Step 3 — Vendor CrabNet + utilities

Moved existing source code under `src/model/`:

- `crab/` → `src/model/crab/` (CrabNet architecture + Model wrapper)
- `utils/` → `src/model/utils/` (composition parsing, EDM loader, optimizers,
  compute device detection, and training-only helpers)

Fixed import paths:

| File | Before | After |
|---|---|---|
| `crab/model.py` | `from utils.utils import ...` | `from model.utils.utils import ...` |
| `crab/model.py` | `from utils.optim import SWA` | `from model.utils.optim import SWA` |
| `utils/utils.py` | `from utils.composition import ...` | `from model.utils.composition import ...` |
| `utils/modelselectionhelper.py` | `from utils.utils import get_cbfv` | `from model.utils.utils import get_cbfv` |
| `utils/modelselectionhelper.py` | `from utils.estimatorselectionhelper import ...` | `from model.utils.estimatorselectionhelper import ...` |

Added `__init__.py` to `src/model/crab/` and `src/model/utils/` to make them
proper Python subpackages.

No other changes were made to the vendored code. The `Embedder` class in
`kingcrab.py` still reads `data/element_properties/mat2vec.csv` via a relative
path from the CWD — this works both locally (CWD = repo root) and in Docker
(CWD = `/app`, data copied to `/app/data/`).

#### Step 4 — Remove old Flask app

Deleted:

- `app/` — `__init__.py`, `forms.py`, `routes.py`, `templates/`
- `wsgi.py`

The Flask web form and REST API are replaced by the LMDS runner's `process()`
interface. The web UI is now provided by the LMDS frontend, which renders the
model contract's declared fields.

#### Step 5 — Write model contract (`schema/model.json`)

Two modes, mirroring the original two interfaces:

**instance** (immediate response):

- Input: `query` (string) — single composition, default `"LiPO3"`
- Output: `query` (string), `classification` (integer), `regression` (number)
- Maps to: web form with a single composition

**dataset** (deferred response):

- Input: `compositions` (array of strings, inline) **or** `file` (CSV with
  `composition` column)
- Output: `result.json` (asset) — JSON with per-composition predictions
- Maps to: REST API (`compositions` list) and CSV batch upload

No model-level parameters — the model has no configurable hyperparameters at
inference time (k, thresholds, etc. are baked into the trained weights).

The top-level `description` field was removed after schema validation failed
— `model.schema.json` has `additionalProperties: false` at the root, allowing
only `$schema`, `name`, `version`, `parameters`, and `execution`.

Validated: `model.json` passes `jsonschema.validate()` against
`model.schema.json`.

#### Step 6 — Write model code (`src/model/model.py`)

`LiIonModel` class implementing the `ModelHandler` protocol:

- `__init__()`: calls `get_compute_device()` (auto-detects CUDA), then loads
  both CrabNet checkpoints from `data/trained_models/` (path configurable via
  `LIION_MODELS_PATH` env var). Raises `FileNotFoundError` with a clear message
  if weights are missing.
- `process(mode, values, files, output_dir, parameters, logger)`: routes to
  `_predict_single()` or `_predict_batch()`.
- `_predict_single()`: calls `_run_predictions([query])`, returns the first
  result as a dict of value outputs.
- `_predict_batch()`: parses compositions from inline list (supports both
  array and comma-separated string) or CSV file, calls `_run_predictions()`,
  writes `result.json` to `output_dir`, returns `{}`.
- `_run_predictions()`: mirrors the original inference pipeline from
  `routes.py`:
  1. `ElMD(comp).pretty_formula` — normalize formulas
  2. Build DataFrame `{"formula": queries, "target": np.ones(len(queries))}`
  3. `model.load_data(df, train=False)` for both reg and clf models
  4. `model.predict(model.data_loader)` for both
  5. Build list of `{"query", "classification", "regression"}` dicts

Comma-separated string handling was added for API compatibility: if
`compositions` is a string, it's split on commas. This covers the original API's
`{"compositions": "LiPO3"}` single-string format.

#### Step 7 — Write `__init__.py`, `pyproject.toml`, Dockerfile

**`src/model/__init__.py`**: `handler = LiIonModel()` — the runner imports
`handler` from here.

**`pyproject.toml`**: matches the app-template format (uses uv's default build
backend, `package = true`). Dependencies:

- Infrastructure: `jsonschema`, `boto3`, `structlog`
- CrabNet runtime: `torch`, `numpy`, `pandas`, `ElMD`, `tqdm`,
  `scikit-learn`, `matplotlib`, `seaborn`
- `setuptools>=69.0,<80` (ElMD's dead `import pkg_resources`)
- Entry points: `model-run`, `model-types`, `validate-model`
- `requires-python = ">=3.10"` (original code is compatible with 3.10+)

**Dockerfile**: multi-stage build:

- Stage 1: `python:3.12-slim` + uv, builds wheel from `src/`
- Stage 2: `python:3.12-slim`, installs wheel, copies `data/trained_models/`
  and `data/element_properties/` into `/app/data/`, sets
  `LIION_MODELS_PATH=/app/data/trained_models`
- GPU note: for GPU support, override the base image to a CUDA-enabled
  PyTorch image

#### Step 8 — Write scripts, tests, README

**Scripts**:

- `run_instance.sh` — single composition (default: LiPO3)
- `run_api.sh` — inline batch (API-style, accepts multiple args)
- `run_dataset.sh` — CSV file input

**Tests**:

- `conftest.py` — creates mock CrabNet checkpoints (real model architecture
  with random weights) in a temp directory so tests run without the real 111 MB
  weights. Skips if `mat2vec.csv` is missing (needed by `Embedder.__init__`).
- `test_model.py` — 12 tests: instance mode (basic prediction, default query,
  formula normalization, binary classification), dataset mode (inline list,
  comma string, single, empty filtering, CSV file, missing column, empty
  rows), error handling (unknown mode, no input).

**README.md**: full documentation — what it does, quick start, project
structure, data files, modes, environment variables, GPU support, Docker,
citations.

#### Step 9 — Verification

- `model.json` validates against `model.schema.json` ✓
- Infrastructure imports work (`infrastructure.logging`, `infrastructure.handler`) ✓
- Wheel builds successfully — 24 source files packaged correctly ✓
- All vendored import paths fixed and resolved ✓

### What could not be verified locally

The development environment is Alpine Linux (musl), which has no PyTorch
wheels. The following require a glibc system (Debian/Ubuntu) or Docker:

- **PyTorch import**: torch, numpy, pandas, ElMD cannot be installed on musl
- **Unit tests**: `conftest.py` creates mock checkpoints using torch
- **Docker build**: uses `python:3.12-slim` (Debian/glibc) — torch installs
  from prebuilt wheels there
- **End-to-end local test**: `./scripts/run_instance.sh LiPO3` requires torch
- **Docker runtime test**: `docker run --rm liion:1.0.0 --local --parameters '...'`

## Resulting layout (after migration)

```
tools/liion/
├── schema/
│   ├── model.json              # Model contract: instance + dataset modes
│   └── model.schema.json       # JSON Schema meta-schema (shared)
├── src/
│   ├── cmds/                   # Copy-as-is from app-template
│   │   ├── run.py              # Entrypoint: from model import handler; run(handler)
│   │   ├── model_types.py
│   │   └── validate_model.py
│   ├── infrastructure/         # Copy-as-is from app-template
│   │   ├── runner.py           # Orchestrator: routes inputs, calls process, builds Output JSON
│   │   ├── handler.py          # ModelHandler Protocol
│   │   ├── s3.py
│   │   ├── storage.py
│   │   └── logging.py
│   └── model/
│       ├── __init__.py         # handler = LiIonModel()
│       ├── model.py            # LiIonModel: process(), _predict_single(), _predict_batch()
│       ├── crab/               # Vendored CrabNet (kept as-is)
│       │   ├── kingcrab.py     # CrabNet nn.Module
│       │   └── model.py        # Model wrapper (load_network, predict, fit)
│       └── utils/              # Vendored utilities (kept as-is)
│           ├── composition.py  # Formula parsing
│           ├── utils.py        # EDM_CsvLoader, Scaler, Lamb, Lookahead, losses
│           ├── optim.py        # SWA optimizer
│           ├── get_compute_device.py
│           ├── figures.py      # Training-only plotting
│           ├── estimatorselectionhelper.py  # Training-only
│           ├── modelselectionhelper.py      # Training-only
│           └── get_core_count.py
├── data/                       # NOT tracked in git
│   ├── trained_models/
│   │   ├── TransferFinalModel_Reg.pth   (55.6 MB)
│   │   └── TransferFinalModel_Clf.pth   (55.6 MB)
│   ├── element_properties/
│   │   └── mat2vec.csv                  (466 KB)
│   ├── compositions.csv         # Sample dataset (tracked)
│   └── output/                 # Runner writes here (gitignored)
├── scripts/
│   ├── run_instance.sh         # Single composition
│   ├── run_api.sh              # Inline batch (API-style)
│   └── run_dataset.sh          # CSV file batch
├── tests/
│   ├── conftest.py             # Mock CrabNet checkpoints
│   └── test_model.py            # 12 unit tests
├── Dockerfile                  # Multi-stage build (python:3.12-slim)
├── pyproject.toml              # Package config
├── .env.example
├── .gitignore
├── .dockerignore
├── README.md
└── CONTEXT.md                  # This file
```

## Mode coverage vs original interfaces

| Original interface | LMDS mode | Input format | Output |
|---|---|---|---|
| Web form (single composition) | `instance` | `{"query": {"value": "LiPO3"}}` | `{"query": "LiPO3", "classification": 1, "regression": -1.23}` |
| Web form (comma-separated batch) | `dataset` | `{"compositions": {"value": ["LiPO3","NaCl"]}}` | `result.json` asset |
| REST API (`compositions: "LiPO3"`) | `dataset` | `{"compositions": {"value": "LiPO3"}}` (string split on commas) | `result.json` asset |
| REST API (`compositions: ["LiPO3","NaCl"]`) | `dataset` | `{"compositions": {"value": ["LiPO3","NaCl"]}}` | `result.json` asset |
| CSV upload (not in original) | `dataset` | `{"file": {"uri": "file://...", "mime_type": "text/csv"}}` | `result.json` asset |

## Dependencies

| Package | Version | Why |
|---|---|---|
| `torch` | >=2.0 | CrabNet neural network |
| `numpy` | >=1.24 | Array operations |
| `pandas` | >=2.0 | DataFrame for EDM_CsvLoader |
| `ElMD` | >=0.5.12 | Composition normalization + mat2vec feature vectors |
| `tqdm` | >=4.0 | Progress bars in EDM_CsvLoader |
| `scikit-learn` | >=1.3 | Metrics in crab/model.py (training code, kept as-is) |
| `matplotlib` | >=3.7 | Imported at module level in utils/utils.py (kept as-is) |
| `seaborn` | >=0.13 | Imported at module level in utils/utils.py (kept as-is) |
| `setuptools` | >=69.0,<80 | ElMD's dead `import pkg_resources` crashes on 80+ |
| `jsonschema` | >=4.26 | Model contract validation |
| `boto3` | >=1.34 | S3 storage (remote mode) |
| `structlog` | >=25.5 | Structured logging |

## Next steps

1. **Docker build + runtime test**: Build `docker build -t liion:1.0.0 .` and run
   `docker run --rm liion:1.0.0 --local --parameters '{"mode":"instance","inputs":{"query":{"value":"LiPO3"}},"parameters":{}}'`
   on a glibc system.

2. **Run unit tests**: `uv run pytest tests/ -v` on a glibc system with the
   real `mat2vec.csv` in `data/element_properties/`.

3. **Register with LMDS**: Build Docker, push to registry, register via
   `curl -X POST http://localhost:42069/api/v1/models -H "Content-Type: application/json" -d @schema/model.json`.

4. **Optional cleanup**: Strip training-only code from vendored `crab/` and
   `utils/` to remove `matplotlib`, `seaborn`, and `scikit-learn` dependencies.
   This would require careful surgery around `Model.load_network()` which
   instantiates `Lamb`/`Lookahead`/`SWA` optimizers.
