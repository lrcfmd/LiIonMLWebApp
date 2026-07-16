import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ElMD import ElMD

from model.crab.kingcrab import CrabNet
from model.crab.model import Model
from model.utils.get_compute_device import get_compute_device
from infrastructure.logging import newLogger


class LiIonModel:
    """Li-Ionics ML conductivity prediction model.

    Loads two CrabNet models at startup — one for regression (predicting
    log10 conductivity) and one for classification (conductivity > 1e-4 S/cm).
    Answers composition queries using the same inference pipeline as the
    original Flask application.

    Two modes:
      - ``instance``: single composition → classification + regression
        (immediate response).
      - ``dataset``: batch from inline list or CSV file → results written
        to ``result.json`` (deferred response).
    """

    def __init__(self):
        self.logger = newLogger("model")
        self.compute_device = get_compute_device()
        self.logger.info(
            "LiIon model initializing",
            compute_device=str(self.compute_device),
        )

        models_dir = Path(
            os.environ.get("LIION_MODELS_PATH", "data/trained_models")
        )
        self.crabnet_reg = self._load_model(
            models_dir / "TransferFinalModel_Reg.pth", classification=False
        )
        self.crabnet_cls = self._load_model(
            models_dir / "TransferFinalModel_Clf.pth", classification=True
        )
        self.logger.info("LiIon model loaded", models=str(models_dir))

    # -- Loading ----------------------------------------------------------

    def _load_model(self, path: Path, classification: bool) -> Model:
        """Load a CrabNet model checkpoint."""
        if not path.exists():
            raise FileNotFoundError(
                f"Required model checkpoint not found: {path}. "
                f"Set LIION_MODELS_PATH or place the file at "
                f"data/trained_models/{path.name}."
            )
        model = Model(
            CrabNet(compute_device=self.compute_device).to(self.compute_device),
            model_name=path.stem,
            verbose=False,
            classification=classification,
        )
        model.load_network(str(path))
        return model

    # -- Entrypoint -------------------------------------------------------

    def process(
        self,
        mode: str,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        parameters: dict,
        logger,
    ) -> dict:
        """Run the selected mode."""
        if mode == "instance":
            return self._predict_single(values, logger)
        elif mode == "dataset":
            return self._predict_batch(values, files, output_dir, logger)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    # -- Instance mode ----------------------------------------------------

    def _predict_single(self, values: dict, logger) -> dict:
        """Single composition → classification + regression."""
        query = values.get("query") or "LiPO3"
        logger.info("instance predict", query=query)

        results = self._run_predictions([query], logger)
        return results[0]

    # -- Dataset mode -----------------------------------------------------

    def _predict_batch(
        self,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        logger,
    ) -> dict:
        """Batch predict from inline list or CSV file → result.json asset."""
        inline = values.get("compositions")
        if inline:
            if isinstance(inline, str):
                compositions = [c.strip() for c in inline.split(",") if c.strip()]
            else:
                compositions = [c.strip() for c in inline if c.strip()]
            logger.info("dataset predict (inline)", count=len(compositions))
        elif files:
            file_path = self._get_input_file(files)
            logger.info("dataset predict (file)", file=str(file_path))
            compositions = self._read_compositions(file_path)
        else:
            raise ValueError(
                "No input provided — supply either 'compositions' (list of strings) "
                "or 'file' (CSV with a 'composition' column)"
            )

        logger.info("compositions to predict", count=len(compositions))

        results = self._run_predictions(compositions, logger)

        # Write the asset file — name must match the declared asset field
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "result.json"
        with output_path.open("w") as f:
            json.dump({"results": results}, f, indent=2)
        logger.info(
            "dataset predict complete",
            queries=len(results),
            path=str(output_path),
        )

        return {}  # asset output; no value outputs

    # -- CrabNet inference ------------------------------------------------

    def _run_predictions(self, compositions: list[str], logger) -> list[dict]:
        """Run CrabNet regression + classification for a list of compositions.

        Mirrors the inference pipeline from the original Flask routes.py:
        1. Normalize formulas via ElMD
        2. Build a DataFrame with dummy targets (inference mode)
        3. Load data into both CrabNet models
        4. Predict and aggregate results
        """
        # Normalize formulas
        queries = [ElMD(comp).pretty_formula for comp in compositions]

        # Build DataFrame — CrabNet expects formula + target columns
        df = pd.DataFrame({"formula": queries, "target": np.ones(len(queries))})

        # Load data into both models (inference mode, no training)
        self.crabnet_reg.load_data(df, train=False)
        self.crabnet_cls.load_data(df, train=False)

        # Predict — predict() returns (act, pred, formulae, uncert)
        _, reg_preds, _, _ = self.crabnet_reg.predict(self.crabnet_reg.data_loader)
        _, clf_preds, _, _ = self.crabnet_cls.predict(self.crabnet_cls.data_loader)

        # Build result dicts
        results = []
        for i, comp in enumerate(compositions):
            results.append(
                {
                    "query": queries[i],
                    "classification": int(clf_preds[i]),
                    "regression": float(reg_preds[i]),
                }
            )
            logger.debug(
                "prediction done",
                query=queries[i],
                classification=int(clf_preds[i]),
                regression=float(reg_preds[i]),
            )

        return results

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _get_input_file(files: dict[str, tuple[Path, str]]) -> Path:
        """Extract the input file from the files dict."""
        for _name, (path, _mime) in files.items():
            return path
        raise ValueError("No input file provided")

    @staticmethod
    def _read_compositions(file_path: Path) -> list[str]:
        """Read compositions from a CSV file with a 'composition' column."""
        compositions = []
        with file_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "composition" not in reader.fieldnames:
                raise ValueError(
                    f"CSV must have a 'composition' column. "
                    f"Found columns: {reader.fieldnames}"
                )
            for row in reader:
                comp = row["composition"].strip()
                if comp:
                    compositions.append(comp)
        if not compositions:
            raise ValueError("No compositions found in the input file")
        return compositions