"""Tests for the Li-Ionics ML model.

These tests verify the model's process() interface, input handling,
and output structure. They use mock CrabNet checkpoints so they can
run without the real 111 MB trained weights.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper to get a model handler instance (loads mock weights)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def handler(mock_element_data):
    """Create a LiIonModel instance with mock weights."""
    from model.model import LiIonModel
    return LiIonModel()


# ---------------------------------------------------------------------------
# Instance mode
# ---------------------------------------------------------------------------

class TestInstanceMode:
    """Tests for instance mode (single composition, immediate response)."""

    def test_basic_prediction(self, handler):
        """Instance mode returns query, classification, and regression."""
        result = handler.process(
            mode="instance",
            values={"query": "LiPO3"},
            files={},
            output_dir=Path("/tmp/test_output"),
            parameters={},
            logger=MagicMock(),
        )
        assert "query" in result
        assert "classification" in result
        assert "regression" in result
        assert isinstance(result["classification"], int)
        assert isinstance(result["regression"], float)

    def test_default_query(self, handler):
        """Default query is LiPO3 when none provided."""
        result = handler.process(
            mode="instance",
            values={},
            files={},
            output_dir=Path("/tmp/test_output"),
            parameters={},
            logger=MagicMock(),
        )
        assert result["query"] == "LiPO3"

    def test_query_normalization(self, handler):
        """Query formula is normalized via ElMD.pretty_formula."""
        result = handler.process(
            mode="instance",
            values={"query": "li po 3"},  # lowercase with spaces
            files={},
            output_dir=Path("/tmp/test_output"),
            parameters={},
            logger=MagicMock(),
        )
        # ElMD normalizes to a pretty formula
        assert result["query"]  # should be a non-empty string

    def test_classification_is_binary(self, handler):
        """Classification output is 0 or 1."""
        for comp in ["LiPO3", "NaCl", "Fe2O3", "SrTiO3"]:
            result = handler.process(
                mode="instance",
                values={"query": comp},
                files={},
                output_dir=Path("/tmp/test_output"),
                parameters={},
                logger=MagicMock(),
            )
            assert result["classification"] in (0, 1)


# ---------------------------------------------------------------------------
# Dataset mode — inline compositions
# ---------------------------------------------------------------------------

class TestDatasetInline:
    """Tests for dataset mode with inline compositions array."""

    def test_inline_list(self, handler, tmp_path):
        """Dataset mode accepts inline compositions list."""
        result = handler.process(
            mode="dataset",
            values={"compositions": ["LiPO3", "NaCl", "SrTiO3"]},
            files={},
            output_dir=tmp_path,
            parameters={},
            logger=MagicMock(),
        )
        assert result == {}  # asset output, no value outputs

        output_file = tmp_path / "result.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert "results" in data
        assert len(data["results"]) == 3
        for entry in data["results"]:
            assert "query" in entry
            assert "classification" in entry
            assert "regression" in entry

    def test_inline_comma_string(self, handler, tmp_path):
        """Dataset mode accepts comma-separated string."""
        result = handler.process(
            mode="dataset",
            values={"compositions": "LiPO3, NaCl, Fe2O3"},
            files={},
            output_dir=tmp_path,
            parameters={},
            logger=MagicMock(),
        )
        assert result == {}

        data = json.loads((tmp_path / "result.json").read_text())
        assert len(data["results"]) == 3

    def test_inline_single(self, handler, tmp_path):
        """Single composition in inline list works."""
        result = handler.process(
            mode="dataset",
            values={"compositions": ["LiPO3"]},
            files={},
            output_dir=tmp_path,
            parameters={},
            logger=MagicMock(),
        )
        data = json.loads((tmp_path / "result.json").read_text())
        assert len(data["results"]) == 1

    def test_inline_empty_filtered(self, handler, tmp_path):
        """Empty strings in inline list are filtered out."""
        result = handler.process(
            mode="dataset",
            values={"compositions": ["LiPO3", "", "  ", "NaCl"]},
            files={},
            output_dir=tmp_path,
            parameters={},
            logger=MagicMock(),
        )
        data = json.loads((tmp_path / "result.json").read_text())
        assert len(data["results"]) == 2


# ---------------------------------------------------------------------------
# Dataset mode — CSV file
# ---------------------------------------------------------------------------

class TestDatasetCSV:
    """Tests for dataset mode with CSV file input."""

    def test_csv_file(self, handler, tmp_path):
        """Dataset mode accepts a CSV file with a 'composition' column."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("composition\nLiPO3\nNaCl\nFe2O3\n")

        result = handler.process(
            mode="dataset",
            values={},
            files={"file": (csv_path, "text/csv")},
            output_dir=tmp_path,
            parameters={},
            logger=MagicMock(),
        )
        assert result == {}

        data = json.loads((tmp_path / "result.json").read_text())
        assert len(data["results"]) == 3

    def test_csv_missing_column(self, handler, tmp_path):
        """CSV without 'composition' column raises ValueError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("formula\nLiPO3\n")

        with pytest.raises(ValueError, match="composition"):
            handler.process(
                mode="dataset",
                values={},
                files={"file": (csv_path, "text/csv")},
                output_dir=tmp_path,
                parameters={},
                logger=MagicMock(),
            )

    def test_csv_empty_rows(self, handler, tmp_path):
        """Empty rows in CSV are skipped."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("composition\nLiPO3\n\n\nNaCl\n")

        result = handler.process(
            mode="dataset",
            values={},
            files={"file": (csv_path, "text/csv")},
            output_dir=tmp_path,
            parameters={},
            logger=MagicMock(),
        )
        data = json.loads((tmp_path / "result.json").read_text())
        assert len(data["results"]) == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    """Tests for error handling."""

    def test_unknown_mode(self, handler, tmp_path):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            handler.process(
                mode="bogus",
                values={},
                files={},
                output_dir=tmp_path,
                parameters={},
                logger=MagicMock(),
            )

    def test_no_input(self, handler, tmp_path):
        """Dataset mode with no input raises ValueError."""
        with pytest.raises(ValueError, match="No input"):
            handler.process(
                mode="dataset",
                values={},
                files={},
                output_dir=tmp_path,
                parameters={},
                logger=MagicMock(),
            )