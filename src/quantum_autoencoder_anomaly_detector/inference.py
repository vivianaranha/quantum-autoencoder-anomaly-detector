"""Saved-model inference for sensor telemetry CSV files.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS
from .models import ScoreNormalizer, pca_reconstruction_error
from .preprocessing import amplitude_normalize
from .quantum_autoencoder import QuantumAutoencoderModel


def validate_sensor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required sensor columns: {', '.join(missing)}")
    validated = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="raise")
    if not np.isfinite(validated.to_numpy(dtype=float)).all():
        raise ValueError("Sensor values must be finite")
    return validated


def _normalizer(values: dict[str, dict[str, float]], name: str) -> ScoreNormalizer:
    return ScoreNormalizer(
        minimum=float(values[name]["minimum"]),
        maximum=float(values[name]["maximum"]),
    )


def predict_frame(artifact_directory: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply every saved detector with its validation-fitted score scaling and threshold."""

    artifact_path = Path(artifact_directory)
    thresholds = json.loads((artifact_path / "thresholds.json").read_text(encoding="utf-8"))
    normalizer_values = json.loads(
        (artifact_path / "score_normalizers.json").read_text(encoding="utf-8")
    )
    scaler = joblib.load(artifact_path / "sensor_scaler.joblib")
    standard = np.asarray(scaler.transform(validate_sensor_frame(frame)), dtype=float)
    output = pd.DataFrame(index=frame.index)
    if "sample_id" in frame:
        output["sample_id"] = frame["sample_id"]

    pca = joblib.load(artifact_path / "pca_autoencoder.joblib")
    pca_raw_scores = pca_reconstruction_error(pca, standard)
    pca_scores = _normalizer(normalizer_values, "pca_autoencoder").transform(pca_raw_scores)
    output["pca_autoencoder_raw_score"] = pca_raw_scores
    output["pca_autoencoder_score"] = pca_scores
    output["pca_autoencoder_prediction"] = (
        pca_scores >= float(thresholds["pca_autoencoder"])
    ).astype(int)

    isolation_forest = joblib.load(artifact_path / "isolation_forest.joblib")
    isolation_raw_scores = -isolation_forest.score_samples(standard)
    isolation_scores = _normalizer(normalizer_values, "isolation_forest").transform(
        isolation_raw_scores
    )
    output["isolation_forest_raw_score"] = isolation_raw_scores
    output["isolation_forest_score"] = isolation_scores
    output["isolation_forest_prediction"] = (
        isolation_scores >= float(thresholds["isolation_forest"])
    ).astype(int)

    quantum_path = artifact_path / "quantum_autoencoder.json"
    if quantum_path.exists():
        model = QuantumAutoencoderModel.from_dict(
            json.loads(quantum_path.read_text(encoding="utf-8"))
        )
        raw_scores = model.score_samples(amplitude_normalize(standard))
        quantum_scores = _normalizer(normalizer_values, "quantum_autoencoder").transform(raw_scores)
        output["quantum_autoencoder_raw_score"] = raw_scores
        output["quantum_autoencoder_reconstruction_fidelity"] = 1.0 - raw_scores
        output["quantum_autoencoder_score"] = quantum_scores
        output["quantum_autoencoder_prediction"] = (
            quantum_scores >= float(thresholds["quantum_autoencoder"])
        ).astype(int)
    return output


def predict_csv(artifact_directory: str | Path, input_csv: str | Path) -> pd.DataFrame:
    return predict_frame(artifact_directory, pd.read_csv(input_csv))
