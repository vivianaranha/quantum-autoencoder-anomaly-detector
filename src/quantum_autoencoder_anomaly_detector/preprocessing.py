"""Train-only standardization and amplitude normalization.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class PreparedData:
    """Standardized classical arrays and normalized quantum amplitudes."""

    standard_train: np.ndarray
    standard_validation: np.ndarray
    standard_test: np.ndarray
    amplitude_train: np.ndarray
    amplitude_validation: np.ndarray
    amplitude_test: np.ndarray
    scaler: StandardScaler


def amplitude_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize rows as real quantum statevectors."""

    matrix = np.asarray(values, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    normalized = matrix / safe_norms
    zero_rows = np.flatnonzero(norms.reshape(-1) < 1e-12)
    if len(zero_rows):
        normalized[zero_rows, 0] = 1.0
    return normalized


def prepare_features(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
) -> PreparedData:
    """Fit the scaler only on known-normal training telemetry."""

    scaler = StandardScaler()
    standard_train = np.asarray(scaler.fit_transform(x_train), dtype=float)
    standard_validation = np.asarray(scaler.transform(x_validation), dtype=float)
    standard_test = np.asarray(scaler.transform(x_test), dtype=float)
    return PreparedData(
        standard_train=standard_train,
        standard_validation=standard_validation,
        standard_test=standard_test,
        amplitude_train=amplitude_normalize(standard_train),
        amplitude_validation=amplitude_normalize(standard_validation),
        amplitude_test=amplitude_normalize(standard_test),
        scaler=scaler,
    )
