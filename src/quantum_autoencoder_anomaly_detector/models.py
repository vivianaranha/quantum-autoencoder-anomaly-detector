"""Classical unsupervised anomaly-detection baselines.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .evaluation import classification_metrics, threshold_search


@dataclass(frozen=True, slots=True)
class ScoreNormalizer:
    """Validation-fitted min-max transformation for anomaly scores."""

    minimum: float
    maximum: float

    @classmethod
    def fit(cls, values: np.ndarray) -> ScoreNormalizer:
        return cls(float(np.min(values)), float(np.max(values)))

    def transform(self, values: np.ndarray) -> np.ndarray:
        width = self.maximum - self.minimum
        if width < 1e-12:
            return np.zeros_like(values, dtype=float)
        return np.clip((np.asarray(values, dtype=float) - self.minimum) / width, 0.0, 1.0)


@dataclass(slots=True)
class ModelResult:
    """Model, scores, threshold audit, timing, and held-out metrics."""

    name: str
    model: Any
    normalizer: ScoreNormalizer
    raw_validation_scores: np.ndarray
    raw_test_scores: np.ndarray
    validation_scores: np.ndarray
    test_scores: np.ndarray
    predictions: np.ndarray
    threshold: float
    metrics: dict[str, Any]
    threshold_history: pd.DataFrame
    training_samples: int


def pca_reconstruction_error(model: PCA, values: np.ndarray) -> np.ndarray:
    latent = model.transform(values)
    reconstructed = model.inverse_transform(latent)
    return np.mean(np.square(values - reconstructed), axis=1)


def _finalize_result(
    name: str,
    model: Any,
    raw_validation_scores: np.ndarray,
    raw_test_scores: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
    training_seconds: float,
    inference_seconds: float,
    training_samples: int,
) -> ModelResult:
    normalizer = ScoreNormalizer.fit(raw_validation_scores)
    validation_scores = normalizer.transform(raw_validation_scores)
    test_scores = normalizer.transform(raw_test_scores)
    threshold, history = threshold_search(
        validation_labels,
        validation_scores,
        false_negative_cost,
        false_positive_cost,
    )
    predictions = (test_scores >= threshold).astype(int)
    metrics = classification_metrics(
        test_labels,
        test_scores,
        threshold,
        false_negative_cost,
        false_positive_cost,
        training_seconds,
        inference_seconds,
    )
    return ModelResult(
        name=name,
        model=model,
        normalizer=normalizer,
        raw_validation_scores=np.asarray(raw_validation_scores, dtype=float),
        raw_test_scores=np.asarray(raw_test_scores, dtype=float),
        validation_scores=validation_scores,
        test_scores=test_scores,
        predictions=predictions,
        threshold=threshold,
        metrics=metrics,
        threshold_history=history,
        training_samples=training_samples,
    )


def train_classical_models(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pca_components: int,
    random_seed: int,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, ModelResult]:
    """Fit PCA reconstruction and Isolation Forest on clean-normal training rows."""

    train_start = perf_counter()
    pca = PCA(n_components=pca_components, random_state=random_seed).fit(x_train)
    pca_training_seconds = perf_counter() - train_start
    inference_start = perf_counter()
    pca_validation = pca_reconstruction_error(pca, x_validation)
    pca_test = pca_reconstruction_error(pca, x_test)
    pca_inference_seconds = perf_counter() - inference_start

    train_start = perf_counter()
    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_seed,
        n_jobs=-1,
    ).fit(x_train)
    isolation_training_seconds = perf_counter() - train_start
    inference_start = perf_counter()
    isolation_validation = -isolation_forest.score_samples(x_validation)
    isolation_test = -isolation_forest.score_samples(x_test)
    isolation_inference_seconds = perf_counter() - inference_start

    return {
        "pca_autoencoder": _finalize_result(
            "pca_autoencoder",
            pca,
            pca_validation,
            pca_test,
            y_validation,
            y_test,
            false_negative_cost,
            false_positive_cost,
            pca_training_seconds,
            pca_inference_seconds,
            len(x_train),
        ),
        "isolation_forest": _finalize_result(
            "isolation_forest",
            isolation_forest,
            isolation_validation,
            isolation_test,
            y_validation,
            y_test,
            false_negative_cost,
            false_positive_cost,
            isolation_training_seconds,
            isolation_inference_seconds,
            len(x_train),
        ),
    }


def finalize_external_result(**kwargs: Any) -> ModelResult:
    """Apply the same validation scaling and threshold logic to a quantum model."""

    return _finalize_result(**kwargs)
