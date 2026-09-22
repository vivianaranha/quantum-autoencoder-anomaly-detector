"""End-to-end anomaly-detection experiment.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import create_run_directory, save_all_artifacts
from .config import ExperimentConfig
from .data import DataPartitions, create_partitions
from .models import ModelResult, train_classical_models
from .preprocessing import PreparedData, prepare_features
from .quantum_autoencoder import (
    QuantumAutoencoderModel,
    build_encoder_circuit,
    circuit_diagnostics,
    train_quantum_autoencoder,
)

ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class ExperimentResult:
    """Outputs shared by the CLI, dashboard, and tests."""

    output_directory: Path
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    partitions: DataPartitions
    prepared: PreparedData
    model_results: dict[str, ModelResult]
    quantum_model: QuantumAutoencoderModel | None
    objective_history: list[float]
    circuit_info: dict[str, Any]
    optimizer_info: dict[str, Any]


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def run_experiment(
    config: ExperimentConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ExperimentResult:
    resolved = (config or ExperimentConfig()).validate()
    destination = create_run_directory(resolved.output_root)
    _notify(progress_callback, "Generating correlated sensor telemetry and anomaly families")
    partitions = create_partitions(
        resolved.sample_count,
        resolved.anomaly_rate,
        resolved.validation_size,
        resolved.test_size,
        resolved.random_seed,
    )
    _notify(progress_callback, "Fitting the scaler on clean-normal training telemetry only")
    prepared = prepare_features(partitions.x_train, partitions.x_validation, partitions.x_test)
    _notify(progress_callback, "Training unsupervised classical baselines")
    model_results = train_classical_models(
        prepared.standard_train,
        prepared.standard_validation,
        partitions.y_validation,
        prepared.standard_test,
        partitions.y_test,
        resolved.pca_latent_components,
        resolved.random_seed,
        resolved.false_negative_cost,
        resolved.false_positive_cost,
    )
    circuit, _ = build_encoder_circuit(resolved.input_qubits, resolved.ansatz_reps)
    circuit_info = circuit_diagnostics(
        circuit,
        resolved.input_qubits,
        resolved.latent_qubits,
        resolved.ansatz_reps,
    )
    quantum_model = None
    objective_history: list[float] = []
    quantum_training_ids = np.array([], dtype=str)
    optimizer_info: dict[str, Any] = {
        "success": False,
        "message": "Quantum autoencoder was skipped",
        "evaluations": 0,
        "objective": None,
    }
    if resolved.include_quantum_autoencoder:
        _notify(progress_callback, "Training the quantum autoencoder on normal amplitude states")
        quantum_output = train_quantum_autoencoder(
            prepared.amplitude_train,
            partitions.train_ids,
            prepared.amplitude_validation,
            partitions.y_validation,
            prepared.amplitude_test,
            partitions.y_test,
            resolved.normal_train_limit,
            resolved.input_qubits,
            resolved.latent_qubits,
            resolved.ansatz_reps,
            resolved.optimizer_maxiter,
            resolved.random_seed,
            resolved.false_negative_cost,
            resolved.false_positive_cost,
        )
        model_results[quantum_output.result.name] = quantum_output.result
        quantum_model = quantum_output.model
        circuit = quantum_output.circuit
        objective_history = quantum_output.objective_history
        quantum_training_ids = quantum_output.selected_ids
        optimizer_info = quantum_output.optimizer_info

    labels = partitions.full_dataset["is_anomaly"].to_numpy(dtype=int)
    dataset_summary = {
        "source": "deterministic synthetic industrial sensor generator",
        "sample_count": len(labels),
        "normal_count": int(np.sum(labels == 0)),
        "anomaly_count": int(np.sum(labels)),
        "observed_anomaly_rate": float(np.mean(labels)),
        "normal_train_count": len(partitions.y_train),
        "excluded_training_anomalies": len(partitions.excluded_train_anomaly_ids),
        "validation_count": len(partitions.y_validation),
        "validation_anomalies": int(np.sum(partitions.y_validation)),
        "test_count": len(partitions.y_test),
        "test_anomalies": int(np.sum(partitions.y_test)),
        "sensor_features": len(partitions.x_train.columns),
        "quantum_training_count": len(quantum_training_ids),
    }
    _notify(progress_callback, "Saving models, thresholds, circuits, plots, and provenance")
    metrics, predictions = save_all_artifacts(
        destination,
        resolved.to_dict(),
        dataset_summary,
        circuit_info,
        optimizer_info,
        partitions,
        prepared,
        model_results,
        circuit,
        quantum_model,
        quantum_training_ids,
        objective_history,
    )
    _notify(progress_callback, f"Complete: {destination}")
    return ExperimentResult(
        output_directory=destination,
        metrics=metrics,
        predictions=predictions,
        partitions=partitions,
        prepared=prepared,
        model_results=model_results,
        quantum_model=quantum_model,
        objective_history=objective_history,
        circuit_info=circuit_info,
        optimizer_info=optimizer_info,
    )
