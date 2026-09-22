"""Artifact persistence and experiment reporting.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .models import ModelResult
from .visualization import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_objective,
    plot_precision_recall,
    plot_qae_reconstruction_fidelity,
    plot_roc,
    plot_score_distributions,
    plot_sensor_signals,
    plot_threshold_costs,
    plot_time_series,
)


def create_run_directory(output_root: str | Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_root) / f"run-{timestamp}"
    suffix = 1
    while destination.exists():
        destination = Path(output_root) / f"run-{timestamp}-{suffix}"
        suffix += 1
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def environment_details() -> dict[str, Any]:
    packages = {}
    for package in (
        "qiskit",
        "qiskit-machine-learning",
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas",
        "streamlit",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def metrics_frame(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = [
        {"model": name, "training_samples": result.training_samples, **result.metrics}
        for name, result in results.items()
    ]
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def predictions_frame(partitions: Any, results: dict[str, ModelResult]) -> pd.DataFrame:
    lookup = partitions.full_dataset.set_index("sample_id")
    frame = pd.DataFrame(
        {
            "sample_id": partitions.test_ids,
            "actual_anomaly": partitions.y_test,
            "anomaly_type": lookup.loc[partitions.test_ids, "anomaly_type"].to_numpy(),
        }
    )
    for name, result in results.items():
        frame[f"{name}_raw_score"] = result.raw_test_scores
        frame[f"{name}_score"] = result.test_scores
        frame[f"{name}_prediction"] = result.predictions
        if name == "quantum_autoencoder":
            frame["quantum_autoencoder_reconstruction_fidelity"] = 1.0 - result.raw_test_scores
    return frame.sort_values("sample_id").reset_index(drop=True)


def anomaly_type_performance_frame(
    partitions: Any, results: dict[str, ModelResult]
) -> pd.DataFrame:
    """Summarize held-out scores and decisions for each injected input family."""

    lookup = partitions.full_dataset.set_index("sample_id")
    anomaly_types = lookup.loc[partitions.test_ids, "anomaly_type"].to_numpy()
    rows: list[dict[str, Any]] = []
    ordered_types = (
        "normal",
        "sensor_spike",
        "process_drift",
        "correlation_break",
        "stuck_sensor",
    )
    for name, result in results.items():
        for anomaly_type in ordered_types:
            selected = anomaly_types == anomaly_type
            if not np.any(selected):
                continue
            row: dict[str, Any] = {
                "model": name,
                "anomaly_type": anomaly_type,
                "samples": int(np.sum(selected)),
                "mean_raw_anomaly_score": float(np.mean(result.raw_test_scores[selected])),
                "mean_normalized_anomaly_score": float(np.mean(result.test_scores[selected])),
                "flagged_rate": float(np.mean(result.predictions[selected])),
            }
            if name == "quantum_autoencoder":
                row["mean_reconstruction_fidelity"] = float(
                    np.mean(1.0 - result.raw_test_scores[selected])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def split_assignments_frame(partitions: Any) -> pd.DataFrame:
    rows = []
    for split, identifiers, labels in (
        ("train_normal", partitions.train_ids, partitions.y_train),
        ("validation", partitions.validation_ids, partitions.y_validation),
        ("test", partitions.test_ids, partitions.y_test),
        (
            "excluded_train_anomaly",
            partitions.excluded_train_anomaly_ids,
            np.ones(len(partitions.excluded_train_anomaly_ids), dtype=int),
        ),
    ):
        rows.append(pd.DataFrame({"sample_id": identifiers, "split": split, "is_anomaly": labels}))
    return pd.concat(rows, ignore_index=True).sort_values("sample_id")


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    summary: dict[str, Any],
    circuit_info: dict[str, Any],
    optimizer_info: dict[str, Any],
) -> None:
    lines = [
        "# Quantum Autoencoder Anomaly Detector — Experiment Report",
        "",
        "**Created by School of AI and School of QC**",
        "",
        "## Dataset",
        "",
        f"- Synthetic sensor samples: {summary['sample_count']}",
        f"- Anomaly samples: {summary['anomaly_count']}",
        f"- Observed anomaly rate: {summary['observed_anomaly_rate']:.4f}",
        f"- Clean-normal training samples: {summary['normal_train_count']}",
        f"- Validation samples: {summary['validation_count']}",
        f"- Test samples: {summary['test_count']}",
        "",
        "## Held-out results",
        "",
    ]
    for row in metrics.to_dict(orient="records"):
        lines.extend(
            [
                f"### {str(row['model']).replace('_', ' ').title()}",
                "",
                f"- Training samples: {int(row['training_samples'])}",
                f"- Validation-selected threshold: {row['threshold']:.3f}",
                f"- Precision: {row['precision']:.4f}",
                f"- Recall: {row['recall']:.4f}",
                f"- F2: {row['f2']:.4f}",
                f"- ROC AUC: {row['roc_auc']:.4f}",
                f"- Average precision: {row['average_precision']:.4f}",
                f"- False positives: {int(row['false_positive'])}",
                f"- False negatives: {int(row['false_negative'])}",
                f"- Configured test cost: {row['business_cost']:.2f}",
                f"- Training time: {row['training_seconds']:.4f} seconds",
                "",
            ]
        )
        if str(row["model"]) == "quantum_autoencoder":
            lines.extend(
                [
                    "- Mean normal-input reconstruction fidelity: "
                    f"{row['mean_normal_reconstruction_fidelity']:.4f}",
                    "- Mean anomalous-input reconstruction fidelity: "
                    f"{row['mean_anomaly_reconstruction_fidelity']:.4f}",
                    f"- Reconstruction-fidelity gap: {row['reconstruction_fidelity_gap']:.4f}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Quantum autoencoder",
            "",
            f"- Input qubits: {circuit_info['input_qubits']}",
            f"- Latent qubits: {circuit_info['latent_qubits']}",
            f"- Trash qubits: {circuit_info['trash_qubits']}",
            f"- Input Hilbert-space dimension: {circuit_info['input_state_dimension']}",
            f"- Latent Hilbert-space dimension: {circuit_info['latent_state_dimension']}",
            "- State-dimension reduction: "
            f"{100 * circuit_info['state_dimension_reduction_fraction']:.1f}%",
            f"- Trainable parameters: {circuit_info['trainable_parameters']}",
            f"- Encoder depth: {circuit_info['encoder_depth']}",
            f"- Final normal-state trash loss: {optimizer_info.get('objective')}",
            f"- Objective evaluations: {optimizer_info.get('evaluations', 0)}",
            "",
            "## Interpretation",
            "",
            "Models learned only from known-normal training telemetry. Thresholds were selected on "
            "mixed validation data and evaluated once on the held-out test partition. The dataset "
            "and cost units are synthetic. This exact-simulator experiment is not evidence of "
            "quantum advantage and is not a predictive-maintenance system.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_all_artifacts(
    destination: Path,
    config: dict[str, Any],
    dataset_summary: dict[str, Any],
    circuit_info: dict[str, Any],
    optimizer_info: dict[str, Any],
    partitions: Any,
    prepared: Any,
    results: dict[str, ModelResult],
    circuit: Any,
    quantum_model: Any | None,
    quantum_training_ids: np.ndarray,
    objective_history: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = metrics_frame(results)
    predictions = predictions_frame(partitions, results)
    anomaly_type_performance = anomaly_type_performance_frame(partitions, results)
    thresholds = {name: result.threshold for name, result in results.items()}
    normalizers = {
        name: {"minimum": result.normalizer.minimum, "maximum": result.normalizer.maximum}
        for name, result in results.items()
    }

    save_json(destination / "config.json", config)
    save_json(destination / "environment.json", environment_details())
    save_json(destination / "dataset_summary.json", dataset_summary)
    save_json(destination / "circuit_diagnostics.json", circuit_info)
    save_json(destination / "optimizer.json", optimizer_info)
    save_json(destination / "thresholds.json", thresholds)
    save_json(destination / "score_normalizers.json", normalizers)
    save_json(destination / "metrics.json", {"models": metrics.to_dict(orient="records")})
    metrics.to_csv(destination / "metrics.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    anomaly_type_performance.to_csv(destination / "anomaly_type_performance.csv", index=False)
    partitions.full_dataset.to_csv(destination / "synthetic_sensor_telemetry.csv", index=False)
    split_assignments_frame(partitions).to_csv(destination / "split_assignments.csv", index=False)
    pd.DataFrame({"sample_id": quantum_training_ids}).to_csv(
        destination / "quantum_training_ids.csv", index=False
    )

    threshold_frames = []
    for name, result in results.items():
        frame = result.threshold_history.copy()
        frame.insert(0, "model", name)
        threshold_frames.append(frame)
    pd.concat(threshold_frames, ignore_index=True).to_csv(
        destination / "threshold_search.csv", index=False
    )
    pd.DataFrame(
        {
            "evaluation": np.arange(1, len(objective_history) + 1),
            "trash_loss": objective_history,
        }
    ).to_csv(destination / "optimization_history.csv", index=False)
    np.savez_compressed(
        destination / "prepared_data.npz",
        standard_train=prepared.standard_train,
        standard_validation=prepared.standard_validation,
        standard_test=prepared.standard_test,
        amplitude_train=prepared.amplitude_train,
        amplitude_validation=prepared.amplitude_validation,
        amplitude_test=prepared.amplitude_test,
        y_validation=partitions.y_validation,
        y_test=partitions.y_test,
    )
    joblib.dump(prepared.scaler, destination / "sensor_scaler.joblib")
    joblib.dump(results["pca_autoencoder"].model, destination / "pca_autoencoder.joblib")
    joblib.dump(results["isolation_forest"].model, destination / "isolation_forest.joblib")
    if quantum_model is not None:
        save_json(destination / "quantum_autoencoder.json", quantum_model.to_dict())
    (destination / "encoder_circuit.txt").write_text(
        str(circuit.draw(output="text", fold=120)) + "\n", encoding="utf-8"
    )

    plot_class_distribution(destination / "class_distribution.png", partitions.full_dataset)
    plot_sensor_signals(destination / "sensor_distributions.png", partitions.full_dataset)
    plot_time_series(destination / "sensor_time_series.png", partitions.full_dataset)
    plot_precision_recall(destination / "precision_recall_curves.png", partitions.y_test, results)
    plot_roc(destination / "roc_curves.png", partitions.y_test, results)
    plot_confusion_matrices(destination / "confusion_matrices.png", partitions.y_test, results)
    plot_score_distributions(
        destination / "anomaly_score_distributions.png", partitions.y_test, results
    )
    plot_threshold_costs(destination / "threshold_costs.png", results)
    plot_objective(destination / "optimization_history.png", objective_history)
    lookup = partitions.full_dataset.set_index("sample_id")
    test_anomaly_types = lookup.loc[partitions.test_ids, "anomaly_type"].to_numpy()
    plot_qae_reconstruction_fidelity(
        destination / "qae_reconstruction_fidelity.png",
        test_anomaly_types,
        results.get("quantum_autoencoder"),
    )
    _write_report(
        destination / "experiment_report.md",
        metrics,
        dataset_summary,
        circuit_info,
        optimizer_info,
    )
    return metrics, predictions
