"""Diagnostic visualizations for anomaly detection.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .models import ModelResult

COLORS = ["#3156D3", "#E0565B", "#7C3AED", "#149B8E"]


def plot_class_distribution(path: Path, dataset: pd.DataFrame) -> None:
    counts = dataset["anomaly_type"].value_counts()
    order = ["normal", "sensor_spike", "process_drift", "correlation_break", "stuck_sensor"]
    counts = counts.reindex(order, fill_value=0)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(counts.index.str.replace("_", " "), counts.values, color=[COLORS[0]] + COLORS)
    axis.set_title("Synthetic Sensor Telemetry Classes")
    axis.set_ylabel("Samples")
    axis.tick_params(axis="x", rotation=20)
    for index, value in enumerate(counts.values):
        axis.text(index, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_sensor_signals(path: Path, dataset: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    columns = ["temperature_c", "pressure_bar", "vibration_mm_s", "motor_current_a"]
    for axis, column in zip(axes.flat, columns, strict=True):
        normal = dataset.loc[dataset["is_anomaly"] == 0, column]
        anomaly = dataset.loc[dataset["is_anomaly"] == 1, column]
        sns.histplot(normal, bins=28, stat="density", color=COLORS[0], alpha=0.35, ax=axis)
        sns.histplot(anomaly, bins=28, stat="density", color=COLORS[1], alpha=0.35, ax=axis)
        axis.set_title(column.replace("_", " ").title())
        axis.set_ylabel("Density")
    axes[0, 0].legend(["Normal", "Anomaly"])
    figure.suptitle("Selected Sensor Distributions", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_time_series(path: Path, dataset: pd.DataFrame) -> None:
    subset = dataset.iloc[: min(500, len(dataset))]
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(subset.index, subset["vibration_mm_s"], color=COLORS[0], linewidth=1)
    anomalies = subset["is_anomaly"] == 1
    axis.scatter(
        subset.index[anomalies],
        subset.loc[anomalies, "vibration_mm_s"],
        color=COLORS[1],
        s=25,
        label="Injected anomaly",
    )
    axis.set_title("Vibration Telemetry with Injected Anomalies")
    axis.set_xlabel("Sequence")
    axis.set_ylabel("Vibration (mm/s)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_precision_recall(path: Path, labels: np.ndarray, results: dict[str, ModelResult]) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6))
    for color, (name, result) in zip(COLORS, results.items(), strict=False):
        precision, recall, _ = precision_recall_curve(labels, result.test_scores)
        axis.plot(
            recall,
            precision,
            color=color,
            label=f"{name} — AP {result.metrics['average_precision']:.3f}",
        )
    axis.axhline(float(np.mean(labels)), color="#666666", linestyle="--", label="Base rate")
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.set_title("Precision–Recall Curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_roc(path: Path, labels: np.ndarray, results: dict[str, ModelResult]) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6))
    for color, (name, result) in zip(COLORS, results.items(), strict=False):
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, result.test_scores)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            label=f"{name} — AUC {result.metrics['roc_auc']:.3f}",
        )
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--")
    axis.set_title("Receiver Operating Characteristic")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(
    path: Path, labels: np.ndarray, results: dict[str, ModelResult]
) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.2), squeeze=False)
    for axis, (name, result) in zip(axes[0], results.items(), strict=True):
        matrix = confusion_matrix(labels, result.predictions, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples", cbar=False, ax=axis)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticklabels(["Normal", "Anomaly"])
        axis.set_yticklabels(["Normal", "Anomaly"], rotation=0)
    figure.suptitle("Held-Out Confusion Matrices", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_score_distributions(
    path: Path, labels: np.ndarray, results: dict[str, ModelResult]
) -> None:
    figure, axes = plt.subplots(len(results), 1, figsize=(9, 3.3 * len(results)))
    axes = np.atleast_1d(axes)
    for axis, (name, result) in zip(axes, results.items(), strict=True):
        axis.hist(
            result.test_scores[labels == 0],
            bins=22,
            alpha=0.55,
            color=COLORS[0],
            label="Normal",
        )
        axis.hist(
            result.test_scores[labels == 1],
            bins=22,
            alpha=0.55,
            color=COLORS[1],
            label="Anomaly",
        )
        axis.axvline(result.threshold, color="#111111", linestyle="--", label="Threshold")
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Normalized anomaly score")
        axis.set_ylabel("Samples")
        axis.legend()
    figure.suptitle("Test Anomaly-Score Distributions", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_threshold_costs(path: Path, results: dict[str, ModelResult]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for color, (name, result) in zip(COLORS, results.items(), strict=False):
        history = result.threshold_history
        axis.plot(history["threshold"], history["cost"], color=color, label=name)
        chosen_cost = float(history.loc[history["threshold"] == result.threshold, "cost"].iloc[0])
        axis.scatter([result.threshold], [chosen_cost], color=color, s=45)
    axis.set_title("Validation Cost by Decision Threshold")
    axis.set_xlabel("Normalized anomaly threshold")
    axis.set_ylabel("Configured validation cost")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_objective(path: Path, objective_history: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    if objective_history:
        evaluations = np.arange(1, len(objective_history) + 1)
        axis.plot(evaluations, objective_history, color=COLORS[2])
        axis.scatter(evaluations, objective_history, color=COLORS[2], s=13)
    else:
        axis.text(0.5, 0.5, "Quantum autoencoder was not run", ha="center", va="center")
    axis.set_title("Quantum Trash-Register Loss")
    axis.set_xlabel("Objective evaluation")
    axis.set_ylabel("Mean trash probability")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_qae_reconstruction_fidelity(
    path: Path,
    anomaly_types: np.ndarray,
    result: ModelResult | None,
) -> None:
    """Show decoder reconstruction fidelity for each held-out input family."""

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    if result is None:
        axis.text(
            0.5,
            0.5,
            "Quantum autoencoder was not run",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
    else:
        frame = pd.DataFrame(
            {
                "Input family": pd.Series(anomaly_types).str.replace("_", " ").str.title(),
                "Reconstruction fidelity": 1.0 - result.raw_test_scores,
            }
        )
        order = [
            label
            for label in (
                "Normal",
                "Sensor Spike",
                "Process Drift",
                "Correlation Break",
                "Stuck Sensor",
            )
            if label in set(frame["Input family"])
        ]
        sns.boxplot(
            data=frame,
            x="Input family",
            y="Reconstruction fidelity",
            order=order,
            color=COLORS[0],
            width=0.62,
            fliersize=2,
            ax=axis,
        )
        axis.tick_params(axis="x", rotation=18)
        axis.set_ylim(-0.02, 1.02)
    axis.set_title("Quantum Decoder Reconstruction Fidelity")
    axis.set_ylabel("State fidelity")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
