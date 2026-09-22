"""Integration tests. Created by School of AI and School of QC."""

import json
import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from quantum_autoencoder_anomaly_detector.config import ExperimentConfig
from quantum_autoencoder_anomaly_detector.experiment import run_experiment
from quantum_autoencoder_anomaly_detector.inference import predict_frame


def small_config(tmp_path, include_quantum=False):
    return ExperimentConfig(
        sample_count=400,
        anomaly_rate=0.10,
        normal_train_limit=40,
        ansatz_reps=1,
        optimizer_maxiter=14,
        include_quantum_autoencoder=include_quantum,
        output_root=str(tmp_path),
    ).validate()


def test_classical_experiment_saves_reloadable_artifacts(tmp_path):
    result = run_experiment(small_config(tmp_path))
    assert len(result.metrics) == 2
    assert (result.output_directory / "experiment_report.md").exists()
    assert (result.output_directory / "anomaly_type_performance.csv").exists()
    assert (result.output_directory / "qae_reconstruction_fidelity.png").exists()
    sample = result.partitions.full_dataset.iloc[:3]
    predictions = predict_frame(result.output_directory, sample)
    assert len(predictions) == 3
    assert "isolation_forest_prediction" in predictions
    assert "pca_autoencoder_raw_score" in predictions


def test_quantum_experiment_saves_reloadable_model(tmp_path):
    result = run_experiment(small_config(tmp_path, include_quantum=True))
    model_data = json.loads((result.output_directory / "quantum_autoencoder.json").read_text())
    assert len(result.metrics) == 3
    assert len(model_data["weights"]) == 12
    predictions = predict_frame(result.output_directory, result.partitions.full_dataset.iloc[:2])
    assert "quantum_autoencoder_score" in predictions
    assert "quantum_autoencoder_raw_score" in predictions
    assert "quantum_autoencoder_reconstruction_fidelity" in predictions
    assert (
        (result.output_directory / "anomaly_type_performance.csv")
        .read_text()
        .startswith("model,anomaly_type,samples")
    )
    assert len(list(result.output_directory.glob("*.png"))) == 10


def test_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "quantum_autoencoder_anomaly_detector.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "quantum autoencoder" in completed.stdout.lower()


def test_app_starts_headlessly():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Quantum Autoencoder Anomaly Detector"


def test_quickstart_notebook_is_valid_and_branded():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "quickstart.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "Created by School of AI and School of QC" in sources
    assert "run_experiment" in sources
