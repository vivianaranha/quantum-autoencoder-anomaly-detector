"""Interactive anomaly-detection workbench.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from quantum_autoencoder_anomaly_detector import ExperimentConfig, run_experiment
from quantum_autoencoder_anomaly_detector.data import FEATURE_COLUMNS
from quantum_autoencoder_anomaly_detector.inference import predict_frame

st.set_page_config(page_title="Quantum Autoencoder Anomaly Detector", layout="wide")


def available_runs() -> list[Path]:
    return sorted(
        [path for path in Path("artifacts").glob("run-*") if (path / "metrics.csv").exists()],
        reverse=True,
    )


def render_metrics(metrics: pd.DataFrame) -> None:
    preferred = [
        "model",
        "training_samples",
        "threshold",
        "precision",
        "recall",
        "f2",
        "roc_auc",
        "average_precision",
        "false_positive",
        "false_negative",
        "business_cost",
    ]
    st.dataframe(
        metrics[[name for name in preferred if name in metrics.columns]].style.format(precision=4),
        use_container_width=True,
        hide_index=True,
    )


def render_run(run_directory: Path) -> None:
    metrics = pd.read_csv(run_directory / "metrics.csv")
    render_metrics(metrics)
    gallery = [
        ("precision_recall_curves.png", "Precision–recall"),
        ("roc_curves.png", "ROC"),
        ("confusion_matrices.png", "Confusion matrices"),
        ("qae_reconstruction_fidelity.png", "QAE reconstruction fidelity"),
        ("threshold_costs.png", "Threshold cost"),
        ("optimization_history.png", "QAE optimization"),
    ]
    columns = st.columns(2)
    for position, (filename, caption) in enumerate(gallery):
        image_path = run_directory / filename
        if image_path.exists():
            columns[position % 2].image(str(image_path), caption=caption, use_container_width=True)
    report_path = run_directory / "experiment_report.md"
    if report_path.exists():
        with st.expander("Experiment report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    circuit_path = run_directory / "encoder_circuit.txt"
    if circuit_path.exists():
        with st.expander("Quantum encoder circuit"):
            st.code(circuit_path.read_text(encoding="utf-8"), language="text")
    st.download_button(
        "Download metrics CSV",
        metrics.to_csv(index=False),
        file_name=f"{run_directory.name}-metrics.csv",
        mime="text/csv",
    )
    st.caption(f"Artifact directory: {run_directory}")


def benchmark_page() -> None:
    with st.sidebar:
        st.header("Experiment settings")
        sample_count = st.slider("Telemetry samples", 400, 3_000, 800, step=100)
        anomaly_rate = st.slider("Anomaly rate", 0.05, 0.25, 0.12, step=0.01)
        random_seed = st.number_input("Random seed", 0, 10_000, 42)
        normal_limit = st.slider("QAE normal training limit", 40, 500, 200, step=10)
        latent_qubits = st.selectbox("Latent qubits", [1, 2], index=0)
        ansatz_reps = st.slider("Encoder repetitions", 1, 4, 2)
        parameter_count = 3 * 2 * (ansatz_reps + 1)
        maxiter = st.slider("COBYLA evaluations", parameter_count + 2, 180, 80)
        pca_components = st.slider("Classical PCA components", 1, 5, 2)
        false_negative_cost = st.number_input("Missed-anomaly cost", 1.0, 50.0, 5.0)
        false_positive_cost = st.number_input("False-alarm cost", 1.0, 20.0, 1.0)
        include_qae = st.checkbox("Train quantum autoencoder", value=True)
        run_button = st.button("Run benchmark", type="primary", use_container_width=True)

    st.info(
        "All telemetry and anomaly labels are synthetic. This educational exact-statevector "
        "simulator is not a monitoring, safety, or predictive-maintenance system."
    )
    if run_button:
        config = ExperimentConfig(
            sample_count=sample_count,
            anomaly_rate=anomaly_rate,
            random_seed=int(random_seed),
            normal_train_limit=normal_limit,
            latent_qubits=latent_qubits,
            ansatz_reps=ansatz_reps,
            optimizer_maxiter=maxiter,
            pca_latent_components=pca_components,
            false_negative_cost=float(false_negative_cost),
            false_positive_cost=float(false_positive_cost),
            include_quantum_autoencoder=include_qae,
            output_root="artifacts",
        ).validate()
        status = st.empty()
        with st.spinner("Training anomaly detectors..."):
            st.session_state["experiment_result"] = run_experiment(
                config, progress_callback=status.write
            )
        status.success("Experiment complete")

    result = st.session_state.get("experiment_result")
    if result is None:
        st.subheader("Leakage-safe benchmark protocol")
        st.write(
            "Only clean-normal training telemetry fits the scaler and unsupervised models. Mixed "
            "validation data selects one cost-sensitive threshold per detector. Frozen thresholds "
            "are then evaluated once on an untouched mixed test partition."
        )
        return

    st.subheader("Held-out model comparison")
    render_metrics(result.metrics)
    left, right = st.columns(2)
    left.image(
        str(result.output_directory / "precision_recall_curves.png"),
        use_container_width=True,
    )
    right.image(
        str(result.output_directory / "qae_reconstruction_fidelity.png"),
        use_container_width=True,
    )
    st.subheader("Inspect a held-out sensor sample")
    position = st.slider("Test sample position", 0, len(result.partitions.y_test) - 1, 0)
    sample_id = result.partitions.test_ids[position]
    source = result.partitions.full_dataset.loc[
        result.partitions.full_dataset["sample_id"] == sample_id
    ]
    predictions = predict_frame(result.output_directory, source)
    actual = int(source["is_anomaly"].iloc[0])
    anomaly_type = str(source["anomaly_type"].iloc[0]).replace("_", " ")
    st.metric("Synthetic status", anomaly_type.title() if actual else "Normal")
    st.dataframe(source, use_container_width=True, hide_index=True)
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.success(f"Experiment artifacts: {result.output_directory}")


def inspect_page() -> None:
    st.subheader("Inspect a saved experiment")
    runs = available_runs()
    if not runs:
        st.warning("No local runs were found. Run the benchmark first.")
        return
    selected = st.selectbox("Saved run", runs, format_func=lambda path: path.name)
    render_run(selected)


def inference_page() -> None:
    st.subheader("Score a sensor CSV with saved models")
    runs = available_runs()
    if not runs:
        st.warning("No local runs were found. Run the benchmark first.")
        return
    selected = st.selectbox("Model artifacts", runs, format_func=lambda path: path.name)
    st.write("Required sensor columns: " + ", ".join(FEATURE_COLUMNS))
    template = pd.DataFrame([{name: 0.0 for name in FEATURE_COLUMNS}])
    st.download_button(
        "Download input template",
        template.to_csv(index=False),
        file_name="sensor-input-template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload sensor CSV", type="csv")
    if uploaded is None:
        return
    try:
        input_frame = pd.read_csv(uploaded)
        predictions = predict_frame(selected, input_frame)
    except (ValueError, OSError, KeyError) as error:
        st.error(str(error))
        return
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions",
        predictions.to_csv(index=False),
        file_name="anomaly-predictions.csv",
        mime="text/csv",
    )


st.title("Quantum Autoencoder Anomaly Detector")
st.caption("Created by School of AI and School of QC")
st.write(
    "Compress amplitude-encoded sensor states into a smaller quantum latent register, reconstruct "
    "them, and flag inputs that do not fit the learned normal-state manifold."
)
with st.sidebar:
    page = st.radio("Workspace", ["Run benchmark", "Inspect saved run", "Score CSV"])

if page == "Run benchmark":
    benchmark_page()
elif page == "Inspect saved run":
    inspect_page()
else:
    inference_page()

st.divider()
st.caption("Created by School of AI and School of QC")
