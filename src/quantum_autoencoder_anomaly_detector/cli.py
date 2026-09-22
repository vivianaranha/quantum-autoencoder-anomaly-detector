"""Command-line interface.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qaad-train",
        description="Train classical and quantum autoencoder anomaly detectors.",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON configuration")
    parser.add_argument("--samples", type=int, help="Synthetic telemetry sample count")
    parser.add_argument("--anomaly-rate", type=float, help="Injected anomaly prevalence")
    parser.add_argument("--validation-size", type=float, help="Validation fraction")
    parser.add_argument("--test-size", type=float, help="Test fraction")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--normal-limit", type=int, help="QAE normal training sample limit")
    parser.add_argument("--latent-qubits", type=int, help="Compressed quantum latent qubits")
    parser.add_argument("--ansatz-reps", type=int, help="Encoder entangling repetitions")
    parser.add_argument("--maxiter", type=int, help="COBYLA objective-evaluation budget")
    parser.add_argument("--pca-components", type=int, help="Classical PCA latent components")
    parser.add_argument("--false-negative-cost", type=float, help="Cost of a missed anomaly")
    parser.add_argument("--false-positive-cost", type=float, help="Cost of a false alarm")
    parser.add_argument("--output", type=str, help="Artifact root directory")
    quantum = parser.add_mutually_exclusive_group()
    quantum.add_argument(
        "--include-quantum-autoencoder",
        action="store_true",
        dest="include_quantum_autoencoder",
    )
    quantum.add_argument(
        "--skip-quantum-autoencoder",
        action="store_false",
        dest="include_quantum_autoencoder",
    )
    parser.set_defaults(include_quantum_autoencoder=None)
    return parser


def resolve_config(arguments: argparse.Namespace) -> ExperimentConfig:
    config = (
        ExperimentConfig.from_json(arguments.config)
        if arguments.config
        else ExperimentConfig().validate()
    )
    return config.with_overrides(
        sample_count=arguments.samples,
        anomaly_rate=arguments.anomaly_rate,
        validation_size=arguments.validation_size,
        test_size=arguments.test_size,
        random_seed=arguments.seed,
        normal_train_limit=arguments.normal_limit,
        latent_qubits=arguments.latent_qubits,
        ansatz_reps=arguments.ansatz_reps,
        optimizer_maxiter=arguments.maxiter,
        pca_latent_components=arguments.pca_components,
        false_negative_cost=arguments.false_negative_cost,
        false_positive_cost=arguments.false_positive_cost,
        include_quantum_autoencoder=arguments.include_quantum_autoencoder,
        output_root=arguments.output,
    )


def main() -> None:
    arguments = build_parser().parse_args()
    config = resolve_config(arguments)
    result = run_experiment(config, progress_callback=lambda message: print(f"[qaad] {message}"))
    columns = [
        "model",
        "training_samples",
        "threshold",
        "precision",
        "recall",
        "f2",
        "roc_auc",
        "average_precision",
        "business_cost",
    ]
    print("\nHeld-out metrics")
    print(result.metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nInput qubits: {result.circuit_info['input_qubits']}")
    print(f"Latent qubits: {result.circuit_info['latent_qubits']}")
    print(f"Trainable parameters: {result.circuit_info['trainable_parameters']}")
    print(f"Artifacts: {result.output_directory.resolve()}")
    print("\nCreated by School of AI and School of QC")


if __name__ == "__main__":
    main()
