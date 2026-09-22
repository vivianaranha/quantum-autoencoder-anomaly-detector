"""Quantum Autoencoder Anomaly Detector.

Created by School of AI and School of QC.
"""

from .config import ExperimentConfig
from .experiment import ExperimentResult, run_experiment

__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment"]
__version__ = "1.0.0"
