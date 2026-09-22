# Quantum Autoencoder Anomaly Detector — Experiment Report

**Created by School of AI and School of QC**

## Dataset

- Synthetic sensor samples: 900
- Anomaly samples: 108
- Observed anomaly rate: 0.1200
- Clean-normal training samples: 475
- Validation samples: 180
- Test samples: 180

## Held-out results

### Isolation Forest

- Training samples: 475
- Validation-selected threshold: 0.460
- Precision: 0.4348
- Recall: 0.9091
- F2: 0.7463
- ROC AUC: 0.9312
- Average precision: 0.6495
- False positives: 26
- False negatives: 2
- Configured test cost: 36.00
- Training time: 0.4518 seconds

### Pca Autoencoder

- Training samples: 475
- Validation-selected threshold: 0.025
- Precision: 1.0000
- Recall: 0.9545
- F2: 0.9633
- ROC AUC: 1.0000
- Average precision: 1.0000
- False positives: 0
- False negatives: 1
- Configured test cost: 5.00
- Training time: 0.0008 seconds

### Quantum Autoencoder

- Training samples: 250
- Validation-selected threshold: 0.270
- Precision: 0.4318
- Recall: 0.8636
- F2: 0.7197
- ROC AUC: 0.9281
- Average precision: 0.6739
- False positives: 25
- False negatives: 3
- Configured test cost: 40.00
- Training time: 0.2691 seconds

- Mean normal-input reconstruction fidelity: 0.8404
- Mean anomalous-input reconstruction fidelity: 0.3924
- Reconstruction-fidelity gap: 0.4480

## Quantum autoencoder

- Input qubits: 3
- Latent qubits: 1
- Trash qubits: 2
- Input Hilbert-space dimension: 8
- Latent Hilbert-space dimension: 2
- State-dimension reduction: 75.0%
- Trainable parameters: 18
- Encoder depth: 12
- Final normal-state trash loss: 0.15721875091513066
- Objective evaluations: 100

## Interpretation

Models learned only from known-normal training telemetry. Thresholds were selected on mixed validation data and evaluated once on the held-out test partition. The dataset and cost units are synthetic. This exact-simulator experiment is not evidence of quantum advantage and is not a predictive-maintenance system.
