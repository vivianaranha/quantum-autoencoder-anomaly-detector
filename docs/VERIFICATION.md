# Verification Record

Created by School of AI and School of QC.

Verification completed on 2026-09-20 with Python 3.12.14 on Linux. The frozen environment
recorded Qiskit 2.5.2, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, SciPy 1.17.0, and
Streamlit 1.64.0. The package supports Python 3.11 and 3.12; CI tests both versions.

Release checks:

- ruff check passed.
- ruff format check passed.
- 60 tests passed.
- Independent Qiskit Statevector parity checks covered two circuit depths and two latent widths.
- Compression produced normalized latent states.
- Decoder fidelity matched the all-zero trash probability to numerical precision.
- Saved classical and quantum models reloaded and scored new frames.
- The CLI help path and Streamlit application startup passed.
- The source distribution and wheel built successfully.
- The final GitHub ZIP was tested again from a fresh extraction.

The default seed-42 run produced 33 artifacts from 900 samples. The circuit used three input
qubits, one latent qubit, two trash qubits, 18 trainable angles, six CX gates, and decomposed
depth 12. It maps an eight-dimensional input state to a two-dimensional conditional latent
state.

COBYLA consumed the configured 100-evaluation budget and ended at mean normal trash loss
0.157219. Its status explicitly reports that the maximum function-evaluation budget was reached;
this is expected for the fixed-budget experiment and is not represented as optimizer
convergence.

Held-out reference results:

- Quantum autoencoder: ROC AUC 0.9281, average precision 0.6739, precision 0.4318, recall 0.8636,
  F2 0.7197, configured cost 40, mean normal fidelity 0.8404, mean anomaly fidelity 0.3924.
- PCA autoencoder: ROC AUC 1.0000, average precision 1.0000, precision 1.0000, recall 0.9545,
  F2 0.9633, configured cost 5.
- Isolation Forest: ROC AUC 0.9312, average precision 0.6495, precision 0.4348, recall 0.9091,
  F2 0.7463, configured cost 36.

The curated machine-readable evidence is in examples/reference-run. Results remain specific to
the synthetic generator, default split, seed, software environment, and asymmetric costs.
