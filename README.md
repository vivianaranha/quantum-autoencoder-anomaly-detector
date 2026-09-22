# Quantum Autoencoder Anomaly Detector

Created by School of AI and School of QC.

A complete, local-first project that trains a variational quantum autoencoder (QAE) on normal
industrial-style sensor telemetry, compresses each three-qubit input into a one-qubit latent
state, reconstructs the input with the learned decoder, and treats low reconstruction fidelity
as evidence of an unusual or corrupted input.

This repository is ready to upload to GitHub. It includes a reusable Python package, CLI,
three-workflow Streamlit app, executable quickstart notebook, saved-model CSV inference,
deterministic synthetic data, classical controls, 60 automated tests, CI, curated reference
evidence, governance files, and an MIT license. Everything runs on a local exact statevector;
no cloud account, API key, paid service, or quantum hardware is required.

## Project goals

- Learn a compact quantum representation from known-normal inputs only.
- Expose the actual latent state, decoded state, and round-trip fidelity.
- Detect sensor spikes, process drift, correlation breaks, and stuck-sensor corruption.
- Compare the QAE honestly with PCA reconstruction and Isolation Forest controls.
- Prevent train/test leakage through a frozen train/validation/test protocol.
- Save enough provenance to reproduce, inspect, and reload every result.

## How it works

Eight standardized sensor features are amplitude-normalized into a real three-qubit state.
A trainable encoder made from RY/RZ rotations and circular CX entanglement tries to move
normal-state information into the lower latent qubit while resetting two trash qubits.

The encoded state is projected onto the all-zero trash subspace. Its normalized one-qubit
component is the compressed representation. To decode, the project attaches |00⟩ trash qubits
and applies the inverse encoder. The round-trip fidelity is exactly the all-zero trash
probability, so the raw anomaly score is

$$s(x)=1-F\left(|x\rangle, |\hat{x}\rangle\right).$$

This makes the compression objective, reconstruction, and anomaly score three views of the same
quantity rather than unrelated demonstrations.

~~~mermaid
flowchart TD
    A["8 sensor features"] --> B["3-qubit amplitude state"]
    B --> C["Trainable encoder"]
    C --> D["1-qubit latent state"]
    D --> E["Attach zero trash + inverse encoder"]
    E --> F["Reconstruction fidelity"]
    F --> G["Anomaly score = 1 − fidelity"]
~~~

The evaluation protocol is separate and leakage-safe:

~~~mermaid
flowchart TD
    A["Deterministic synthetic telemetry"] --> B["Stratified split"]
    B --> C["Clean-normal train"]
    B --> D["Mixed validation"]
    B --> E["Untouched mixed test"]
    C --> F["Fit scaler and detectors"]
    D --> G["Fit score ranges and thresholds"]
    F --> G
    G --> H["Freeze policy"]
    E --> H
~~~

Labels never enter detector fitting. Validation labels are used only for cost-sensitive
threshold selection; the held-out test set is scored once.

## Quick start

Python 3.11 or 3.12 is supported.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
streamlit run app.py
~~~

The app provides three workflows:

- **Run benchmark** — configure, train, evaluate, and inspect a held-out sample.
- **Inspect saved run** — review metrics, circuit text, reports, and plots.
- **Score CSV** — upload sensor rows and download predictions from saved models.

For a reproducible command-line run:

~~~bash
qaad-train --config configs/default.json
~~~

For a smaller educational run:

~~~bash
qaad-train --samples 400 --normal-limit 60 --ansatz-reps 1 --maxiter 30
~~~

## Quickstart notebook

Open [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) after installing the project. It
runs a compact experiment, compares detectors, inspects latent states and reconstruction
fidelity, and reloads the saved inference pipeline.

To install a local JupyterLab interface as well:

~~~bash
python -m pip install -e ".[notebook]"
jupyter lab notebooks/quickstart.ipynb
~~~

## Score a CSV

Generate a sample file, run a benchmark, and score the file:

~~~bash
python scripts/generate_sensor_data.py
qaad-train --config configs/default.json
python scripts/predict_csv.py artifacts/<run-name> data/synthetic_sensor_telemetry.csv \
  --output predictions.csv
~~~

The CSV must contain the eight numeric feature columns documented in
[data/README.md](data/README.md). A sample_id column is carried into the output when present.
Predictions contain raw scores, validation-normalized scores, binary decisions, and QAE
reconstruction fidelity.

## Reference evidence

The frozen seed-42 configuration trained on 900 synthetic samples and evaluated 180 held-out
samples. The quantum circuit used three input qubits, one latent qubit, two trash qubits, 18
trainable angles, and 100 COBYLA objective evaluations.

The QAE achieved:

- ROC AUC: **0.9281**
- Average precision: **0.6739**
- Precision: **0.4318**
- Recall: **0.8636**
- F2: **0.7197**
- Configured test cost: **40**
- Mean normal-input reconstruction fidelity: **0.8404**
- Mean anomalous-input reconstruction fidelity: **0.3924**
- Reconstruction-fidelity gap: **0.4480**

The PCA control reached ROC AUC and average precision of **1.0000** on these deliberately
structured synthetic anomalies. Isolation Forest reached ROC AUC **0.9312** and average
precision **0.6495**. The classical win is reported prominently: using a quantum circuit does
not imply better performance.

![Quantum decoder reconstruction fidelity by input family](docs/assets/qae_reconstruction_fidelity.png)

![Held-out precision–recall curves](docs/assets/precision_recall_curves.png)

The exact metrics, environment, thresholds, circuit diagnostics, optimizer trace, and report are
checked into [examples/reference-run](examples/reference-run). See
[docs/RESULTS.md](docs/RESULTS.md) for interpretation and
[docs/VERIFICATION.md](docs/VERIFICATION.md) for the verification record.

## Generated artifacts

Each timestamped run records:

- configuration, environment, dataset summary, and split assignments;
- fitted scaler, PCA model, Isolation Forest, and serialized QAE weights;
- raw and normalized scores, frozen thresholds, decisions, and held-out metrics;
- per-anomaly-family behavior and reconstruction fidelity;
- circuit diagnostics, circuit text, optimizer history, and selected QAE training IDs;
- ten plots and a self-contained Markdown experiment report.

Generated runs stay under artifacts/ and are ignored by Git. The compact
examples/reference-run/ evidence bundle is intentionally versioned.

## Quality checks

~~~bash
ruff check .
ruff format --check .
pytest
python -m build
~~~

GitHub Actions repeats these gates on Python 3.11 and 3.12. The tests include independent Qiskit
Statevector parity checks for trash probabilities, compression normalization, decoder fidelity,
artifact reloads, the CLI, and the Streamlit startup path.

## Repository map

- app.py — interactive benchmark, run inspector, and CSV-scoring workbench
- src/quantum_autoencoder_anomaly_detector/ — reusable experiment package
- notebooks/quickstart.ipynb — end-to-end learning notebook
- configs/default.json — frozen reference configuration
- examples/reference-run/ — compact reproducibility evidence
- scripts/ — dataset generation and saved-model CSV inference
- tests/ — unit, parity, integration, CLI, serialization, and app tests
- docs/ — architecture, mathematics, evaluation, results, safety, and upload guides
- .github/workflows/ci.yml — Python 3.11/3.12 quality gates

## Scope and limitations

This is an educational exact-simulation benchmark, not a predictive-maintenance product.
Synthetic anomalies are cleaner than physical failures. Three simulated qubits do not establish
scalability. Exact statevectors omit shot noise, state-preparation cost, device noise,
transpilation, and calibration drift. Scores identify unusual patterns, not causes.

Do not use this project for safety shutdowns, employment, insurance, healthcare, finance, or
other consequential decisions. Real deployment would require representative data, uncertainty
analysis, drift monitoring, threshold governance, security review, domain validation, and human
oversight.

## Technical references

The architecture follows the quantum-autoencoder objective introduced by Romero, Olson, and
Aspuru-Guzik in [Quantum autoencoders for efficient compression of quantum
data](https://arxiv.org/abs/1612.02806). Exact state evolution is implemented with Qiskit's
[Statevector and quantum-information APIs](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector).
Classical controls use scikit-learn's
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and
[IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

## License and citation

Released under the MIT License. If this project supports teaching or research, cite the included
CITATION.cff and disclose changes to the generator, split, threshold costs, circuit, or optimizer
budget.

Created by School of AI and School of QC.
