# Experiment Guide

Created by School of AI and School of QC.

Start with `configs/default.json`. Change one factor at a time and save each timestamped artifact
directory. Useful experiments include varying latent qubits, encoder repetitions, optimizer
evaluations, normal training limit, anomaly prevalence, and the false-negative cost ratio.

For a faster classical-only check:

```bash
qaad-train --samples 400 --skip-quantum-autoencoder --output artifacts
```

For a small QAE run:

```bash
qaad-train --samples 400 --normal-limit 60 --ansatz-reps 1 --maxiter 30
```

For a guided Python workflow, open `notebooks/quickstart.ipynb`. It includes latent-state
compression, reconstruction fidelity, and saved-model inference.

Do not compare runs solely by accuracy. Inspect average precision, recall, false positives,
false negatives, business cost, runtime, and seed sensitivity. For a serious study, repeat each
configuration across several seeds and report uncertainty rather than only the best run.

When comparing latent widths, remember that one latent qubit retains a two-dimensional
conditional state while two latent qubits retain four dimensions. Record the state-dimension
reduction alongside accuracy metrics; otherwise an easier, less-compressed model may appear
better without making the tradeoff visible.
