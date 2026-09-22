# Responsible Use

Created by School of AI and School of QC.

Treat this repository as an educational simulator. Before adapting it to real telemetry, obtain
permission to use the data, document sensor provenance, remove identifiers, evaluate operating
regimes separately, test drift, quantify uncertainty, and include domain experts in threshold
selection and incident review.

Never interpret an anomaly score as proof of failure, misconduct, fraud, or unsafe behavior.
Require human review and corroborating evidence. Maintain a safe fallback when the model or
quantum service is unavailable. Log model versions, thresholds, alerts, overrides, and outcomes.

Do not infer that a lower-dimensional quantum state is automatically private, secure, unbiased,
or useful. Compression fidelity measures state overlap in this simulator; it does not establish
semantic preservation, causal diagnosis, or protection against reconstruction attacks.

Report security issues through `SECURITY.md`. Do not include sensitive production telemetry in a
public issue or repository.
