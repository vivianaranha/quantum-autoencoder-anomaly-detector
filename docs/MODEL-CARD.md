# Model Card

Created by School of AI and School of QC.

## Intended use

Education, local experimentation, and research prototyping for unsupervised quantum machine
learning and anomaly-evaluation methodology.

## Inputs and outputs

Inputs are eight finite numeric industrial-style sensor fields. Each model outputs a raw score,
a validation-normalized anomaly score, and a binary prediction using a validation-selected
threshold. The QAE also outputs reconstruction fidelity and can return the conditional latent
state and reconstructed state.

## Training data

Models train only on deterministic synthetic rows labeled normal. The generator creates four
anomaly families for validation and testing: sensor spike, process drift, correlation break, and
stuck sensor. No real or personal data is included.

## Risks

Performance may not transfer to physical equipment. Exact simulation omits quantum-device
noise. Synthetic labels encode the generator designer's assumptions. Scores do not diagnose a
root cause. Distribution shift can invalidate both normalization and thresholds. Conditional
projection success is part of the QAE score; the exposed latent state alone must not be treated
as a calibrated representation-quality guarantee.

## Out of scope

Autonomous shutdown, safety monitoring, predictive-maintenance claims, worker evaluation,
medical use, and any consequential decision without independent validation and human oversight.
