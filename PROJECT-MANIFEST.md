# Project Manifest

Created by School of AI and School of QC.

This release contains a reusable Python package, three-workflow Streamlit application,
command-line interface, quickstart notebook, default experiment configuration, data and
inference scripts, 60 automated tests, GitHub Actions workflow, MIT license, citation metadata,
governance templates, curated evidence, and nine focused technical guides.

The repository intentionally excludes generated telemetry, benchmark runs, caches, virtual
environments, and build outputs. Running the default configuration recreates 33 auditable files
inside a timestamped `artifacts` directory, including models, thresholds, raw and normalized
predictions, per-anomaly-family evidence, ten plots, provenance, circuit text, optimizer history,
and a report.

The compact `examples/reference-run` bundle and four `docs/assets` plots are intentionally
tracked so claims in the README can be audited without committing multi-megabyte fitted models.
