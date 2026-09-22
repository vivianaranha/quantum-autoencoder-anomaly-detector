# GitHub Upload

Created by School of AI and School of QC.

Create an empty GitHub repository, extract this project, and run:

```bash
git init
git add .
git commit -m "Initial quantum autoencoder anomaly detector"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

Before pushing, run `ruff check .`, `ruff format --check .`, `pytest`, and
`python -m build`. Generated runs and data
are ignored. Review the files staged by `git status`; never commit credentials or private sensor
records. GitHub Actions will repeat the quality checks on Python 3.11 and 3.12 after upload.

The curated `examples/reference-run` files and `docs/assets` plots are intentionally tracked.
They make the README evidence inspectable without committing large fitted-model artifacts.
