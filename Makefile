.PHONY: install run train test lint format build

install:
	python -m pip install -e ".[dev]"

run:
	streamlit run app.py

train:
	qaad-train --config configs/default.json

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

build:
	python -m build
