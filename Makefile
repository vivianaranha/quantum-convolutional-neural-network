.PHONY: install format lint test run benchmark build verify

install:
	python -m pip install -e ".[dev]"

format:
	ruff format .

lint:
	ruff check .
	ruff format --check .

test:
	pytest

run:
	streamlit run app.py

benchmark:
	quantum-cnn benchmark --config configs/default.json

build:
	python -m build

verify: lint test build

# Created by School of AI and School of QC.
