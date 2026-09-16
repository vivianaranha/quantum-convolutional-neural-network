# Project manifest

**Created by School of AI and School of QC**

## Entry points

- `app.py` — interactive Streamlit benchmark and inference dashboard.
- `quantum-cnn benchmark` — installed benchmark CLI.
- `quantum-cnn predict` — installed saved-model inference CLI.
- `scripts/predict_saved_models.py` — direct inference script.
- `notebooks/quickstart.ipynb` — notebook walkthrough.

## Source package

- `src/quantum_cnn/__init__.py` — public package surface and version.
- `src/quantum_cnn/config.py` — immutable validated settings.
- `src/quantum_cnn/data.py` — image generation and patch features.
- `src/quantum_cnn/qcnn.py` — simulator, circuit, optimizer, gradients, and training.
- `src/quantum_cnn/models.py` — QCNN persistence wrapper and classical baselines.
- `src/quantum_cnn/metrics.py` — per-split and aggregate metrics.
- `src/quantum_cnn/benchmark.py` — experiment orchestration and probes.
- `src/quantum_cnn/artifacts.py` — provenance, data, model, report, and plot persistence.
- `src/quantum_cnn/visualization.py` — non-interactive result plots.
- `src/quantum_cnn/inference.py` — image validation and trusted-run inference.
- `src/quantum_cnn/cli.py` — command parser and terminal output.

## Configuration and environment

- `configs/default.json` — frozen seed-42 reference configuration.
- `pyproject.toml` — package metadata, dependencies, entry point, and tool settings.
- `requirements.txt` and `requirements-dev.txt` — pip-compatible dependency lists.
- `.streamlit/config.toml` — light Streamlit theme.
- `.env.example` — documents that no secret is required.
- `.gitignore` — excludes environments, caches, builds, and generated runs.
- `MANIFEST.in` — source-distribution inclusions.
- `Makefile` — install, format, lint, test, run, benchmark, build, and verify shortcuts.

## Tests

- `tests/conftest.py` — shared deterministic tiny benchmark.
- `tests/test_config.py` — configuration behavior and validation.
- `tests/test_data.py` — generator, image, and feature invariants.
- `tests/test_qcnn.py` — gates, normalization, Qiskit parity, gradients, training, and serialization.
- `tests/test_metrics_models.py` — metrics, model factories, and complexity accounting.
- `tests/test_integration.py` — artifact, CLI, inference, notebook, split, and Streamlit integration.

## Documentation

- `README.md` — project overview and primary user guide.
- `docs/ARCHITECTURE.md` — components, data flow, topology, and tradeoffs.
- `docs/QCNN.md` — feature map, gates, readout, optimization, and simulator validation.
- `docs/DATASET.md` — dataset generation, fields, uses, and limits.
- `docs/BENCHMARK-METHODOLOGY.md` — frozen protocol, models, metrics, and leakage controls.
- `docs/MODEL-CARD.md` — intended use, inputs, evaluation, limits, and maintenance.
- `docs/RESPONSIBLE-USE.md` — interpretation and artifact-safety boundaries.
- `docs/EXPERIMENT-GUIDE.md` — running, varying, and interpreting experiments.
- `docs/RESULTS.md` — frozen seed-42 results.
- `docs/VERIFICATION.md` — software and archive verification record.
- `docs/GITHUB-UPLOAD.md` — repository initialization and upload steps.
- `examples/reference-run/` — selected machine-readable reference outputs and figures.

## Community and release files

- `LICENSE` — MIT license.
- `CONTRIBUTING.md` — development and review expectations.
- `CODE_OF_CONDUCT.md` — participation standards.
- `SECURITY.md` — vulnerability reporting and joblib warning.
- `ROADMAP.md` — research and engineering directions.
- `CHANGELOG.md` — release history.
- `CITATION.cff` — citation metadata.
- `.github/workflows/ci.yml` — CI across Python 3.11 and 3.12.
- `.github/ISSUE_TEMPLATE/` and `.github/pull_request_template.md` — contribution templates.

**Created by School of AI and School of QC**
