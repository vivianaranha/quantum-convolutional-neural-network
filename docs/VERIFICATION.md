# Verification record

**Created by School of AI and School of QC**

Release 1.0.0 was verified on 2026-09-16 with Python 3.12.14.

## Source checks

- `ruff check .` — passed.
- `ruff format --check .` — passed for 40 formatted Python files.
- `pytest` — 65 passed.
- `python -m compileall -q src tests app.py scripts` — passed.
- `python -m pip check` — no broken requirements.
- Notebook JSON parse — passed.
- Local Markdown target scan — passed.
- Common placeholder and credential-pattern scan — passed.

The test suite includes deterministic data checks, validation failures, state normalization, independent NumPy/Qiskit statevector parity, analytical-gradient/finite-difference parity, training-loss behavior, model serialization, metric aggregation, split isolation, artifact completeness, CLI inference, direct-script inference, notebook structure, and Streamlit rendering.

## Reference-run checks

- The frozen seed-42 configuration completed all three repeated holdouts.
- Every expected data, model, metric, provenance, circuit, report, and plot artifact was written.
- All eight generated plots were visually inspected.
- Saved-model inference produced four valid probability pairs for a generated test image.
- Machine-readable result copies match the completed run artifacts.

## Distribution checks

- `python -m build` created a wheel and source distribution.
- Wheel contents include the complete `quantum_cnn` runtime package and metadata.
- Source-distribution contents include code, tests, documentation, reference figures, and selected reference outputs.
- The wheel imported as version 1.0.0 with provisioned dependencies, built Qiskit diagnostics, and exposed the benchmark CLI.

## Clean archive checks

A release-candidate ZIP was created with caches, bytecode, build products, egg metadata, and the full generated run directory excluded. The archive contained 86 curated entries and passed `unzip -t`. From an unrelated temporary extraction directory:

- Ruff lint and format checks passed;
- all 65 tests passed;
- wheel and source-distribution builds succeeded;
- the installed wheel imported successfully; and
- the installed benchmark command displayed its complete help interface.

The final release ZIP is produced from the same curated file rules after this record is included, then checked again for integrity and forbidden entries.

## Scope

Verification covers source formatting and linting, unit and integration tests, Python package creation, command-line entry points, notebook structure, archive integrity, and a clean extracted-copy rerun. Scientific limitations are documented separately; software verification does not validate claims outside the benchmark protocol.

**Created by School of AI and School of QC**
