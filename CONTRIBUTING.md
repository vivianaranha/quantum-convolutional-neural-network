# Contributing

**Created by School of AI and School of QC**

Thank you for improving the project. Contributions should preserve reproducibility, matched-input comparison, explicit scientific boundaries, and local execution without paid services.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make verify
```

## Pull requests

- Open an issue first for material architecture, dataset, metric, or circuit changes.
- Keep each pull request focused and explain the motivation and tradeoffs.
- Add or update tests for behavioral changes.
- Update the README, model card, methodology, and changelog when semantics change.
- Do not overwrite the frozen reference result without documenting the new version and reason.
- Do not commit credentials, private data, generated run directories, caches, or build outputs.
- Retain the attribution: “Created by School of AI and School of QC.”

## Scientific changes

Changes to the generator, feature order, circuit gate sequence, readout, label mapping, splits, hyperparameters, or metric aggregation can invalidate comparisons. Include a migration note, a before/after experiment, and a clear statement of which hypotheses were formed before seeing results.

## Style

Ruff is the formatter and linter. Use type hints for public functions, validate external inputs, favor deterministic seeds, and keep files small enough to inspect. Avoid hidden network calls and implicit downloads.

## Tests

Run `make verify` before submitting. Tests should be deterministic and complete quickly. Numerical tests should state their tolerance and, when possible, compare independent formulations.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

**Created by School of AI and School of QC**
