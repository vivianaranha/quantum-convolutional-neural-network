# Roadmap

**Created by School of AI and School of QC**

The roadmap prioritizes scientific clarity before adding scale.

## Near term

- Add confidence intervals over a larger, predeclared seed panel.
- Add a separate validation split for hyperparameter selection.
- Add probability-calibration plots and expected calibration error.
- Add a configurable patch map while preserving explicit feature order.
- Benchmark simulator memory and time by batch size.

## Research extensions

- Compare alternative data encodings at a fixed qubit count.
- Add circuit-depth and two-qubit-gate ablations.
- Compare shared versus unshared stage parameters.
- Add a small real public dataset with a documented license and an independent test set.
- Add a device-noise study tied to a named calibration snapshot.
- Add optional execution on quantum hardware behind an explicit, separately tested adapter.

## Engineering extensions

- Version the saved-model schema and add migration tests.
- Export a static HTML benchmark report.
- Add property-based tests for image validation and gates.
- Add signed release artifacts and software-bill-of-materials generation.

Hardware support will not be labeled complete until the repository records backend identity, transpilation settings, circuit layout, queue/execution metadata, shot budgets, error-mitigation choices, and a classical simulation control.

**Created by School of AI and School of QC**
