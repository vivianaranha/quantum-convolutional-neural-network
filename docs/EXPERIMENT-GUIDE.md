# Experiment guide

**Created by School of AI and School of QC**

## Run the frozen benchmark

```bash
python -m pip install -e ".[dev]"
quantum-cnn benchmark --config configs/default.json
```

The command prints progress, a compact summary, and the timestamped artifact directory. Keep the committed default unchanged when reproducing the reference. Create a separate config for new experiments.

## Run a controlled variant

Copy `configs/default.json`, change one factor, and give the output a separate root. For example, to study stronger image corruption:

```json
{
  "samples": 240,
  "image_size": 8,
  "image_noise": 0.25,
  "occlusion_probability": 0.25,
  "repeats": 3,
  "test_size": 0.3,
  "qcnn_epochs": 80,
  "qcnn_learning_rate": 0.05,
  "qcnn_l2": 0.0005,
  "mlp_hidden_units": 8,
  "mlp_max_iterations": 2000,
  "shot_count": 256,
  "shot_trials": 10,
  "parameter_noise": 0.04,
  "noise_trials": 10,
  "random_seed": 42,
  "split_seed_offset": 10000,
  "output_root": "artifacts/noise-025"
}
```

Run it with:

```bash
quantum-cnn benchmark --config configs/noise-025.json
```

## CLI overrides

The benchmark subcommand supports focused overrides:

```bash
quantum-cnn benchmark \
  --config configs/default.json \
  --samples 120 \
  --repeats 1 \
  --qcnn-epochs 20 \
  --seed 123 \
  --output-root artifacts/smoke
```

Configuration validation rejects unsupported image sizes, invalid probability ranges, insufficient samples, and unreasonable iteration limits before training begins.

## Interpret the artifact bundle

- Start with `benchmark_report.md` and `summary_overall.csv`.
- Inspect `metrics_per_split.csv` before trusting the mean.
- Check `split_wins_or_ties.csv` for consistency across repeats.
- Use `test_predictions.csv` for error analysis.
- Compare `occlusion_subgroup_metrics.csv` groups cautiously because their counts vary.
- Inspect `qcnn_training_history.csv` for optimization behavior.
- Treat `qcnn_robustness.csv` as sensitivity analysis, not hardware evidence.
- Use `config.json`, `dependency_versions.json`, and `split_assignments.csv` for provenance.

## Score a saved model

Generated input:

```bash
quantum-cnn predict \
  --artifacts artifacts/run-YYYYMMDDTHHMMSSZ \
  --orientation vertical \
  --position 2 \
  --thickness 2 \
  --noise 0.10 \
  --seed 9 \
  --occluded \
  --output saved_predictions.csv \
  --save-image generated_image.npy
```

Existing input:

```bash
quantum-cnn predict \
  --artifacts artifacts/run-YYYYMMDDTHHMMSSZ \
  --image my_image.csv
```

Only load joblib artifacts from a trusted run.

## Good experiment hygiene

- Form a hypothesis before reading the outcome.
- Change one factor at a time when possible.
- Never tune on the reported test rows.
- Record failed runs and software versions.
- Use more seeds or an independent test set before making comparative claims.
- Separate simulator findings from hardware findings.

**Created by School of AI and School of QC**
