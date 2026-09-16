# Model card

**Created by School of AI and School of QC**

## Model details

This repository trains four binary classifiers over the same eight spatial patch features:

- an eight-qubit quantum convolutional neural network;
- logistic regression;
- a calibrated radial-basis-function support-vector machine;
- a one-hidden-layer neural network.

The primary subject is the QCNN. It uses 12 shared circuit parameters and a two-parameter classical sigmoid head. Its output is the probability assigned to the synthetic horizontal-line class.

## Intended purpose

The models support education, software testing, and controlled comparison of a small variational quantum architecture with classical baselines. Saved models also demonstrate a complete train-persist-load-infer workflow.

They are not designed for deployment, consequential decisions, general image classification, or claims about quantum advantage.

## Inputs and outputs

Input is an 8×8 finite floating-point array with values in `[0, 1]`. A fixed transform produces eight 2×4 patch means. Output includes predicted class (`vertical` or `horizontal`) and probabilities for both classes.

Inputs outside this schema are rejected. Inputs inside the schema but outside the generator’s distribution may still produce unjustified confident predictions.

## Training data

Training data is generated locally. It contains balanced horizontal and vertical line images with variable position, thickness, intensity, noise, a faint nuisance patch, and optional square occlusion. See [DATASET.md](DATASET.md).

## Evaluation

The frozen reference uses three deterministic stratified 70/30 holdouts. Accuracy is primary; balanced accuracy, precision, recall, F1, ROC AUC, log loss, fit time, occlusion subgroup accuracy, and two QCNN sensitivity probes are saved. See [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md) and [RESULTS.md](RESULTS.md).

## Limitations

- Synthetic performance may not transfer to any real image source.
- Patch averaging discards most local structure.
- Three holdouts provide limited uncertainty characterization.
- Probabilities are not validated for real-world calibration.
- Classical and quantum models do not receive equal compute budgets.
- Exact simulation omits hardware noise, compilation, connectivity, queueing, and execution constraints.
- The finite-shot and angle-noise probes are simplified interventions.

## Ethical considerations

The generated dataset contains no people or personal information. That does not make the models appropriate for human-impacting use. Any adaptation to real data requires a new dataset audit, task-specific risk analysis, subgroup evaluation, privacy review, security review, and human accountability.

## Maintenance

The test suite and pinned major dependency ranges define the supported behavior. Changes to circuit gates, feature order, labels, default hyperparameters, or probability semantics should be treated as model-version changes and recorded in `CHANGELOG.md`.

**Created by School of AI and School of QC**
