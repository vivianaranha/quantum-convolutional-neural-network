# Benchmark methodology

**Created by School of AI and School of QC**

## Research question

How does a compact, shared-parameter QCNN compare with three familiar classical classifiers when all models receive the same eight spatial image features and exactly the same held-out rows?

The benchmark is exploratory and educational. It does not test quantum advantage.

## Frozen reference protocol

- Generate 240 balanced 8×8 images with seed 42.
- Use image-noise level 0.16 and occlusion probability 0.25.
- Extract eight fixed patch means before splitting.
- Create three stratified 70/30 train/test holdouts.
- Derive each split seed and model seed deterministically from the experiment seed.
- Fit each model on the training rows only.
- Evaluate probabilities and labels on the matching test rows.
- Aggregate each metric as the population mean and population standard deviation across repeats.

The committed `configs/default.json` is the source of truth. Hyperparameters were fixed before the seed-42 reference benchmark was executed.

## Models

### Quantum convolutional neural network

Eight angle-encoded qubits, three shared convolution/pooling stages, 12 circuit parameters, and a two-parameter sigmoid head. It is trained for 80 full-batch Adam epochs using exact occurrence-wise parameter-shift gradients and L2 penalty 0.0005.

### Logistic regression

StandardScaler followed by L2 logistic regression. Its recorded complexity is the coefficient and intercept count.

### RBF SVM

StandardScaler followed by an RBF-kernel SVC with `C=3` and scaled gamma. `CalibratedClassifierCV` with three folds and `ensemble=False` provides held-out probabilities without using the deprecated `SVC(probability=True)` path. Its recorded complexity is the fitted base SVC support-vector count.

### Neural network

StandardScaler followed by an eight-unit tanh MLP, L2 coefficient 0.001, and deterministic L-BFGS optimization. Its recorded complexity is the fitted weight and bias count.

## Primary and supporting metrics

Accuracy is the primary descriptive metric because the generated classes are balanced. Balanced accuracy, precision, recall, F1, ROC AUC, and log loss provide supporting views. Fit time is observed wall-clock time and should be interpreted only within the recorded environment.

`split_wins_or_ties.csv` counts a model when its accuracy equals the best accuracy on a repeat, including ties. It is descriptive and is not a significance test.

## Subgroup check

Accuracy is separately reported for generated rows with and without the square occlusion intervention. These groups can be small in an individual holdout, so the result is a diagnostic rather than a fairness assessment.

## QCNN sensitivity probes

Two post-training interventions are evaluated on each held-out set:

- `parameter_noise` adds independent Gaussian noise with standard deviation 0.04 to fitted circuit angles over ten trials.
- `finite_shots` draws ten binomial estimates at 256 shots from the exact final-qubit measurement probability.

The original trained model is not mutated. These are controlled probes, not device runs and not a calibrated physical-noise simulation.

## Leakage controls

- The generator never reads model outputs.
- No test labels participate in model fitting.
- Scaling is inside each scikit-learn pipeline and fits on training rows only.
- The feature transform is fixed, label-independent, and stateless.
- Split assignments are persisted and tests assert disjoint train/test row IDs.
- Seed-42 reference results were generated after the configuration was frozen.

## Interpretation limits

Three repeated holdouts do not justify inferential claims. The data is synthetic, the input is aggressively compressed, hyperparameter budgets differ, and QCNN execution occurs on a classical exact simulator. Results describe this repository’s protocol only.

**Created by School of AI and School of QC**
