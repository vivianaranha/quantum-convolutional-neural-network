# QCNN Benchmark Report

Created by School of AI and School of QC.

## Protocol

- Images: `240` synthetic 8×8 line-orientation samples
- Occlusion probability: `0.25`
- Repeated stratified holdouts: `3`
- Test fraction: `0.30`
- QCNN epochs per split: `80`
- Qubits: `8`
- QCNN trainable parameters: `14`

## Overall model results

- Quantum Convolutional Neural Network: accuracy `0.8704 ± 0.0625`, F1 `0.8586`, ROC AUC `0.9334`, fit time `38.7397s`.
- Logistic Regression: accuracy `0.4630 ± 0.0285`, F1 `0.4524`, ROC AUC `0.4730`, fit time `0.0032s`.
- RBF SVM: accuracy `0.9954 ± 0.0065`, F1 `0.9953`, ROC AUC `1.0000`, fit time `0.0068s`.
- Neural Network: accuracy `0.9954 ± 0.0065`, F1 `0.9953`, ROC AUC `1.0000`, fit time `0.0052s`.

## Split wins or ties

- RBF SVM: `2`
- Neural Network: `2`

## QCNN sensitivity probes

- parameter_noise: clean accuracy `0.8704`, probe accuracy `0.8639`.
- finite_shots: clean accuracy `0.8704`, probe accuracy `0.8583`.

## Interpretation boundary

The QCNN uses exact statevector simulation. Finite-shot and parameter-noise results are controlled sensitivity probes, not physical-device experiments. The data is synthetic, the spatial compression is fixed, and the benchmark does not demonstrate quantum advantage.

Created by School of AI and School of QC.
