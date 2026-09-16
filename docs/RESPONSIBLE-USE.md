# Responsible use

**Created by School of AI and School of QC**

## Safe scope

Use this repository for learning, demonstrations, reproducibility exercises, and experiments where every participant understands that the data and noise interventions are synthetic.

## Do not infer more than the benchmark measures

A high held-out score here means that a model separated two generated line orientations after a fixed eight-feature compression. It does not establish:

- superiority on natural images;
- quantum advantage;
- efficiency on quantum hardware;
- robustness to realistic corruption or adversarial input;
- reliable probabilities outside the generator distribution;
- suitability for decisions affecting people.

## Required work before a new application

If adapting the code, define the intended user and harm model first. Then replace the dataset card and model card with task-specific versions; isolate an external test set; review sampling, labels, consent, privacy, and licensing; choose meaningful subgroups; establish human oversight; and validate security and failure handling.

## Honest quantum reporting

Always distinguish among exact statevector simulation, finite-shot simulation, simplified parameter perturbation, calibrated device-noise simulation, and execution on named hardware. This repository performs the first and exposes sensitivity probes corresponding to the next two categories. It does not perform a device experiment.

Do not compare the included fit-time numbers with accelerator or QPU runtime. QCNN fit time here is dominated by a Python/NumPy training implementation on a classical CPU.

## Artifact safety

CSV, JSON, NPZ, PNG, Markdown, and the QCNN JSON can be inspected as data. The classical-model joblib artifact uses Python pickle semantics and can execute code when loaded. Only run inference against artifacts you created or obtained from a trusted source.

## Reporting results

Publish the configuration, software versions, seeds, split strategy, aggregation rule, full metric set, and known limitations. Report negative or mixed results. Avoid selecting only favorable seeds or metrics.

**Created by School of AI and School of QC**
