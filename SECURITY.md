# Security policy

**Created by School of AI and School of QC**

## Supported versions

Security fixes target the latest release on the `main` branch.

## Reporting a vulnerability

Please use GitHub’s private vulnerability-reporting feature for the repository. Do not open a public issue for an unpatched vulnerability. Include affected versions, reproduction steps, impact, and any suggested mitigation. Maintainers will acknowledge a complete report as soon as practical and coordinate disclosure after a fix is available.

## Model artifact warning

The QCNN model is plain JSON. Classical models are saved with joblib, which uses pickle-compatible deserialization and can execute malicious code. Load `classical_models.joblib` only from benchmark runs you created or sources you fully trust.

## Data and secrets

The project needs no API key and generates its dataset locally. Never place credentials in configs, notebooks, artifacts, or screenshots. `.env` files are ignored; `.env.example` documents that no secret is required.

## Dependency and input safety

Install from the declared dependency ranges and review automated dependency updates. Inference accepts only 8×8 `.npy` or `.csv` arrays with finite values in `[0, 1]`. NumPy loading disables pickle.

**Created by School of AI and School of QC**
