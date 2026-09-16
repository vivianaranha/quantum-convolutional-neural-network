# GitHub upload guide

**Created by School of AI and School of QC**

The repository is ready to upload as-is. The commands below create a new Git history and push it to a repository you own.

## Create the remote

Create an empty GitHub repository named `quantum-convolutional-neural-network`. Do not ask GitHub to add a README, license, or `.gitignore`, because those files already exist here.

## Initialize and inspect

```bash
git init
git branch -M main
git status --short
```

The generated `artifacts/run-*` directories, virtual environments, build products, and caches should not appear because `.gitignore` excludes them. The curated `examples/reference-run/` files should appear.

## Commit

```bash
git add .
git diff --cached --stat
git commit -m "Add reproducible QCNN image-classification benchmark"
```

Inspect the staged diff before committing. In particular, verify that no credentials, personal data, large unneeded binaries, or untrusted joblib files are present.

## Connect and push

HTTPS:

```bash
git remote add origin https://github.com/YOUR-ACCOUNT/quantum-convolutional-neural-network.git
git push -u origin main
```

SSH:

```bash
git remote add origin git@github.com:YOUR-ACCOUNT/quantum-convolutional-neural-network.git
git push -u origin main
```

## Repository settings

After the first push:

- confirm the Actions workflow passes on Python 3.11 and 3.12;
- enable branch protection and require the CI check before merge;
- enable Dependabot and secret scanning if available;
- add a short repository description and the topics `quantum-computing`, `quantum-machine-learning`, `qcnn`, and `image-classification`;
- verify that the README images and internal links render correctly.

## Pre-release verification

```bash
make verify
quantum-cnn benchmark --samples 80 --repeats 1 --qcnn-epochs 5 --seed 7
```

The project does not require a secret or paid service. Do not commit a local `.env`, private data, or benchmark artifacts containing models from an untrusted source.

**Created by School of AI and School of QC**
