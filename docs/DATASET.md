# Dataset card

**Created by School of AI and School of QC**

## Summary

The project generates a balanced binary dataset of 8×8 grayscale images. Label `0` is a vertical line and label `1` is a horizontal line. The default reference run contains 240 samples.

This synthetic dataset is included to make the full experiment offline, deterministic, quick to inspect, and legally uncomplicated. It is not intended to represent natural images or any human population.

## Generation process

For each row, the generator samples:

- an orientation from the balanced, shuffled label sequence;
- a line position from 0 through 7;
- a thickness of one or two pixels;
- line intensity from a fixed uniform range;
- a faint 2×2 nuisance patch;
- Gaussian background and image noise;
- an optional 2×2 attenuation block according to the configured occlusion probability.

Values are clipped to `[0, 1]`. Every random choice derives from a NumPy random generator initialized with the configured dataset seed.

## Features

Each image is reshaped into non-overlapping 2×4 regions arranged as a 4×2 spatial grid. The eight patch means, in row-major order, become the inputs for all four models.

This representation keeps the model comparison matched, but it can make the task harder when line position causes the signal to straddle patch boundaries. It also discards fine detail by design.

## Saved fields

`generated_images.csv` includes row ID, label, human-readable class, position, thickness, occlusion flag, and 64 pixels. `generated_images.npz` stores the same arrays compactly. `patch_features.csv` stores row ID and the eight inputs. `split_assignments.csv` identifies the train/test membership for every repeat.

## Intended use

- teaching variational quantum classification;
- testing QCNN training and persistence code;
- demonstrating matched-input benchmarking and subgroup reporting;
- CI and reproducibility exercises.

## Out-of-scope use

Do not treat performance on this dataset as evidence for production computer vision, medical imaging, surveillance, biometric classification, quantum advantage, or a particular quantum device. The labels and nuisance factors are constructed, not observed from the world.

## Known limitations

- The two classes are simple and balanced.
- The source distribution is fully specified by one generator.
- Occlusion is a simplified square attenuation, not realistic missingness.
- Only one coarse representation is evaluated.
- Samples are independent and contain no real demographic or contextual variation.

**Created by School of AI and School of QC**
