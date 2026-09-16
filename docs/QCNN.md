# QCNN design and mathematics

**Created by School of AI and School of QC**

## Feature map

Each 8×8 image is divided into a 4×2 grid of non-overlapping 2×4 patches. The mean of each patch gives a feature vector (x \in [0,1]^8). Feature (x_j) is encoded on qubit (j) as

\[
|\phi_j(x_j)\rangle = R_Z(\pi x_j^2) R_Y(\pi x_j)|0\rangle.
\]

The full encoded state is the tensor product of the eight single-qubit states. No fitted preprocessing is applied to QCNN inputs.

## Shared convolution block

For source qubit (s), sink qubit (t), and stage parameters ((\theta_0,\theta_1,\theta_2)), the project applies:

1. (R_Y(\theta_0)) to (s);
2. (R_Y(\theta_1)) to (t);
3. `CX(s, t)`;
4. (R_Z(\theta_2)) to (t);
5. `CX(t, s)`.

Every pair in one stage reuses the same three values. This is the project’s quantum analogue of a shared convolutional filter.

## Pooling block

Each stage then applies `CX(s, t)` followed by (R_Y(\theta_3)) on the sink. Later stages act only on sink qubits, producing the logical active-width sequence 8 → 4 → 2 → 1.

## Readout

The circuit returns the exact expectation

\[
z(x,\theta)=\langle Z_7\rangle.
\]

A two-parameter classical head converts the expectation to the horizontal-class probability:

\[
p(y=1\mid x)=\sigma(a\,z(x,\theta)+b).
\]

The model therefore has 12 shared quantum parameters and two classical-head parameters.

## Objective and gradients

Training minimizes full-batch binary cross-entropy plus L2 regularization on the circuit weights and head scale. Adam updates all parameters.

A shared value occurs in multiple gates. Its derivative must therefore sum the contribution of every occurrence. For an occurrence (k), the parameter-shift contribution is

\[
\frac{\partial z}{\partial \theta_k}
=\frac{1}{2}\left[z(\theta_k+\pi/2)-z(\theta_k-\pi/2)\right].
\]

The implementation shifts one occurrence at a time and sums those derivatives before applying the chain rule. Treating all shared occurrences as one simultaneous shifted circuit would compute a different quantity.

There are 28 parameterized gate occurrences in the three-stage topology. One epoch needs one unshifted forward statevector pass plus 56 shifted passes, or 57 passes total. The default 80-epoch fit therefore uses 4,560 exact statevector passes.

## Simulator parity

The optimized simulator stores a batch of (2^8=256) amplitudes and applies gates in place using NumPy reshaping and indexed swaps. Tests independently construct the symbolic Qiskit circuit, bind random features and weights, and compare the resulting statevector with the NumPy implementation up to numerical precision. Additional tests compare the shared parameter-shift gradient with central finite differences.

Qiskit is therefore used as an independent representation and validation oracle, while the benchmark’s training loop uses the faster specialized NumPy path.

## What this circuit is—and is not

It is a compact variational circuit inspired by hierarchical QCNN convolution and pooling. It is not a claim that the gates reproduce a classical convolution exactly, and it is not a reproduction of the architecture or learning task in the original QCNN paper.

The final readout is evaluated exactly in the primary benchmark. The 256-shot result is a post-training statistical sensitivity probe derived from the exact measurement probability. The parameter-noise probe perturbs fitted angles with independent Gaussian noise. Neither probe models a specific device.

**Created by School of AI and School of QC**
