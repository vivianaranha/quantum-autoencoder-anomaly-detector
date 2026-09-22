# Quantum Autoencoder

Created by School of AI and School of QC.

## State preparation

For a standardized eight-feature vector $x$, amplitude normalization creates

$$|x\rangle = \frac{1}{\lVert x\rVert_2}\sum_{j=0}^{7} x_j |j\rangle.$$

Because $2^3=8$, the state occupies three qubits without padding. This simulator treats state
preparation as given; a hardware study would have to measure and report its cost.

## Encoder and latent state

The parameterized unitary $U(\theta)$ attempts to place normal-state information in latent
qubit 0 while resetting trash qubits 1 and 2. Qubit 0 is Qiskit's least-significant qubit. The
all-zero trash projector is

$$P_0 = I_{\text{latent}} \otimes |00\rangle\langle 00|_{\text{trash}}.$$

For the encoded state $|z\rangle=U(\theta)|x\rangle$, the successful-compression probability is

$$p_0(x)=\langle z|P_0|z\rangle
=\sum_{j=0}^{1}|\langle j|U(\theta)|x\rangle|^2.$$

Conditional on that projection, the one-qubit latent state is

$$|\ell(x)\rangle =
\frac{(I\otimes\langle 00|)U(\theta)|x\rangle}{\sqrt{p_0(x)}}.$$

The public model API exposes this normalized latent state and $p_0$. A zero-probability
projection receives a deterministic zero-state fallback so downstream code remains defined.

## Decoder and fidelity

The decoder attaches a clean trash register and applies the inverse encoder:

$$|\hat{x}\rangle=U^\dagger(\theta)(|\ell(x)\rangle\otimes|00\rangle).$$

For pure states, the round-trip fidelity satisfies

$$F(|x\rangle,|\hat{x}\rangle)=|\langle x|\hat{x}\rangle|^2=p_0(x).$$

Therefore the raw anomaly score is both trash probability and reconstruction infidelity:

$$s(x)=1-p_0(x)=1-F(|x\rangle,|\hat{x}\rangle).$$

The implementation tests this identity numerically and independently checks trash-register
probabilities with Qiskit's Statevector API across multiple circuit depths and latent widths.

## Training

The encoder alternates trainable RY and RZ rotations with a circular CX pattern. The default
two repetitions produce 18 angles and six CX gates. COBYLA minimizes mean $s(x)$ over a seeded,
deterministic subset of clean-normal training states. Mixed validation labels select only the
score normalization and decision threshold; they do not train the circuit.

## Simulator boundary

The project obtains exact probabilities from a local unitary/statevector representation. It
does not sample shots, estimate uncertainty, model noise, or submit circuits to hardware.
A hardware experiment would additionally need a state-preparation circuit, trash-register
measurement protocol, finite-shot intervals, transpilation, backend calibration records, and
noise-aware comparisons.

This project demonstrates a valid small-scale QAE computation. It does not demonstrate quantum
advantage, practical compression of classically stored vectors, or production hardware
performance.

## References

- Romero, Olson, and Aspuru-Guzik,
  [Quantum autoencoders for efficient compression of quantum data](https://arxiv.org/abs/1612.02806).
- Qiskit,
  [Statevector API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector).
