# Adaptive Fractional Fourier Neural Operator (fr-FNO)

Official implementation of the Adaptive Fractional Fourier Neural Operator (fr-FNO) for learning solution operators of fractional partial differential equations.

## Overview

This repository introduces a novel fractional-inspired spectral filtering mechanism integrated into the Fourier Neural Operator (FNO) framework.

Unlike classical FNOs that employ fixed spectral parameterizations, fr-FNO incorporates learnable channel-wise fractional parameters that adaptively control spectral attenuation during training.

The proposed architecture aims to improve the modeling of nonlocal transport phenomena commonly observed in fractional PDEs.

---

## Key Features

- Adaptive channel-wise fractional spectral filtering
- Compatible with standard FNO architectures
- Resolution-invariant operator learning
- Computational complexity remains O(N log N)
- Improved convergence on fractional PDE benchmarks
- Zero-shot super-resolution evaluation

---

## Proposed Filter

The adaptive spectral filter is defined as

\[
\Psi(\alpha,k)
=
\frac{1}
{\exp(1-\alpha)+\alpha \|k\|^2}
\]

where

- \(k\) denotes the Fourier frequency,
- \(\alpha\) is a learnable channel-wise parameter.

The filter suppresses high-frequency noise while preserving dominant low-frequency structures.

---

## Benchmarks

The framework is evaluated on:

1. Space-Fractional Burgers Equation
2. Fractional Allen–Cahn Equation
3. Fractional FitzHugh–Nagumo System

---

## Model Variants

### Vanilla FNO

Standard Fourier Neural Operator.

### Fixed fr-FNO

Fractional-inspired spectral filter with fixed \(\alpha\).

### Adaptive fr-FNO

Learnable channel-wise \(\alpha\) optimized jointly with network weights.
## License

MIT Licens
