# The formal specification

`photonics_helper` has a companion Lean 4 development in
[`formal/`](https://github.com/hitaishi/photonics_helper/tree/main/formal) that
states — and proves — the mathematics the numerical kernels implement.

The relationship is one-directional, and worth being precise about:

```
        Lean 4 + Mathlib          the specification
              │
              ▼
     photonics_helper (Python)   the implementation, which follows the spec
              │
              ▼
        findings → literature     where they disagree, the literature decides
```

Where a Python kernel and a Lean theorem disagree, **neither is presumed
correct**. That disagreement is a finding, and it is resolved against the
primary source — Agrawal, MacLeod, Poletti & Horák, or the paper under
`reproductions/`. If the literature agrees with Lean, the Python is wrong and
gets fixed. If it agrees with Python, the *theorem* was misstated and gets
fixed. Every such resolution is recorded in `formal/CHANGELOG.md`.

## What is machine-checked

The conventions below are not documentation that happens to be true — they are
theorems, and a theorem that failed to compile would fail the build.

| Area | Theorem | Governs |
|---|---|---|
| Spectral-axis orientation | `Conventions.beta_flip` | Odd-order dispersion is antisymmetric under axis reversal. A sign error in the axis *is* a sign error in β₁, β₃, … |
| Fourier pairing | `Grid.dft_pair` | `fft`/`ifft` are a matched pair, so the `dt` and `1/dt` scalings are not a convention to remember |
| Conjugate symmetry | `Grid.conj_dft` | The conjugated-transform fast path `conj(DFT(conj(·)))` is an identity, not an approximation |
| Hermitian symmetry | `Grid.hermitian` | A real input has a conjugate-paired spectrum — this is why the Raman `H(Ω)` is real |
| Raman gain bound | `Nonlinear.ramanDrive_le_max` | A normalised, non-negative response cannot amplify past unity, whatever delay table is set |
| Multimode coefficients | `Nonlinear.xpmFactor_coPolarized`, `fwmFactor_coPolarized` | Co-polarized spatial-mode SPM/XPM/FWM limits (SPM 1, XPM 2, FWM 1) — not the two-polarization 2/3 strip-down |
| Multimode FWM gate | `Nonlinear.fwmAllowed_partner` | One angular-momentum check serves both arms of an exchange pair |
| Multimode energy | `Nonlinear.depletedFwm_energy_conserved` | The depleted pump set conserves `Σ|A|²`; the pair-only approximation does not |
| Shock once | `Nonlinear.shockSymbol_double_ne_single` | Applying `(1+Ωτ)` twice is `(1+Ωτ)²`, the regression the step10 fix pins |
| Raman causality | `Nonlinear.ramanCausal`, `ramanDrive_early` | The retarded drive draws only on intensities at `t ≤ t′` |
| Energy conservation | `Optics.DBR.energyConservation` | `R + T = 1` for a lossless stack |

The full register — one row per governed quantity, with the primary source for
each — is [`formal/CONFORMANCE.md`](https://github.com/hitaishi/photonics_helper/blob/main/formal/CONFORMANCE.md).

## Using the theorems

Theorem names are cited in the docstrings of the Python symbols they govern.
The specification is therefore traceable from the code without a build-time
dependency on Lean — you can read the kernel and see which law it is asserting.

Current status: **260 theorems, no outstanding proofs**, `lake build` green.

## What this does *not* verify

Stated plainly, because a formal development that overclaims is worse than no
formal development.

- **Floating-point accuracy.** Rounding, FFT-backend parity, and error estimates
  are not verified. Lean checks the algebra, not the `float64`.
- **The Python control flow.** A theorem about an operator does not check that
  the shipped code composes its sub-operators in that order.
- **Experimental parameters** of the reproduction decks — only the closed-form
  laws they are compared against.
- **The material database.** Provenance and licensing are data-governance
  concerns.
- **Measure-theory statements**, notably the modal-overlap integrals in
  `structured.py`.
- **Conformance itself.** The register is a reviewed artefact, not a compiler
  guarantee. Nothing mechanically ties a Python symbol to its row.

## Building it yourself

```bash
cd formal
LAKE_JOBS=2 LEAN_NUM_THREADS=2 lake exe cache get   # ~5 GB, once
LAKE_JOBS=2 LEAN_NUM_THREADS=2 lake build
```

The toolchain (`v4.34.1`) and the Mathlib revision are both pinned and
committed, so the build is reproducible. Do not run `lake update` casually — it
rewrites `lean-toolchain` to whatever it resolves and invalidates the cache.

See [`formal/README.md`](https://github.com/hitaishi/photonics_helper/blob/main/formal/README.md)
for the status table and [`formal/PLAN.md`](https://github.com/hitaishi/photonics_helper/blob/main/formal/PLAN.md)
for the roadmap.