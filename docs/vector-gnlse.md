# Vector / polarization-coupled GNLSE

The scalar `SplitStepEngine` (see [GNLSE physics](gnlse-physics.md)) solves a
single slowly-varying envelope. This page documents the **vector** engine
(`photonics_helper.vector_gnlse`), which propagates the field as a
two-component vector `A = (A_x, A_y)` and therefore resolves the physics
that only exists between polarization channels.

- API reference: `photonics_helper.vector_gnlse`
  (`VectorSplitStepEngine`, `RandomBirefringenceEngine`, `MANAKOV_FACTOR`).

## What the scalar engine cannot model

| Regime | Scalar GNLSE | Vector engine |
|---|---|---|
| Single-mode, single-polarization fiber | exact model | scalar limit of `coupling="incoherent"` (bit-exact, tested) |
| PM / high-birefringence fiber | wrong (no axes) | per-axis β, XPM `2/3`, no coherent mixing |
| Randomly birefringent fiber (telecom) | γ overestimated by 12.5% | `coupling="manakov"` / `RandomBirefringenceEngine` |
| Deterministic birefringence + polarization FWM | no FWM | `coupling="coherent"` with `delta_beta` |
| Differential group delay (PMD walk-off) | none | `walkoff` (s/m) on the y-channel |

## Model

In the retarded frame of the x axis (Agrawal, *Nonlinear Fiber Optics*,
5th ed.):

```
∂A_x/∂z = L_x A_x + iγ( P_x + (2/3)P_y ) A_x + (i/3)γ A_y² A_x* e^{−2iΔβz}
∂A_y/∂z = L_y A_y + iγ( P_y + (2/3)P_x ) A_y + (i/3)γ A_x² A_y* e^{+2iΔβz}
          − Δβ₁ ∂A_y/∂T
```

with per-axis Taylor dispersion operators `L_j = −i Σ β_k⁽ʲ⁾/k! ∂ᵀᵏ`,
shared scalar loss `e^{−αz/2}`, the `2/3` XPM anisotropy of the degenerate
linearly-polarized mode pair, and the coherent polarization FWM pair
(`coupling="coherent"`, phase mismatch `Δβ = β_x − β_y` in rad/m).

`coupling` selects the nonlinear model:

- `"incoherent"` — FWM term dropped (averaged out over the beat length in
  real PM fiber). Reduces **exactly** to the scalar `SplitStepEngine` when
  `A_y ≡ 0`; all scalar reproductions remain the tool of record there.
- `"coherent"` — full Agrawal coupled GNLSE with the FWM mixing term.
  The FWM pair is energy-conserving (Hamiltonian substructure) and the
  diagonal SPM/XPM phase is advanced exactly while the mixing term is
  integrated with frequency-domain RK4 substeps (same RK4IP structure the
  scalar shock integrator uses). Validated against a dense-RK4 reference
  (`tests/test_vector_gnlse.py::TestCoherentFWM`).
- `"manakov"` — polarization-averaged model with the `8/9` coefficient
  (Wai & Menyuk 1996): the correct *effective* description of a fiber whose
  beat length is far shorter than the nonlinear length. Requires identical
  per-axis dispersion and zero walkoff. Validated: a Manakov CW acquires
  exactly `(8/9)γP₀L` phase, and the random-birefringence ensemble
  converges to the Manakov spectrum (<2% L2 over 6 seeds).

## Random birefringence

`RandomBirefringenceEngine(coupling='incoherent' locally, seed=...)` applies
an `SU(2)` frame rotation (uniform axis, uniform angle) before and after
every nonlinear step, mimicking a correlation length equal to the step.
Total energy is conserved at machine precision, CW shape is exactly
preserved, and the ensemble-averaged spectrum converges to the deterministic
Manakov (8/9) run.

## Scope / limits (v1)

- Raman is the *scalar* per-channel response `Pⱼ = (1−f_R)|A_j|² +
  f_R h_R ⊛ |A_j|²`; the full vector Raman response (Lin & Agrawal 2006)
  is a planned extension.
- Self-steepening, TPA and free carriers are scalar-engine features; the
  vector engine rejects them explicitly rather than silently ignoring them.
- Fixed-step propagation (`num_steps` or `step_size`), with the same
  snapshot / energy-drift-monitor semantics as the scalar engine.
- Multimode (fibers with >2 guided modes) coupling is future work; the
  polarization channels here are the ±linear-polarization pair.
