---
quadrant: how-to
---

# Choosing a solver

The library has several propagation models and they are not interchangeable. They
differ in how many fields they move, how dispersion is supplied, which physics is
switched on, and how the step is taken.

Answer the five questions below in order. The first one that applies to you names
the engine.

## Five questions

**Do you need to propagate a field along z at all?** If you only want modes,
dispersion curves, overlap integrals, phase matching, a multilayer stack or pulse
retrieval, you do not need a propagator. Use `fiber` for dispersion,
[`vector_modes`](api/vector-modes.md) for analytic circular-guide eigenmodes,
[`mode_analysis`](api/mode-analysis.md) to import an external solver's output,
[`phase_matching`](api/phase-matching.md) for dispersive waves and MI gains,
[`dbr.TMM`](api/dbr.md) for stacks, or `FROGTrace` for retrieval.

**How many fields move together?** One spatial mode in one polarization is the
common case and [`GNLSESolver`](api/gnlse.md) is built for it. Two polarizations
(PM fiber, birefringent walk-off, polarization FWM) needs
[`VectorSplitStepEngine`](api/vector-gnlse.md); add
`RandomBirefringenceEngine` when the birefringence is random along z and you want
an ensemble-averaged Manakov limit. A few spatial modes with intermodal XPM and
FWM needs [`MultimodeSplitStepEngine`](api/multimode-gnlse.md).

**Does the fiber itself change along z?** A taper, or any guide whose dispersion
and nonlinearity drift with position, needs
[`TaperedGNLSESolver`](api/gnlse.md) with a `dispersion_profile` callable.

**Is your spectrum wider than about one octave?** A truncated Taylor series stops
describing the physics there, and you should pass `full_beta=` instead of
`betas=`. See [how to choose the dispersion input](#how-to-choose-the-dispersion-input)
below.

**Are you going around a cavity many times?** Use
`MultimodeSplitStepEngine(cavity_...)` plus `cavity_iterate`. N=1 is a valid
single-mode cavity.

For χ⁽²⁾ processes (SHG, SFG, DFG, QPM, cascaded Kerr) use
[`chi2.solve_*`](api/chi2.md), which is a separate z-only CW model, not a
temporal envelope propagator.

## The engines side by side

| Engine | Fields propagated | Dispersion input | Stepping |
|---|---|---|---|
| `SplitStepEngine` | 1 scalar envelope | Taylor `betas` or static `full_beta` | fixed or adaptive; 4 schemes |
| `GNLSESolver` | 1 scalar envelope | same | high-level wrapper; adaptive, preflight and `verify_convergence` |
| `TaperedGNLSESolver` | 1 scalar envelope | `β(ω, z)` profile/callable | fixed or adaptive |
| `VectorSplitStepEngine` | 2 polarization envelopes | shared or per-axis Taylor `betas` | fixed |
| `RandomBirefringenceEngine` | 2 polarization envelopes | shared Taylor `betas` | fixed; random SU(2) rotations |
| `MultimodeSplitStepEngine` | N modal envelopes | per-mode Taylor or per-mode `full_beta` | fixed |

`GNLSESolver` and `SplitStepEngine` share one core. Reach for the lower-level
engine only when you need direct control of `evolution` and `z_array`.

## What each engine currently implements

| Physics | Scalar | Vector | Multimode | Tapered |
|---|---|---|---|---|
| Kerr SPM/XPM | ✅ | ✅ (2/3 or 8/9) | ✅ (models or overlap tensors) | ✅ |
| Raman | ✅ `include_raman` | ✅ per channel | ✅ per channel plus intermodal tensors | ✅ |
| Self-steepening | ✅ `include_self_steepening` | ⏳ | ✅ `include_shock` | ✅ |
| TPA / free carriers | ✅ `include_tpa` / `include_free_carriers` | ⏳ | ✅ same flags | ✅ |
| Full β(ω) | ✅ static `full_beta` | ⏳ | ✅ per mode | ✅ z-dependent only |
| FWM between propagated fields | ⏳ | ✅ `coupling="coherent"` (polarization) | ✅ `include_fwm` plus pump depletion | ⏳ |
| OAM selection gate / per-ℓ budget | ⏳ | ⏳ | ✅ `oam_l`, `power_by_l` | ⏳ |
| Cavity round trips | ⏳ | ⏳ | ✅ `cavity_iterate` | ⏳ |
| Spontaneous Raman noise | ✅ `raman_noise=True` | ⏳ | ⏳ | ✅ |
| Adaptive z-stepping | ✅ | ⏳ | ⏳ | ✅ |
| Convergence self-check | ✅ `verify_convergence` | ⏳ | ⏳ | ⏳ |

⏳ means the feature is not implemented in that engine. It does not mean the
physics is negligible. If you need it, move up a question or add it to the model
explicitly.

Defaults differ. `GNLSESolver` turns Raman on; `SplitStepEngine`, vector and
multimode turn it off.

## How to choose the dispersion input

Taylor `betas` (`[β₂, β₃, ...]` in `ps^k/m`) are the default. They are fine while
the pulse spectrum stays well inside the fitted band: below roughly one octave,
and away from material resonances. Build them with `Dispersion.get_betas()` from
a `WaveguideMode` or a measured curve.

`full_beta` is a static `β(ω)` array on the grid, or a callable `β(ω_abs)`. Use it
for octave-spanning supercontinuum and for any spectrum where a truncated Taylor
series visibly changes the answer. On the scalar engine it is mutually exclusive
with `dispersion_profile`. On the multimode engine it is one curve per mode and
carries the group delay and phase itself.

`dispersion_profile` is the z-dependent path only: a `β(ω, z)` callable or a
`ZDependentDispersion`. A static `full_beta` belongs on `GNLSESolver`, not on the
taper solver.

Do not hand-build `betas` or tensors for a few-mode deck. Use
[`build_multimode_deck`](api/mode-analysis.md), which returns per-mode dispersion,
overlap tensors, `oam_l`, and the `gamma_reference` scale.

## Runnable starting points

This page decides which engine to use. The complete, runnable code for each one
lives in the [example gallery](gallery.md), where every script is a full program
with its figure. Start from the row that matches your answer above.

| Engine | Runnable example |
|---|---|
| `GNLSESolver`, Taylor dispersion | [Basic propagation](examples/12_gnlse_basic_propagation.md) |
| `GNLSESolver`, `full_beta` | [NLSE_dudley](examples/21_gnlse_nlse_dudley.md) |
| `GNLSESolver`, units and sign conventions | [Beta-coefficient units](examples/23_gnlse_beta_units.md) |
| `TaperedGNLSESolver` | [Tapered waveguide](examples/19_gnlse_tapered_waveguide.md) |
| `VectorSplitStepEngine` | [Vector GNLSE](examples/37_vector_polarization_gnlse.md) |
| `MultimodeSplitStepEngine` | [Few-mode, one parameter at a time](examples/38_multimode_fewmode_gnlse.md) |
| `MultimodeSplitStepEngine`, cavity | No example yet. See the `cavity_iterate` API page. |
| `chi2.solve_*` | [QPM design and inverse fit](examples/42_inverse_design_qpm_fwm.md) |
| `dbr.TMM` | [TMM media and field profile](examples/24_tmm_media_and_field_profile.md) |

## What these solvers are not

This page tells you which model to pick. It does not tell you whether the library
covers your problem at all. For that, read
[what this library does not do](scope.md) first.

## Before trusting a number

1. Reduce to the adjacent model and check. Multimode at N=1 must equal the scalar
   engine; vector with one axis empty must equal scalar. Both are
   regression-tested.
2. Run `GNLSESolver.verify_convergence()` (scalar) or a step-size ladder (vector
   and multimode, which are fixed-step).
3. Check the physics flags you did not enable, and say so when reporting the
   result. A missing Raman term looks like a bandwidth prediction, not an error.
4. For symmetry-sensitive quantities (OAM splitting, polarization splitting), use
   the exact reference path: `vector_modes` or a symmetry-adapted FEM mesh, not a
   Cartesian mode solve.
