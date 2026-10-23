# Choosing a solver

The library has several propagation models. They are not interchangeable.
They differ in how many fields they propagate, how dispersion is supplied,
which physics is enabled, and how the step is taken. Pick the smallest model
that answers your question, then check it against a reduced case (single
channel, one polarization, no nonlinearity) before trusting a production
number.

## Fast path

| Your problem | Use |
|---|---|
| One spatial mode, one polarization, isolated pulse, β₂ to βₙ known | [`GNLSESolver`](api/gnlse.md) |
| One mode, octave-spanning spectrum (Taylor β fails) | `GNLSESolver(..., full_beta=...)` |
| Two polarizations: PM fibre, birefringent walk-off, polarization FWM | [`VectorSplitStepEngine`](api/vector-gnlse.md) |
| Random-birefringence fibre; ensemble-averaged Manakov limit | [`RandomBirefringenceEngine`](api/vector-gnlse.md) |
| Few spatial modes: intermodal XPM/FWM, OAM gating, mode-resolved γ | [`MultimodeSplitStepEngine`](api/multimode-gnlse.md) |
| z-dependent dispersion/nonlinearity: taper, dispersion-managed guide | [`TaperedGNLSESolver`](api/gnlse.md) |
| Cavity / ring / amplifier round trips | `MultimodeSplitStepEngine(cavity_...)` + `cavity_iterate` (N=1 is a valid single-mode cavity) |
| χ⁽²⁾: SHG, SFG, DFG, QPM, cascaded Kerr | [`chi2.solve_*`](api/chi2.md) |
| Linear multilayer stack / DBR | [`dbr.TMM`](api/dbr.md) |
| No propagation: modes, dispersion, overlaps, phase matching | `fiber`, [`mode_analysis`](api/mode-analysis.md), [`vector_modes`](api/vector-modes.md), [`phase_matching`](api/phase-matching.md) |
| Pulse retrieval or parameter fitting | `FROGTrace`, [`inverse_design`](cascaded-chi23.md) |

The DBR, analysis and fitting rows are not time-domain nonlinear propagation.

## Propagation engines side by side

| Engine | Fields propagated | Dispersion input | Stepping |
|---|---|---|---|
| `SplitStepEngine` | 1 scalar envelope | Taylor `betas` or static `full_beta` | fixed or adaptive; 4 schemes |
| `GNLSESolver` | 1 scalar envelope | same | high-level wrapper; adaptive, preflight and `verify_convergence` |
| `TaperedGNLSESolver` | 1 scalar envelope | `β(ω, z)` profile/callable | fixed or adaptive |
| `VectorSplitStepEngine` | 2 polarization envelopes | shared or per-axis Taylor `betas` | fixed |
| `RandomBirefringenceEngine` | 2 polarization envelopes | shared Taylor `betas` | fixed; random SU(2) rotations |
| `MultimodeSplitStepEngine` | N modal envelopes | per-mode Taylor or per-mode `full_beta` | fixed |

`GNLSESolver` and `SplitStepEngine` share one core. Use `GNLSESolver` unless
you need the lower-level engine's direct `evolution`/`z_array` control.

## Physics availability

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

⏳ means the feature is not implemented in that engine, not that it is
physically negligible. If you need it, move up a row or add it to the model
explicitly. `GNLSESolver` defaults Raman on. The lower-level
`SplitStepEngine`, vector and multimode engines default it off.

## How to choose the dispersion input

Taylor `betas` (`[β₂, β₃, ...]`, `ps^k/m`) are the default. They are fine
while the pulse spectrum stays well inside the fitted band: below roughly one
octave, and away from material resonances. Build them with
`Dispersion.get_betas()` from a `WaveguideMode` or a measured curve.

`full_beta` is a static `β(ω)` array on the grid or a callable `β(ω_abs)`. Use
it for octave-spanning supercontinuum and for any spectrum where a truncated
Taylor series visibly changes the answer. On the scalar engine it is mutually
exclusive with `dispersion_profile`. On the multimode engine it is one curve
per mode and carries the group delay and phase itself.

`dispersion_profile` is the z-dependent path only: a `β(ω, z)` callable or a
`ZDependentDispersion`. A static `full_beta` belongs on `GNLSESolver`, not on
the taper solver.

Few-mode decks should not hand-build `betas` or tensors. Use
[`build_multimode_deck`](api/mode-analysis.md), which returns per-mode
dispersion, overlap tensors, `oam_l`, and the `gamma_reference` scale.

## Minimal examples

Scalar, Taylor dispersion:

```python
from photonics_helper import (
    Area, Envelope, Fiber, GNLSESolver, Length, TemporalGrid, Time, Wave, Wavelength,
)

grid = TemporalGrid(N=2**12, Tmax=Time(10e-12, "s"))
pulse = Wave(
    grid=grid,
    envelope=Envelope(shape="sech", peak_amplitude=100.0, pulse_width=Time(50e-15, "s")),
    central_wavelength=Wavelength(1550, "nm"),
)
solver = GNLSESolver(
    pulse=pulse,
    fiber=Fiber(n2=2.6e-20, alpha=0.0, A_eff=Area(80e-12, "m^2"), length=Length(0.1, "m")),
    betas=[-2e-3, 1e-4],          # ps^k/m
    include_raman=True,
)
solver.propagate(num_steps=200, nsaves=100)
```

Two polarizations:

```python
from photonics_helper.vector_gnlse import VectorSplitStepEngine

engine = VectorSplitStepEngine(
    pulse_x, pulse_y, fiber,
    betas=[-2e-3, 1e-4],
    coupling="incoherent",        # or "coherent" / "manakov"
    walkoff=2e-12,                # s/m, β₁_y - β₁_x
)
engine.propagate(num_steps=200, nsaves=100)
```

Few-mode deck from a solver export:

```python
from photonics_helper.mode_analysis import build_multimode_deck
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine

deck = build_multimode_deck("exports/rib", lambda0=2.0, dispersion_order=4, n2=1e-18)
engine = MultimodeSplitStepEngine(
    waves, fiber,
    betas=deck["betas"],
    group_delays=deck["group_delays"],
    xpm_weights=deck["xpm_weights"],
    fwm_weights=deck["fwm_weights"],
    oam_l=deck["oam_l"],
    coupling_tensor=deck["coupling_tensor"],
    gamma_reference=deck["gamma_reference"],
    include_fwm=True,
)
```

Taper:

```python
from photonics_helper import TaperedGNLSESolver

solver = TaperedGNLSESolver(
    pulse=pulse, fiber=fiber,
    dispersion_profile=beta_of_omega_and_z,   # callable(omega, z) -> β (rad/m)
    a_eff_fn=lambda z: 80e-12 * (1 + 2 * z / 0.01),
)
solver.propagate(num_steps=500, nsaves=200)
```

## What these solvers are not

These engines are not beam-propagation or diffraction solvers. They are
one-way (z) envelope models: the transverse structure is frozen into mode
profiles, overlaps and γ. Transverse diffraction, self-focusing and multi-mode
imaging need a different code.

Propagation is one-way. Backscattering is not modelled except the opt-in
multimode Rayleigh term.

The library is not a general mode solver. `vector_modes` gives exact analytic
modes for a circular step-index guide. `mode_analysis` consumes external
femwell/tidy3d exports. Neither replaces a full-vector FEM solve for an
arbitrary cross-section.

The χ⁽²⁾ functions are z-only CW models, not temporal envelope propagators.
For χ⁽²⁾ plus χ⁽³⁾ pulse dynamics you need the cascaded model or an external
solver.

## Before trusting a number

1. Reduce to the adjacent model and check. Multimode at N=1 must equal the
   scalar engine; vector with one axis empty must equal scalar. Both are
   regression-tested.
2. Run `GNLSESolver.verify_convergence()` (scalar) or a step-size ladder
   (vector and multimode, which are fixed-step).
3. Check the physics flags you did not enable, and say so when reporting the
   result. A missing Raman term looks like a bandwidth prediction, not an
   error.
4. For symmetry-sensitive quantities (OAM splitting, polarization splitting),
   use the exact reference path: `vector_modes` or a symmetry-adapted FEM mesh,
   not a Cartesian mode solve.
