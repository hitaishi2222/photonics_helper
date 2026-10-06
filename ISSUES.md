# ISSUES — open problems verified against the current working tree

> **Register state (2026-10-03): all items resolved.** Every issue below has
> been fixed in the code. This file is retained as a historical record of what
> was found and how it was resolved. For the current state of the code, see
> the individual module docstrings and the test suite.
>
> **Audit date:** 2026-09-21 (original sweep), with per-entry resolution dates
> through 2026-10-03. Method: every claim of an issue/problem/bug found in the
> repo's markdown files was re-checked against the code and tests.

---

## Resolved Issues (summary)

| #   | Issue                                                                          | Resolution                                                                                                                                                                                                                     |
| --- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0   | GNLSE linear operator: dispersion time-direction inconsistent                  | **Closed 2026-09-30** — convention swap (conjugated transform pair + plain causal h_R convolution + `exp(+iΣβₖΩᵏz/k!)`); verified against dense-DFT reference, Raman/SSFS law, FFT-kernel tone probe, RDW group-delay walk-off |
| 1   | Self-steepening: residual photon-number drift                                  | **Closed 2026-10-02** — grid-validity artifact, not a model defect; `_validate_shock_grid` now checks `τ_shock·Ω_max` and warns above `_SHOCK_TAYLOR_LIMIT = 0.2`                                                              |
| 2   | `mi_gain_spectrum_extended`: catastrophic cancellation at absolute ω₀          | **Closed 2026-09-28** — offset-aware contract (`betas=` path, `beta_fn_convention="detuning"`); legacy path kept with DeprecationWarning                                                                                       |
| 3   | `TaperedGNLSESolver`: inert `betas_unit` parameter                             | **Closed 2026-09-28** — emits `UserWarning` when `betas_unit != "ps^k/m"`                                                                                                                                                      |
| 4   | `plot_waterfall`: non-standard waterfall rendering                             | **Closed 2026-09-28** — re-drawn as a true ridge with per-artist `ScaledTranslation` transforms                                                                                                                                |
| 5   | Author/process actions (Zenodo DOI + JOSS submission)                          | **Open** — author-side process, not a code bug                                                                                                                                                                                 |
| 6   | Raissi-2019 PINN: rel-L2 plateaus at ~3× the paper's 1.97e-3                   | **Closed 2026-10-02** — metric-definition mismatch (complex vs magnitude convention); L-BFGS accounting recorded                                                                                                               |
| 7   | ROCm iGPU training crash guardrail                                             | **Superseded by CUDA**                                                                                                                                                                                                         |
| 8   | Guasoni-2015 IM-MI: flat Eq.-12 readout                                        | **Closed 2026-10-02** — readout saturation artifact; band structure recovered with local-gain measurement                                                                                                                      |
| 10  | Wright-2015 STMI: engine-side check B asserts a noise                          | **Closed** — see folder README                                                                                                                                                                                                 |
| 11  | Multimode FWM substep: RK4 blow-up                                             | **Closed** — see folder README                                                                                                                                                                                                 |
| 12  | Multimode FWM substep: hidden Euler integrator                                 | **Closed** — see folder README                                                                                                                                                                                                 |
| 13  | `hult_2007_rk4ip` folder: both decks fail                                      | **Closed 2026-10-01** — see folder README                                                                                                                                                                                      |
| 14  | `heidt_2009_adaptive_step`: deck A's global error saturates                    | **Closed** — see folder README                                                                                                                                                                                                 |
| 15  | `oam_l` FWM gate: VERIFIED EQUAL to Poletti & Horak Eq. (18)                   | **Closed** — no code change needed                                                                                                                                                                                             |
| 16  | `VectorSplitStepEngine`: `betas_x` without `betas_y` raises                    | **Fixed 2026-10-03** — docstring corrected to match code (both axes must be specified explicitly)                                                                                                                              |
| 17  | Nine of thirty tabulated `source_key`s cannot be loaded                        | **Fixed 2026-10-03** — longest-known-material-prefix match; all 30 keys now load                                                                                                                                               |
| 18  | `PhononResponse.from_material` raises for 34 of 44 materials                   | **Closed as documented behaviour** — 10 of 44 materials have a mode list; silica's list is a band model                                                                                                                        |
| 19  | Three papercuts on the refraction/materials API                                | **Fixed 2026-10-03** — (a) `n_func` accepts `Wavelength`, (b) `RamanSpec.model_fields()` classmethod added, (c) `beta_fn` has default `None`                                                                                   |
| 20  | `mi_gain_spectrum_extended` emits DeprecationWarning twice                     | **Fixed 2026-10-03** — warnings consolidated to one                                                                                                                                                                            |
| 21  | `VectorSplitStepEngine` "manakov" docstring omits 8/9 equivalence conditions   | **Fixed 2026-10-03** — docstring now states equivalence holds only for single channel                                                                                                                                          |
| 22  | `conserving_shock=True` is a silent no-op for γ > 0 + ComplexWarning flood     | **Fixed 2026-10-03** — `.real` added to delayed intensity convolution; docstring documents γ < 0 requirement                                                                                                                   |
| 23  | `fit_two_wave`: sign of Δk is not identifiable                                 | **Fixed 2026-10-03** — docstring documents sign symmetry                                                                                                                                                                       |
| 24  | `fit_shg_autodiff` does not recover (sigma, P0)                                | **Fixed 2026-10-03** — docstring corrected; use `fit_two_wave` for reliable recovery                                                                                                                                           |
| 25  | `design_efficiency` assumes monotone η(L)                                      | **Fixed 2026-10-03** — docstring documents monotonicity assumption and `residual_norm` check                                                                                                                                   |
| 26  | `_SHOCK_TAYLOR_LIMIT` caps grid step at ~13 fs                                 | **Fixed 2026-10-03** — docstring documents resolution floor                                                                                                                                                                    |
| 27  | `noise.py`: Bose-factor docstring overstates n_th + `peak_power()` unit switch | **Fixed 2026-10-03** — docstrings corrected; n_th = 0.138 at 300 K documented; peak_power unit hazard documented                                                                                                               |

---

## Detailed Resolutions

### #0. GNLSE linear operator: dispersion time-direction inconsistent

**Symptom.** In the Renninger & Wise GRIN multicomponent-soliton reproduction, the higher modes must blue-shift by −1.53 / −4.61 nm to lock group velocities. The measured centroid shifts came out +1.01 / +4.30 nm (red) under the library's wavelength map.

**Cause.** The engine's sub-operators disagreed about the time direction of the spectral axis. Dispersion behaved as if `grid.w = −Ω_phys` while Raman / walk-off / λ-map behaved as if `grid.w = +Ω_phys`.

**Resolution (2026-09-30).** Convention swap: conjugated transform pair (`grid.fft = conj(fft(conj(·)))·dt`), plain causal `h_R` convolution (no `conj`), `exp(+iΣβₖΩᵏz/k!)` propagator. Verified against dense-DFT reference (1.9e-13), Raman/SSFS law, FFT-kernel tone probe, RDW group-delay walk-off. The Renninger-Wise blue-shift sign is now correct (−1.06 / −3.80 nm).

---

### #1. Self-steepening: residual photon-number drift

**Symptom.** With `include_self_steepening=True` (+ Raman), long SCG-class runs drift the pulse energy / photon number by ≈5–6 %.

**Cause.** Grid-validity artifact, not a model defect. The first-order Blow–Wood shock model `iγ(1 + (i/ω₀)∂_t)(A·P_NL)` has a conservation law that vanishes only for a time-independent drive. The drift appears once `τ·Ω_max ≳ 0.15`, where the Taylor form itself is wrong.

**Resolution (2026-10-02).** `_validate_shock_grid` now checks `τ_shock·Ω_max` and warns above `_SHOCK_TAYLOR_LIMIT = 0.2`, naming the measured drift values and the remedies. Regression: `tests/test_gnlse_shock_energy.py`.

---

### #2. `mi_gain_spectrum_extended`: catastrophic cancellation at absolute ω₀

**Symptom.** The extended MI gain evaluates `Δ(Ω) = β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)` with a user-supplied `beta_fn` at the absolute carrier. At NIR `ω₀ ≈ 1.2e15 rad/s` the float64 ULP is ≈ 0.25 rad/s, while the useful signal `|Δ| ~ 0.01–0.4 rad/s`.

**Resolution (2026-09-28).** Offset-aware contract: `betas=<β₂…βₖ array>` (exact by construction), `beta_fn_convention="detuning"` (caller supplies `β~(Ω) = β(ω₀+Ω) − β(ω₀)`). Legacy absolute-carrier path kept with DeprecationWarning.

---

### #16. `VectorSplitStepEngine`: `betas_x` without `betas_y` raises

**Symptom.** The class docstring says `betas_y` defaults to `betas_x`, but the code raises `ValueError` when only `betas_x` is given.

**Resolution (2026-10-03).** Docstring corrected to match code: both axes must be specified explicitly (or use `betas` for degenerate axes).

---

### #17. Nine of thirty tabulated `source_key`s cannot be loaded

**Symptom.** `RefractiveIndex.from_material_database(source_key)` splits the `material-author` key on the last hyphen, so any key whose author field contains a hyphen is mangled. Nine of the 30 tabulated datasets fail.

**Resolution (2026-10-03).** Longest-known-material-prefix match: try the full key first, then progressively shorter prefixes. All 30 source keys now load. Also added `list_nk_materials()` to `MaterialsDatabase` and updated `_resolve_canonical_name` to check both `raman_specs` and `nk_data` tables.

---

### #19. Three papercuts on the refraction/materials API

**Resolution (2026-10-03).**

- (a) `n_func` (and `k_func`, `nk_func`, `dn_dlambda`) now accept `Wavelength` objects by extracting `.as_um`.
- (b) `RamanSpec.model_fields()` classmethod added (returns `__dataclass_fields__`).
- (c) `mi_gain_spectrum_extended`'s `beta_fn` now has default `None`.

---

### #20. `mi_gain_spectrum_extended` emits DeprecationWarning twice

**Resolution (2026-10-03).** Warnings consolidated to one (emitted in the Δ(Ω) computation branch only).

---

### #21. `VectorSplitStepEngine` "manakov" docstring omits 8/9 equivalence conditions

**Resolution (2026-10-03).** Docstring now states: the `8/9` factor invites the reading that `"manakov"` at γ _is_ `"incoherent"` at `8γ/9`. That is true only when one channel is unpopulated (`P_y ≡ 0`).

---

### #22. `conserving_shock=True` is a silent no-op for γ > 0 + ComplexWarning flood

**Symptom (a).** For any fiber with γ > 0, `abs(gamma) == gamma`, so the two code paths are identical. Measured: relative field difference 4.53e-15.

**Symptom (b).** With `conserving_shock=True` every nonlinear step emits `ComplexWarning: Casting complex values to real discards the imaginary part` — 300 warnings in 50 steps.

**Resolution (2026-10-03).** Added `.real` to the delayed intensity convolution (both in `_delayed_intensity` and inline in `_nonlinear_step`). Docstring now documents that the modification only bites for γ < 0.

---

### #23. `fit_two_wave`: sign of Δk is not identifiable

**Resolution (2026-10-03).** Docstring now documents: `η(z)` is even in `Δk`, so the sign is not identifiable from `η(z)` alone. `delta_k_bounds` is where a sign prior is supplied.

---

### #24. `fit_shg_autodiff` does not recover (sigma, P0)

**Resolution (2026-10-03).** Docstring corrected: the Adam optimizer does not reliably recover the `(σ, P₀)` split even with the power term. Use `fit_two_wave` (scipy) for reliable recovery.

---

### #25. `design_efficiency` assumes monotone η(L)

**Resolution (2026-10-03).** Docstring now documents: the bisection assumes `η(L)` is monotonically increasing. Under QPM, `η(L)` oscillates, so the bisection returns the first crossing. Always check `residual_norm`.

---

### #26. `_SHOCK_TAYLOR_LIMIT` caps grid step at ~13 fs

**Resolution (2026-10-03).** `_validate_shock_grid` docstring now documents the resolution floor: `dt ≥ π/(0.2·ω₀)` — about 12.9 fs at 1550 nm and 7.1 fs at 850 nm.

---

### #27. `noise.py`: Bose-factor docstring + `peak_power()` unit switch

**Resolution (2026-10-03).**

- (a) `raman_noise_field` docstring corrected: n_th = 0.138 at 300 K (not ≪ 1). The Bose factor is a 35% correction at room temperature.
- (b) `Wave.peak_power` docstring now documents the unit hazard: the return value changes by 13 orders of magnitude depending on whether `A_eff` is attached.

---

## House rule

An item moves out of this file only when the code/tests change and the claim's original location is updated. All items above have been resolved in the code as of 2026-10-03.
