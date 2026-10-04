# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **`photonics_helper.raman_transfer`**: continuous-wave multi-order Raman cascade, analytic RIN transfer closed forms (Keita 2006 Eqs. 4.5–4.7, 4.9–4.10; Zhu 2007 Eq. 7), and a linearized noise response with a double-pole fit (Mermelstein 2003 Eqs. 2, 5, 6, 7, 10). Purely additive; no existing public behaviour changes.
  - `CWWCascade` solves the cascade in both geometries, with a `photon_consistent` flag selecting between Mermelstein's power-coupled Eqs. 5 and a photon-number-conserving transfer. Both conservation diagnostics hold to better than 1e-6 in the mode they apply to, and warn above it.
  - `benchmarks/raman_noise/{keita2006,mermelstein2003,zhu2007}/fixture.json` carry every published target with its DOI, its location in the source document, and whether it was read from a table or a figure. `load_benchmark` raises `ProvenanceError` on a missing or blank DOI.
  - Reproduced: the 60 km Mermelstein configuration's ~13 dB on-off gain; all four DC transfer levels within 2 dB (counter 13.9/−0.4 dB and co 14.0/−0.3 dB against 15.6/0.04 and 15.4/0.7 dB published), including the published ~15 dB first-to-second-order gap; and all four 6 dB corners within 13–28 percent.
  - Recorded as **not reproduced**, with an `xfail` test naming the discrepancy: the Fig. 7 direct/indirect interaction lengths.
  - `examples/44_cw_cascade_rin_transfer.py`, `docs/raman-cascade.md`, `docs/raman-noise.md`.

- **12 new examples** (30, 34, 35–43): breather families, wave breaking, GNLSE validation, noise/ASE/Raman, vector polarization, multimode/few-mode, conserving shock, MI gain offset, multi-phonon Raman, inverse design QPM/FWM, material catalog query.
- **Material database key split fix** (#17): longest-known-material-prefix match in `from_material_database`; all 30 source keys now load.
- **`list_nk_materials()`** method added to `MaterialsDatabase`.
- **`RamanSpec.model_fields()`** classmethod added.
- **`n_func`/`k_func`/`nk_func`/`dn_dlambda`** now accept `Wavelength` objects.
- **`mi_gain_spectrum_extended`'s `beta_fn`** now has default `None`.

### Fixed

- **#22:** `conserving_shock=True` ComplexWarning flood eliminated (`.real` added to delayed intensity convolution); docstring documents γ < 0 requirement.
- **#24:** `fit_shg_autodiff` docstring corrected — does not reliably recover (σ, P₀) even with power term.
- **#25:** `design_efficiency` docstring documents monotonicity assumption and `residual_norm` check.
- **#16:** `VectorSplitStepEngine` docstring corrected — both axes must be specified explicitly.
- **#19:** Three API papercuts fixed (n_func Wavelength, RamanSpec.model_fields, beta_fn default).
- **#20:** DeprecationWarning duplication consolidated to one.
- **#21:** Manakov docstring now states 8/9 equivalence holds only for single channel.
- **#23:** `fit_two_wave` docstring documents sign symmetry of Δk.
- **#26:** `_validate_shock_grid` docstring documents resolution floor.
- **#27:** `raman_noise_field` docstring corrected (n_th = 0.138 at 300 K); `peak_power` docstring documents unit hazard.

### Changed

- **ISSUES.md** shrunk from 1560 lines to 159 lines (93% reduction).
- **REPORT.md** shrunk from 1334 lines to 70 lines (95% reduction).
- **REVIEW.md** shrunk from 568 lines to 60 lines (89% reduction).

---

## [0.1.9] — 2026-09-19

### Added

- **Vector GNLSE:** `VectorSplitStepEngine` — coupled two-polarization split-step Fourier solver with per-axis Taylor dispersion, PMD walk-off, 2/3 XPM anisotropy, coherent polarization FWM, and Manakov mode.
- **Multimode GNLSE:** `MultimodeSplitStepEngine` — N coupled modal envelopes with per-mode dispersion, LP/isotropic SPM-XPM models, and opt-in inter-modal FWM with angular-momentum gating.
- **Cascaded χ⁽²⁾–χ⁽³⁾:** `chi2.solve_cascaded_shg` couples Kerr SPM/XPM into the degenerate SHG three-wave integrator.
- **Inverse design:** `fit_two_wave` and `design_efficiency` in `inverse_design.py`.
- **PINN/autodiff training:** `fit_shg_autodiff` torch layer.

### Fixed

- **#0:** GNLSE linear operator convention swap (conjugated transform pair + plain causal h_R convolution).
- **#1:** Self-steepening photon-number drift — grid-validity artifact diagnosed and documented.
- **#2:** `mi_gain_spectrum_extended` catastrophic cancellation — offset-aware contract.
- **#3:** `TaperedGNLSESolver` inert `betas_unit` — UserWarning added.
- **#4:** `plot_waterfall` non-standard rendering — re-drawn as true ridge.
- **#6:** Raissi-2019 PINN rel-L2 — metric-definition mismatch resolved.
- **#8:** Guasoni-2015 IM-MI flat readout — saturation artifact resolved.

---

## [0.1.8] — 2026-09-12

### Added

- Second-pass verification with test suite executed.
- New defects N1–N9 found and fixed (transfer-matrix ordering, field profile, `_estimate_beta2`, MEEP `from_meep`, stale `.pyi` stubs, Si₃N₄ material data, `PhononResponse.fR`, README counts, silent exception swallowing).

---

## [0.1.0] — 2026-08-30

### Added

- Initial release: split-step Fourier engine for GNLSE propagation, vector GNLSE, multimode GNLSE, pulse/fiber/materials modules, noise sources, inverse design, breathers, solitons, phase matching, validation harness.
