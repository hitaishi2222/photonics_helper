# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **`photonics_helper.raman_cavity`**: N-channel Raman cavity solver — arbitrary orders, per-channel direction, per-end FBG mirrors, **six steady-state solvers** (bidirectional fixed point with under-relaxation; Newton on the shooting residual; continuation in launched power; **trapezoidal collocation with a sparse Newton**; **null-vector branch switching**; and **full continuation that also ladders the backward-pump fraction**), an implicit `method="BDF"` integrator, and a linearized noise transfer solved by superposition. All are validated against the single-order RFL (exact 0.12449 W). Key numerical finding: a bidirectional pump breaks a plain collocation Newton even at 0.01 W backward power, and the backward-fraction continuation fixes it — see PLAN.md.
- **`benchmarks/raman_noise/alcon-camas2010/fixture.json`**: ref [14] of Rizzelli 2016 (Alcon-Camas & Ania-Castanon, doi:10.1364/OE.18.023569), Table 1 read directly — the Raman gain coefficients and losses the Rizzelli paper defers to it. Ref [15] (Nuno, Alcon-Camas & Ania-Castanon, doi:10.1364/OE.20.027376) is also in `literature/`.
- **`tests/test_raman_cavity.py`** (13 tests).

- **`photonics_helper.raman_rfl`**: Raman fiber laser cavity model — steady-state BVP with FBG boundaries, threshold, exact linear-superposition pump-to-Stokes RIN transfer, and the Krause 2006 Q-factor penalty. **Krause 2006 is now fully reproduced** (FSR comb spacing = `v_g/(2L)`, transfer falls with pump power, and the output-coupler, gain and length design rules), turning the stage 2 cavity target from *attempted* into *reproduced*.
- **`benchmarks/raman_noise/generate_raman_dataset.py`** and **`tests/test_raman_dataset.py`**: the stage 3 coarse RFL sweep dataset (60 samples, resumable, atomic shards) and its gates. The Krause 2006 experimental configuration is recovered from the dataset alone to 1e-9.
- **Three sweep-driver correctness fixes**: atomic shards were written zero-byte because `np.savez` appended `.npz` to a `.tmp` name; results were stored as pickled object arrays that `allow_pickle=False` could not load; and the resume trusted the manifest without verifying shard integrity, so a truncated shard was never repaired. The resume now verifies every shard and recomputes affected shards in full.
- **`tests/test_raman_rfl.py`** (8 tests) and **`tests/test_raman_dataset.py`** (5 tests).
- **Mermelstein 2003 Fig. 7 interaction lengths now reproduced.** `interaction_length_km` measured the *relative* modulation index from the peak; Fig. 7 plots the *absolute* pump power fluctuation `dP_2 = m_2 P_2` and marks the *full width at 1/e* of its peak. With the correct measurement the model gives 20.6 km direct (published 20.5) and 27.8 km indirect (published 25.5); the `xfail` is removed. This was a measurement-definition error, not a physics change.

- **`photonics_helper.raman_multipump`**, **`photonics_helper.raman_materials`**, **`photonics_helper.raman_reproductions`**: stage 2 multiple-pump cascade, material-parameterized Raman coupling, and four published reproductions. Additive; the amplifier default is unchanged.
  - `RamanChannel` now carries a `direction` (`PropagationDirection.FORWARD`/`BACKWARD`), so an arbitrary pump set can be forward, backward, or a mixture. Inter-channel coupling includes pump-to-pump and signal-to-signal pairs, not only pump-to-signal.
  - Coherent and incoherent pump combination are both implemented (`combine_coherent`, `combine_incoherent`, `virtual_equivalent_pump`). The incoherent rule is the square root of the summed mean squares and is the physically typical case; the coherent equivalent pump modulation is larger, so the coherent mean RIN transfer is smaller, matching Zhu 2007 Fig. 1.
  - `load_material_raman_gain` builds Raman coupling from `materials.db` rows and fails loudly (`ProvenanceError`) when a row lacks a reference or DOI. `raman_shift_ratio("Si", "Silica")` returns 1.18 (520/440 cm⁻¹); the design document's "roughly three times" claim is recorded as unsupported rather than tuned.
  - `assert_not_tuned` and `ReproductionReport` enforce that a target requiring tuned parameters cannot pass, and report attempted targets separately from passed ones with the missing parameter named.
  - New fixtures `benchmarks/raman_noise/{krause2006,rizzelli2016,ma2012}/fixture.json`; `zhu2007` extended with `tuned_parameters` and parameter-sufficiency records.
  - Reproduced: Zhu 2007 Eq. 7 corner (within 1 kHz) and roll-off (within 3 dB/decade of −20), the real-versus-virtual gap direction, and the pump-count trend; the silicon/silica Raman-shift ratio. Marked **attempted** with the missing parameter named: the Zhu absolute DC levels (inter-pump gain matrix deferred to refs [11],[12]) and the Krause/Rizzelli/Ma cavity-dependent targets (reflective cavity, stage 3). No parameter was tuned.
  - `examples/46_cascade_rin_vs_published.py`, `tests/test_raman_reproduction.py` (29 tests).

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

### Known limitations

- **Keita 2006 strongly walk-off-limited regime is ill-conditioned.** The solver vs. closed-form cross-check now **passes** (see Fixed), agreeing with Eq. 4.5 to ~1 dB over `Δk·L = 0.2` to `66` rad. Once `Δk·L ≳ 220` rad the transfer is below −45 dB and sits on a `sinc²` null where the closed form is hypersensitive to the exact walk-off; those points are reported by `benchmarks/raman_noise/keita2006/dispersion_solver_vs_closed.py` but not gated, and results there remain unvalidated against an independent solver.

### Fixed

- **Keita 2006 dispersion cross-check (was a false failure).** The reported −17.6 dB offset and 2π-early roll-off were three benchmark convention errors, not a physics limit: the monochromatic-pump solver was compared against the large-bandwidth Eq. 4.9/4.10 instead of Eq. 4.5; the walk-off omitted the 2π that turns the offset frequency into the paper's angular `ν_s`; and the demodulation used `4|F|/(P₀·T)` instead of the `|F|/(P₀·T)` the field-based Eq. 2.16 needs. With all three corrected the solver matches Eq. 4.5 to ~1 dB and reproduces Eq. 4.6's low-frequency 13 dB. The `D·f` invariance still holds exactly.
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
