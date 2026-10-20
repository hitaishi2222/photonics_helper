# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Documented PEP 810 (Python 3.15) lazy-import path.** `docs/install.md` now
  covers `lazy from photonics_helper import …` and the process-wide
  `-X lazy_imports=all` / `sys.set_lazy_imports("all")` opt-ins, which defer the
  `from`-imported submodule (and its numpy/scipy/pydantic chain) until first use
  — the one eager path the PEP 562 lazy surface does not cover. The `lazy`
  keyword is a `SyntaxError` below 3.15.
- **3.15-gated regression tests** in `tests/test_lazy_import.py`
  (`test_pep810_lazy_from_import_defers_submodule`,
  `test_pep810_lazy_import_reifies_to_same_object`). They skip below 3.15 and
  activate automatically once CI has a 3.15 interpreter; they assert the
  deferred statement loads no heavy dependency and reifies to the same object
  as the PEP 562 attribute path.

## [0.1.11] — 2026-10-21

### Added

- **Static full-β(ω) for the scalar engines (Gap E).** `SplitStepEngine` and
  `GNLSESolver` now accept `full_beta=` (callable `β(ω_abs)` or an array on
  `grid.w`); `betas` may be `None`. The retarded-frame phase uses the same
  4th-order centre value/slope as the multimode engine, is supported by
  `scheme="etdrk4"`, and the adaptive step heuristic estimates β₂ from the
  curve. `full_beta` and `dispersion_profile` are mutually exclusive. N=1
  multimode ≡ scalar to machine precision. Tests:
  `tests/test_gnlse_full_beta.py`.
- **`Envelope.from_energy` (Gap F1).** Builds an envelope directly in the
  engines' power convention (`E = A0²·T0·F_shape`, `|A|²` in W) for gaussian,
  sech, lorentzian, rectangular, super-gaussian, triangular, parabolic,
  cosine and exponential. Per-shape energy factors are derived from
  `Envelope.field` (the external GAP draft's lorentzian/triangular/
  exponential/super-gaussian values were wrong). Airy is rejected: the
  oscillating tail is not in L². Tests: `tests/test_pulse_energy.py`.
- **`raman_response=` kwarg on the scalar engines (Gap F3).** `SplitStepEngine`
  and `GNLSESolver` accept a response override, with `fiber.raman_response` as
  the fallback — parity with `MultimodeSplitStepEngine`.

### Fixed

- **Amplitude-convention guard (Gap F2).** `from_peak_power` /
  `from_fwhm_peak_power` mark their envelopes `amplitude_convention="field"`
  (V/m). Both engines now raise a clear error instead of silently treating a
  field-unit amplitude as √W (a 13–14 order-of-magnitude error for a
  high-index rib). `Wave.to_engine_units(A_eff)` /
  `Envelope.to_engine_units` convert to the engine convention.
- **Airy metadata vs measurement + parabolic FWHM (Gap G).** `Envelope` grows
  `measured_width()` (shape-aware; Airy is documented as the main-lobe FWHM)
  and `Envelope.calc_width` / `Wave.calc_width` defer to it for canonical
  shapes, so `env.fwhm` and the measured width can no longer disagree. The
  parabolic FWHM had been falling through to `T0`; it now uses
  `2√(1−1/√2)·T0`. Tests: `tests/test_pulse_energy.py` and the updated
  `tests/test_pulse.py`.
- **Multimode GNLSE literature audit (2026-10) — coefficient scale, XPM
  basis, tensor scaling, stability estimate.** Four conformance defects
  found against Poletti & Horak (2008) Eq. 6/7, Mumtaz, Essiambre &
  Agrawal (2013) Eq. 6/8, and Wright et al. (2018) Eq. 1:
  1. `mode_analysis.overlap_tensors_from_fields` built the absolute
     `coupling_tensor` as `½(γ_i+γ_j)·xpm`. That underestimates SPM on a
     larger mode by `A_eff,0/A_eff,i` (0.75× on the committed rib deck) and
     averages a quantity the overlap tensor already symmetrises. The
     literature form is `C_ij = γ₀·xpm_ij = (n₂ω₀/c)·Q⁽¹⁾_iijj`, whose
     diagonal is exactly the per-mode `γ_i`. Fixed, with a new regression
     asserting `diag(C) == gamma`.
  2. The default `coef_model="lp_degenerate"` (SPM 1, XPM 2/3) is the
     **two-polarization** vector-GNLSE strip-down, not the XPM of two
     co-polarized spatial modes. Added `coef_model="co_polarized"`
     (SPM 1, XPM 2, FWM 1; Mumtaz Eq. 6 scalar limit) and made the
     `mode_analysis` bridge default to `xpm_model="co_polarized"`, which
     doubles only the off-diagonal `S⁽¹⁾` ratio. `xpm_model="vector"`
     keeps the bare tensor ratio for polarization-resolved users.
  3. `S⁽²⁾/S_0000`-normalised `fwm_weights`/`raman_tensors` inherited the
     caller's `Fiber.A_eff`. Added an engine `gamma_reference` override and
     returned `gamma_reference = γ₀`, `gamma`, `A_eff` from
     `build_multimode_deck`, so the tensor arms keep their absolute scale
     independent of the fiber choice.
  4. `_fwm_rate_max` estimated RK4 stability with `γ·f` and ignored an
     absolute `fwm_coupling_tensor`; the substep count could be
     underestimated. It now uses `_fwm_coupling` on both exchange arms.
  5. `MultimodeSplitStepEngine._photon_number` omitted the spectral measure
     `dω` that the scalar engine carries (relative drift was unaffected,
     absolute photon number was low by `dω`). Fixed and cross-checked
     against the scalar `_photon_number` at N=1.
  Tests: `tests/test_multimode_audit.py` (14). Docs updated in
  `docs/multimode-gnlse.md` and `docs/api/mode-analysis.md`.
- **`GNLSESolver.propagate_auto` converges on the conserved quantity**: with
  `include_self_steepening=True` the drift check now runs on the Blow & Wood
  App. Eq. (A5) photon number (`report["photon_drift"]`,
  `report["drift_metric"] == "photon"`) instead of the unweighted energy,
  which Raman scattering legitimately changes. Previously a shock+Raman run
  with 15 % physical energy drift and 1e-6 photon drift burned every round
  doubling steps and reported non-convergence; it now converges in round 1.
  `report["drift"]` keeps the raw energy drift for inspection; behaviour
  without self-steepening is unchanged.

### Changed (smooth-UX pass)

- **`Envelope.from_peak_power` / `from_fwhm_peak_power`**: build envelopes from
  physical `Power` + `A_eff` (`A0 = sqrt(P / (½·n·c·ε₀·A_eff))`) instead of raw V/m
  amplitude. Raw `peak_amplitude` constructors are unchanged.
- **`Wave.from_envelope`** (new, auto-grid) + **`Wave.from_pulse_train` with optional
  `grid`/`repetition_rate`**: single pulses never need a grid or rep-rate;
  trains auto-build via `TemporalGrid.for_pulse_train`. `average_power(rate=None)`
  falls back to the stored rate.
- **`ElectricField.from_array`** (new) + **`ElectricField.from_envelope(Envelope, ...)`**:
  the old `from_envelope(array, ...)` still works with a deprecation warning.
- **`Fiber` renames `FiberProfile`** (`FiberProfile` kept as alias); **gamma-first**
  `Fiber.from_gamma(gamma, A_eff=...)` / `Fiber.from_gamma(gamma, ...)` needs no
  `n2` (legacy `from_gamma(gamma, n2, omega0)` still works with a warning), plus
  `Fiber.from_n2`, `Fiber.effective_gamma`, and `float | AngularFrequency` omega0
  everywhere.
- **Raman single entry point**: `RamanSpec.response()` → `RamanResponse`,
  `RamanResponse.frequency` / `RamanSpec.frequency_response()` →
  `RamanFrequencyResponse`, plus `plot_response` / `plot_frequency_response`
  delegates. The three classes stay separate internally.
- Docs (`quickstart`, `gnlse-physics`, `README`) updated to the new forms.
- **`TemporalGrid.for_gnlse` + `GNLSESolver.propagate_auto`**: fission-aware grid
  sizing (dispersive spread × soliton order) plus a converge-and-retry loop —
  doubles steps on energy drift, doubles the window at fixed dt on edge-energy
  (wrap-around) detection. Returns a report dict; warns instead of raising
  when `max_rounds` is exhausted.
- **Reproductions migrated to the new syntax** (`Fiber`, gamma-first
  `from_gamma` without `n2`/`omega0`, `spec.response()` facade). No numerics
  changed: every migrated `from_gamma` stored the same γ the legacy path
  back-computed, and all grids/steps are untouched.
- **Examples migrated too**, plus hand-rolled `2πc/λ` conversions now go
  through `Wavelength.to_omega()` (scalars), `AngularFrequency.to_wl()` /
  `AngularFrequencyArray.to_wl()` (spectra/plot axes) and `WavelengthArray`
  (tabulated spectra). Verified bit-identical except three plot axes that
  used the truncated `3e8`/`2.99792458e8` and now use exact *c*. Composite
  physics formulas (D→β₂, detunings) are deliberately untouched.

- **`photonics_helper.mode_analysis`** (step7): mode-solver → multimode-engine bridge — `load_modes` (N-mode export, permutation-invariant ordering, fundamental first by mid-grid `n_eff`; cross-λ continuation is the solver-export contract, one physical mode per file), `mode_dispersion` (per-mode Taylor betas, `group_delays[0] == 0`, GRIN-only `phase_offsets` note), `overlap_tensors_from_fields` (grid quadrature, no circular symmetry; `S⁽²⁾` kept complex when the fields are — never clipped — and the engine accepts complex `fwm_weights` accordingly; confinement `Γ_m` honestly unity — the export contract carries no core mask), `build_multimode_deck` (one call to engine kwargs). Per-mode γ routed through existing `coupling_tensor` slots (no engine change). Honesty note: MM-SCG runs state Raman/shock/full-β as off until steps 8–10. Tests: `tests/test_mode_analysis.py` (7 tests).
- **`MultimodeSplitStepEngine` Raman** (step8): per-channel intramodal `include_raman` + `raman_response` (N=1 == scalar engine) plus intermodal delayed-coupling `raman_tensors` memory kernel on the RK4 path (Wright JSTQE Eq. 1, `S^R = S^K`; step7 `S⁽²⁾` admissible as `T_R`, hand-supplied fallback documented). `docs/multimode-gnlse.md` Raman claim now matches code; stale pump-depletion future-work line corrected. Honesty note: shock/TPA/full-β still off until steps 9–10. `raman_tensors` requires `include_raman=True` (they share the h_R kernel) — enforced at init. Multi-peak ChG phonon sets (Wu et al. 2023) deferred as follow-up. Tests: `tests/test_multimode_raman.py` (4 tests).
- **`MultimodeSplitStepEngine` shock + TPA/carriers** (step9): per-channel `include_shock` self-steepening (`τ_shock ≈ 1/ω₀`, Wright JSTQE §III spectral form on coupled paths, scalar-identical Strang scheme on the default path — N=1 == scalar shock deck; `conserving_shock` pcGNLSE variant mirrored) with shock-grid guards and Blow & Wood photon tracking, plus opt-in time-resolved TPA/FCA (`include_tpa`/`include_free_carriers`; per-mode `beta_tpa`/`sigma_fca` are caller-scaled, since step7 `Γ_m` is unity). `conserving_shock` is pcGNLSE-only: rejected at init on the coupled (tensor/FWM) paths. Honesty note: full-β/FWM-widening still off until step10. Tests: `tests/test_multimode_shock_tpa.py` (5 tests).
- **`MultimodeSplitStepEngine` full-β + full-tensor FWM** (step10): per-mode full-β(ω) spectral dispersion operator (`full_beta`, scalar `TaperedGNLSESolver` precedent; decision: full operator, not Taylor 8–10) agreeing with high-order Taylor in the narrowband limit, plus full-`S⁽²⁾_mnpq` FWM contraction (`fwm_full_tensor=True`, Wright Eq. 1 tensor term image-verified, Poletti Eq. 18/19 + `oam_l` selection preserved, diagonal arms excluded). Phase 3/4 check: degenerate-only weights reproduce the slice exactly; off-slice entries add arms the slice misses. With `include_shock` the shock operator is applied once to the assembled nonlinear RHS — never pre-applied inside the FWM arm (regression test enforces exactly-once). Octave deck stable at dt ≤ 1 fs. **Retires the switched-off-physics disclaimer — all four GAP steps landed.** Tests: `tests/test_multimode_full_beta_fwm.py` (5 tests).
- **Reproductions 38/38 green**: `scripts/test_reproductions.py` all pass post-convention-fix; the 3 dudley-SCG failures were the pre-existing missing `C_MS` import in `reproductions/dudley_2006_scg/common.py` (re-added as an explicit `# noqa: F401` re-export — the fig* modules read `common.C_MS`; figure suite re-run green). Guasoni walk-off audit closed (linear layer vindicated, no new arm); Wright gain figure re-render as measured-vs-analytic (task 6.4) — renderer shipped, sweep run pending (render is slow; do not block the release on it); PINN closure-count correction landed (accepted-otherwise).

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
