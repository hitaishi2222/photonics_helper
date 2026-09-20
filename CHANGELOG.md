# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Vector / polarization-coupled GNLSE** (`photonics_helper.vector_gnlse`,
  2-channel split-step Fourier engine, Agrawal §6.1–6.3):
  `VectorSplitStepEngine` with per-axis Taylor dispersion (`betas_x` /
  `betas_y`), differential group delay walk-off (`walkoff`, s/m), the
  `2/3` XPM anisotropy, coherent polarization FWM (`coupling="coherent"`,
  `delta_beta` rad/m, energy-conserving mixing pair advanced with an RK4IP
  frequency-domain substep) and the Manakov polarization-averaged mode
  (`coupling="manakov"` — the 8/9 coefficient, Wai & Menyuk 1996).
  `RandomBirefringenceEngine` propagates with SU(2) random-frame rotations;
  its ensemble spectrum converges to the deterministic Manakov run and total
  energy is conserved at machine precision in every mode. Scalar contract:
  with `A_y ≡ 0` the vector engine equals the scalar `SplitStepEngine` to
  machine precision (tested), so all existing scalar reproductions are
  unaffected. Docs: `docs/vector-gnlse.md`, API page, README section.
  Regression: `tests/test_vector_gnlse.py` (14 tests).
- **Multimode (few-mode) coupled GNLSE** (`photonics_helper.multimode_gnlse`):
  `MultimodeSplitStepEngine` propagates N guided spatial modes as coupled
  envelopes — per-mode Taylor dispersion and modal group delay (`group_delays`,
  s/m), SPM/XPM coefficient sets (degenerate LP `1, 2/3` or isotropic), and
  opt-in pump-driven inter-modal FWM (`include_fwm=True`): each pump channel
  exchanges the pair (m, q) through conjugate Hamiltonian partners, gated by
  the angular-momentum rule `ℓ_m = 2ℓ_n − ℓ_q` when `oam_l` is supplied.
  Contracts (regression-tested): single channel = scalar engine (machine
  precision); 2-channel LP-degenerate = the polarization vector engine
  (machine precision); mode walk-off = Δβ₁·L; FWM vs an independent dense
  RK4 <5%; forbidden FWM triplets unmixed to machine precision. Modes as
  nonlinear channels complete the Phase 4 item 2 group ("structured.py
  stops being a linear-only portrait"). Docs: `docs/multimode-gnlse.md` +
  API page. Regression: `tests/test_multimode_gnlse.py` (11 tests).
- **Cascaded χ⁽²⁾–χ⁽³⁾** (`chi2.solve_cascaded_shg`): degenerate SHG
  integrator extended with the bulk Kerr SPM/XPM terms; all χ⁽²⁾
  reproductions unaffected (pure-quadratic limit = `solve_shg` exactly).
  Limit contracts regression-tested: pure-Kerr = the analytic SPM phase
  `exp(iγ P₀ L)` exactly; the cascaded-Kerr limit at large phase mismatch
  recovers the effective coefficient `γ_φ = σ²P₀/Δk` (Epstein / Saltiel /
  Agrawal §10.5) within 5%. XPM defaults to the 2/3 degenerate
  linearly-polarized mode-pair factor consistent with the vector GNLSE.
  Regression: `tests/test_cascaded_chi23.py` (4 tests).
- **Inverse-design layer (Phase 4 item 5, v1)**
  (`photonics_helper.inverse_design`): `fit_two_wave` identifies the
  well-posed invariants (κ = σ√P₀, Δk) of a χ⁽²⁾ run from measured η(z)
  data via multi-start least squares over the exact forward solver —
  validated to recover κ and |Δk| exactly; the (σ, P₀) degeneracy inside
  κ is documented as non-identifiable from η(z). `design_efficiency`
  recovers the analytic tanh²(κL) design length exactly and fails loudly
  for unreachable targets. Regression: `tests/test_inverse_design.py`
  (5 tests). PINN/differentiable training is declared future work.
- **Multimode GNLSE v2 — overlap tensors + pump-depletion**
  (`multimode_gnlse`, Mumtaz JLT 31, 398 (2013), doi:10.1109/JLT.2012.2235414,
  Eq. 6/8; Poletti & Horak JOSA B 25, 1645 (2008), doi:10.1364/JOSAB.25.001645):
  - `xpm_weights` (N×N) and `fwm_weights` (N⁴) — mode-specific nonlinear
    overlap tensors override the uniform `coef_model` factors pair-wise
    (Mumtaz Eq. 8), including the SPM slot.
  - `fwm_pump_depletion=True` — Manley–Rowe-consistent FWM exchange:
    creation arms `+iγ f A_n²A_q*` / `+iγ f A_n²A_m*` plus the
    back-conversion pump arm `+2iγ f* A_m A_q A_n*`. The FWM pair arm
    itself was corrected from the previous complex-conjugate form to the
    Mumtaz creation form (the old arm leaked photon number pointwise;
    the depleted set conserves Σ|A|² to <1e-6 split-step / machine-prec
    pointwise on the RHS, verified against a dense Manley–Rowe RK4 to
    <5%).
  - Regression: `tests/test_multimode_v2.py` (7 tests); the pre-v2 suite
    is updated where the corrected mixing changes the physics contract.
- **Inverse-design v2: differentiable (torch) fitting**
  (`inverse_design.fit_shg_autodiff`, `[pinns]` extra): the degenerate SHG
  three-wave physics written as a batched multi-start fixed-step RK4
  integrator in torch.float64 with gradients flowing through the
  integration; Adam over (log σ, log P₀, Δk) recovers the full triple
  within ~10% when the absolute SH power along z is supplied (breaking
  the `κ = σ√P₀` degeneracy of the η-only inverse problem — the flat
  direction is documented and NOT asserted). torch forward validated
  against `solve_shg` to 1.7e-8. Regression:
  `tests/test_inverse_design_autodiff.py` (2 tests, ~32 s).

## [0.1.9] - 2026-09-19

### Added

- **Interaction-picture self-steepening (RK4IP)** in `SplitStepEngine`: the
  shock step factors out the exactly-integrable Kerr/Raman phase and advances
  only the shock correction `iγτ_shock ∂_t(A·P_NL)` with frequency-domain RK4
  (Hult 2007; Hochbruck & Ostermann 2010). The historical
  `max(1 + Ω·τ_shock, 0)` clamp is removed, and a spectral-validity guard
  requires `Ω_max < ω₀` when self-steepening is enabled (raises with the grid
  values and the remedy). Validated against a fully-resolved reference of the
  same nonlinear flow (`tests/test_gnlse_unitarity.py`); the photon number is
  conserved to machine precision in the constant-drive limit.
- **Multi-phonon Raman response in the solver**: `PhononResponse.h_R(t)` is a
  causal, unit-integral superposition of damped oscillators built from the
  phonon-mode data (Hollenbeck & Cantrell 2002), and the solver's Raman
  dispatch reads `fR` and `h_R(t)` from any response object on the fiber, so
  multi-mode crystalline materials can be propagated
  (`tests/test_gnlse_phonon_raman.py`).
- **Time-resolved TPA / free-carrier model (opt-in)**:
  `SplitStepEngine(include_free_carriers=True)` resolves the carrier density
  over the retarded-time grid (exact TPA attenuation, carrier
  generation/recombination, free-carrier absorption), opt-in and separate
  from the legacy spatially-averaged TPA
  (`tests/test_gnlse_free_carrier.py`).
- **GNLSE validation & convergence harness** (`photonics_helper.gnlse_validation`,
  exported at top level): `convergence_study(build_solver, refinements,
  observables, tolerance)` with per-observable values, successive relative
  changes and a converged verdict (Sinkin et al. 2003), plus cited analytical
  checks `check_spm`, `check_mi`, `check_soliton`, `check_gordon_ssfs` that
  raise `ValidationFailure` on a closed-form mismatch. Regression:
  `tests/test_gnlse_convergence.py`.
- `docs/gnlse-physics.md`: the (corrected) shock photon-number balance, the
  multi-mode response, the free-carrier model and the harness, each with
  references; an API reference page (`api/gnlse-validation.md`) and mkdocs nav
  entries.

### Changed

- README gained a "Solver physics hardening" section; version bumped to 0.1.9
  (also `CITATION.cff`).

## [0.1.8] - 2026-09-19

### Fixed

- **Actionable errors when material data cannot be retrieved** (the stable core
  surface is unchanged; only the messages are):
  - a missing, empty or corrupt database now names the path it tried, the
    underlying SQLite error, and the remedy (reinstall, or `python seed_db.py`);
  - an unknown material suggests close matches (`Did you mean: Silica?`) and
    points at `RamanDatabase().list_materials()` and the `material-author`
    tabulated keys;
  - requesting `axis=` on a material with no ordinary/extraordinary rows now
    explains that and lists the birefringent materials the database does have;
  - blank or non-string material names are rejected up front instead of
    producing a confusing “No Sellmeier data for  in materials.db”.
- `RefractiveIndex.from_material_database` accepts an optional existing
  `RamanDatabase` handle (`db=`), which makes the failure paths testable. This
  is a compatible addition to a provisional module; no stable-core signature
  changed (the API-surface snapshot is unchanged).

### Added

- **`material_catalog(name=None)` / `print_material_catalog(name=None)`** —
  discover every dataset in the bundled database as a rich table (material,
  kind, wavelength range, DOI, licence), with a case-insensitive name filter.
  Exported at top level. The catalogue is complete: it includes Sellmeier rows
  with no Raman spec (Silicon, Sapphire, Germanium, the LiNbO₃ ordinary /
  extraordinary sub-rows) and tabulated datasets without a Raman spec (GaP).
- **`RamanDatabase.list_sellmeier_datasets()`** — every Sellmeier entry as a
  summary (fixes the omission of rows without a `raman_specs` entry).
- `provenance.doi` is now backfilled from the citation text, so DOIs no longer
  have to be parsed out of citations by consumers.
- `tests/test_material_errors.py` (error-message contract) and
  `tests/test_material_catalog.py`.

## [0.1.7] - 2026-09-19

### Added

- **Published stability contract** (`docs/stability.md`): the stable surface
  (`photonics_helper.core` + the database schema) vs the provisional satellites,
  the versioning rules, the deprecation lifecycle, the support window and the
  data-migration rule.
- **Deprecation machinery** (`photonics_helper._deprecation`):
  `deprecated(...)` (functions and classes) and `warn_deprecated(...)`, emitting
  a once-per-process `DeprecationWarning` that names the replacement and the
  removal release, without changing signatures or behaviour.
- **Core API-surface guard**: `scripts/update_api_snapshot.py` +
  `tests/core_api_snapshot.json` + `tests/test_core_api_surface.py` pin every
  stable module's symbol set and public signature shape; removals, renames and
  new required parameters fail CI, optional additions are allowed.
- **Build-on-core guide** (`docs/building-on-core.md`) and a runnable example
  (`examples/33_build_on_core.py`): a dispersive-broadening calculator written
  against the core only, validated to ~1e-15 against the closed-form
  `sqrt(1 + (z/L_D)^2)`, with a test asserting no satellite module is imported.
- **Governance artefacts**: `CONTRIBUTING.md`, `CITATION.cff`, and a JOSS-format
  software-paper draft (`paper/paper.md`, `paper/paper.bib`).

### Changed

- `pyyaml` added to the `dev` extra (governance tests parse `CITATION.cff`).

### Notes

- Nothing is deprecated yet; the machinery exists so the first rename follows
  the published policy instead of breaking callers.
- The JOSS submission itself is an author action (needs a Zenodo DOI and the
  JOSS bot); this release prepares the draft.

## [0.1.6] - 2026-09-19

### Added

- **Material provenance registry** — a `provenance` table
  (`source_key`, `kind`, `citation`, `doi`, `url`, `license`) joinable via
  `nk_data.source`, plus `RamanDatabase.get_provenance()` /
  `list_provenance()`.
- **Per-row licence** on `nk_data`, `sellmeier`, `raman_specs` and
  `phonon_modes`, backfilled by an idempotent schema migration (documented
  sentinels where a source declares no blanket licence).
- **`PhononResponse.from_material(name)`** — resolves phonon modes through the
  database first, falling back to the canonical `PHONON_MATERIALS` table.
- **`Material.license`** on `photonics_helper.core.materials`, populated from
  the provenance registry.
- **`docs/data-schema.md`** documenting every table, the licence model and the
  regeneration procedure.
- Golden-value regression tests (`tests/test_material_data_golden.py`) pinning
  `n(λ)`/`k(λ)`, Raman shift/linewidth/`f_R` and phonon modes.

### Changed

- **Populated `phonon_modes`** (0 → 52 modes for 10 crystals); `seed_db.py` now
  seeds phonon data, and the empty-home-database fallback does too.
- **Lazy database initialisation** — constructing a `RamanDatabase` no longer
  opens, creates, migrates or seeds the file; that happens on first use.
- **Drift guards** — the shipped `materials.db` is now asserted to match the
  canonical Python seed tables (`raman_specs`, `phonon_modes`).

### Data

- `materials.db` migrated in place: `nk_data` preserved (24 767 rows), licence
  columns and provenance rows added. No `nk_data` or `sellmeier` value changed.

## [0.1.5] - 2026-09-19

### Added

- **`photonics_helper.core` foundation namespace** — the stable,
  dependency-light primitives other projects build on:
  `core.units`, `core.constants`, `core.grids` (the extracted `TemporalGrid`)
  and `core.materials` (a runtime-checkable `OpticalMaterial` protocol, a
  `Material` wrapper, and a `material()` database lookup). Importing it pulls
  in numpy/scipy/pydantic only — no matplotlib, plotly, dash, or solver
  modules.
- **Lazy package import** (PEP 562): `import photonics_helper` no longer
  eagerly imports the plotting/web/simulation stack; each public name is
  resolved on first access. `__all__` is unchanged.
- **Property-based tests** (Hypothesis) for unit round-trips, defining
  relations and grid FFT/Parseval.
- Import-budget, wheel-content and lazy-surface tests (`tests/test_core.py`,
  `tests/test_lazy_import.py`, `tests/test_import_budget.py`) plus a
  `wheel smoke` CI job.

### Changed

- matplotlib imports in `pulse.py` and `materials.py` are now function-local,
  so importing a solver or `RefractiveIndex` does not require a plotting
  backend.

### Fixed

- `Permiability` → `Permeability` spelling (hard rename; the misspelled name
  was only in the 0.1.1 public surface).
- Packaging: add `LICENSE`/`NOTICE`, adopt PEP 639 licence expressions, and
  pin `[tool.setuptools.packages.find]` so flat-layout discovery cannot leak
  `reproductions/` into the wheel.


## [0.1.1] - 2026-09-19

### Fixed

- **BREAKING** — `chi2.shg_coupling` was exactly 2× the standard Boyd/Miller
  plane-wave coupling; η from it was 4× too large. `σ` now follows
  `(ω d_eff/(n c))·√(2 Z₀/(n³ A_eff))`: the published 0.1.0 χ⁽²⁾ efficiencies
  must be re-derived (see the `chi2-nonlinear` spec change for details).

### Added

- **χ⁽²⁾ mode-overlap coupling** — `chi2.shg_coupling_overlap(E_pump, E_sh, dx, dz)`
  computes the modal overlap `g` from transverse mode arrays
  (Wang et al. 2017, Eq. 2), plus `chi2.pgln_overlap` for the periodically-
  grooved-LN quasi-phase-matched `g'` (Eqs. 3–5). Higher-order / multi-lobe
  SH modes — the case of TE₀(ω)↔TE₃(2ω) modal phase matching — are now
  supported without hand-rolled code.
- **Propagation loss in `chi2.solve_shg`** — per-field power loss
  (`loss_db_per_cm`), reproducing the lossy coupled-mode limit of Wang et
  al. Eq. (10); the loss-free path stays bitwise identical.
- **LiNbO₃ birefringence** — the wrong, near-dispersion-free Zeiger
  Sellmeier entry (n(1.55 µm) ≈ 2.261) is replaced by the widely-used
  Edwards & Lawrence (1984) curves. `from_material_database("LiNbO3")` now
  returns the extraordinary index (d33-active for x-cut work); select the
  ordinary axis via `axis="ordinary"` (rows `LiNbO3_er` / `LiNbO3_or`).
- **`PropagationConstant.beta(omega)` / `__call__`** — callable β(ω) on the
  stored table; usable directly as `beta_fn` in `phase_matching` and `chi2`
  without the adaptor indirection (fixes an SHG-replication `AttributeError`).
- **χ⁽³⁾ / χ⁽²⁾ phase-matching docmap** in both module docstrings.

### Deprecated

- _None._

## [0.1.0] - 2026-09-19

First release to PyPI (`photonics-helper` 0.1.0) via GitHub Actions Trusted
Publishing.

### Added

- **GPU FFT backend** — optional `cupy` path in the FFT chain
  (`PHOTONICS_FFT_BACKEND=cupy`), opt-in only, with transparent CPU fallback and
  a warning when the GPU backend is unavailable. Benchmark:
  `benchmarks/fft_backend_benchmark.py`.
- **Waveguide mode import** — `WaveguideMode.from_csv` / `from_npz` load
  external FEM `n_eff(λ)` tables into `PropagationConstant` / `Dispersion`, plus
  `PropagationConstant.beta2()`. Real femwell / Tidy3D export examples under
  `examples/` and `examples/data/`.
- **χ⁽²⁾ nonlinear optics** — new `photonics_helper.chi2` module: coupled
  SHG / SFG / DFG solver (RK4IP), QPM square-wave grating, `Lambda_qpm`,
  `delta_k_shg`, `shg_coupling`; textbook `tanh²(κL)` reproduction.
- **Physical power/energy scaling** — `PeakPower.from_envelope` and
  `Wave.with_effective_area`; normalized-unit metrics now warn once.
- **Structured light** — Laguerre–Gaussian / OAM modes (`structured.py`).
- **Release engineering** — PyPI Trusted Publishing workflow, mkdocs API docs,
  and a Python 3.12/3.13 × ±pyfftw CI matrix with ruff + mypy gates.
- **Unified dashboard** — `photonics_helper.dashboard.app()` combines the Raman
  explorer and an interactive GNLSE result viewer behind one Dash app.
- CHANGELOG, `docs/`, and `mkdocs.yml`.

### Changed

- `_fftw.py` backend chain documented and extended; automatic selection stays on
  the CPU backends (cupy is never auto-selected).
- `fiber.py` numerical-fit failures raise `ValueError` instead of the previous
  odd exception type.

### Removed

- The `xy` pulse-visualization backend (dead and undeclared in practice); the
  `xy` optional extra was dropped from `pyproject.toml`. `visualize_2d` now
  accepts only `"plotly"` and `"matplotlib"`.

### Fixed

- Inverted GNLSE dispersion sign and Raman response direction (exposed by the
  paper reproductions).
- MI gain convention (`Ω_c² = 4γP/|β₂|`, `g_max = 2γP`).
- FROG PCGPA retrieval for chirped pulses (seeded principal-component projection).
- CI on NumPy ≥ 2.4: scalar-conversion errors in `phase_matching.py` (`float()`
  on size-1 arrays); graceful FFTW-backend fallback when `pyfftw` is absent;
  `laserfun`-dependent test now skips when the package is not installed.
- CI runner pinned to `ubuntu-24.04` (clears the ubuntu-latest → Ubuntu 26
  migration warning).

[Unreleased]: https://github.com/hitaishi2222/photonics_helper/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/hitaishi2222/photonics_helper/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/hitaishi2222/photonics_helper/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/hitaishi2222/photonics_helper/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/hitaishi2222/photonics_helper/compare/v0.1.1...v0.1.5
[0.1.1]: https://github.com/hitaishi2222/photonics_helper/releases/tag/v0.1.1
[0.1.0]: https://github.com/hitaishi2222/photonics_helper/releases/tag/v0.1.0
