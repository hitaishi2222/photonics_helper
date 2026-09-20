# Roadmap

Status of the library's feature evolution. ✅ = shipped, ~~strikethrough~~ =
originally planned and now delivered.

- ~~Add methods to convert wavelengths to energy (in eV)~~
- ~~Add functionality for dispersion calculations~~
- ~~Modeling Envelopes~~
- ~~Modeling Pulse~~
- **Transfer Matrix Method (TMM)** — available (see `dbr.py`)
  - DBR multilayer stack simulation
  - Characteristic-matrix (Macleod) formalism with correct layer ordering
  - Spectral response (R, T) including absorbing and oblique stacks, plus `TMM.plot_spectrum`
  - Electric-field profiling via `TMM.field_profile`
  - `Pattern` layer stack
- **FROG** ✅
  - SHG-FROG trace generation
  - PCGPA pulse retrieval
  - Fidelity metric
- **Raman Modeling** ✅
  - Time-domain Raman response h_R(t)
  - Frequency-domain gain spectrum H(Ω)
  - Raman pulse interaction (R(t) ⊗ |E|²)
  - Material comparison overlays with 6 panel types
  - Pump wavelength explorer (Stokes/anti-Stokes)
  - SQLite material database (44 entries)
  - Interactive Dash dashboard
  - Catalog explorer example (`examples/11_raman_material_catalog.py`)
- **GNLSE** ✅
  - Dispersion (arbitrary-order β_k)
  - Kerr effect
  - Raman scattering (delayed response)
  - Self-steepening (optional; default off — set `include_self_steepening=True` to match laserfun's `shock=True`)
  - Two-photon absorption (TPA)
  - Adaptive step-size (SSFM)
  - Soliton propagation, fission, supercontinuum
- **Soliton Analysis** ✅
  - Soliton order, dispersion/nonlinear/fission lengths
  - Dispersive wave (Cherenkov) wavelength
  - Soliton trajectory extraction and RSFS rate
  - Soliton counting via peak detection
  - Publication-ready visualization (trajectories, fission dynamics, DW spectrum)
- **Waveguide Support** ✅
  - Confinement factor Γ in γ formula: `γ = n₂·ω₀·Γ/(c·A_eff)`
  - Backward compatible (Γ=1.0 recovers fiber behavior)
- **Waveguide mode import (FEM)** ✅ — `WaveguideMode.from_csv/from_npz` → `PropagationConstant`/`Dispersion`, `PropagationConstant.beta2`
- **χ⁽²⁾ nonlinear optics** ✅ — `chi2` SHG/SFG/DFG RK4IP solver, QPM grating, `Lambda_qpm`, textbook `tanh²(κL)` reproduction
- **GPU FFT backend** ✅ — opt-in cupy path (`PHOTONICS_FFT_BACKEND=cupy`) with transparent CPU fallback and a benchmark
- **Unified dashboard** ✅ — `photonics_helper.dashboard.app()` combines the Raman Explorer and an interactive GNLSE result viewer
- **Release engineering** ✅ — PyPI Trusted Publishing, Zensical API docs, and a Python 3.12/3.13 × ±pyfftw CI matrix (ruff + mypy gates)
- **Chalcogenide Materials** ✅
  - GeAsSe added (n₂=6e-18 m²/W, 44 materials total)
  - Suitable for soliton fission in chalcogenide waveguides
- **Structured Light** ✅ — Laguerre–Gaussian / OAM modes, Gaussian-beam propagation helpers, modal overlap integrals and transverse-profile plotting (`structured.py`)
- ~~Add methods for bandwidth calculations~~
- ~~Add methods for power/intensity conversions~~ (physical scaling via `Wave.with_effective_area` / `PeakPower.from_envelope`)

---

## Foundation backbone

The strategy is to make the library a **foundation other photonics projects
build on**, with `photonics_helper.core` (units, constants, grids, materials) as
the stable contract.

- **Phase 0 — hygiene** ✅ — `LICENSE` + data `NOTICE`, PEP 639 metadata, wheel
  smoke tests, import-budget guards.
- **Phase 1 — core namespace** ✅ — `photonics_helper.core`, lazy package import
  (PEP 562), `TemporalGrid` extraction, `OpticalMaterial` protocol.
- **Phase 2 — data layer** ✅ — populated phonon modes, provenance registry +
  per-row licences, lazy database access, drift + golden tests.
- **Phase 3 — stability contract** ✅ — published policy, deprecation machinery,
  core API-surface guard, build-on-core example, contribution/citation/paper
  artefacts.
- **Phase 4 — capability growth** (satellites; in progress):
  - *Physics hardening* ✅ (0.1.9): interaction-picture self-steepening,
    multi-phonon Raman dispatch, time-resolved TPA/free carriers,
    `gnlse_validation` convergence + analytical checks harness.
  - *Vector / polarization-coupled GNLSE* ✅ (this release):
    `vector_gnlse.VectorSplitStepEngine` (per-axis dispersion, PMD walk-off,
    XPM 2/3 + coherent polarization FWM, Manakov 8/9 mode) and
    `RandomBirefringenceEngine` (ensemble → Manakov, validated); exact
    scalar-limit reduction enforced by tests. Multimode/few-mode OAM
    coupling remains future work (Phase 4 item 2 partially delivered).
  - *Multimode/few-mode OAM coupling* ✅ (this release):
    `multimode_gnlse.MultimodeSplitStepEngine` — per-mode dispersion +
    group delay, LP (`1, 2/3`) / isotropic SPM-XPM models, opt-in pump-driven
    inter-modal FWM (Manley–Rowe exchange) gated by the
    angular-momentum rule `ℓ_m = 2ℓ_n − ℓ_q`; contracts: single-channel =
    scalar engine (machine precision), 2-channel LP-degenerate =
    `VectorSplitStepEngine` (machine precision), FWM vs dense RK4 <5%,
    forbidden triplets unmixed to machine precision.
    *Extension (2026-09-20, Mumtaz et al., JLT 31, 398 (2013),
    doi:10.1109/JLT.2012.2235414, Eq. 6/8; Poletti & Horak, JOSA B 25, 1645
    (2008), doi:10.1364/JOSAB.25.001645, Eq. 15):* `xpm_weights` and
    `fwm_weights` mode-specific overlap tensors override the uniform
    coefficient models pair-wise; `fwm_pump_depletion=True` adds the
    Manley–Rowe pump back-conversion arm `+2iγf* A_mA_qA_n*` — the FWM
    pair arm was also corrected to the Mumtaz creation form `+iγfA_n²A_m*`
    (the previous conjugate partner leaked photon number pointwise);
    depleted runs conserve Σ|A|² to <1e-6 and match the dense
    Manley–Rowe reference to <5%. Distribution-mode-specific overlap
    tensors and pump depletion: done.
  - *Cascaded χ⁽²⁾–χ⁽³⁾* ✅ (this release): `chi2.solve_cascaded_shg`
    (three-wave + Kerr SPM/XPM integrator; limit contracts: pure quadratic
    = `solve_shg` exactly, pure Kerr = analytic SPM exactly, large-Δk
    cascaded effective-γ = `σ²P₀/Δk` <5%).
  - *Inverse-design layer (v1)* ✅ (this release):
    `inverse_design.fit_two_wave` (identifiable κ/|Δk| recovery, exact on
    synthetic data; the (σ, P₀) degeneracy documented) and
    `design_efficiency` (exact against the analytic tanh²(κL) design
    formula).
  - *Inverse-design layer (v2, autodiff)* ✅ (2026-09-20):
    `fit_shg_autodiff` — differentiable batched multi-start RK4 SHG
    model in torch (=[pinns] extra), trained with Adam over (σ, P₀, Δk);
    forward validated vs `solve_shg` (1.7e-8), full-triple recovery
    within ~10% when the absolute SH power breaks the κ = σ√P₀
    degeneracy (documented; not asserted for η-only data).
  - *Pending*: none — Phase 4 scope is decoupled from unpublished
    own-work reproduction targets.

## Pending — author action

- [ ] **JOSS software-paper submission.** The prerequisites are in place
  (`paper/paper.md` + `paper/paper.bib` in JOSS format, `CITATION.cff`,
  `CONTRIBUTING.md`, the stability contract and provenance docs the JOSS bot
  pre-check looks for). What remains is author-side: mint a **Zenodo DOI** for
  the release (link the GitHub repo to Zenodo and cut a release), then open the
  submission issue at `openjournals/joss-reviews` with the repo URL, version and
  DOI. To be done once the project reaches a suitable milestone.
- [ ] Mint the Zenodo DOI and add it to `CITATION.cff` (`identifiers:`) and the
  README once the archive exists.
- [x] Material database backend relocated to the foundation layer
  (2026-09-20): `photonics_helper/core/data.py` now owns ``materials.db``
  (path resolution, schema, generic queries) as ``MaterialsDatabase``;
  `core.materials._lookup_metadata` and `materials.from_material_database`
  reach it directly instead of importing the Raman satellite (which stays
  importable via a documented alias in `raman/db.py`). Regression:
  `tests/test_core.py` (satellite-free `core.data` import + alias identity).
