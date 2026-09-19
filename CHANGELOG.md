# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] (planned: 0.1.1)

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

[Unreleased]: https://github.com/hitaishi2222/photonics_helper/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hitaishi2222/photonics_helper/releases/tag/v0.1.0
