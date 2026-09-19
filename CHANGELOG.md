# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet — next release planned as 0.1.1._

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
