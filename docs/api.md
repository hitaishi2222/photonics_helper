# API Reference

The public surface, grouped by module. Docstrings are the source of truth and
are rendered with [mkdocstrings](https://mkdocstrings.github.io/) — each
module has its own page.

| Section | Page | What's in it |
|---|---|---|
| [Foundation core](api/core.md) | `core` | Stable primitives: units/constants/grids/materials |
| [Units and constants](api/base.md) | `base` | Unit-safe scalars and constants |
| [Pulse envelopes & grids](api/pulse.md) | `pulse` | ``Wave``/``Envelope``, grid tools |
| [Fiber & propagation constants](api/fiber.md) | `fiber` | Dispersion objects, splines |
| [GNLSE](api/gnlse.md) | `gnlse` | The nonlinear-Schrödinger solver |
| [Phase-matching diagnostics (χ³)](api/phase-matching.md) | `phase_matching` | FWM / MI / dispersive waves |
| [χ⁽²⁾ nonlinear optics](api/chi2.md) | `chi2` | SHG / SFG / DFG + modal coupling |
| [Raman scattering](api/raman.md) | `raman` | Response, gains, database |
| [Soliton analysis](api/soliton.md) | `soliton` | Order, fission, trajectories |
| [Breathers](api/breathers.md) | `breathers` | Exact/periodic solutions |
| [Noise sources](api/noise.md) | `noise` | Shot noise / OPPM seeds |
| [Optical wave breaking](api/wave-breaking.md) | `wave_breaking` | OWB diagnostics |
| [Structured light](api/structured.md) | `structured` | LG/OAM modes |
| [DBR / TMM](api/dbr.md) | `dbr` | Multilayer stacks |
| [Materials & n/k database](api/materials.md) | `materials` | `RefractiveIndex`, materials.db |
| [Phonons](api/phonon.md) | `phonon` | Phonon helpers |
| [FFT backends](api/fftw.md) | `_fftw` | Backend chain selection |
| [Unified dashboard](api/dashboard.md) | `dashboard` | Dash app |
