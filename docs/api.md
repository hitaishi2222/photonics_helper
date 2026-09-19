# API Reference

The public surface, grouped by module. Docstrings are the source of truth and
are rendered with [mkdocstrings](https://mkdocstrings.github.io/).

## Units and constants

::: photonics_helper.base

## Pulse envelopes, grids and trains

::: photonics_helper.pulse

## Fiber dispersion and propagation constants

::: photonics_helper.fiber

## Generalized nonlinear Schrödinger equation

::: photonics_helper.gnlse

## Phase-matching diagnostics

::: photonics_helper.phase_matching

## χ⁽²⁾ nonlinear optics (SHG / SFG / DFG)

::: photonics_helper.chi2

### Modal (overlap) SHG coupling

For waveguide χ⁽²⁾ devices (x-cut LNOI, PGLN-style grooves) use the
mode-overlap entry points instead of the scalar ``A_eff``:

```python
import numpy as np
from photonics_helper.base import Wavelength
from photonics_helper.chi2 import shg_coupling_overlap, pgln_overlap

# transverse mode fields (nx, nz) from a FEM solver, arbitrary units
wl = Wavelength(1550, "nm")
g = shg_coupling_overlap(E_pump, E_sh, dx, dz, wavelength=wl, d=27e-12,
                         n_pump=2.14, n_sh=2.19)
# g in 1/(√W·m) — for a uniform mode this equals shg_coupling(A_eff)

res = pgln_overlap(E_pump, E_sh, dx, dz, wavelength=wl,
                   d0=27e-12, d1=d1_profile,          # d^(m) harmonic profiles
                   delta_eps1_pump=de1, delta_eps1_sh=de1,
                   delta_k=2*np.pi/2.77e-6)           # QPM residual mismatch
res["g_eff"]  # the quasi-phase-matched overlap g' (Wang et al. 2017 Eq. 5)
```

## Raman scattering

::: photonics_helper.raman

## Soliton analysis

::: photonics_helper.soliton

## Breathers and exact solutions

::: photonics_helper.breathers

## Noise sources

::: photonics_helper.noise

## Optical wave breaking

::: photonics_helper.wave_breaking

## Structured light (Laguerre–Gaussian / OAM)

::: photonics_helper.structured

## Distributed Bragg reflectors / transfer-matrix method

::: photonics_helper.dbr

## Materials and the n/k database

::: photonics_helper.materials

## Phonons

::: photonics_helper.phonon

## FFT backends

::: photonics_helper._fftw

## Unified dashboard

::: photonics_helper.dashboard
