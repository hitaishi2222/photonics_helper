# photonics-helper

A helper library for photonics, fiber optics and nonlinear-optics calculations
in Python — unit-safe types, dispersion and propagation constants, pulse
envelopes, the GNLSE, Raman modelling, χ⁽²⁾ mixing, DBR/TMM stacks, and
literature reproductions.

## Install

```bash
pip install photonics-helper
```

Optional extras:

| Extra | What it adds |
|---|---|
| `plotting` | Plotly interactive figures |
| `webapp` | Dash dashboards |
| `fftw` | FFTW3 FFT backend (`pyfftw`) |
| `extras` | `imageio` (GIF/video export) |
| `mode-export` | `femwell` + `tidy3d` for generating waveguide mode exports |
| `docs` | mkdocs + mkdocstrings toolchain |
| `all` | everything above |

## Quick start

```python
from photonics_helper import Wavelength

wl = Wavelength(1550, "nm")
print(wl.to_freq().as_THz)      # 193.4 THz
print(wl.to_energy().as_eV)     # 0.8 eV
```

```python
import numpy as np
from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

grid = TemporalGrid(N=2**12, Tmax=Time(5e-12, "s"))
pulse = Wave(
    grid=grid,
    envelope=Envelope(shape="sech", peak_amplitude=np.sqrt(1000.0), pulse_width=Time(50, "fs")),
    central_wavelength=Wavelength(1550, "nm"),
)
fiber = FiberProfile(n2=2.6e-20, alpha=0.0, A_eff=Area(80e-12, "m^2"), length=Length(0.1, "m"))
solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=np.array([-0.02, 1e-4]))
solver.propagate(num_steps=200)
```

## Where to go next

- **[API Reference](api.md)** — every public module and function.
- **[Reproductions](reproductions.md)** — validated literature results.
- The [README](https://github.com/hitaishi2222/photonics_helper#readme) has the
  full feature tour and runnable `examples/`.

## Conventions

All internal math is SI (m, s, rad/s, W, J). Unit classes convert on
construction and expose `as_*` accessors. The GNLSE uses the standard
`β_k` in `ps^k/m` with `Ω` in `rad/ps` unless `betas_unit="s^k/m"` is passed.
