# photonics-helper

A helper library for photonics, fiber optics and nonlinear-optics calculations
in Python — unit-safe types, dispersion and propagation constants, pulse
envelopes, the GNLSE, Raman modelling, χ⁽²⁾ mixing, DBR/TMM stacks, and
literature reproductions.

## Install

See [**Installation**](install.md) — the short version:

```bash
pip install photonics-helper
```

## Quick start

```python
from photonics_helper import Wavelength

wl = Wavelength(1550, "nm")
print(wl.to_freq().as_THz)      # 193.4 THz
print(wl.to_energy().as_eV)     # 0.8 eV
```

See the [full quick start](quickstart.md) for a GNLSE run and the
[API Reference](api.md) for the module map.

## Where to go next

- **[Install](install.md)** — extras (`fftw`, `webapp`, …) and FFT backends.
- **[Quick start](quickstart.md)** — first pulses through the solvers.
- **[API Reference](api.md)** — every public module, one page per module.
- **[Reproductions](reproductions.md)** — validated literature results.
- The [README](https://github.com/hitaishi2222/photonics_helper#readme) has the
  full feature tour and runnable `examples/`.

## Conventions

All internal math is SI (m, s, rad/s, W, J). Unit classes convert on
construction and expose `as_*` accessors. The GNLSE uses the standard
`β_k` in `ps^k/m` with `Ω` in `rad/ps` unless `betas_unit="s^k/m"` is passed.
