# Foundation core

`photonics_helper.core` is the **foundation layer** of the library — the small,
dependency-light set of primitives that everything else (and anything you build)
can rely on. It is deliberately separate from the solvers and plotting helpers,
which are *satellites* layered on top.

```python
from photonics_helper.core import units, constants, grids, materials
```

## Why the separation exists

The library aims to be something other photonics projects build on. That only
works if the foundation can be used without paying for the rest: importing
`photonics_helper.core` pulls in **numpy, scipy and pydantic only** — no
matplotlib, no plotly, no dash, and none of the solver modules.

```python
# a fresh interpreter
import photonics_helper.core   # fast, no plotting stack
```

This is enforced by tests (`tests/test_core.py`, `tests/test_import_budget.py`),
not just documented.

## What is in the core

| Module | Contents |
|---|---|
| `core.units` | Typed unit classes: `Wavelength`, `Frequency`, `AngularFrequency`, `Energy`, `Time`, `Length`, `Area`, `Power`, `Wavenumber`, `PeakPower`, `Permittivity`, `Permeability`, and their array variants |
| `core.constants` | `C_MS`, `EPS_0`, `MU_0`, `Z0`, `H_PLANCK`, `HBAR`, `PI` |
| `core.grids` | `TemporalGrid` — time/frequency sampling and the documented FFT convention |
| `core.materials` | `OpticalMaterial` protocol, `Material`, and the `material()` database lookup |

### Units

```python
from photonics_helper.core import units

wl = units.Wavelength(1550, "nm")
wl.to_freq().as_THz      # 193.41...
wl.to_energy().as_eV     # 0.799...
```

### Grids

```python
from photonics_helper.core import grids, units

grid = grids.TemporalGrid(N=4096, Tmax=units.Time(10, "ps"))
A_w = grid.fft(A_t)      # FFT(A) * dt, fftshifted
A_t = grid.ifft(A_w)     # paired inverse, / dt
```

### Materials

`OpticalMaterial` is a runtime-checkable protocol; `material()` returns a
concrete implementation backed by the bundled database.

```python
from photonics_helper.core.materials import OpticalMaterial, material

silica = material("Silica")
isinstance(silica, OpticalMaterial)   # True
silica.n_func(1.55)                   # n at 1.55 um
silica.source                         # provenance string
```

You can also type your own material against the protocol:

```python
from photonics_helper.core.materials import OpticalMaterial

def group_index(mat: OpticalMaterial, wavelength_um: float) -> float:
    ...
```

## Stable core, evolving satellites

| Layer | Modules | Guarantee |
|---|---|---|
| **Core (stable)** | `core.units`, `core.constants`, `core.grids`, `core.materials` | Foundation contract; additive-only changes within a major version |
| **Satellites** | `gnlse`, `chi2`, `phase_matching`, `soliton`, `pulse`, `raman`, `dbr`, `frog`, `structured`, `breathers`, `noise`, `wave_breaking`, `phonon`, `dashboard` | First-class, but may evolve as physics and features develop |

Both layers keep their **existing import paths**. Nothing in this namespace
replaces `photonics_helper.base` or `photonics_helper.materials` — for example,
`from photonics_helper.base import Wavelength` and
`from photonics_helper.core.units import Wavelength` refer to the *same class
object*.

## Building on the core

A minimal external tool only needs the foundation:

```python
"""A tiny free-space propagator built on core primitives."""
import numpy as np
from photonics_helper.core import constants, grids, units


def propagate(gaussian_E, dz_m: float):
    grid = grids.TemporalGrid(N=gaussian_E.size, Tmax=units.Time(20, "ps"))
    beta = 2 * np.pi / units.Wavelength(1550, "nm").as_m
    return np.fft.ifft(np.fft.fft(gaussian_E) * np.exp(1j * beta * dz_m))
```

See the [API reference](api/core.md) for the full symbol list.
