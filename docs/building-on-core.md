# Building on the core

`photonics_helper.core` is meant to be built on. This page walks through a
complete, runnable external tool — a dispersive-broadening calculator — written
using **only** the core primitives, and validates it against a closed-form
result. The full script is
[`examples/33_build_on_core.py`](https://github.com/hitaishi2222/photonics_helper/blob/main/examples/33_build_on_core.py).

```sh
python examples/33_build_on_core.py
```

A test runs the same script in a fresh interpreter and asserts that it imports
no solver/plotting module (`tests/test_build_on_core.py`), so the example is an
executable guarantee that the foundation stands alone.

## What the tool does

Given a material's `n(λ)`, it computes the group-velocity dispersion `β₂` and
propagates a Gaussian pulse through a length `z` of glass, then checks the
pulse broadening against the textbook result

```
T(z) / T₀ = sqrt(1 + (z / L_D)²),    L_D = T₀² / |β₂|.
```

## Step 1 — define a material

The core does not force you to use the bundled database. `OpticalMaterial` is a
runtime-checkable protocol, so any object with the right shape works — that is
the intended extension point:

```python
from dataclasses import dataclass
import numpy as np
from photonics_helper.core import units
from photonics_helper.core.materials import OpticalMaterial


@dataclass
class CauchyGlass:
    name: str = "Cauchy glass"
    A: float = 1.4437
    B_um2: float = 0.00354
    source: str | None = "two-term Cauchy fit (illustrative)"
    license: str | None = None

    def n_func(self, wavelength_um: float) -> float:
        return self.A + self.B_um2 / wavelength_um**2

    def k_func(self, wavelength_um: float) -> float:
        return 0.0

    @property
    def wl(self) -> units.WavelengthArray:
        return units.WavelengthArray(np.linspace(1.0, 2.0, 11), "um")

    @property
    def n(self) -> np.ndarray:
        return np.array([self.n_func(w) for w in self.wl.as_um])

    @property
    def k(self) -> np.ndarray:
        return np.zeros_like(self.n)


assert isinstance(CauchyGlass(), OpticalMaterial)
```

Alternatively, load a bundled material with
`photonics_helper.core.materials.material("Silica")` — note that the _bundled
database_ is read through the Raman SQLite backend today (a naming holdover),
so the protocol path above is what keeps an external tool free of that
dependency.

## Step 2 — dispersion from units and constants

`units` handles the conversions (wavelength ↔ angular frequency) and
`constants` supplies `c`:

```python
import math
from photonics_helper.core import constants, units


def group_velocity_dispersion(material, wavelength) -> float:
    omega0 = wavelength.to_freq().to_omega().as_rad_s

    def beta(omega):  # beta(omega) = n(omega) * omega / c
        wavelength_um = (2 * math.pi * constants.C_MS / omega) * 1e6
        return material.n_func(wavelength_um) * omega / constants.C_MS

    step = omega0 * 1e-3  # central finite difference
    return (beta(omega0 + step) - 2 * beta(omega0) + beta(omega0 - step)) / step**2


beta2 = group_velocity_dispersion(CauchyGlass(), units.Wavelength(1550, "nm"))
print(f"beta2 = {beta2 * 1e27:.2f} ps^2/km")  # -> 24.27 ps^2/km
```

## Step 3 — propagate on a core grid

`TemporalGrid` provides the time/frequency sampling and a documented FFT
convention (`FFT(A)·dt` forward, `/dt` inverse, both fftshifted), and
`Wavelength`/`Time` keep the units honest:

```python
import numpy as np
from photonics_helper.core import grids, units

grid = grids.TemporalGrid(N=1 << 14, Tmax=units.Time(20, "ps"))
t0 = units.Time(50, "fs").as_s
field = np.exp(-(grid.t**2) / (2 * t0**2)).astype(complex)

dispersion_length = t0**2 / abs(beta2)


def propagate(field, beta2, length_m):
    spectral = grid.fft(field)
    phase = np.exp(1j * beta2 * grid.w**2 * length_m / 2.0)
    return grid.ifft(spectral * phase)


final = propagate(field, beta2, dispersion_length)
```

## Step 4 — validate

Measure the RMS width before and after; the ratio must reproduce the closed
form. The script prints:

```
  z/L_D    numeric    analytic   |error|
   0.5    1.118034   1.118034   2.22e-16
   1.0    1.414214   1.414214   8.88e-16
   2.0    2.236068   2.236068   1.78e-15
   4.0    4.123106   4.123106   2.66e-15
```

## What you get, and what you don't

| From the core                                           | Not from the core                                             |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| Typed units and constants                               | The GNLSE / solver stack                                      |
| `TemporalGrid` + FFT conventions                        | Plotting, dashboards                                          |
| The `OpticalMaterial` protocol and the bundled database | The specific physics of your problem — that is yours to write |

That is the point: the core removes the unit/numerics/data plumbing so your code
is about your physics.

## Next

- [Stability contract](stability.md) — what you can rely on and for how long.
- [Foundation core](./core.md) — the full surface.
- [API reference](api/core.md).

---

## Building a Raman cascade on the multimode engine

`MultimodeSplitStepEngine` propagates N coupled channels and already provides the
painful parts — retarded-frame group-velocity management, adaptive stepping, and the
energy-drift diagnostic. A Raman cascade is that engine with different coefficients, so
build the coefficients rather than a second engine: writing a parallel solver would
duplicate all of it and give two places for conservation to break.

### 1. Build the configuration

```python
from photonics_helper.base import Area, Wavelength
from photonics_helper.raman_cascade import (
    RamanCascadeConfig,
    RamanGainTable,
    build_raman_coupling_tensor,
)

cfg = RamanCascadeConfig(
    wavelengths=(
        Wavelength(1375, "nm"),
        Wavelength(1465, "nm"),
        Wavelength(1560, "nm"),
    ),
    gains=RamanGainTable(
        [0.527, 0.419], "Mermelstein 2003 Table I", "10.1109/JLT.2003.812461"
    ),
    a_eff=(Area(55, "um^2"),) * 3,  # per order, not shared
    n2=3.2e-20,
    alpha_per_m=[...],  # per order, in 1/m
)
tensor = build_raman_coupling_tensor(cfg)
```

`RamanGainTable` **refuses** an empty DOI. Channel 0 is the highest-frequency order and
wavelengths must ascend.

### 2. Feed it to the engine

```python
engine = MultimodeSplitStepEngine(
    waves,
    fiber,
    betas,
    coupling_tensor=tensor,
    channel_alpha=cfg.alpha_per_m,  # loss is NOT part of the tensor
)
engine.propagate(n_steps)
```

Two things that will bite otherwise:

- **Loss does not go in `coupling_tensor`.** The tensor multiplies `|A_j|²`, so loss
  folded into it becomes amplitude-dependent. Use `channel_alpha`.
- **Build waves as `A = sqrt(P_W)` and do not attach an effective area.**
  `with_effective_area()` leaves the field alone but switches `peak_power()` to the
  `n c ε₀ A_eff |A|²/2` relation, so a 0.78 W pump then reports 8e-14 W. Read powers
  back with `channel_powers_w`.

### 3. Check conservation

`engine.energy_vs_z` gives `Σ ∫|A_i|²dt` per snapshot. For a photon-conserving
cascade the conserved quantity is `Σ P_k λ_k`, not `Σ P_k` — see
`raman_transfer.photon_flux_drift`. Measured drift on a deep-depletion 60 km cascade is
2e-9 relative.

### 4. Add noise and extract

```python
from photonics_helper.raman_noise import (
    NoiseBand,
    pump_noise_field,
    relative_intensity_rin,
    rin_g2_consistency_db,
)

A = pump_noise_field(grid, A_pump, 1e-12, NoiseBand(1e8, 2e10), seed=s)
f_hz, psd = relative_intensity_rin(ensemble, grid, band)
```

`g2` must come from the intensity statistics, not from `noise.coherence_g12` — the
latter measures _mutual_ coherence between realizations and vanishes for independent
seeds while `g2` stays finite. `rin_g2_consistency_db` is the check.
