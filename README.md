# Photonics Helper

A comprehensive helper library for photonics and optics calculations.

**Documentation:** <https://hitaishi2222.github.io/photonics_helper/>

---

## Installation

```bash
pip install photonics-helper
```

Optional extras: `fftw`, `plotting`, `webapp`, `extras`, `mode-export`, `pinns`.

---

## Key Features

- **Type Safety**: Full inline type hints (PEP 561 `py.typed`)
- **Unit Conversions**: Wavelength, frequency, angular frequency
- **Pulse Visualization**: Interactive 2D/3D pulse envelope plots
- **DBR Simulation**: Transfer Matrix Method for multilayer stacks
- **FROG**: SHG-FROG trace generation and PCGPA pulse retrieval
- **Raman Modeling**: Full Raman response physics for 40+ materials
- **GNLSE Engine**: Split-step Fourier solver for nonlinear propagation
- **Vector GNLSE**: Coupled two-polarization solver
- **Multimode GNLSE**: N coupled modal envelopes with inter-modal FWM
- **Inverse Design**: `fit_two_wave`, `design_efficiency`, `fit_shg_autodiff`
- **Breathers**: Exact analytic breather solutions
- **Solitons**: Soliton solutions and analysis
- **Noise**: Stochastic noise sources (ASE, Raman)
- **Validation**: Convergence studies and analytical checks

---

## Quick Start

```python
from photonics_helper import Wavelength, Frequency, AngularFrequency

# Unit conversions
wl = Wavelength(1550, "nm")
freq = Frequency.from_wavelength(wl)
omega = AngularFrequency.from_frequency(freq)

print(f"λ = {wl.as_nm} nm")
print(f"f = {freq.as_THz} THz")
print(f"ω = {omega.as_rad_s:.4e} rad/s")
```

### GNLSE Propagation

```python
from photonics_helper import (
    Area, Length, Time, Wavelength,
    FiberProfile, GNLSESolver,
    Envelope, TemporalGrid, Wave,
    RamanResponse, RamanSpec,
)

# Create a pulse
grid = TemporalGrid(N=2**13, Tmax=Time(16e-12, "s"))
env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550, "nm"))

# Create a fiber
silica = RamanSpec(name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18)
raman = RamanResponse(spec=silica, grid=grid, tau1=12.2e-15, tau2=32e-15)
fiber = FiberProfile(n2=2.6e-20, alpha=0.0, A_eff=Area(80, "um^2"),
                     length=Length(1.0, "m"), raman_response=raman)

# Propagate
solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=[0, 0, -21e-3],
                     include_raman=True, include_self_steepening=True)
solver.propagate(num_steps=100)
```

---

## Examples

See `examples/` for 43 runnable examples covering:

- Pulse visualization and FROG
- Raman response and material comparison
- GNLSE propagation (basic, soliton, Raman, steepening, TPA)
- Breather families and wave breaking
- Vector polarization and multimode GNLSE
- Noise, ASE, and Raman thermal floor
- Inverse design (QPM, FWM, autodiff)
- Material catalog query

---

## Reproductions

See `reproductions/` for literature reproductions (12 papers reproduced).

---

## Material Database

The SQLite material database (`materials.db`) ships with 44 materials,
39 Sellmeier equations, and 30 tabulated n/k datasets.
All 30 source keys are loadable through `RefractiveIndex.from_material_database`.

---

## Testing

```bash
python -m pytest tests/
```

---

## License

MIT
