# Photonics Helper

> ⚠️ **Disclaimer**: This library is under active development. APIs, interfaces, and internals may change between versions without notice. It is intended as a learning resource and research aid — not a production-grade simulation tool. If you rely on it for published results, please verify all outputs independently and cite the underlying physical models rather than this library.

A comprehensive helper library for photonics and optics calculations, providing easy-to-use tools for wavelength, frequency, and angular frequency conversions.

# Installation

```bash
pip install photonics-helper
```

# Key Features

- **Type Safety**: Full type hints support with stub files
- **Unit Conversions**:
  - Wavelength (nm, μm, m)
  - Frequency (THz, GHz, MHz, Hz)
  - Angular Frequency (rad/s, rad/ps)
- **Array Operations**: NumPy-based array operations for batch processing
- **Pulse Visualization**: Interactive 2D/3D pulse envelope plots
  - Gaussian, sech, chirped, Airy, and custom pulse shapes
  - Temporal intensity, spectral intensity, phase, and polar plots
  - FWHM markers and pulse width (T₀) annotations
  - Light/dark theme support
  - Interactive Plotly HTML or static Matplotlib PNG export
- **DBR Simulation**: Transfer Matrix Method for multilayer stacks
- **FROG**: SHG-FROG trace generation and PCGPA pulse retrieval
  - Generate FROG traces from electric fields
  - Retrieve pulse shape, chirp, and phase from measured traces
  - Fidelity metric for retrieval quality assessment
- **Raman Modeling**: Full Raman response physics for 30+ materials
  - Time-domain response (electronic Kerr + delayed lattice oscillation)
  - Frequency-domain gain spectrum
  - Stokes / anti-Stokes wavelength calculation
  - Pump-wavelength explorer and material comparison overlays
  - SQLite material database (`materials.db`) with 31 materials
  - Interactive Dash dashboard for comparing Raman properties
- **Comprehensive Documentation**: Clear documentation with examples
- **Easy to Use**: Intuitive API design

# Quick Start

To get started, import the library and use its functions:

```python
from photonics_helper import Wavelength

# Convert wavelength to frequency
wl = Wavelength(1550, "nm")
freq = wl.to_freq()
print(f"Frequency: {freq.as_THz:.2f} THz")

# Convert frequency to angular frequency
omega = freq.to_omega()
print(f"Angular frequency: {omega.as_rad_ps:.2f} rad/ps")
```

# Pulse Visualization

Visualize pulse envelopes with interactive 2D/3D plots:

```python
from photonics_helper.pulse import Envelope

# Create a 50 fs Gaussian pulse
pulse = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=50e-15)

# Generate interactive Plotly HTML
fig = pulse.visualize_2d(backend="plotly", theme="light")
fig.write_html("pulse.html")

# Or generate static Matplotlib PNG
fig = pulse.visualize_2d(backend="matplotlib", figsize=(14, 10))
fig.savefig("pulse.png", dpi=150, bbox_inches="tight")

# 3D spectrogram
fig_3d = pulse.visualize_3d()
fig_3d.write_html("spectrogram.html")
```

The visualization shows:
- **Temporal intensity** with FWHM and T₀ markers
- **Spectral intensity** (Fourier transform)
- **Instantaneous phase** (for chirped pulses)
- **Polar plot** (Re vs Im of the field)
- **Parameters box**: shape, T₀, FWHM, chirp

# Pulse Trains

Model mode-locked laser output with pulse trains:

```python
from photonics_helper.pulse import Envelope, Wave
from photonics_helper.base import Wavelength, Frequency

# Single pulse
pulse = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=50e-15)

# Mode-locked laser: 1 GHz rep rate, 10 pulses
wave = Wave.from_pulse_train(
    envelope=pulse,
    central_wavelength=Wavelength(1550, "nm"),
    grid=Envelope._make_grid(pulse, N=2**14),
    repetition_rate=Frequency(1, "GHz"),
    n_pulses=10,
)

# Visualize
fig = wave.visualize(t_unit="ns", w_unit="THz", t_scale=1e9, w_scale=1e12)
fig.savefig("pulse_train.png", dpi=150, bbox_inches="tight")
```

# FROG (Frequency-Resolved Optical Gating)

Generate FROG traces and retrieve ultrashort pulse shapes using PCGPA.

```python
import numpy as np
from photonics_helper import generate_trace, retrieve, fidelity

# Create a chirped pulse
T0 = 50e-15  # 50 fs
N = 2**10
dt = 10 * T0 / N
t = np.arange(N) * dt - N * dt / 2
E = np.exp(-t**2 / (2 * T0**2)) * np.exp(1j * 0.5 * 2.0 * (t / T0)**2)

# Generate FROG trace
trace = generate_trace(E, dt=dt)

# Retrieve pulse using PCGPA
result = retrieve(trace, max_iter=100)
f = fidelity(trace, result)  # → ~1.0 for good retrieval

# Visualize trace + retrieved field
fig = trace.visualize(retrieved=result)
fig.savefig("frog.png", dpi=150, bbox_inches="tight")
```

# Raman Material Database

Load Raman material parameters from the built-in SQLite database (30 materials):

```python
from photonics_helper.raman import RamanSpec

# Load from bundled materials.db (falls back to hardcoded dict)
silica = RamanSpec.from_database("Silica")
print(silica.summary())
# Material: Silica
# Raman shift: 440.0 cm⁻¹ = 13.19 THz
# Linewidth: 45.0 cm⁻¹ = 1.35 THz
# fR: 0.18
# n₂: 3.2e-20 m²/W
```

All 31 available materials:

| Category | Materials |
|----------|-----------|
| Glasses | Silica, GeO₂, As₂S₃, As₂Se₃, ZBLAN |
| Semiconductors | Si, Ge, GaAs, GaN, AlN, InP, InGaAs, AlGaAs, SiC, Si₃N₄ |
| II-VI | CdS, CdTe, ZnO |
| Oxides | Ga₂O₃, Al₂O₃ (sapphire), BaTiO₃, LiNbO₃, LiTaO₃, KTP |
| Chalcogenides | GeAsSe |
| Crystals & Hosts | Diamond, YAG, YLF |
| NLO Crystals | LBO, AgGaS₂, AgGaSe₂ |

```python
# Stokes / anti-Stokes for a given pump
from photonics_helper.base import Wavelength

pump = Wavelength(800, "nm")
stokes = silica.stokes_wavelength(pump)
anti = silica.anti_stokes_wavelength(pump)
print(f"Stokes: {stokes.as_nm:.1f} nm, Anti-Stokes: {anti.as_nm:.1f} nm")
```

# Soliton Analysis

Analyze soliton dynamics from GNLSE simulation results:

```python
from photonics_helper.gnlse import GNLSESolver, FiberProfile
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.soliton import SolitonAnalyzer, plot_soliton_trajectories
from photonics_helper.base import Wavelength, Time, Area, Length
import numpy as np

# Create pulse and fiber (waveguide with confinement factor)
grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
pulse = Wave(
    grid=grid,
    envelope=Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(100e-15, "s")),
    central_wavelength=Wavelength(1550, "nm"),
)
fiber = FiberProfile(
    n2=6.0e-18,  # GeAsSe
    alpha=0.0,
    A_eff=Area(0.2e-12, "m^2"),
    length=Length(5e-3, "m"),
    confinement_factor=0.8,  # waveguide
)

# Run GNLSE
solver = GNLSESolver(
    pulse, fiber,
    betas=np.array([-0.2, 0.001]),  # β₂=-0.2 ps²/m, β₃=0.001 ps³/m
    include_raman=True,
)
solver.propagate(num_steps=200)

# Analyze soliton dynamics
analyzer = SolitonAnalyzer(
    pulse, fiber, solver.betas, solver.z_array, solver.spectra_vs_z
)
print(f"Soliton order: {analyzer.soliton_order():.2f}")
print(f"Fission length: {analyzer.fission_length():.2f} mm")
print(f"DW wavelength: {analyzer.dispersive_wave_wavelength()*1e9:.1f} nm")

# Plot results
fig = plot_soliton_trajectories(solver)
fig.savefig("trajectories.png", dpi=150)
```

# Development

To install for development:

```sh
git clone https://github.com/yourusername/photonics-helper
cd photonics-helper
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
pip install -e .
```

# Roadmap

- ~~Add methods to convert wavelengths to energy (in eV)~~
- ~~Add functionality for dispersion calculations~~
- ~~Modeling Envelopes~~
- ~~Modeling Pulse~~
- **Transfer Matrix Method (TMM)** — *in development* (see `dbr.py`)
  - DBR multilayer stack simulation
  - Fresnel interface & propagation matrices
  - Spectral response (R, T)
  - Electric field profiling
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
  - SQLite material database (30 entries)
  - Interactive Dash dashboard
  - Catalog explorer example (`examples/11_raman_material_catalog.py`)
- **GNLSE** ✅
  - Dispersion (arbitrary-order β_k)
  - Kerr effect
  - Raman scattering (delayed response)
  - Self-steepening (energy-conserving RK45 integrator)
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
- **Chalcogenide Materials** ✅
  - GeAsSe added (n₂=6e-18 m²/W, 31 materials total)
  - Suitable for soliton fission in chalcogenide waveguides
- Structured Light
- Add methods for bandwidth calculations
- Add methods for power/intensity conversions

> ⚠️ **Note**: The TMM / DBR module (`dbr.py`) is still under active development. The API and internals may change.
