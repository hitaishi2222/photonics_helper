# Photonics Helper

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
- Structured Light
- Add methods for bandwidth calculations
- Add methods for power/intensity conversions
- Modeling GNLSE
  - Kerr Effect
  - Raman Effect
  - Self Steepening
  - Mode profile Dispersion
- SSFM

> ⚠️ **Note**: The TMM / DBR module (`dbr.py`) is still under active development. The API and internals may change.
