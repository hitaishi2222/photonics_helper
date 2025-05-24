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
- **Comprehensive Documentation**: Clear documentation with examples
- **Easy to Use**: Intuitive API design

# Quick Start

To get started, import the library and use its functions:

```python
from photonics_helper import Wavelength, Frequency, Omega

# Convert wavelength to frequency
wl = Wavelength(1550, "nm")
freq = wl.to_freq()
print(f"Frequency: {freq.as_THz:.2f} THz")

# Convert frequency to angular frequency
omega = freq.to_omega()
print(f"Angular frequency: {omega.as_rad_ps:.2f} rad/ps")
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

- Add methods to convert wavelengths to energy (in eV)
- Add support for wavenumber (cm^-1) conversions
- Add methods for bandwidth calculations
- Add functionality for dispersion calculations
- Add methods for power/intensity conversions
