# Quick start

## Units

```python
from photonics_helper import Wavelength

wl = Wavelength(1550, "nm")
print(wl.to_freq().as_THz)      # 193.4 THz
print(wl.to_energy().as_eV)     # 0.8 eV
```

## GNLSE propagation

```python
import numpy as np
from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

grid = TemporalGrid(N=2**12, Tmax=Time(5e-12, "s"))
pulse = Wave(
    grid=grid,
    envelope=Envelope(shape="sech", peak_amplitude=np.sqrt(1000.0),
                      pulse_width=Time(50, "fs")),
    central_wavelength=Wavelength(1550, "nm"),
)
fiber = FiberProfile(n2=2.6e-20, alpha=0.0, A_eff=Area(80e-12, "m^2"),
                     length=Length(0.1, "m"))
solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=np.array([-0.02, 1e-4]))
solver.propagate(num_steps=200)
```

## Next steps

- [API Reference](api.md) — one page per module, rendered from docstrings.
- [Raman](api/raman.md) and [χ⁽²⁾](api/chi2.md) pages for those workflows.
