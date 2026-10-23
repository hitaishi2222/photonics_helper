---
quadrant: how-to
---

# Recipes

Short, copyable forms for the things you will do most often. Each block is
complete on its own, except the last one, which continues from the
[tutorial](tutorial.md). If you have not run a propagation yet, start there.

## Convert between wavelength, frequency and photon energy

```python
from photonics_helper import Wavelength

wl = Wavelength(1550, "nm")
print(wl.to_freq().as_THz)     # 193.4
print(wl.to_energy().as_eV)    # 0.8
```

```
193.41448903225802
0.7998980544077436
```

## Build a single pulse with a physical peak power

`Envelope.from_fwhm_peak_power` takes power, not field amplitude. It builds the
temporal grid for you, sized to about ten times the pulse width.

```python
from photonics_helper.base import Area, Power, Time, Wavelength
from photonics_helper.pulse import Envelope, Wave

envelope = Envelope.from_fwhm_peak_power(
    "sech", Power(1, "kW"), Time(50, "fs"), Area(80, "um^2")
)
pulse = Wave.from_envelope(envelope, Wavelength(1550, "nm"))
```

## Build a pulse train

The grid and the pulse spacing both follow from the repetition rate.

```python
from photonics_helper.base import Frequency, Time, Wavelength
from photonics_helper.pulse import Envelope, Wave

envelope = Envelope.from_fwhm("sech", peak_amplitude=12.2, fwhm=Time(50, "fs"))
train = Wave.from_pulse_train(
    envelope,
    Wavelength(1550, "nm"),
    repetition_rate=Frequency(10, "GHz"),
    n_pulses=5,
)
```

## Set up a fiber when you know gamma instead of n2

No effective area needed. Useful when the gamma came from a measurement.

```python
from photonics_helper.base import Area, Length
from photonics_helper.gnlse import Fiber

fiber = Fiber.from_gamma(0.001, A_eff=Area(80, "um^2"), length=Length(0.1, "m"))
```

## Read the spectrum back out

`solver.evolution` holds one saved field per output step, starting with the input.
This is a complete run you can paste into a file.

```python
import numpy as np
from photonics_helper.base import Area, Length, Power, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

grid = TemporalGrid(N=2**11, Tmax=Time(10e-12, "s"))
envelope = Envelope.from_fwhm_peak_power(
    "sech", Power(1, "kW"), Time(50, "fs"), Area(80, "um^2")
)
pulse = Wave.from_envelope(envelope, Wavelength(1550, "nm"))
pulse = pulse.to_engine_units(Area(80, "um^2"))   # W-safe amplitudes
solver = GNLSESolver(
    pulse=pulse,
    fiber=Fiber.from_gamma(gamma=1e-3, length=Length(0.01, "m")),
    betas=[-0.02],
    include_raman=False,
)
solver.propagate(num_steps=20)

A_out = solver.evolution[-1].envelope_field
spectrum = np.abs(pulse.grid.fft(A_out)) ** 2
detuning_rad_per_ps = pulse.grid.w * 1e-12

print(f"saved steps = {len(solver.evolution)}")
print(f"spectral bins = {spectrum.size}")
print(f"peak at {detuning_rad_per_ps[np.argmax(spectrum)]:.1f} rad/ps")
```

```
saved steps = 26
spectral bins = 4096
peak at 0.0 rad/ps
```

Detuning is measured from the carrier, so `grid.w = 0` is the wavelength you passed
to `Wave`. Add `pulse.central_frequency` to get absolute angular frequency.
