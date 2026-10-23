---
quadrant: how-to
---

# Read a spectrum without getting the carrier wrong

## Before you start

You have `solver.evolution[-1].envelope_field` and you want a plot with
wavelength or frequency on the x axis. Almost every wrong plot in this library's
issue history comes from one confusion: `grid.w` is a **detuning**, not a
frequency.

## Three rules

1. `grid.w` is `Omega`, the offset from the carrier you passed to `Wave`, in
   rad/s. It runs from `-Omega_max` to `+Omega_max` and starts near zero.
2. Absolute angular frequency is `pulse.central_frequency + grid.w`.
3. Wavelength comes from the absolute frequency only: `lambda = 2*pi*c/omega`.
   Converting a detuning to a wavelength is meaningless.

```python
import numpy as np
from photonics_helper.base import Area, Length, Power, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

c = 2.99792458e8
grid = TemporalGrid(N=2**11, Tmax=Time(10e-12, "s"))
envelope = Envelope.from_fwhm_peak_power(
    "sech", Power(1, "kW"), Time(50, "fs"), Area(80, "um^2")
)
pulse = Wave.from_envelope(envelope, Wavelength(1550, "nm")).to_engine_units(
    Area(80, "um^2")
)
solver = GNLSESolver(
    pulse=pulse,
    fiber=Fiber.from_gamma(gamma=1e-3, length=Length(0.05, "m")),
    betas=[-0.02],
    include_raman=False,
)
solver.propagate(num_steps=200)

spectrum = np.abs(pulse.grid.fft(solver.evolution[-1].envelope_field)) ** 2
omega0 = pulse.central_frequency

print(f"carrier omega0 = {omega0:.4e} rad/s ({omega0 / (2 * np.pi) * 1e-12:.2f} THz)")
print(f"grid N         = {pulse.grid.N}")
print()
print(f"{'Omega rad/ps':>12} {'absolute THz':>13} {'wavelength nm':>14}")
for detuning_rad_ps in (-40.0, -20.0, 0.0, 20.0, 40.0):
    omega = omega0 + detuning_rad_ps * 1e12
    print(f"{detuning_rad_ps:>12.1f} {omega / (2 * np.pi) * 1e-12:>13.2f} {2 * np.pi * c / omega * 1e9:>14.2f}")
```

```
carrier omega0 = 1.2153e+15 rad/s (193.41 THz)
grid N         = 4096

Omega rad/ps  absolute THz  wavelength nm
       -40.0        187.05        1602.75
       -20.0        190.23        1575.94
         0.0        193.41        1550.00
        20.0        196.60        1524.90
        40.0        199.78        1500.61
```

The spacing is the point: the same 20 rad/ps step is 25.94 nm wide on the red side
and 25.10 nm wide on the blue side, and the asymmetry grows as you move out.

## See what the wrong conversion does

```python
for detuning_rad_ps in (20.0, 40.0):
    omega = detuning_rad_ps * 1e12
    print(f"Omega={detuning_rad_ps:>5.1f} rad/ps -> {2 * np.pi * c / omega * 1e9:.0f} nm")
```

```
Omega= 20.0 rad/ps -> 94183 nm
Omega= 40.0 rad/ps -> 47091 nm
```

That is the far infrared, and it is the number you get if you convert a detuning
instead of a frequency. If your spectrum plot claims the light moved to 90 um,
this is why.

## A plot that is safe by construction

```python
# Read the axes from pulse.grid, not from a grid you built earlier: the solver may
# enlarge the grid while propagating, and the saved variable keeps the old size.
detuning_rad_ps = pulse.grid.w * 1e-12
wavelength_nm = 2 * np.pi * c / (omega0 + pulse.grid.w) * 1e9

peak = int(np.argmax(spectrum))
print(f"peak detuning   = {detuning_rad_ps[peak]:+.2f} rad/ps")
print(f"peak wavelength = {wavelength_nm[peak]:.2f} nm")
print(f"bins            = {spectrum.size}")
```

```
peak detuning   = +0.00 rad/ps
peak wavelength = 1550.00 nm
bins            = 4096
```

Compute the wavelength axis once from the absolute frequency and index it with the
same indices as the spectrum. Never recompute it per feature.

## You know it worked when

- The peak of your unpropagated spectrum lands on the wavelength you passed to
  `Wave`, to within one bin.
- A feature you identify as "the dispersive wave at 700 nm" is 700 nm when you
  index `wavelength_nm` at the same index as the spectrum peak.
- The axis is monotonic in the direction you expect: wavelength decreases as
  detuning increases.

## Where the background is

[GNLSE physics](../gnlse-physics.md) defines the detuning convention and the sign
of the time-to-frequency transform. [Choosing a solver](../choose-a-solver.md)
covers which engines report `full_beta` spectra rather than Taylor ones.
