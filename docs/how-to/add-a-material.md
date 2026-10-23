---
quadrant: how-to
---

# Add a material the database does not carry

## Before you start

You have Sellmeier coefficients, a measured n(lambda) curve, or a single
refractive index at one wavelength, for a glass or crystal that is not in
`materials.db`. You want beta coefficients you can hand to a solver.

The path is always the same two steps: build `n(omega)`, then differentiate the
propagation constant. The library gives you the first step and the conversions.

## Build the refractive index

Sellmeier coefficients of the form `n^2 = A0 + sum(A_i * lambda^2 / (lambda^2 - B_i))`
go in directly, with `B_i` as the square of the resonance wavelength in microns.

```python
import numpy as np
from photonics_helper.base import Wavelength
from photonics_helper.materials import RefractiveIndex

# Malitson 1965 fused silica.
silica = RefractiveIndex.from_sellmeier(
    A0=1.0,
    A=[0.6961663, 0.4079426, 0.8974794],
    B=[0.004679148, 0.013512075, 97.953962],
    wl_from_to_in_um=(0.5, 2.0),
    n_points=500,
)

print(f"n(1.55 um) = {silica.n_func(1.55):.6f}")
print(f"dn/dlambda = {silica.dn_dlambda(1.55):.5f} per um")
```

```
n(1.55 um) = 1.444025
dn/dlambda = -0.01198 per um
```

`n(1.55) = 1.444` is the published value for fused silica. If your coefficients
give something else, they are for a different glass or the `B_i` are not squared
resonance wavelengths. Fix that before going further.

## Differentiate to get beta2

`beta(omega) = n(omega) * omega / c`. A central second difference on that gives
`beta2` with no analytic differentiation and no sign ambiguity.

```python
import numpy as np
from photonics_helper.base import Wavelength

c = 2.99792458e8
omega0 = Wavelength(1550, "nm").to_omega().as_rad_s
d_omega = 2 * np.pi * 1e12          # 1 THz spacing


def beta(omega):
    wavelength_um = 2 * np.pi * c / omega * 1e6
    return silica.n_func(wavelength_um) * omega / c


beta2 = (beta(omega0 + d_omega) - 2 * beta(omega0) + beta(omega0 - d_omega)) / d_omega**2

print(f"beta2 = {beta2 * 1e27:.2f} ps^2/km")
```

```
beta2 = -27.94 ps^2/km
```

## Read the sign

Bulk silica at 1550 nm sits past its zero-dispersion wavelength of 1.27 um, so it
is anomalous and `beta2` is negative. The magnitude is the check worth keeping: a
standard single-mode fibre at the same wavelength has `beta2` near -21.7 ps^2/km.
Bulk material dispersion and fibre dispersion differ by the waveguide
contribution, so expect this order of magnitude and not equality.

If you get `+3000` or `-0.3`, you have a unit error, not new physics. The factor
`1e27` converts `s^2/m` to `ps^2/km`.

## Hand it to a solver

```python
import numpy as np
from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

pulse = Wave(
    grid=TemporalGrid(N=2**10, Tmax=Time(4e-12, "s")),
    envelope=Envelope.from_fwhm("sech", peak_amplitude=12.0, fwhm=Time(50, "fs")),
    central_wavelength=Wavelength(1550, "nm"),
)
solver = GNLSESolver(
    pulse=pulse,
    fiber=Fiber.from_gamma(gamma=1.3e-3, length=Length(0.01, "m")),
    betas=[beta2 * 1e24],              # ps^2/m, the engine's unit
    include_raman=False,
)
solver.propagate(num_steps=50)
print(f"betas passed = [{beta2 * 1e24:.4f}] ps^2/m")
print(f"saved steps  = {len(solver.evolution)}")
```

```
betas passed = [-0.0279] ps^2/m
saved steps  = 51
```

The solver wants `betas` in `ps^k/m`, so a value in `ps^2/km` is divided by
`1e24`. Keep the value you computed in SI and convert at the boundary; converting
twice is the mistake that hides.

## You know it worked when

- `n` at a wavelength you can look up matches to the published digits.
- `beta2` has the sign the material's zero-dispersion wavelength predicts.
- Your propagation reproduces a limiting case you already trust, for example the
  broadening formula in [building on the core](../building-on-core.md).

## Where the background is

[Database schema](../data-schema.md) describes the bundled tables, and
[data provenance](../data-provenance.md) says where each coefficient came from.
[Group dispersion](../examples/17_group_dispersion.md) is a runnable script that
shows `n`, `ng` and `vg` for the same material.
