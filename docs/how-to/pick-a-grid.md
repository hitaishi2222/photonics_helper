---
quadrant: how-to
---

# Pick a grid and a time window

## Before you start

You need the pulse width and roughly how much spectrum you expect. For a
transform-limited sech pulse of full width at half maximum `FWHM`, the spectral
half-width is about `1/T0` in angular frequency, with `T0 = FWHM/1.763`.

Three numbers have to be right at once: the time window, the sample count, and
the frequency span the grid can carry.

## Know the three constraints

- **Time window.** `Tmax` must hold the pulse plus everything it spreads into.
  Start at 10 to 20 times the pulse width.
- **Frequency resolution.** `dOmega = 2*pi/Tmax`. A longer window resolves finer
  spectral structure but costs resolution in time.
- **Frequency span.** The grid covers detunings up to `Omega_max = pi*N/Tmax`.
  Your spectrum has to fit inside it.

There is a fourth constraint that only appears when self-steepening is on:
`Omega_max` must stay below the carrier frequency `omega0`, otherwise the
positive and negative frequency sides overlap and the shock operator is
meaningless.

```python
import numpy as np
from photonics_helper.base import Time, Wavelength
from photonics_helper.pulse import TemporalGrid

N, TMAX_PS = 4096, 12.0

grid = TemporalGrid(N=N, Tmax=Time(TMAX_PS * 1e-12, "s"))
omega0 = Wavelength(1550, "nm").to_omega().as_rad_s
omega_max = float(np.abs(grid.w).max())

print(f"dt          = {(grid.t[1] - grid.t[0]) * 1e15:.2f} fs")
print(f"Omega_max   = {omega_max:.3e} rad/s")
print(f"omega0      = {omega0:.3e} rad/s")
print(f"ratio       = {omega_max / omega0:.3f}")
print(f"dOmega      = {2 * np.pi / TMAX_PS * 1e-12:.3e} rad/s")
```

```
dt          = 2.93 fs
Omega_max   = 1.072e+15 rad/s
omega0      = 1.215e+15 rad/s
ratio       = 0.882
dOmega      = 5.236e-13 rad/s
```

Read this as a pass list. `dt = 2.93 fs` samples a 50 fs pulse about seventeen
times per pulse width, which is enough. `ratio = 0.882` means the grid is inside
the carrier, so self-steepening will be allowed.

## Check the guard before you trust it

The solver refuses a grid it cannot use rather than returning a wrong answer.

```python
import numpy as np
from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

BAD_N, BAD_TMAX_PS = 4096, 7.0

grid = TemporalGrid(N=BAD_N, Tmax=Time(BAD_TMAX_PS * 1e-12, "s"))
envelope = Envelope.from_fwhm("sech", peak_amplitude=12.0, fwhm=Time(50, "fs"))
pulse = Wave(grid=grid, envelope=envelope, central_wavelength=Wavelength(1550, "nm"))

try:
    solver = GNLSESolver(
        pulse=pulse,
        fiber=Fiber.from_gamma(gamma=1e-3, length=Length(0.01, "m")),
        betas=[-0.02],
        include_self_steepening=True,
    )
    solver.propagate(num_steps=10)          # the grid guard fires here
    print("accepted")
except ValueError as exc:
    print("rejected by", type(exc).__name__)
    print("reason starts:", str(exc).split(".")[0][:60])
```

```
rejected by ValueError
reason starts: self-steepening requires the grid to resolve only positive a
```

The fix is in the message: widen `Tmax` by the ratio it reports, or reduce `N`.

## You know it worked when

- `ratio` is below 1 whenever self-steepening is on.
- `dt` is a small fraction of the pulse width, roughly `FWHM/20` or finer.
- Halving `N` or doubling `Tmax` changes your answer by less than the tolerance
  you care about. That is the [convergence check](check-convergence.md).

## Where the background is

[GNLSE physics](../gnlse-physics.md) explains why the shock term needs
`Omega_max < omega0`. [Choosing a solver](../choose-a-solver.md) covers which
engine you are preparing the grid for.
