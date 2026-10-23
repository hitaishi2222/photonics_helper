---
quadrant: how-to
---

# Check that a propagation is converged

## Before you start

You have a run that finished, and you want to know whether the step size was
small enough. You do not need a second hand-written run: the solver can compare
its result against a finer one and tell you in one table.

## Run the check

`verify_convergence()` re-runs the same problem at twice the step count and
compares the two results on metrics that survive a phase difference between
them.

```python
import numpy as np
from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

grid = TemporalGrid(N=2**10, Tmax=Time(4e-12, "s"))
envelope = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(100.0), fwhm=Time(50, "fs"))
pulse = Wave(grid=grid, envelope=envelope, central_wavelength=Wavelength(1550, "nm"))

solver = GNLSESolver(
    pulse=pulse,
    fiber=Fiber.from_gamma(gamma=1.3e-3, length=Length(0.01, "m")),
    betas=[-0.02],
    include_raman=False,
    include_self_steepening=False,
)
solver.propagate(num_steps=100)

diag = solver.verify_convergence(warn=False)   # prints the table below
print("converged:", diag.converged)
```

```
GNLSE conservation + convergence diagnostics
  scheme=strang  steps=100  dz=1.000e-04 m  probe=200 (2x)
  invariant drift (max): 0.0000 %  [energy, tol 1.00 %]
  self-convergence: rms 0.000 dB (tol 0.5) | peak shift 0.000 nm (tol 2) | overlap 1.00000 (tol 0.99)
  note: adaptive run: probe uses uniform steps at the nominal dz = length/num_steps
  verdict: CONVERGED
converged: True
```

## Read the four numbers

- **drift** is how far the conserved quantity moved across the recorded steps.
  With self-steepening the invariant is photon number, otherwise it is energy.
- **rms** is the in-band spectral difference between your run and the finer one,
  in dB.
- **peak shift** is how far the spectral peak moved, in nm.
- **overlap** is the normalised overlap of the two spectra.

A run is `converged` only when the drift is inside its tolerance and all three of
the other metrics are inside theirs. One green column does not carry the others.

## If you want to see it fail

Coarsen `num_steps` until `converged` is `False`. Which metric moves first
depends on the physics: a dispersion-dominated run fails on `rms` while its
energy stays flat, and a strongly nonlinear run fails on `drift` before the
spectra separate. Knowing your problem's first-failing metric is what tells you
which tolerance to tighten later.

## You know it worked when

`diag.converged` is `True`, and doubling `num_steps` again does not change the
result you care about. If the verdict is `CONVERGED` but your published number
still moves, the tolerance is too loose for that number: tighten `rms_tol_db`,
`peak_tol_nm` or `overlap_tol` and re-run.

## Where the background is

[Stability and numerical conventions](../gnlse-physics.md) covers the invariants
the drift column reports. [Choosing a solver](../choose-a-solver.md) says which
engines have this check at all: today it is the scalar engine.
