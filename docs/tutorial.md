---
quadrant: tutorial
---

# Tutorial: one fiber, one pulse, one answer you can check

You are going to launch a short pulse into 10 mm of fiber and predict the output
before the code runs. Then you will confirm the code agrees.

The physics is a fundamental soliton: dispersion and the Kerr effect cancel
exactly, so the pulse comes out the way it went in. That gives you a check you
can do with three multiplications and a stopwatch.

Every code block below runs in order, as written.

```bash
pip install photonics-helper
```

## Step 1: write down the pulse and the fiber

You need four numbers, all of which you would already have from a laser
datasheet and a fiber spec sheet.

```python
import numpy as np
from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

FWHM_PS = 0.050            # pulse full width at half maximum
BETA2_PS2_PER_M = -0.12    # group-velocity dispersion, anomalous
GAMMA_PER_W_M = 1.0        # nonlinear coefficient
LENGTH_M = 0.010           # 10 mm of fiber
```

`np` is used later for the check. Nothing has been simulated yet.

## Step 2: do the arithmetic by hand

Two derived quantities decide everything about a soliton: the dispersion length
`L_D` and the fundamental peak power `P0`.

```python
T0_PS = FWHM_PS / 1.763                     # sech T0 from FWHM
L_D_MM = T0_PS**2 / abs(BETA2_PS2_PER_M) * 1e3
P0_W = abs(BETA2_PS2_PER_M) / (GAMMA_PER_W_M * T0_PS**2)

print(f"T0   = {T0_PS*1e3:.2f} fs")
print(f"L_D  = {L_D_MM:.2f} mm")
print(f"P0   = {P0_W:.2f} W")
```

```
T0   = 28.36 fs
L_D  = 6.70 mm
P0   = 149.19 W
```

Your 10 mm of fiber is therefore 1.49 dispersion lengths long. Hold on to that
number.

Note the unit convention. `betas` are in `ps^k/m` and the pulse width comes in
as a `Time`, so `T0**2 / betas` lands in meters without any extra factor. If you
ever get a soliton power off by 10⁶, this is the line to look at.

## Step 3: launch a fundamental soliton

Peak power is `P0`, so the soliton order `N` is 1. The envelope wants a peak
*power*, not a peak field amplitude, because that is the quantity you measured.

```python
grid = TemporalGrid(N=4096, Tmax=Time(12e-12, "s"))
envelope = Envelope.from_fwhm(
    "sech", peak_amplitude=np.sqrt(P0_W), fwhm=Time(FWHM_PS, "ps")
)
pulse = Wave(grid=grid, envelope=envelope, central_wavelength=Wavelength(1550, "nm"))

solver = GNLSESolver(
    pulse=pulse,
    fiber=Fiber.from_gamma(gamma=GAMMA_PER_W_M, length=Length(LENGTH_M, "m")),
    betas=[BETA2_PS2_PER_M],
    include_raman=False,
    include_self_steepening=False,
    include_tpa=False,
)
solver.propagate(num_steps=1000)
```

`Fiber.from_gamma` skips `n2` and effective area entirely when you already know
γ. The `include_*` flags are all off, so this is the pure nonlinear Schrödinger
equation.

Two grid numbers matter here. `N=4096` over `Tmax=12 ps` resolves the 28 fs
pulse and keeps the spectrum far from the edges. Widen `Tmax` and the
grid gets coarser; raise `N` and the run gets slower but the answer does not
change. Check that before trusting anything below.

## Step 4: check it against Step 2

```python
P_in = np.abs(pulse.envelope_field) ** 2
P_out = np.abs(solver.evolution[-1].envelope_field) ** 2

print(f"peak in  = {P_in.max():.4f} W")
print(f"peak out = {P_out.max():.4f} W")
print(f"ratio    = {P_out.max() / P_in.max():.6f}")
```

```
peak in  = 149.1921 W
peak out = 149.2207 W
ratio    = 1.000192
```

The peak came back 0.02 % higher. That residue is the split-step error, and it
shrinks when you add steps. The prediction from Step 2 survived.

That is the whole point of the exercise. A solver you cannot check against a
number you computed yourself is a drawing, not a measurement.

## Step 5: break it on purpose

Now raise the peak power 9×, so `N = 3`. Integer-N solitons are periodic: over
one soliton period `(π/2)L_D` they come back to their starting shape. Your fiber
is 1.49 `L_D`, which is 0.95 of a period, so almost nothing should change.

```python
def propagate(n_squared, shock=False):
    env = Envelope.from_fwhm(
        "sech", peak_amplitude=np.sqrt(P0_W * n_squared), fwhm=Time(FWHM_PS, "ps")
    )
    p = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550, "nm"))
    s = GNLSESolver(
        pulse=p,
        fiber=Fiber.from_gamma(gamma=GAMMA_PER_W_M, length=Length(LENGTH_M, "m")),
        betas=[BETA2_PS2_PER_M],
        include_raman=False,
        include_self_steepening=shock,
        include_tpa=False,
    )
    s.propagate(num_steps=1000)
    return p, s

for n_squared, shock in ((1.0, False), (9.0, False), (9.0, True)):
    p, s = propagate(n_squared, shock)
    t, w = p.grid.t * 1e12, p.grid.w * 1e-12
    Pi, Po = np.abs(p.envelope_field) ** 2, np.abs(s.evolution[-1].envelope_field) ** 2
    Si = np.abs(p.grid.fft(p.envelope_field)) ** 2
    So = np.abs(p.grid.fft(s.evolution[-1].envelope_field)) ** 2
    tw = lambda a: np.sqrt(np.sum(t**2 * a) / np.sum(a))
    sw = lambda a: np.sqrt(np.sum(w**2 * a) / np.sum(a))
    print(
        f"N²={n_squared:>2} shock={shock!s:<5}"
        f"  peak {Pi.max():7.1f} -> {Po.max():7.1f} W"
        f"  t-width {tw(Pi):.4f} -> {tw(Po):.4f} ps"
        f"  Ω-width {sw(Si):.1f} -> {sw(So):.1f} rad/ps"
    )
```

```
N²=1.0 shock=False  peak   149.2 ->   149.2 W  t-width 0.0257 -> 0.0257 ps  Ω-width 20.4 -> 20.4 rad/ps
N²=9.0 shock=False  peak  1342.7 ->  1372.3 W  t-width 0.0257 -> 0.0257 ps  Ω-width 20.4 -> 23.5 rad/ps
N²=9.0 shock=True   peak  1342.7 ->  1745.4 W  t-width 0.0257 -> 0.0310 ps  Ω-width 20.4 -> 29.3 rad/ps
```

Read the three rows in order. The first is your soliton, unchanged. The second
is a higher-order soliton that has almost finished its period, so it is still
close to where it started. The third has self-steepening switched on, which
breaks the periodicity and lets the pulse start to fission: the peak climbs 30 %
and the spectrum almost doubles in width over the same fiber.

That third row is why you would open this library on a real day.

## Where to go next

You now have the pattern the rest of the documentation assumes: physical
inputs, a derived scale length, one propagation, a check against arithmetic.

- [Choosing a solver](choose-a-solver.md) once your problem stops being one
  mode, one polarization, and one beta2.
- [GNLSE physics](gnlse-physics.md) for what each flag in `GNLSESolver`
  actually adds.
- [Raman noise](raman-noise.md) and [cascaded χ⁽²⁾](cascaded-chi23.md) for the
  two physics branches this tutorial left switched off.
- [`examples/`](https://github.com/hitaishi2222/photonics_helper/tree/main/examples)
  for 40 runnable scripts, each with a figure at the end.
