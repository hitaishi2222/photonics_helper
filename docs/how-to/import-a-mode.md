---
quadrant: how-to
---

# Import a mode or dispersion curve from an external solver

## Before you start

You have eigenmode data from femwell or tidy3d: for each mode, `n_eff` over
wavelength, and ideally the transverse field. You want the per-mode dispersion,
group delay, overlap weights and gamma that `MultimodeSplitStepEngine` needs.

Do not hand-build those tensors. One call produces the whole deck, and it fails
loudly when the export is not what it expects.

## Point the bridge at an export directory

The repository ships a three-mode rib export, so the example runs as written.
An export is a directory of `mode-*.npz` files plus `meta.json`.

```python
from photonics_helper.mode_analysis import build_multimode_deck

deck = build_multimode_deck(
    "examples/data/multimode_rib",
    lambda0=2.0,            # pump wavelength, um
    dispersion_order=3,     # Taylor order per mode
    n2=1e-18,               # m^2/W for the core material
)

print("keys:", sorted(deck))
print(f"modes           = {deck['n_modes']}")
print(f"gamma_reference = {deck['gamma_reference']:.4e} 1/(W m)")
print(f"oam_l           = {deck['oam_l']}")
print(f"group_delays    = {[f'{g:.2e}' for g in deck['group_delays']]}")
```

```
keys: ['A_eff', 'betas', 'coupling_tensor', 'fwm_weights', 'gamma', 'gamma_reference', 'group_delays', 'lambda0_um', 'n_modes', 'oam_l', 'phase_offsets', 'pump', 'xpm_weights']
modes           = 3
gamma_reference = 1.2346e+00 1/(W m)
oam_l           = [0, 1, -1]
group_delays    = ['0.00e+00', '-4.00e-11', '-4.34e-11']
```

`group_delays[0]` is zero by construction: the engine retards every mode into the
frame of mode 0. A nonzero first entry is the error the engine rejects, not a
physical result.

## Pass the deck straight to the engine

```python
from photonics_helper import Area, Length
from photonics_helper.gnlse import Fiber
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Time, Wave, Wavelength
import numpy as np

grid = TemporalGrid(N=2**10, Tmax=Time(1e-12, "s"))
waves = [
    Wave(
        grid=grid,
        envelope=Envelope.from_fwhm("sech", peak_amplitude=10.0, fwhm=Time(100, "fs")),
        central_wavelength=Wavelength(2000, "nm"),
    )
    for _ in range(deck["n_modes"])
]

engine = MultimodeSplitStepEngine(
    waves,
    Fiber.from_gamma(gamma=deck["gamma_reference"], length=Length(0.01, "m")),
    betas=deck["betas"],
    group_delays=deck["group_delays"],
    xpm_weights=deck["xpm_weights"],
    fwm_weights=deck["fwm_weights"],
    oam_l=deck["oam_l"],
    gamma_reference=deck["gamma_reference"],
    include_fwm=False,
)
engine.propagate(num_steps=20)
print(f"modes propagated = {len(waves)}")
print(f"gamma_reference  = {deck['gamma_reference']:.4e}")
```

```
modes propagated = 3
gamma_reference  = 1.2346e+00
```

`gamma_reference` is the load-bearing scale. Every tensor weight is normalised to
the fundamental mode's gamma, and passing it makes the deck independent of
whatever `A_eff` you put on the `Fiber`. Leave it out and the engine falls back to
the fiber area, which silently rescales every nonlinear term.

## Know it worked

Reduce the deck to one mode and check it against the scalar engine. The multimode
engine at `N=1` must reproduce `GNLSESolver` to numerical precision; if it does
not, the overlap weights or the group delay are wrong, and no amount of tuning
downstream will fix that.

## Where the background is

[Mode analysis API](../api/mode-analysis.md) lists the export format field by
field. [Multimode GNLSE](../multimode-gnlse.md) explains the coefficient models
and what `xpm_weights` and `fwm_weights` mean. The runnable script is
[few-mode, one parameter at a time](../examples/38_multimode_fewmode_gnlse.md).
