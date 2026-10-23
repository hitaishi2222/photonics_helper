# Quick start

> Not sure which engine to use? See **[Choosing a solver](choose-a-solver.md)**.

## Units

```python
from photonics_helper import Wavelength

wl = Wavelength(1550, "nm")
print(wl.to_freq().as_THz)  # 193.4 THz
print(wl.to_energy().as_eV)  # 0.8 eV
```

## GNLSE propagation

```python
import numpy as np
from photonics_helper.base import Area, Length, Power, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver
from photonics_helper.pulse import Envelope, Wave

# Single pulse — grid is built automatically (~10x T0 window).
# Prefer physical peak power over raw V/m amplitude:
envelope = Envelope.from_fwhm_peak_power(
    "sech", Power(1, "kW"), Time(50, "fs"), Area(80, "um^2")
)
pulse = Wave.from_envelope(envelope, Wavelength(1550, "nm"))
# Pulse train — grid + spacing derived from the repetition rate:
# Wave.from_pulse_train(envelope, Wavelength(1550, "nm"),
#                        repetition_rate=Frequency(10, "GHz"), n_pulses=5)
# Gamma-first fiber — no n2 needed when gamma is known:
# Fiber.from_gamma(0.001, A_eff=Area(80, "um^2"), length=Length(0.1, "m"))
fiber = Fiber(n2=2.6e-20, alpha=0.0, A_eff=Area(80, "um^2"), length=Length(0.1, "m"))
solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=np.array([-0.02, 1e-4]))
solver.propagate(num_steps=200)
```

> ``FiberProfile`` is a deprecated alias of ``Fiber``; ``Wave.from_pulse_train``
> still works but ``grid`` and (for single pulses) ``repetition_rate`` are now
> optional, and ``ElectricField.from_array`` is the raw-samples constructor
> (``from_envelope`` takes an ``Envelope``).

## Next steps

- [API Reference](api/index.md) — one page per module, rendered from docstrings.
- [Raman](api/raman.md) and [χ⁽²⁾](api/chi2.md) pages for those workflows.
