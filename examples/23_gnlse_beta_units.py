"""
Example: GNLSE Beta-Coefficient Units (`betas_unit`)
====================================================

Demonstrates the explicit dispersion-unit contract added to
``SplitStepEngine`` / ``GNLSESolver`` / ``TaperedGNLSESolver``.

``betas`` may now be supplied in either of two explicitly named units:

- ``"ps^k/m"``  (default) — the native internal convention (`Ω` in rad/ps),
  as returned by ``Dispersion.get_betas()``.
- ``"s^k/m"`` or ``"SI"`` — SI Taylor coefficients.

The ``betas_unit`` flag converts the SI form into ps^k/m at construction time
(`β_k[s^k/m] · 10^{12k}`), so both routes describe the *same physical
propagation*. Invalid inputs (unknown unit string, NaN, non-finite, wrong
shape) now raise a clear error instead of silently producing a factor-10^24
wrong pulse.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


# ── 1. The same physical fiber, expressed two ways ───────────────────────
# beta2 = -2.0 ps^2/m  = -2.0e-24 s^2/m
# beta3 = -0.05 ps^3/m = -0.05e-36 s^3/m  (1 ps^3 = 1e-36 s^3!)
betas_ps = np.array([-2.0, -0.05])
betas_si = np.array([-2.0e-24, -0.05e-36])

fiber = FiberProfile(
    n2=3.0e-20,
    alpha=0.0,
    A_eff=Area(80e-12, "m^2"),
    length=Length(0.05, "m"),  # 5 cm
)


def make_pulse() -> Wave:
    grid = TemporalGrid(N=2**10, Tmax=Time(5e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(100e-15, "s"))
    return Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )


# ── 2. Propagate identically with both unit conventions ──────────────────
solver_ps = GNLSESolver(
    pulse=make_pulse(), fiber=fiber, betas=betas_ps, include_raman=False
)
solver_si = GNLSESolver(
    pulse=make_pulse(),
    fiber=fiber,
    betas=betas_si,
    betas_unit="s^k/m",  # equivalently "SI"
    include_raman=False,
)

solver_ps.propagate(num_steps=60)
solver_si.propagate(num_steps=60)

print("Beta-unit contract demo")
print("=" * 60)
print(f"  solver_ps.betas (native ps^k/m)                : {solver_ps.betas}")
print(f"  solver_si.betas (converted internally)         : {solver_si.betas}")
print(
    f"  → classes hold identical values: "
    f"{np.allclose(solver_ps.betas, solver_si.betas)}"
)

# Normalized spectra must agree to float precision
spec_ps = np.fft.fftshift(solver_ps.evolution[-1].spectrum)
spec_si = np.fft.fftshift(solver_si.evolution[-1].spectrum)
print(f"  spectral agreement |max| : {np.max(np.abs(spec_ps - spec_si)):.3e}")

# ── 3. Invalid inputs now fail loudly ────────────────────────────────────
for label, kwargs in [
    ("unknown unit nm", {"betas": betas_ps, "betas_unit": "nm"}),
    ("NaN coefficient", {"betas": np.array([-2.0, np.nan])}),
    ("Inf coefficient", {"betas": np.array([-2.0, np.inf]), "betas_unit": "SI"}),
]:
    try:
        GNLSESolver(pulse=make_pulse(), fiber=fiber, include_raman=False, **kwargs)
    except (ValueError, TypeError) as exc:
        print(f"  rejected {label}: {type(exc).__name__}: {str(exc)[:70]}...")

# ── 4. Visual check: both routes produce the same propagation ────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
omega, spectra = solver_ps.spectra_vs_z
wl = 2 * np.pi * 299792458.0 / (omega + solver_ps.omega0) * 1e9
o = np.argsort(wl)
axes[0].pcolormesh(
    wl[o],
    solver_ps.z_array * 1e3,
    10 * np.log10(spectra[:, o] + 1e-30),
    shading="auto",
    cmap="hot",
)
axes[0].set_xlabel("Wavelength (nm)")
axes[0].set_ylabel("Distance (mm)")
axes[0].set_title("betas in ps^k/m (default)")

_, spectra2 = solver_si.spectra_vs_z
axes[1].pcolormesh(
    wl[o],
    solver_si.z_array * 1e3,
    10 * np.log10(spectra2[:, o] + 1e-30),
    shading="auto",
    cmap="hot",
)
axes[1].set_xlabel("Wavelength (nm)")
axes[1].set_title('betas in s^k/m with betas_unit="s^k/m"')
plt.tight_layout()
fig.savefig("examples/images/gnlse_beta_units.png", dpi=150, bbox_inches="tight")
print("  Saved examples/images/gnlse_beta_units.png")
plt.close(fig)

print("\n✓ Beta-unit example complete")
