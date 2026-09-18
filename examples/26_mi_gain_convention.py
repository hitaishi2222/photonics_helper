"""
Example: Modulation-Instability Gain (Corrected Convention)
===========================================================

Demonstrates the corrected modulation-instability (MI) gain in
``photonics_helper.phase_matching``.

The exact linear-stability result for the codebase's NLSE convention
(``i∂A/∂z = (β₂/2)∂²A/∂T² − γ|A|²A``) is (Agrawal, Nonlinear Fiber Optics,
Eq. 5.1.9):

    g(Ω) = |β₂ Ω| √(Ω_c² − Ω²),   Ω_c² = 4γP/|β₂|,
    Ω_peak = Ω_c/√2,              g_max = 2γP,

with sideband detunings given by ``β₂Ω² + 4γP = 0``. (The previous versions
used √2-scaled cutoff/peak frequencies while keeping the same peak height.)

Also demonstrates the exact extended (full β(ω)) gain
``g(Ω) = √[−Δ(Ω)(Δ(Ω) + 4γP)]`` with the un-doubled even dispersion mismatch
``Δ(Ω) = β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)``, which reduces *exactly* to the
classical result under a Taylor expansion of β(ω).
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from photonics_helper.phase_matching import (
    mi_gain_spectrum,
    mi_gain_spectrum_extended,
    mi_sideband_frequencies,
)

# ── Physics setup: SMF-28-like fiber pumped in the anomalous regime ─────
beta2 = -20.0e-27  # s^2/m  (−20 fs²/m)
gamma = 5.0e-3  # 1/(W·m)
P0 = 20.0  # W pump power

print("Modulation-instability gain (corrected convention)")
print("=" * 60)

# 1. Closed forms the solver must reproduce
Omega_c = np.sqrt(4 * gamma * P0 / abs(beta2))
Omega_peak = np.sqrt(2 * gamma * P0 / abs(beta2))
g_max = 2 * gamma * P0
print(f"  Ω_c     = {Omega_c:.4e} rad/s  (= √(4γP/|β₂|))")
print(f"  Ω_peak  = {Omega_peak:.4e} rad/s  (= Ω_c/√2)")
print(f"  g_max   = {g_max:.4f} 1/m       (= 2γP)")

# 2. Solver's summary dict
info = mi_gain_spectrum(beta2, gamma, P0)
print(
    f"  solver  : Ω_c = {info['Omega_cutoff']:.4e}, "
    f"Ω_peak = {info['Omega_peak']:.4e}, g_max = {info['g_max']:.4f}"
)
assert np.isclose(info["Omega_cutoff"], Omega_c)
assert np.isclose(info["Omega_peak"], Omega_peak)
assert np.isclose(info["g_max"], g_max)

# 3. Sideband frequencies and their wavelengths relative to a 1550 nm pump
omega_sb = mi_sideband_frequencies(beta2, gamma, P0)
lam0 = 1550e-9
lam_side = (
    2 * np.pi * 299792458.0 / (2 * np.pi * 299792458.0 / lam0 + omega_sb[1]) * 1e9
)
print(f"  sideband detuning       : ±{omega_sb[1]:.4e} rad/s")
print(f"  sideband wavelength     : {lam_side:.2f} nm (pump 1550 nm)")

# 4. Full gain curve + the extended full-β(ω) version (pure β₂ Taylor case)
Omega = np.linspace(0, 1.5 * Omega_c, 800)
gain = mi_gain_spectrum(beta2, gamma, P0, Omega)


omega0 = 2 * np.pi * 299792458.0 / lam0
# NOTE (numerics): the extended solver evaluates beta_fn at ω₀ ± Ω with a
# carrier ω₀ ≈ 1.2e15 rad/s; float64 gives it a ~0.25 rad/s ULP, so a β_fn
# that returns the *absolute* β(ω) including the 2πc/λ₀ offset swamps the
# tiny mismatch Δ(Ω) ~ 0.1 rad/s with round-off (catastrophic cancellation).
# The β₁ offset cancels analytically in Δ, so supply β_fn *without* that
# linear term to make the mismatch resolvable — this is a user-side
# workaround for a known library numerical caveat.
ext = mi_gain_spectrum_extended(
    beta_fn=lambda w: 0.5 * beta2 * (w - omega0) ** 2,  # Δ carries the signal
    omega0=omega0,
    gamma=gamma,
    P=P0,
    omega_m=Omega,
)

print(
    f"  extended vs classical   : max |\u0394g| = "
    f"{np.max(np.abs(ext['gain'] - gain)):.2e} (exact reduction)"
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Omega * 1e-12, gain, lw=2, label="classical $|β₂Ω|√(Ω_c²−Ω²)$")
ax.plot(
    ext["omega_m"] * 1e-12,
    ext["gain"],
    "--",
    lw=2,
    label="extended (full $β(ω)$, Taylor)",
)
ax.axvline(Omega_c * 1e-12, color="k", ls=":", lw=0.8, label=r"$Ω_c$")
ax.axvline(
    Omega_peak * 1e-12, color="C3", ls=":", lw=0.8, label=r"$Ω_{peak} = Ω_c/\sqrt{2}$"
)
ax.axhline(g_max, color="C2", ls=":", lw=0.8, label=r"$g_{max}=2γP$")
ax.set_xlabel("Ω (rad/ps)")
ax.set_ylabel("MI gain g (1/m)")
ax.set_title("MI gain — exact linear-stability convention (Agrawal 5.1.9)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig("examples/images/mi_gain_convention.png", dpi=150, bbox_inches="tight")
print("  Saved examples/images/mi_gain_convention.png")
plt.close(fig)

print("\n✓ MI-gain example complete")
