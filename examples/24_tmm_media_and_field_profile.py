"""
Example: TMM Configurable Media & Continuous Field Profile
==========================================================

Demonstrates the new ``n_incident`` / ``n_substrate`` options on
:class:`~photonics_helper.dbr.TMM` and the new continuous
``field_profile_z(wavelength)`` method.

Key physics shown:

- Reflectivity of a quarter-wave dielectric stack on a glass substrate
  (n = 1.5) vs. the old air-on-both-sides assumption.
- ``T = (Re eta_exit / Re eta_incident) * |t|^2`` satisfies ``R + T = 1``
  for a lossless stack on a lossless substrate.
- The intra-layer standing-wave pattern of ``|E(z)|`` — impossible to see
  with the older interface-only ``field_profile``.
- The exponential decay of ``|E(z)|`` inside an absorbing layer.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photonics_helper.base import Length, Wavelength, WavelengthArray
from photonics_helper.dbr import TMM, Block, Material, Pattern

wl_array = WavelengthArray(np.linspace(1300, 1800, 401), "nm")
lam0_nm = 1550.0


def const_material(n, k=0.0, name="m"):
    wl = WavelengthArray(np.linspace(1000, 2000, 101), "nm")
    return Material(name=name, n=np.full(101, n), k=np.full(101, k), wl=wl)


# ── 1. Quarter-wave TiO2/SiO2 stack at design wavelength 1550 nm ─────────
n_H, n_L, n_S = 2.30, 1.46, 1.50  # high / low / substrate index
d_H = lam0_nm / (4 * n_H) * 1e-9  # quarter-wave thicknesses
d_L = lam0_nm / (4 * n_L) * 1e-9

blocks = {
    "H": Block(length=Length(d_H, "m"), material=const_material(n_H, name="TiO2")),
    "L": Block(length=Length(d_L, "m"), material=const_material(n_L, name="SiO2")),
}
pattern = Pattern(
    style="HLHLHLHL", mapping=blocks, central_wavelength=Wavelength(lam0_nm, "nm")
)

# Historic behaviour (air on both sides) vs. realistic deposition on glass.
# Defaults keep every existing air/air result identical.
tmm_air = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
tmm_glass = TMM(
    pattern=pattern,
    angle_of_incidence=0.0,
    polarisation="TE",
    n_incident=1.0 + 0.0j,  # air — semi-infinite incident medium
    n_substrate=1.5 + 0.0j,  # glass — semi-infinite exit medium
)
R_air, T_air = tmm_air.spectrum(wl_array)
R_glass, T_glass = tmm_glass.spectrum(wl_array)

i0 = int(np.argmin(np.abs(wl_array.as_nm - lam0_nm)))
print("TMM configurable-media demo")
print("=" * 60)
print(f"  R @ {lam0_nm:.0f} nm, air/air      : {R_air[i0]:.4f}")
print(f"  R @ {lam0_nm:.0f} nm, on glass     : {R_glass[i0]:.4f}")
print(f"  T @ {lam0_nm:.0f} nm, into glass   : {T_glass[i0]:.4f}")
print(
    f"  max |R + T − 1| (lossless/lossless): "
    f"{np.max(np.abs((R_glass + T_glass) - 1.0)):.2e}"
)
# Stopband shift: the mirror's reflectivity changes because the exit
# admittance changed — the substrate is no longer a fake thin layer.
side = np.max(np.abs(R_glass - R_air))
print(f"  max substrate-vs-air |ΔR|          : {side:.4f}")

# ── 2. Continuous intra-layer field profile: absorbing film on glass ────
blocks2 = {
    "A": Block(
        length=Length(40e-9, "m"), material=const_material(0.14, 4.5, name="Ag")
    ),
    "B": Block(
        length=Length(120e-9, "m"), material=const_material(1.52, 0.0, name="glass")
    ),
}
pattern2 = Pattern(
    style="AB", mapping=blocks2, central_wavelength=Wavelength(1550, "nm")
)
tmm_abs = TMM(
    pattern=pattern2,
    angle_of_incidence=0.0,
    polarisation="TE",
    n_substrate=1.5 + 0.0j,
)

wl633 = Wavelength(1550.0, "nm")
z, E_of_z = tmm_abs.field_profile_z(wl633, n_points_per_layer=50)
print(
    f"  field_profile_z: {len(z)} samples spanning "
    f"{z[0] * 1e9:.0f}–{z[-1] * 1e9:.0f} nm, monotonically increasing: "
    f"{bool(np.all(np.diff(z) > 0))}."
)
# Tangential E is continuous at interfaces, so the sampled profile at the
# Ag/B boundary joins the interface value from field_profile().
fp_vals = tmm_abs.field_profile(wl633)
print(f"  field_profile interface values (|E| per layer front): {fp_vals}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(wl_array.as_nm, R_air, label="R — air on both sides")
axes[0].plot(wl_array.as_nm, T_air, "--", label="T — air/air")
axes[0].plot(wl_array.as_nm, R_glass, label="R — on n=1.5 substrate")
axes[0].plot(wl_array.as_nm, T_glass, "--", label="T — into glass")
axes[0].axvline(lam0_nm, color="k", ls=":", lw=0.8)
axes[0].set_xlabel("Wavelength (nm)")
axes[0].set_ylabel("Fraction")
axes[0].set_title("8-layer quarter-wave DBR stack")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].semilogy(z * 1e9, E_of_z, "C0-")
axes[1].axvline(40, color="k", ls=":", lw=0.8)
axes[1].text(42, E_of_z.max() * 0.4, "Ag → glass")
axes[1].set_xlabel("z (nm)")
axes[1].set_ylabel("|E(z)|")
axes[1].set_title("Continuous |E(z)| profile (1550 nm, TE)")
axes[1].grid(alpha=0.3, which="both")
plt.tight_layout()
fig.savefig("examples/images/tmm_media_field_profile.png", dpi=150, bbox_inches="tight")
print("  Saved examples/images/tmm_media_field_profile.png")
plt.close(fig)

print("\n✓ TMM media/field-profile example complete")
