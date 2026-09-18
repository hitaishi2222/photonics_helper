"""
Example: Structured light — Laguerre-Gaussian / OAM modes
=========================================================

Demonstrates :mod:`photonics_helper.structured`:

- analytic Laguerre-Gaussian modes ``LG(p, l)`` carrying orbital angular
  momentum (OAM),
- the modal overlap integral and the orthonormality of the modes,
- the ``2 pi l`` phase winding of an OAM mode around the optical axis,
- Gaussian-beam waist evolution and the Gouy phase,
- transverse intensity/phase plots.

Usage::

    python examples/28_structured_light.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from photonics_helper.structured import (
    LaguerreGaussianMode,
    beam_waist,
    gouy_phase,
    overlap,
    plot_transverse_profile,
    rayleigh_range,
)

W0 = 1e-3  # 1 mm waist
WAVELENGTH = 1064e-9  # 1064 nm (Nd:YAG)
Z_R = rayleigh_range(W0, WAVELENGTH)

MODES = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]


def lg(p: int, l: int) -> LaguerreGaussianMode:
    return LaguerreGaussianMode(p, l, W0, WAVELENGTH)


print("Structured light — Laguerre-Gaussian / OAM modes")
print("=" * 62)
print(
    f"  w0 = {W0 * 1e6:.2f} um, lambda = {WAVELENGTH * 1e9:.0f} nm, "
    f"z_R = {Z_R * 1e3:.3f} mm"
)

# ── 1. Orthonormality of the modes ─────────────────────────────────────
print("\n  Overlap matrix  <LG(p,l) | LG(p',l')>")
header = "        " + "".join(f"{f'({p},{l})':>10}" for p, l in MODES)
print(header)
for p, l in MODES:
    row = "".join(f"{overlap(lg(p, l), lg(p2, l2)).real:10.4f}" for p2, l2 in MODES)
    print(f"  ({p},{l}) {row}")

# ── 2. OAM phase winding around the axis ───────────────────────────────
print("\n  OAM phase winding on a closed loop (r = 2 w0)")
phi = np.linspace(0.0, 2.0 * np.pi, 2881)
for l in (1, 2, 3, -1):
    values = lg(0, l).evaluate(2 * W0 * np.cos(phi), 2 * W0 * np.sin(phi))
    winding = float(np.unwrap(np.angle(values))[-1] - np.unwrap(np.angle(values))[0])
    print(
        f"    l = {l:+d}: winding = {winding:+.4f} rad (2 pi l = {2 * np.pi * l:+.4f})"
    )

# ── 3. Propagation: waist evolution and Gouy phase ─────────────────────
print("\n  Beam propagation of LG(1,1) (Gouy order 2p+|l|+1 = 4)")
print("      z/z_R     w(z) [um]     Gouy [rad]")
for ratio in (0.0, 0.5, 1.0, 2.0, 5.0):
    z = ratio * Z_R
    w = beam_waist(W0, z, WAVELENGTH)
    print(f"    {ratio:6.2f}      {w * 1e6:9.3f}     {gouy_phase(4, z, Z_R):+.4f}")

# ── 4. Transverse intensity / phase map of one OAM mode ────────────────
fig_lg = plot_transverse_profile(lg(1, 3), title="LG(p=1, l=3) intensity and phase")
fig_lg.savefig(
    "examples/images/28_structured_light_profile.png", dpi=150, bbox_inches="tight"
)
print("\n  Saved examples/images/28_structured_light_profile.png")
plt.close(fig_lg)

# ── 5. Montage of mode intensities at the waist ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
for ax, (p, l) in zip(axes.ravel(), MODES):
    field = lg(p, l).structured()
    extent_um = tuple(v * 1e6 for v in field.extent)
    im = ax.imshow(
        field.intensity,
        origin="lower",
        extent=extent_um,
        cmap="inferno",
        aspect="equal",
    )
    ax.set_title(f"LG(p={p}, l={l})  |  OAM = {l} hbar")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax, shrink=0.85)
fig.suptitle("Laguerre-Gaussian mode intensities at the beam waist")
fig.savefig(
    "examples/images/28_structured_light_modes.png", dpi=150, bbox_inches="tight"
)
print("  Saved examples/images/28_structured_light_modes.png")
plt.close(fig)

print("\n" + "=" * 62)
print("✓ Structured-light example complete")
