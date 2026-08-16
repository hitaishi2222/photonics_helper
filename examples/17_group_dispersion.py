"""
Example: Group Dispersion (n_g, v_g)
======================================

Demonstrates the group index and group velocity methods on `RefractiveIndex`.

Group index:    n_g(λ) = n(λ) − λ · dn/dλ
Group velocity: v_g(λ) = c / n_g(λ)

These quantify how fast a pulse envelope (vs. the carrier) travels through a
medium. Crucial for GNLSE solvers, dispersive-wave calculations, and
ultrafast optics.

This example:
  1. Builds silica from its Sellmeier coefficients
  2. Plots n, n_g, and v_g vs. wavelength
  3. Shows dn/dλ (material dispersion)
  4. Compares with a constant-n material (no dispersion → n_g = n)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.materials import RefractiveIndex
from photonics_helper.base import WavelengthArray


def main():
    # ── 1. Silica from Sellmeier ─────────────────────────────────────────────

    A0 = 1.0
    A = [0.6961663, 0.4079426, 0.8974794]
    B = [0.0684043, 0.1162414, 9.896161]

    silica = RefractiveIndex.from_sellmeier(
        A0=A0, A=A, B=B, wl_from_to_in_um=(0.5, 2.0), n_points=500
    )

    print("=== Silica Group Dispersion ===")
    print(f"  n(1.0 μm)       = {silica.n_func(1.0):.6f}")
    print(f"  dn/dλ(1.0 μm)   = {silica.dn_dlambda(1.0):.6e} μm⁻¹")
    print(f"  n_g(1.0 μm)     = {silica.group_index(1.0):.6f}")
    print(f"  v_g(1.0 μm)     = {silica.group_velocity(1.0):.2e} m/s")
    print(f"  v_g / c         = {silica.group_velocity(1.0) / 2.998e8:.6f}")
    print()

    # ── 2. Constant-n reference (no dispersion) ──────────────────────────────

    wl_um = np.linspace(0.5, 2.0, 500)
    n_const = np.full_like(wl_um, 2.0)
    flat = RefractiveIndex(n=n_const, k=np.zeros_like(wl_um), wl=WavelengthArray(wl_um, "um"))

    print("=== Constant n=2.0 (no dispersion) ===")
    print(f"  n(1.0 μm)       = {flat.n_func(1.0):.6f}")
    print(f"  dn/dλ(1.0 μm)   = {flat.dn_dlambda(1.0):.6e} μm⁻¹")
    print(f"  n_g(1.0 μm)     = {flat.group_index(1.0):.6f}  (equals n)")
    print(f"  v_g(1.0 μm)     = {flat.group_velocity(1.0):.2e} m/s")
    print()

    # ── 3. Plots ─────────────────────────────────────────────────────────────

    wl_um = silica.wl.as_um
    n_arr = silica.n
    dndl_arr = silica._dn_spline(wl_um)
    ng_arr = silica.group_index_array()
    vg_arr = silica.group_velocity_array()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Refractive index n(λ)
    axes[0, 0].plot(wl_um, n_arr, label="n(λ)", color="tab:blue", linewidth=1.5)
    axes[0, 0].set_xlabel("Wavelength (μm)")
    axes[0, 0].set_ylabel("n")
    axes[0, 0].set_title("Refractive index")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (b) dn/dλ (material dispersion)
    axes[0, 1].plot(wl_um, dndl_arr, label="dn/dλ", color="tab:orange", linewidth=1.5)
    axes[0, 1].set_xlabel("Wavelength (μm)")
    axes[0, 1].set_ylabel("dn/dλ (μm⁻¹)")
    axes[0, 1].set_title("Material dispersion")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (b) Group index n_g(λ)
    axes[1, 0].plot(wl_um, ng_arr, label="n_g(λ)", color="tab:green", linewidth=1.5)
    axes[1, 0].plot(wl_um, n_arr, "--", label="n(λ)", color="tab:blue", linewidth=1)
    axes[1, 0].set_xlabel("Wavelength (μm)")
    axes[1, 0].set_ylabel("n_g")
    axes[1, 0].set_title("Group index (n_g = n − λ·dn/dλ)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (c) Group velocity v_g(λ)
    c = 2.99792458e8
    axes[1, 1].plot(wl_um, vg_arr, label="v_g(λ)", color="tab:red", linewidth=1.5)
    axes[1, 1].axhline(c, color="gray", linestyle="--", label="c (vacuum)", linewidth=1)
    axes[1, 1].set_xlabel("Wavelength (μm)")
    axes[1, 1].set_ylabel("v_g (m/s)")
    axes[1, 1].set_title("Group velocity")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/images/17_group_dispersion.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/images/17_group_dispersion.png")

    # ── 4. Scalar lookup at a specific wavelength ────────────────────────────

    wl_probe = 1.55  # telecom C-band
    print(f"\n=== Probe at λ = {wl_probe} μm ===")
    print(f"  n        = {silica.n_func(wl_probe):.6f}")
    print(f"  dn/dλ    = {silica.dn_dlambda(wl_probe):.6e} μm⁻¹")
    print(f"  n_g      = {silica.group_index(wl_probe):.6f}")
    print(f"  v_g      = {silica.group_velocity(wl_probe):.2e} m/s")
    print(f"  v_g/c    = {silica.group_velocity(wl_probe) / c:.6f}")

    # ── 5. Error handling demo ───────────────────────────────────────────────

    print("\n=== Error handling ===")
    try:
        silica.group_index(5.0)
    except ValueError as e:
        print(f"  Out-of-range: {e}")


if __name__ == "__main__":
    main()
