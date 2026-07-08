"""
Example: Pump Wavelength Explorer (Layer 5)
==============================================

Demonstrates the key educational concept of Raman scattering:
the frequency shift is constant, but the wavelength shift depends on
the pump wavelength.

This is the hyperbolic relationship λ = c/ν: equal frequency intervals
map to unequal wavelength intervals. The farther from the pump, the
larger the wavelength shift.

Key observations:
- Frequency axis: Anti-Stokes, Pump, Stokes are equidistant
- Wavelength axis: Stokes shift > Anti-Stokes shift (in nm)
- Sweeping pump wavelength shows nonlinear relationship

Material: Silica (440 cm⁻¹ Raman shift)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.raman import (
    RamanSpec, PumpWavelengthExplorer,
)
from photonics_helper.base import Wavelength


def main():
    # ── Create explorers for different materials ─────────────────────────────

    materials = {
        "Silica": RamanSpec.from_database("Silica"),
        "CdS": RamanSpec.from_database("CdS"),
        "Diamond": RamanSpec.from_database("Diamond"),
        "As2Se3": RamanSpec.from_database("As2Se3"),
    }

    print("Pump Wavelength Analysis — Silica @ 800 nm")
    print("=" * 70)
    pump = Wavelength(800, "nm")
    spec = materials["Silica"]
    explorer = PumpWavelengthExplorer(spec=spec)

    stokes = explorer.pump_to_stokes(pump)
    anti = explorer.pump_to_anti_stokes(pump)

    print(f"Pump:           {pump.as_nm:.1f} nm ({pump.to_freq().as_THz:.2f} THz)")
    print(f"Stokes:         {stokes.as_nm:.1f} nm ({stokes.to_freq().as_THz:.2f} THz)")
    print(f"Anti-Stokes:    {anti.as_nm:.1f} nm ({anti.to_freq().as_THz:.2f} THz)")
    print(f"Δλ (Stokes):    {stokes.as_nm - pump.as_nm:.1f} nm")
    print(f"Δλ (Anti-Stokes): {pump.as_nm - anti.as_nm:.1f} nm")
    print(f"Δν̃ (Raman):     {spec.raman_shift_cm:.0f} cm⁻¹ = {spec.raman_shift_THz:.2f} THz")
    print()

    # ── Plot 1: Frequency axis (equidistant) ────────────────────────────────

    fig = explorer.plot_frequency_axis(pump, backend="matplotlib",
                                        freq_range_THz=30,
                                        figsize=(12, 4))
    plt.savefig("examples/images/08_raman_frequency_axis.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/08_raman_frequency_axis.png")
    plt.close()

    # ── Plot 2: Wavelength axis (unequal spacing) ───────────────────────────

    fig = explorer.plot_wavelength_axis(pump, backend="matplotlib",
                                         wl_range_nm=100,
                                         figsize=(12, 4))
    plt.savefig("examples/images/08_raman_wavelength_axis.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/08_raman_wavelength_axis.png")
    plt.close()

    # ── Plot 3: Side by side — the key insight ─────────────────────────────

    fig = explorer.plot_both(pump, backend="matplotlib",
                              freq_range_THz=30, wl_range_nm=100,
                              figsize=(16, 5))
    plt.savefig("examples/images/08_raman_both_axes.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/08_raman_both_axes.png")
    plt.close()

    # ── Plot 4: Material comparison — unequal spacing varies ────────────────

    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    fig.suptitle("Raman Wavelength Shift: Material Comparison (800 nm pump)",
                 fontsize=14, fontweight="bold")

    colors = {"Silica": "#00d4ff", "CdS": "#a78bfa",
              "Diamond": "#34d399", "As2Se3": "#fbbf24"}

    for idx, (name, mat_spec) in enumerate(materials.items()):
        ax = axes[idx]
        exp = PumpWavelengthExplorer(spec=mat_spec)
        s = exp.pump_to_stokes(pump).as_nm
        a = exp.pump_to_anti_stokes(pump).as_nm
        wl_p = pump.as_nm

        # Draw axis
        ax.hlines(0, a - 20, s + 20, colors="gray", linewidth=2)
        ax.vlines(a, -0.1, 0.15, colors=colors[name], linewidth=3,
                  label=f"AS ({a:.0f} nm)")
        ax.vlines(wl_p, -0.1, 0.15, colors="#ef4444", linewidth=3,
                  label=f"Pump ({wl_p:.0f} nm)")
        ax.vlines(s, -0.1, 0.15, colors="#22c55e", linewidth=3,
                  label=f"S ({s:.0f} nm)")

        delta_s = s - wl_p
        delta_a = wl_p - a
        ax.annotate(f"Δλ_S={delta_s:.1f} nm", xy=((wl_p + s) / 2, -0.05),
                    ha="center", fontsize=8, color="#22c55e")
        ax.annotate(f"Δλ_AS={delta_a:.1f} nm", xy=((a + wl_p) / 2, -0.12),
                    ha="center", fontsize=8, color=colors[name])

        ax.set_xlim(a - 20, s + 20)
        ax.set_ylim(-0.3, 0.35)
        ax.set_title(f"{name}  (Δν̃ = {mat_spec.raman_shift_cm:.0f} cm⁻¹)",
                     fontsize=11)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig("examples/images/08_raman_material_comparison.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/08_raman_material_comparison.png")
    plt.close()

    # ── Plot 5: Sweep pump wavelength — nonlinear relationship ──────────────

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Pump Wavelength Sweep — Stokes/Anti-Stokes vs Pump",
                 fontsize=14, fontweight="bold")

    pump_range_um = (0.5, 2.5)  # 500 nm to 2500 nm
    n_points = 200

    for idx, (name, mat_spec) in enumerate(materials.items()):
        ax = axes[idx] if idx < 2 else None
        if ax is None:
            break

    # Left: Stokes wavelength vs pump
    ax1 = axes[0]
    pump_wls_um = np.linspace(pump_range_um[0], pump_range_um[1], n_points)
    for name, mat_spec in materials.items():
        exp = PumpWavelengthExplorer(spec=mat_spec)
        stokes_wls = [exp.pump_to_stokes(Wavelength(wl, "um")).as_um
                      for wl in pump_wls_um]
        ax1.plot(pump_wls_um, stokes_wls, color=colors[name], linewidth=2,
                 label=name)

    ax1.plot(pump_wls_um, pump_wls_um, color="gray", linewidth=1,
             linestyle="--", label="Identity (pump = Stokes)")
    ax1.set_xlabel("Pump Wavelength (μm)", fontsize=11)
    ax1.set_ylabel("Stokes Wavelength (μm)", fontsize=11)
    ax1.set_title("Stokes Wavelength vs Pump", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_aspect("auto")

    # Right: Anti-Stokes wavelength vs pump
    ax2 = axes[1]
    for name, mat_spec in materials.items():
        exp = PumpWavelengthExplorer(spec=mat_spec)
        anti_wls = [exp.pump_to_anti_stokes(Wavelength(wl, "um")).as_um
                    for wl in pump_wls_um]
        ax2.plot(pump_wls_um, anti_wls, color=colors[name], linewidth=2,
                 label=name)

    ax2.plot(pump_wls_um, pump_wls_um, color="gray", linewidth=1,
             linestyle="--", label="Identity (pump = Anti-Stokes)")
    ax2.set_xlabel("Pump Wavelength (μm)", fontsize=11)
    ax2.set_ylabel("Anti-Stokes Wavelength (μm)", fontsize=11)
    ax2.set_title("Anti-Stokes Wavelength vs Pump", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_aspect("auto")

    plt.tight_layout()
    plt.savefig("examples/images/08_raman_pump_sweep.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/08_raman_pump_sweep.png")
    plt.close()

    # ── Plot 6: Wavelength shift magnitude comparison ───────────────────────

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Wavelength Shift Magnitude vs Pump Wavelength",
                 fontsize=14, fontweight="bold")

    for name, mat_spec in materials.items():
        exp = PumpWavelengthExplorer(spec=mat_spec)
        delta_stokes = []
        delta_anti = []
        for wl_um in pump_wls_um:
            p = Wavelength(wl_um, "um")
            s = exp.pump_to_stokes(p).as_um
            a = exp.pump_to_anti_stokes(p).as_um
            delta_stokes.append(s - wl_um)
            delta_anti.append(wl_um - a)

        delta_stokes = np.array(delta_stokes)
        delta_anti = np.array(delta_anti)

        ax.plot(pump_wls_um, delta_stokes * 1000, color=colors[name],
                linewidth=2, label=f"{name} Stokes (nm)")
        ax.plot(pump_wls_um, delta_anti * 1000, color=colors[name],
                linewidth=2, linestyle=":", alpha=0.7,
                label=f"{name} Anti-Stokes (nm)")

    ax.set_xlabel("Pump Wavelength (μm)", fontsize=12)
    ax.set_ylabel("Wavelength Shift (nm)", fontsize=12)
    ax.set_title("Equal Frequency Shift → Unequal Wavelength Shift",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig("examples/images/08_raman_shift_magnitude.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/08_raman_shift_magnitude.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    print("\n— Key Observations —")
    print("• Frequency shift is CONSTANT for a given material (property of the molecule)")
    print("• Wavelength shift DEPENDS on pump wavelength (λ = c/ν is hyperbolic)")
    print("• Stokes shift (in nm) > Anti-Stokes shift (in nm) for visible/NIR pump")
    print("• Materials with larger Raman shift (cm⁻¹) produce larger wavelength shifts")
    print("• At 800 nm pump with Silica: Δλ_S ≈ 29 nm, Δλ_AS ≈ 27 nm")
    print("• At 1550 nm pump with Silica: Δλ_S ≈ 113 nm, Δλ_AS ≈ 99 nm")


if __name__ == "__main__":
    main()
