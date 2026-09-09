"""
Example: Material Catalog Explorer
=====================================

Loads all 30 materials from the database, prints a terminal summary,
and generates plots organized by material category.

Highlights:
- Materials ranked by Raman shift, n2, and gain coefficient
- Category-based overlays from COMMON_COMPARISONS
- Side-by-side spectra for different material families
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.raman import (
    RamanSpec, RamanResponse, MaterialComparison, COMMON_COMPARISONS,
)
from photonics_helper.pulse import TemporalGrid
from photonics_helper.base import Time


def print_table(materials, title="Material Properties"):
    """Print a formatted table of material properties."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Name':<12s} {'Shift':>7s} {'FWHM':>7s} {'n2':>12s} {'fR':>6s} {'Q':>7s}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*12} {'-'*6} {'-'*7}")
    for spec in materials:
        n2_str = f"{spec.n2:.1e}" if spec.n2 else "N/A"
        fr_str = f"{spec.fR:.2f}" if spec.fR is not None else "N/A"
        print(f"  {spec.name:<12s} {spec.raman_shift_cm:>7.0f} "
              f"{spec.raman_linewidth_cm:>7.0f} {n2_str:>12s} "
              f"{fr_str:>6s} {spec.quality_factor:>7.1f}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*12} {'-'*6} {'-'*7}")
    print(f"  Total: {len(materials)} materials")
    print()


def main():
    # ── 1. Load all 30 materials ──────────────────────────────────────────

    material_names = [
        "Silica", "As2S3", "As2Se3", "GeO2", "ZBLAN",          # glasses
        "Si3N4", "GaN", "AlN", "SiC_4H", "Ga2O3",              # wide-gap
        "GaAs", "InP", "AlGaAs", "InGaAs",                      # III-V
        "CdS", "CdTe", "ZnO", "Ge", "Si",                      # II-VI / elemental
        "Diamond", "YAG", "Al2O3", "YLF",                       # crystals / hosts
        "LiNbO3", "LiTaO3", "KTP", "BaTiO3",                   # ferroelectrics
        "LBO", "AgGaS2", "AgGaSe2",                             # NLO
    ]
    all_materials = {}
    for name in material_names:
        all_materials[name] = RamanSpec.from_database(name)

    # ── 2. Terminal summary table ───────────────────────────────────────

    print_table(list(all_materials.values()), "Photonics Helper Material Catalog")

    # Ranked highlights
    by_shift = sorted(all_materials.values(), key=lambda s: s.raman_shift_cm, reverse=True)
    print("  Ranked by Raman shift (cm⁻¹):")
    for i, spec in enumerate(by_shift[:5]):
        print(f"    {i+1}. {spec.name}: {spec.raman_shift_cm} cm⁻¹")
    print()

    by_n2 = [s for s in all_materials.values() if s.n2 is not None]
    by_n2.sort(key=lambda s: abs(s.n2 or 0), reverse=True)
    print("  Ranked by |n2| (m²/W):")
    for i, spec in enumerate(by_n2[:5]):
        print(f"    {i+1}. {spec.name}: n2 = {spec.n2:.2e}")
    print()

    by_gain = [s for s in all_materials.values() if s.gain_coeff is not None]
    by_gain.sort(key=lambda s: s.gain_coeff or 0, reverse=True)
    print("  Materials with gain_coeff data:")
    for i, spec in enumerate(by_gain):
        print(f"    {i+1}. {spec.name}: {spec.gain_coeff} m/GW")
    print()

    # ── 3. Plot: category-comparison panels ─────────────────────────────

    grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle("Photonics Helper — Material Category Comparison",
                 fontsize=15, fontweight="bold")

    categories = [
        ("Glasses / Fibers", ["Silica", "As2S3", "As2Se3", "GeO2", "ZBLAN"]),
        ("Wide-Bandgap Semiconductors", ["GaN", "AlN", "SiC_4H", "Ga2O3", "Si3N4"]),
        ("Elemental / III-V Semiconductors", ["Si", "Ge", "GaAs", "InP", "InGaAs"]),
        ("Nonlinear / Ferroelectric Crystals", ["LiNbO3", "KTP", "BaTiO3", "LiTaO3"]),
        ("Laser Host / Raman Crystals", ["YAG", "Al2O3", "YLF", "Diamond"]),
        ("NLO Crystals", ["LBO", "AgGaS2", "AgGaSe2"]),
    ]

    for idx, (title, names) in enumerate(categories):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        for name in names:
            spec = all_materials[name]
            resp = RamanResponse(spec=spec, grid=grid)
            delayed = resp.delayed_response()
            t_ps = resp.grid.t * 1e12
            mask = (t_ps >= 0) & (t_ps <= 5)
            ax.plot(t_ps[mask], delayed[mask], linewidth=1.5,
                    label=f"{name}  ({spec.raman_shift_cm} cm⁻¹, fR={spec.fR})")

        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("h_R(t) (arb.)")
        ax.set_title(f"{title}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig("examples/images/11_raman_material_catalog.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/images/11_raman_material_catalog.png")
    plt.close()

    # ── 4. Plot: overlay spectra for COMMON_COMPARISONS groups ──────────

    for group_name, mat_names in sorted(COMMON_COMPARISONS.items()):
        comp = MaterialComparison()
        for name in mat_names:
            comp.add(RamanSpec.from_database(name))

        safe_name = group_name.replace("_", "_")
        fig = comp.plot_spectra_overlay(backend="matplotlib", shift_range_cm=2000)
        fig.suptitle(f"Spectra: {group_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"examples/images/11_raman_comparison_spectra_{safe_name}.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = comp.plot_response_overlay(backend="matplotlib", grid=grid)
        fig.suptitle(f"Response: {group_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"examples/images/11_raman_comparison_response_{safe_name}.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = comp.plot_frequency_overlay(backend="matplotlib", grid=grid)
        fig.suptitle(f"Frequency: {group_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"examples/images/11_raman_comparison_freq_{safe_name}.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved: {group_name} (3 plots)")

    print("\nDone. See examples/images/ for all plots.")


if __name__ == "__main__":
    main()
