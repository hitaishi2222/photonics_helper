"""
Example: Material Comparison (Layer 6)
=======================================

Compare Raman properties across multiple materials: spectra overlay,
time-domain response overlay, frequency-domain gain spectrum overlay,
and a comparison table.

This layer answers: "Which material is best for my Raman application?"
by letting you visually compare materials side by side.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.raman import (
    MaterialComparison, RamanSpec, RamanResponse,
    COMMON_COMPARISONS,
)
from photonics_helper.pulse import TemporalGrid
from photonics_helper.base import Time


def main():
    # ── 1. Build a comparison: glass vs chalcogenide ────────────────────────

    print("=== Glass vs Chalcogenide ===")
    comp = MaterialComparison()
    for name in COMMON_COMPARISONS["glass_vs_chalcogenide"]:
        comp.add(RamanSpec.from_database(name))

    print(comp.comparison_table())
    print()

    # Spectra overlay
    fig = comp.plot_spectra_overlay(backend="matplotlib", shift_range_cm=600)
    plt.savefig("examples/images/09_raman_spectra_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/09_raman_spectra_overlay.png")
    plt.close()

    # ── 2. Semiconductor comparison ─────────────────────────────────────────

    print("\n=== Semiconductor Comparison ===")
    comp2 = MaterialComparison()
    for name in COMMON_COMPARISONS["semiconductor"]:
        comp2.add(RamanSpec.from_database(name))

    print(comp2.comparison_table())

    # Frequency overlay — shows gain spectra
    grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
    fig = comp2.plot_frequency_overlay(backend="matplotlib", grid=grid)
    plt.savefig("examples/images/09_raman_frequency_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/09_raman_frequency_overlay.png")
    plt.close()

    # ── 3. Full comparison: all three panels ────────────────────────────────

    print("\n=== Full Comparison: Silica, CdS, As2Se3 ===")
    comp3 = MaterialComparison()
    for name in ["Silica", "CdS", "As2Se3"]:
        comp3.add(RamanSpec.from_database(name))

    # Time-domain response overlay
    fig = comp3.plot_response_overlay(backend="matplotlib", grid=grid)
    plt.savefig("examples/images/09_raman_response_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/09_raman_response_overlay.png")
    plt.close()

    # ── 4. All-in-one: 3-panel comparison ───────────────────────────────────

    fig = comp3.plot_all(backend="matplotlib", grid=grid, figsize=(12, 12))
    plt.savefig("examples/images/09_raman_material_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/09_raman_material_comparison.png")
    plt.close()

    # ── 5. Programmatic comparison ──────────────────────────────────────────

    print("\n=== Programmatic: High-n₂ Materials ===")
    high_n2 = COMMON_COMPARISONS["high_n2"]
    n2_ranking = []
    for name in high_n2:
        spec = RamanSpec.from_database(name)
        n2_ranking.append((name, spec.n2 or 0))

    n2_ranking.sort(key=lambda x: x[1], reverse=True)
    for name, n2 in n2_ranking:
        print(f"  {name}: n₂ = {n2:.2e} m²/W")

    # ── 6. Add/remove/demo workflow ─────────────────────────────────────────

    print("\n=== Interactive Workflow Demo ===")
    comp_demo = MaterialComparison()

    # Start with one material
    comp_demo.add(RamanSpec.from_database("Silica"))
    print(f"After adding Silica: {len(comp_demo.materials)} material(s)")

    # Add more
    comp_demo.add(RamanSpec.from_database("CdS"))
    comp_demo.add(RamanSpec.from_database("Diamond"))
    print(f"After adding CdS + Diamond: {len(comp_demo.materials)} material(s)")

    # Replace Silica with custom data
    custom_silica = RamanSpec(
        name="Silica",
        raman_shift_cm=440,
        raman_linewidth_cm=45,
        fR=0.25,  # higher than default
    )
    comp_demo.add(custom_silica)
    print(f"After replacing Silica: {len(comp_demo.materials)} material(s), "
          f"fR={comp_demo.materials[0].fR}")

    # Remove one
    comp_demo.remove("Diamond")
    print(f"After removing Diamond: {len(comp_demo.materials)} material(s)")
    print(comp_demo.comparison_table())

    # Clear all
    comp_demo.clear()
    print(f"After clear: {len(comp_demo.materials)} material(s)")


if __name__ == "__main__":
    main()
