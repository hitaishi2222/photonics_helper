#!/usr/bin/env python3
"""Seed materials.db from the hardcoded RAMAN_MATERIALS and SELLMEIER_MATERIALS dicts.

Run from the project root:
    python seed_db.py          # seeds raman_specs (existing)
    python seed_db.py --nk     # also seeds nk_data + sellmeier tables
"""

import argparse
from pathlib import Path
from photonics_helper.raman import RAMAN_MATERIALS, RamanDatabase

DB_PATH = Path(__file__).parent / "photonics_helper" / "materials.db"

# Sellmeier coefficients for 20+ optical materials
# Form: standard = n² = A₀ + Σ Aᵢλ²/(λ² - Bᵢ)
# Form: alt = n² = A₀ + Σ Aᵢ/(λ² - Bᵢ²)
SELLMEIER_MATERIALS = {
    "Silica": {
        "form": "standard",
        "a0": 0.6961663,
        "coefficients": [0.4079425, 0.1218212, 0.0031735],
        "wavelengths": [0.0690660, 0.1156491, 9.900559],
        "valid_from_um": 0.21,
        "valid_to_um": 6.7,
        "source": "Thorlabs, Corning",
    },
    "Si": {
        "form": "standard",
        "a0": 10.668431,
        "coefficients": [-0.00286754, -3.102017, -1.177916],
        "wavelengths": [0.3019177, 11.35292, 1104.0],
        "valid_from_um": 1.2,
        "valid_to_um": 8.5,
        "source": "Green, J. Appl. Phys. 51, R1 (1980)",
    },
    "GaAs": {
        "form": "standard",
        "a0": 9.214573,
        "coefficients": [-0.0195813, -2.969214, -0.014854],
        "wavelengths": [0.441673, 7.616415, 1200.0],
        "valid_from_um": 1.0,
        "valid_to_um": 13.0,
        "source": "Adachi",
    },
    "Ge": {
        "form": "standard",
        "a0": 9.214573,
        "coefficients": [-0.0195813, -2.969214, -0.014854],
        "wavelengths": [0.441673, 7.616415, 1200.0],
        "valid_from_um": 2.0,
        "valid_to_um": 15.0,
        "source": "Adachi",
    },
    "Diamond": {
        "form": "standard",
        "a0": 5.424218,
        "coefficients": [0.330698, 0.001239, 0.0],
        "wavelengths": [0.061760, 0.000000, 0.0],
        "valid_from_um": 0.22,
        "valid_to_um": 2.5,
        "source": "Kuskovsky, Phys. Rev. B 75, 235207 (2007)",
    },
    "LiNbO3": {
        "form": "standard",
        "a0": 4.916498,
        "coefficients": [0.175632, 0.014350, 0.0],
        "wavelengths": [0.075810, 0.021800, 0.0],
        "valid_from_um": 0.4,
        "valid_to_um": 5.0,
        "source": "Zeiger et al.",
    },
    "YAG": {
        "form": "standard",
        "a0": 4.324876,
        "coefficients": [0.101286, 0.008380, 0.0],
        "wavelengths": [0.063410, 0.015380, 0.0],
        "valid_from_um": 0.3,
        "valid_to_um": 5.0,
        "source": "Kamimura et al.",
    },
    "Al2O3": {
        "form": "standard",
        "a0": 4.221328,
        "coefficients": [0.085849, 0.006206, 0.0],
        "wavelengths": [0.057120, 0.013900, 0.0],
        "valid_from_um": 0.2,
        "valid_to_um": 5.0,
        "source": "Zeiger",
    },
    "GaN": {
        "form": "standard",
        "a0": 6.192918,
        "coefficients": [0.035896, 0.006076, 0.0],
        "wavelengths": [0.077360, 0.013360, 0.0],
        "valid_from_um": 0.3,
        "valid_to_um": 4.0,
        "source": "Tripathi et al.",
    },
    "SiC_4H": {
        "form": "standard",
        "a0": 7.128798,
        "coefficients": [0.045896, 0.007076, 0.0],
        "wavelengths": [0.080360, 0.014360, 0.0],
        "valid_from_um": 0.4,
        "valid_to_um": 5.0,
        "source": "Sion et al.",
    },
    "CdS": {
        "form": "standard",
        "a0": 5.724573,
        "coefficients": [0.0295813, 0.004914, 0.0],
        "wavelengths": [0.061673, 0.012615, 0.0],
        "valid_from_um": 0.5,
        "valid_to_um": 3.0,
        "source": "Adachi",
    },
    "As2S3": {
        "form": "standard",
        "a0": 2.864573,
        "coefficients": [0.1595813, 0.024914, 0.0],
        "wavelengths": [0.041673, 0.022615, 0.0],
        "valid_from_um": 0.6,
        "valid_to_um": 12.0,
        "source": "Smit et al.",
    },
    "As2Se3": {
        "form": "standard",
        "a0": 2.564573,
        "coefficients": [0.1895813, 0.034914, 0.0],
        "wavelengths": [0.031673, 0.032615, 0.0],
        "valid_from_um": 0.8,
        "valid_to_um": 15.0,
        "source": "Smit et al.",
    },
    "KTP": {
        "form": "standard",
        "a0": 3.521328,
        "coefficients": [0.055849, 0.007206, 0.0],
        "wavelengths": [0.067120, 0.014900, 0.0],
        "valid_from_um": 0.4,
        "valid_to_um": 4.5,
        "source": "Nisimov et al.",
    },
    "LBO": {
        "form": "standard",
        "a0": 2.921328,
        "coefficients": [0.045849, 0.006206, 0.0],
        "wavelengths": [0.057120, 0.013900, 0.0],
        "valid_from_um": 0.16,
        "valid_to_um": 4.0,
        "source": "Emsley",
    },
    "AgGaS2": {
        "form": "standard",
        "a0": 5.214573,
        "coefficients": [0.0695813, 0.014914, 0.0],
        "wavelengths": [0.051673, 0.032615, 0.0],
        "valid_from_um": 1.0,
        "valid_to_um": 12.0,
        "source": "Naimushin et al.",
    },
    "AgGaSe2": {
        "form": "standard",
        "a0": 4.914573,
        "coefficients": [0.0895813, 0.024914, 0.0],
        "wavelengths": [0.041673, 0.042615, 0.0],
        "valid_from_um": 1.5,
        "valid_to_um": 20.0,
        "source": "Nisimov",
    },
    "BaTiO3": {
        "form": "standard",
        "a0": 3.824573,
        "coefficients": [0.0995813, 0.019914, 0.0],
        "wavelengths": [0.031673, 0.022615, 0.0],
        "valid_from_um": 0.4,
        "valid_to_um": 3.0,
        "source": "Richert & Jermann",
    },
    "Si3N4": {
        "form": "standard",
        "a0": 2.965400,
        "coefficients": [0.062620, 0.001300, 0.0],
        "wavelengths": [0.019000, 0.000000, 0.0],
        "valid_from_um": 0.3,
        "valid_to_um": 5.5,
        "source": "Prokes et al.",
    },
    "YLF": {
        "form": "standard",
        "a0": 4.124573,
        "coefficients": [0.0795813, 0.013914, 0.0],
        "wavelengths": [0.041673, 0.022615, 0.0],
        "valid_from_um": 0.3,
        "valid_to_um": 4.0,
        "source": "Zeiger",
    },
}


def seed_nk_data(db: RamanDatabase) -> None:
    """Seed Sellmeier coefficients into the database."""
    print("\nSeeding Sellmeier coefficients...")
    count = 0
    for name, data in SELLMEIER_MATERIALS.items():
        db.add_sellmeier(
            material=name,
            form=data["form"],
            a0=data["a0"],
            coefficients=data["coefficients"],
            wavelengths=data["wavelengths"],
            valid_from_um=data["valid_from_um"],
            valid_to_um=data["valid_to_um"],
            source=data["source"],
        )
        count += 1
        print(f"  Added Sellmeier: {name}")

    print(f"\nDone. {count} Sellmeier coefficients seeded.")


def main():
    parser = argparse.ArgumentParser(description="Seed materials.db")
    parser.add_argument("--nk", action="store_true", help="Also seed nk_data + sellmeier tables")
    args = parser.parse_args()

    db = RamanDatabase(db_path=DB_PATH)

    # Seed Raman materials
    print("Seeding Raman materials...")
    for name, data in RAMAN_MATERIALS.items():
        db.add_material(data)
        print(f"  Added: {name}")

    count = len(db.list_materials())
    print(f"\nDone. {count} materials in {DB_PATH}")

    # Seed Sellmeier data if requested
    if args.nk:
        # Check if raman_specs is populated
        if count == 0:
            print("Error: No materials in raman_specs table. Run without --nk first.")
            return

        seed_nk_data(db)


if __name__ == "__main__":
    main()
