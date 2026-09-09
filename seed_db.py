#!/usr/bin/env python3
"""Seed materials.db from the hardcoded RAMAN_MATERIALS and SELLMEIER_MATERIALS dicts.

Run from the project root:
    python seed_db.py          # seeds raman_specs (existing)
    python seed_db.py --nk     # seeds sellmeier + tabulated nk_data

With ``--nk`` the seeder also loads ``nk_datasets/manifest.json`` (produced by
``nk_datasets/collect.py`` from refractiveindex.info) and populates the
tabulated ``nk_data`` table (n, k vs wavelength) with per-source provenance and
citations. This second step is idempotent: previously seeded tabulated rows are
cleared and re-inserted from the manifest on each run.
"""

import argparse
from pathlib import Path
from photonics_helper.raman import (
    RAMAN_MATERIALS,
    THORLABS_SUBSTRATE_MATERIALS,
    RamanDatabase,
)

DB_PATH = Path(__file__).parent / "photonics_helper" / "materials.db"

# Sellmeier coefficients for optical materials (standard form: n² = A₀ + Σ Aᵢλ²/(λ² - Bᵢ))
# SCHOTT/refractiveindex.info use Bᵢ in μm² for glass and crystal data.
SELLMEIER_MATERIALS = {
    "Silica": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [0.6961663, 0.4079426, 0.8974794],
        "wavelengths": [0.004679148, 0.013512075, 97.953962],
        "valid_from_um": 0.21,
        "valid_to_um": 6.7,
        "source": "Thorlabs UVFS; Malitson, JOSA 55, 1205 (1965)",
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

# Thorlabs optical substrates — Sellmeier from SCHOTT / refractiveindex.info
THORLABS_SELLMEIER_MATERIALS = {
    "N-BK7": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [1.03961212, 0.231792344, 1.01046945],
        "wavelengths": [0.00600069867, 0.0200179144, 103.560653],
        "valid_from_um": 0.3,
        "valid_to_um": 2.5,
        "source": "Thorlabs; SCHOTT N-BK7 (refractiveindex.info)",
    },
    "N-SF11": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [1.73759695, 0.313747346, 1.89878101],
        "wavelengths": [0.013188707, 0.0623068142, 155.23629],
        "valid_from_um": 0.37,
        "valid_to_um": 2.5,
        "source": "Thorlabs; SCHOTT N-SF11 (refractiveindex.info)",
    },
    "N-F2": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [1.39757037, 0.159201403, 1.2686543],
        "wavelengths": [0.00995906143, 0.0546931752, 119.248346],
        "valid_from_um": 0.365,
        "valid_to_um": 2.5,
        "source": "Thorlabs; SCHOTT N-F2 (refractiveindex.info)",
    },
    "F2": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [1.34533359, 0.209073176, 0.937357162],
        "wavelengths": [0.00997743871, 0.0470450767, 111.886764],
        "valid_from_um": 0.32,
        "valid_to_um": 2.5,
        "source": "Thorlabs; SCHOTT F2 (refractiveindex.info)",
    },
    "CaF2": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [
            0.443749998,
            0.444930066,
            0.150133991,
            8.85319946,
        ],
        "wavelengths": [
            0.00178027854,
            0.00788536061,
            0.0124119491,
            2752.28175,
        ],
        "valid_from_um": 0.138,
        "valid_to_um": 2.326,
        "source": "Thorlabs; Daimon & Masumura, Appl. Opt. 41, 5275 (2002)",
    },
    "BaF2": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [0.64336, 0.50760, 3.8261],
        "wavelengths": [0.057789, 0.10664, 46.386],
        "valid_from_um": 0.22,
        "valid_to_um": 10.0,
        "source": "Thorlabs; Zheng et al., Opt. Mater. Express 13, 2380 (2023)",
    },
    "MgF2": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [0.48755108, 0.39875031, 2.3120353],
        "wavelengths": [0.04338408, 0.09461442, 23.793604],
        "valid_from_um": 0.2,
        "valid_to_um": 7.0,
        "source": "Thorlabs; Dodge, Appl. Opt. 23, 1980 (1984), o-ray",
    },
    "ZnSe": {
        "form": "standard",
        "a0": 3.0,
        "coefficients": [1.90],
        "wavelengths": [0.113],
        "valid_from_um": 0.48,
        "valid_to_um": 15.0,
        "source": "Thorlabs; Marple, J. Appl. Phys. 35, 539 (1964)",
    },
    "YVO4": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [2.7665],
        "wavelengths": [0.026884],
        "valid_from_um": 0.488,
        "valid_to_um": 3.39,
        "source": "Thorlabs; Birnbaum & DeShazer, NASA CR (1976), o-ray",
    },
    "KBr": {
        "form": "standard",
        "a0": 0.39408,
        "coefficients": [0.79221, 0.01981, 0.15587, 0.17673, 2.06217],
        "wavelengths": [0.146, 0.173, 0.187, 60.61, 87.72],
        "valid_from_um": 0.20,
        "valid_to_um": 42.0,
        "source": "Thorlabs; Li, J. Phys. Chem. Ref. Data 5, 329 (1976)",
    },
    "Zerodur": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [1.3182408, 0.0244, 1.08915181],
        "wavelengths": [0.00879, 0.0609, 110.0],
        "valid_from_um": 0.365,
        "valid_to_um": 2.325,
        "source": "Thorlabs; SCHOTT ZERODUR (refractiveindex.info)",
    },
    "PMMA": {
        "form": "standard",
        "a0": 1.0,
        "coefficients": [0.99654, 0.18964, 0.00411],
        "wavelengths": [0.00787, 0.02191, 3.85727],
        "valid_from_um": 0.4047,
        "valid_to_um": 1.083,
        "source": "Thorlabs; Szczurowski (refractiveindex.info)",
    },
}

ALL_SELLMEIER_MATERIALS = {**SELLMEIER_MATERIALS, **THORLABS_SELLMEIER_MATERIALS}


def seed_nk_data(db: RamanDatabase) -> None:
    """Seed Sellmeier coefficients into the database."""
    print("\nSeeding Sellmeier coefficients...")
    count = 0
    for name, data in ALL_SELLMEIER_MATERIALS.items():
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


MANIFEST_PATH = Path(__file__).parent / "nk_datasets" / "manifest.json"


def seed_tabulated_nk(db: RamanDatabase, manifest_path: Path = MANIFEST_PATH) -> None:
    """Seed tabulated nk_data from the refractiveindex.info manifest.

    The run is idempotent: all previously seeded (attributed) tabulated rows
    are cleared first, then re-inserted from the manifest. Sellmeier data is a
    separate table and is left untouched. Datasets that fail validation are
    logged and skipped rather than partially seeded.
    """
    from photonics_helper.materials import validate_nk_dataset

    if not manifest_path.exists():
        print(f"Warning: manifest not found at {manifest_path}")
        return

    import json

    data = json.loads(manifest_path.read_text())
    entries = data.get("datasets", [])

    print("\nSeeding tabulated nk_data from manifest...")
    removed = db.clear_all_tabulated_nk()
    if removed:
        print(f"  Cleared {removed} stale tabulated row(s).")

    seeded = 0
    skipped = 0
    for entry in entries:
        errors = validate_nk_dataset(entry)
        if errors:
            print(f"  Skip {entry.get('source', '?')}: {'; '.join(errors)}")
            skipped += 1
            continue
        for wl, n, k in zip(
            entry["wavelengths"], entry["n"], entry["k"]
        ):
            db.add_nk_data(
                material=entry["material"],
                wl_um=wl,
                n=n,
                k=k,
                source=entry["source"],
                citation=entry.get("citation") or "",
            )
        seeded += 1
    total_rows = 0
    for entry in entries:
        if validate_nk_dataset(entry) == []:
            total_rows += len(entry["wavelengths"])
    print(f"Done. {seeded} datasets, {total_rows} tabulated rows seeded; {skipped} skipped.")


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

    print("\nSeeding Thorlabs optical substrates...")
    for name, data in THORLABS_SUBSTRATE_MATERIALS.items():
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
        seed_tabulated_nk(db)


if __name__ == "__main__":
    main()
