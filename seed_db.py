#!/usr/bin/env python3
"""Seed materials.db from the hardcoded RAMAN_MATERIALS dict.

Run from the project root:
    python seed_db.py
"""

from pathlib import Path
from photonics_helper.raman import RAMAN_MATERIALS, RamanDatabase

DB_PATH = Path(__file__).parent / "photonics_helper" / "materials.db"


def main():
    db = RamanDatabase(db_path=DB_PATH)

    for name, data in RAMAN_MATERIALS.items():
        db.add_material(data)
        print(f"  Added: {name}")

    count = len(db.list_materials())
    print(f"\nDone. {count} materials in {DB_PATH}")


if __name__ == "__main__":
    main()
