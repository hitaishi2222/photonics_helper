"""Material data integrity tests (review N6, N9).

Checks that the shipped `materials.db` and the README agree, and that the
Si3N4 Sellmeier entries reproduce physically sensible dispersion (the Ligentec
entry historically had a mistyped B1 that made n(lambda) nearly flat).
"""

import re
import sqlite3
from pathlib import Path

import numpy as np

from photonics_helper.materials import RefractiveIndex

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "photonics_helper" / "materials.db"
README_PATH = ROOT / "README.md"


def _raman_count() -> int:
    con = sqlite3.connect(DB_PATH)
    try:
        return int(con.execute("SELECT COUNT(*) FROM raman_specs").fetchone()[0])
    finally:
        con.close()


def test_readme_material_count_matches_db():
    count = _raman_count()
    readme = README_PATH.read_text()
    declared = [int(m) for m in re.findall(r"with (\d+) materials", readme)]
    assert declared, "README does not declare a material count"
    assert all(d == count for d in declared), (declared, count)


def test_si3n4_ligentec_dispersion_is_physical():
    ri = RefractiveIndex.from_material_database("Si3N4-Ligentec")
    n_633 = ri.n_func(0.633)
    n_1550 = ri.n_func(1.55)
    # Stoichiometric LPCVD SiN: n ~ 2.04 at 633 nm, ~ 2.00 at 1550 nm.
    assert 1.98 < n_1550 < 2.02
    assert 0.02 < (n_633 - n_1550) < 0.06


def test_si3n4_sellmeier_agrees_with_tabulated():
    """The Philipp Sellmeier and the tabulated SiN dataset agree near 633 nm."""
    sell = RefractiveIndex.from_material_database("Si3N4")
    n_sell = sell.n_func(0.633)
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            "SELECT wavelength_um, n FROM nk_data "
            "WHERE material = 'Si3N4' AND source = 'si3n4-beliaev' "
            "ORDER BY wavelength_um"
        ).fetchall()
    finally:
        con.close()
    wl = np.array([r[0] for r in rows])
    n = np.array([r[1] for r in rows])
    n_tab = float(np.interp(0.633, wl, n))
    assert abs(n_sell - n_tab) / n_tab < 0.02
