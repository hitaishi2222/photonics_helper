"""Material data integrity tests (review N6, N9).

Checks that the shipped `materials.db` and the README agree, and that the
Si3N4 Sellmeier entries reproduce physically sensible dispersion (the Ligentec
entry historically had a mistyped B1 that made n(lambda) nearly flat).
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

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


# ─── Write-path integrity (audit findings 7, 11, 12, 14) ───────────────────


def _temp_db(tmp_path):
    from photonics_helper.raman import RamanDatabase

    return RamanDatabase(db_path=tmp_path / "t.db")


def test_update_material_rejects_unknown_keys(tmp_path):
    db = _temp_db(tmp_path)
    db.add_material({"name": "X", "crystal": "cubic"})
    with pytest.raises(ValueError, match="unknown column"):
        db.update_material("X", **{"crystal='hacked', n2": 5})
    # The typo case is refused too, instead of an OperationalError.
    with pytest.raises(ValueError, match="unknown column"):
        db.update_material("X", n22=5.0)
    db.update_material("X", n2=2.0)
    assert db.get_material("X")["n2"] == 2.0


def test_delete_material_removes_every_dependent_row(tmp_path):
    db = _temp_db(tmp_path)
    db.add_material({"name": "X"})
    db.add_nk_data("X", 1.55, 2.0, 0.0, source="x-src")
    db.add_sellmeier("X", "standard", 1.0, [0.1], [0.01], 0.4, 2.0, "src")
    from photonics_helper.phonon import PhononMode

    db.add_phonon_mode("X", PhononMode(shift_cm=254, linewidth_cm=14))
    db.delete_material("X")

    con = sqlite3.connect(db.db_path)
    try:
        for table in ("raman_specs", "nk_data", "sellmeier", "phonon_modes"):
            n = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE "
                + ("name = ?" if table == "raman_specs" else "material = ?"),
                ("X",),
            ).fetchone()[0]
            assert n == 0, table
    finally:
        con.close()


def test_replace_tabulated_nk_is_one_transaction(tmp_path):
    db = _temp_db(tmp_path)
    db.add_material({"name": "X"})
    db.replace_tabulated_nk([("X", 1.5, 2.0, 0.0, "a", "cite")])
    written = db.replace_tabulated_nk([("X", 1.5, 2.1, 0.0, "b", "cite2")])
    assert written == 1
    assert db.list_nk_sources("X") == ["b"]


def test_seed_tabulated_nk_validates_before_deleting(tmp_path):
    """A manifest with one invalid entry keeps the shipped rows."""
    import json

    from seed_db import seed_tabulated_nk

    db = _temp_db(tmp_path)
    db.add_material({"name": "X"})
    db.replace_tabulated_nk([("X", 1.5, 2.0, 0.0, "shipped", "cite")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {"material": "X", "source": "bad"},  # missing arrays
                ]
            }
        )
    )
    seed_tabulated_nk(db, manifest_path=manifest)
    assert db.list_nk_sources("X") == ["shipped"]


# ─── nk uniqueness and partial material upsert (audit findings) ────────────


def test_duplicate_nk_rows_are_migrated_to_one_per_key(tmp_path):
    from photonics_helper.raman import RamanDatabase

    path = tmp_path / "t.db"
    db = RamanDatabase(db_path=path)
    db.add_material({"name": "X"})
    db.add_nk_data("X", 1.55, 2.0, 0.0, source="x")

    con = sqlite3.connect(path)
    try:
        con.execute("DROP INDEX nk_data_unique")
        con.execute(
            "INSERT INTO nk_data (material, wavelength_um, n, k, source) "
            "VALUES ('X', 1.55, 2.1, 0.0, 'x')"
        )
        con.commit()
    finally:
        con.close()

    from photonics_helper.core import data as data_mod

    data_mod._INITIALIZED_PATHS.discard(path.resolve())  # simulate a new process
    reopened = RamanDatabase(db_path=path)
    assert reopened.list_materials() == ["X"]  # first use runs the migration

    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT n FROM nk_data WHERE material = 'X'"
        ).fetchall()
        indexes = {r[1] for r in con.execute("PRAGMA index_list(nk_data)")}
    finally:
        con.close()
    assert rows == [(2.1,)]  # highest rowid kept
    assert "nk_data_unique" in indexes


def test_repeated_nk_add_yields_one_sample(tmp_path):
    db = _temp_db(tmp_path)
    db.add_material({"name": "X"})
    for _ in range(3):
        for wl_um, n in ((1.5, 2.0), (1.6, 2.1), (1.7, 2.2), (1.8, 2.3)):
            db.add_nk_data("X", wl_um, n, 0.0, source="x-a")
    wl, _n, _k, _src = db.get_nk_data("X", with_source=True)
    assert len(wl) == 4
    ri = RefractiveIndex.from_material_database("x-a", db=db)
    assert ri.n_func(1.55) > 1.0


def test_partial_add_material_preserves_unspecified_columns(tmp_path):
    db = _temp_db(tmp_path)
    db.add_material(
        {"name": "Silica", "fR": 0.18, "raman_shift_cm": 440.0, "n2": 3.2e-20}
    )
    db.add_material({"name": "Silica"})
    row = db.get_material("Silica")
    assert (row["fR"], row["raman_shift_cm"], row["n2"]) == (
        0.18,
        440.0,
        3.2e-20,
    )
    db.add_material({"name": "Silica", "n2": 2.0e-20})
    assert db.get_material("Silica")["fR"] == 0.18


def test_partial_add_material_new_row_has_nulls_elsewhere(tmp_path):
    db = _temp_db(tmp_path)
    db.add_material({"name": "New", "crystal": "cubic"})
    row = db.get_material("New")
    assert row["crystal"] == "cubic"
    assert row["fR"] is None and row["n2"] is None


def test_add_material_rejects_unknown_keys(tmp_path):
    db = _temp_db(tmp_path)
    with pytest.raises(ValueError, match="unknown column"):
        db.add_material({"name": "X", "n22": 5.0})
    assert db.get_material("X") is None


def test_import_from_dict_is_a_partial_upsert(tmp_path):
    db = _temp_db(tmp_path)
    db.add_material({"name": "X", "fR": 0.1})
    db.import_from_dict({"X": {"n2": 2.0}, "Y": {"fR": 0.2}})
    x, y = db.get_material("X"), db.get_material("Y")
    assert (x["fR"], x["n2"]) == (0.1, 2.0)
    assert (y["fR"], y["n2"]) == (0.2, None)
