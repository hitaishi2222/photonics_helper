"""Data-layer consistency: canonical tables, drift, provenance and lazy access.

These tests enforce the Phase 2 ("consolidate-material-data") contract:

- the shipped ``materials.db`` is *derived* from the canonical Python tables and
  must match them (drift guard);
- every data row carries a source and a licence, and nk rows join to the
  provenance registry;
- ``RamanDatabase`` initialises lazily;
- phonon data is reachable through the database interface.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from photonics_helper.phonon import PHONON_MATERIALS, PhononResponse
from photonics_helper.raman import RamanDatabase
from photonics_helper.raman.reference import (
    RAMAN_MATERIALS,
    THORLABS_SUBSTRATE_MATERIALS,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "photonics_helper" / "materials.db"

DATA_TABLES = ("nk_data", "sellmeier", "raman_specs", "phonon_modes")
LICENSE_SENTINELS = {"see-source-publication", "see-references", "see-note"}


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _norm(value: object) -> object:
    """Normalise a field for comparison (strip strings, empty -> None)."""
    if isinstance(value, str):
        return value.strip() or None
    return value


# ─── Drift: the DB is derived from the canonical tables ───────────────────


def test_raman_specs_match_canonical_dicts() -> None:
    canonical = {**RAMAN_MATERIALS, **THORLABS_SUBSTRATE_MATERIALS}
    con = _connect()
    try:
        rows = con.execute("SELECT * FROM raman_specs").fetchall()
    finally:
        con.close()

    db_names = {row["name"] for row in rows}
    assert db_names == set(canonical), (
        f"raman_specs differs from the canonical tables: "
        f"only-in-db={db_names - set(canonical)}, only-in-python={set(canonical) - db_names}"
    )

    for row in rows:
        spec = canonical[row["name"]]
        for key in (
            "crystal",
            "bandgap_eV",
            "n2",
            "raman_shift_cm",
            "raman_linewidth_cm",
            "fR",
            "gain_coeff",
            "lo_phonon_cm",
            "to_phonon_cm",
            "references",
        ):
            assert _norm(row[key]) == _norm(spec.get(key)), (
                f"{row['name']}.{key}: db={row[key]!r} python={spec.get(key)!r}"
            )


def test_phonon_modes_match_canonical_table() -> None:
    con = _connect()
    try:
        rows = con.execute(
            "SELECT material, shift_cm, linewidth_cm, symmetry, relative_strength "
            "FROM phonon_modes"
        ).fetchall()
    finally:
        con.close()

    assert rows, "phonon_modes is empty — run seed_db.py"

    db_set = {
        (r["material"], round(r["shift_cm"], 6), round(r["linewidth_cm"], 6),
         r["symmetry"], round(r["relative_strength"], 6))
        for r in rows
    }
    py_set = {
        (material, round(m.shift_cm.as_1_cm, 6), round(m.linewidth_cm.as_1_cm, 6),
         m.symmetry, round(m.relative_strength, 6))
        for material, modes in PHONON_MATERIALS.items()
        for m in modes
    }
    assert db_set == py_set, (
        f"phonon_modes differs from PHONON_MATERIALS: "
        f"db_only={len(db_set - py_set)}, py_only={len(py_set - db_set)}"
    )


def test_every_phonon_material_is_populated() -> None:
    con = _connect()
    try:
        materials = {
            r[0] for r in con.execute("SELECT DISTINCT material FROM phonon_modes")
        }
    finally:
        con.close()
    assert materials == set(PHONON_MATERIALS)


# ─── Provenance and licences ──────────────────────────────────────────────


def test_provenance_registry_matches_nk_sources() -> None:
    con = _connect()
    try:
        nk_sources = {
            r[0]
            for r in con.execute(
                "SELECT DISTINCT source FROM nk_data WHERE source IS NOT NULL"
            )
        }
        prov_keys = {r[0] for r in con.execute("SELECT source_key FROM provenance")}
    finally:
        con.close()
    assert nk_sources, "no attributed nk_data rows"
    assert nk_sources <= prov_keys, f"unregistered nk sources: {nk_sources - prov_keys}"


def test_no_row_is_missing_a_licence() -> None:
    con = _connect()
    try:
        for table in DATA_TABLES:
            missing = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE license IS NULL OR license = ''"
            ).fetchone()[0]
            assert missing == 0, f"{table} has {missing} rows without a licence"
    finally:
        con.close()


def test_nk_licence_is_cc0() -> None:
    con = _connect()
    try:
        licences = {
            r[0] for r in con.execute("SELECT DISTINCT license FROM nk_data")
        }
    finally:
        con.close()
    assert licences == {"CC0-1.0"}


def test_sentinels_are_the_documented_set() -> None:
    """Every non-CC0 licence column value is a documented sentinel."""
    con = _connect()
    try:
        values: set[str] = set()
        for table in DATA_TABLES:
            values |= {
                r[0]
                for r in con.execute(f"SELECT DISTINCT license FROM {table}")
                if r[0]
            }
    finally:
        con.close()
    # CC0-1.0 is a concrete identifier; everything else must be a sentinel.
    assert values - {"CC0-1.0"} <= LICENSE_SENTINELS, values


def test_get_provenance_roundtrip() -> None:
    db = RamanDatabase()
    records = db.list_provenance()
    assert records, "provenance registry is empty"
    key = records[0]["source_key"]
    record = db.get_provenance(key)
    assert record is not None
    assert record["source_key"] == key
    assert record["citation"]
    assert db.get_provenance("definitely-not-a-source") is None


# ─── Lazy initialisation ──────────────────────────────────────────────────


def test_construction_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "lazy.db"
    RamanDatabase(db_path=db_path)
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_first_use_initialises_once(tmp_path: Path) -> None:
    db_path = tmp_path / "lazy.db"
    db = RamanDatabase(db_path=db_path)
    db.list_materials()
    assert db_path.exists()
    marker = db._initialized  # noqa: SLF001 - asserting the internal flag
    db.list_materials()
    assert db._initialized is marker


# ─── Phonon access through the database ───────────────────────────────────


def test_phonon_response_from_material_uses_database() -> None:
    response = PhononResponse.from_material("LiNbO3")
    assert len(response.modes) == len(PHONON_MATERIALS["LiNbO3"])

    # Cross-check against the database rows directly.
    db = RamanDatabase()
    db_modes = db.get_phonon_modes("LiNbO3")
    shifts_response = sorted(m.shift_cm.as_1_cm for m in response.modes)
    shifts_db = sorted(m.shift_cm.as_1_cm for m in db_modes)
    assert shifts_response == shifts_db


def test_phonon_response_falls_back_to_canonical(tmp_path: Path) -> None:
    # An empty temp database has no phonon modes, so the canonical table is used.
    empty_db = RamanDatabase(db_path=tmp_path / "empty.db")
    empty_db.list_materials()  # initialise (stays empty)
    response = PhononResponse.from_material("YAG", db=empty_db)
    assert len(response.modes) == len(PHONON_MATERIALS["YAG"])


def test_phonon_response_unknown_material() -> None:
    with pytest.raises(KeyError) as excinfo:
        PhononResponse.from_material("Unobtainium")
    assert "Unobtainium" in str(excinfo.value)


def test_tabulated_material_license_from_provenance() -> None:
    """A tabulated key resolves its licence through the provenance registry."""
    from photonics_helper.core.materials import material

    mat = material("si-aspnes")
    assert mat.license == "CC0-1.0"
    assert mat.source


def test_importing_phonon_does_not_import_database() -> None:
    import subprocess
    import sys

    code = (
        "import sys, photonics_helper.phonon; "
        "print('photonics_helper.raman.db' in [m for m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
