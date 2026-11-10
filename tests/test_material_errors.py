"""Error-message contract for material retrieval.

A foundation that other projects build on must fail *usefully*: a missing or
corrupt database, an unknown material, or an unavailable axis should tell the
user what is wrong and what to do, not surface a bare SQLite error or a
misleading message. These tests pin those messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from photonics_helper.core.materials import material
from photonics_helper.materials import RefractiveIndex
from photonics_helper.raman import RamanDatabase


def _empty_db(tmp_path: Path) -> RamanDatabase:
    db = RamanDatabase(db_path=tmp_path / "empty.db")
    db.list_materials()  # force (idempotent) schema creation
    return db


# ─── Bad input ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_blank_name_is_rejected_helpfully(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        material(name)


def test_non_string_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        RefractiveIndex.from_material_database(None)  # type: ignore[arg-type]


# ─── Unknown material ─────────────────────────────────────────────────────


def test_unknown_material_suggests_a_close_match() -> None:
    with pytest.raises(ValueError) as excinfo:
        material("Silicaa")
    message = str(excinfo.value)
    assert "Did you mean" in message
    assert "Silica" in message


def test_unknown_material_points_at_the_browser() -> None:
    with pytest.raises(ValueError) as excinfo:
        material("Unobtainium")
    message = str(excinfo.value)
    assert "list_materials()" in message, message
    assert "material-author" in message, message


# ─── Axis selection ───────────────────────────────────────────────────────


def test_unknown_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown axis"):
        material("LiNbO3", axis="sideways")


def test_axis_on_non_birefringent_material_explains_why() -> None:
    with pytest.raises(ValueError) as excinfo:
        material("Silica", axis="ordinary")
    message = str(excinfo.value)
    assert "has no ordinary/extraordinary rows" in message
    assert "birefringent materials" in message


# ─── Database problems ────────────────────────────────────────────────────


def test_empty_database_message_names_the_path_and_remedy(tmp_path: Path) -> None:
    db = _empty_db(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        RefractiveIndex.from_material_database("Silica", db=db)
    message = str(excinfo.value)
    assert "contains no materials" in message
    assert str(db.db_path) in message
    assert "seed_db.py" in message


def test_corrupt_database_is_wrapped_with_context(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"definitely not a sqlite database")
    with pytest.raises(ValueError) as excinfo:
        RefractiveIndex.from_material_database(
            "Silica", db=RamanDatabase(db_path=corrupt)
        )
    message = str(excinfo.value)
    assert "could not read the material database" in message
    assert str(corrupt) in message
    assert "regenerate" in message


def test_missing_library_style_db_creates_nothing_on_construction(
    tmp_path: Path,
) -> None:
    """A broken install must not have side effects before it is even used."""
    db_path = tmp_path / "nested" / "missing.db"
    RamanDatabase(db_path=db_path)
    assert not db_path.exists()


# ─── Happy path with an explicit handle ───────────────────────────────────


def test_explicit_db_handle_still_works(tmp_path: Path) -> None:
    from photonics_helper.raman.reference import RAMAN_MATERIALS

    db = RamanDatabase(db_path=tmp_path / "seeded.db")
    for spec in RAMAN_MATERIALS.values():
        db.add_material(spec)
    db.add_sellmeier(
        material="Silica",
        form="standard",
        a0=1.0,
        coefficients=[0.6961663],
        wavelengths=[0.0684043**2],
        valid_from_um=0.21,
        valid_to_um=3.7,
        source="test",
    )
    ri = RefractiveIndex.from_material_database("Silica", db=db)
    assert ri.n_func(1.0) > 1.0


# ─── Read-only installs and honest database errors (audit findings 3, 6) ───


def _readonly_copy(tmp_path: Path) -> Path:
    import shutil

    src = Path(__import__("photonics_helper").__file__).parent / "materials.db"
    dst = tmp_path / "materials.db"
    shutil.copyfile(src, dst)
    dst.chmod(0o444)
    return dst


def test_read_only_database_serves_read_paths(tmp_path: Path) -> None:
    """A current schema is never written, so a read-only file still works."""
    readonly = _readonly_copy(tmp_path)
    db = RamanDatabase(db_path=readonly)
    assert db.list_materials()
    assert db.get_material("Silica")
    assert readonly.stat().st_mode & 0o222 == 0


def test_read_only_database_that_needs_migration_says_so(tmp_path: Path) -> None:
    import sqlite3

    readonly = _readonly_copy(tmp_path)
    readonly.chmod(0o644)
    conn = sqlite3.connect(readonly)
    conn.execute("DROP TABLE provenance")
    conn.commit()
    conn.close()
    readonly.chmod(0o444)
    with pytest.raises(sqlite3.OperationalError, match="not writable|writable copy"):
        RamanDatabase(db_path=readonly).list_materials()


def test_migration_runs_once_per_path(tmp_path: Path, monkeypatch) -> None:
    from photonics_helper.core.data import MaterialsDatabase

    import shutil
    from pathlib import Path as _Path

    src = _Path(__import__("photonics_helper").__file__).parent / "materials.db"
    copied = tmp_path / "cached" / "materials.db"
    copied.parent.mkdir()
    shutil.copyfile(src, copied)
    calls: list[int] = []
    original = MaterialsDatabase._init_db

    def counted(self) -> None:
        calls.append(1)
        original(self)

    monkeypatch.setattr(MaterialsDatabase, "_init_db", counted)
    MaterialsDatabase(db_path=copied).list_materials()
    MaterialsDatabase(db_path=copied).list_materials()
    assert len(calls) == 1


def test_from_database_raises_for_an_explicit_path(tmp_path: Path) -> None:
    from photonics_helper.raman.spec import RamanSpec

    missing = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="not found in database"):
        RamanSpec.from_database("Silica", db_path=missing)


def test_from_database_propagates_a_corrupt_explicit_path(tmp_path: Path) -> None:
    import sqlite3

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a database")
    from photonics_helper.raman.spec import RamanSpec

    with pytest.raises(sqlite3.DatabaseError):
        RamanSpec.from_database("Silica", db_path=corrupt)
