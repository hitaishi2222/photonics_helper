"""Tests for the tabulated n/k database feature (add-tabulated-nk-database).

Covers the four behaviours called out by task 6.1:

* ``material–author`` key parsing in ``RefractiveIndex.from_material_database``
* tabulated-vs-Sellmeier precedence in ``RamanSpec.nk()``
* validation rejection in ``photonics_helper.materials.validate_nk_dataset``
* idempotent seeding via ``seed_db.seed_tabulated_nk``

The parsing / precedence tests exercise the *shipped* ``materials.db`` (which
already carries a validated batch of tabulated datasets). The seeding tests use
throw-away databases so they never touch the shipped data.
"""

import json

import pytest

from photonics_helper.raman import RamanDatabase, RamanSpec
from photonics_helper.materials import RefractiveIndex, validate_nk_dataset


# ─── Validation (task 4.1) ────────────────────────────────────────────────────


def _valid_entry(**overrides):
    base = dict(
        material="Si",
        source="si-test",
        citation="Test reference",
        wavelengths=[0.5, 1.0, 1.5, 2.0, 2.5],
        n=[3.4, 3.42, 3.45, 3.48, 3.5],
        k=[0.0, 0.0, 1e-4, 1e-2, 0.1],
    )
    base.update(overrides)
    return base


def test_valid_dataset_passes():
    assert validate_nk_dataset(_valid_entry()) == []


def test_rejects_negative_k():
    entry = _valid_entry(k=[0.0, 0.0, -1e-4, 1e-2, 0.1])
    assert any("negative" in e for e in validate_nk_dataset(entry))


def test_rejects_non_monotonic_wavelengths():
    entry = _valid_entry(wavelengths=[0.5, 1.0, 0.9, 2.0, 2.5])
    errs = validate_nk_dataset(entry)
    assert any("increasing" in e for e in errs)


def test_rejects_non_finite():
    entry = _valid_entry(n=[3.4, float("nan"), 3.45, 3.48, 3.5])
    assert any("finite" in e for e in validate_nk_dataset(entry))


def test_rejects_n_le_1_everywhere():
    entry = _valid_entry(n=[0.9, 0.95, 1.0, 0.9, 1.0])
    assert any("n <= 1" in e for e in validate_nk_dataset(entry))


def test_rejects_missing_source():
    errs = validate_nk_dataset(_valid_entry(source=None))
    assert any("source" in e for e in errs)


def test_rejects_too_few_points():
    entry = _valid_entry(wavelengths=[0.5, 1.0], n=[3.4, 3.5], k=[0.0, 0.0])
    assert any("fewer than" in e for e in validate_nk_dataset(entry))


# ─── material–author parsing (task 3.1 / 3.2) ─────────────────────────────────


def test_from_material_database_parses_material_author_key():
    """A material–author key loads the matching tabulated spectrum."""
    ri = RefractiveIndex.from_material_database("silica-franta")
    n = ri.n_func(1.0)
    assert n > 1.0
    # The shipped value is a real tabulated lookup, not a Sellmeier fit.
    assert abs(n - 1.4506682588957254) < 1e-6


def test_from_material_database_missing_author_raises():
    """A missing material–author key raises ValueError (no silent fallback)."""
    with pytest.raises(ValueError, match="silica-palik"):
        RefractiveIndex.from_material_database("silica-palik")


def test_from_material_database_source_is_case_insensitive():
    ri = RefractiveIndex.from_material_database("SILICA-franta")
    assert abs(ri.n_func(1.0) - 1.4506682588957254) < 1e-6


def test_unknown_material_prefix_raises():
    with pytest.raises(ValueError, match="Unknown material"):
        RefractiveIndex.from_material_database("nope-author")


def test_plain_canonical_name_still_works():
    """A bare material name (no author) still resolves (Sellmeier fallback)."""
    ri = RefractiveIndex.from_material_database("Silica")
    assert ri.n_func(1.0) > 1.0


# ─── tabulated-vs-Sellmeier precedence (task 3.3) ─────────────────────────────


def test_raman_spec_nk_routes_tabulated_lookup():
    """RamanSpec.nk() returns the tabulated n/k, matching from_material_database."""
    # Al2O3 has exactly one tabulated source, so no explicit source is needed.
    source = "al2o3-hagemann"
    ri = RefractiveIndex.from_material_database(source)
    wl = ri.wl.as_um[50]
    from_db = ri.n_func(wl) + 1j * ri.k_func(wl)

    spec = RamanSpec.from_database("Al2O3")
    nk = spec.nk(wl, source=source)

    assert abs(nk.real - from_db.real) < 1e-9
    assert abs(nk.imag - from_db.imag) < 1e-9
    # The tabulated Al2O3 data has real absorption at short wavelengths.
    assert nk.imag > 0


def test_multiple_sources_require_selection():
    """Silica has two tabulated sources; nk() without a source must disambiguate."""
    spec = RamanSpec.from_database("Silica")
    with pytest.raises(ValueError, match="Multiple tabulated"):
        spec.nk(1.0)  # no source given, two datasets available


# ─── Idempotent seeding (task 2.3) ────────────────────────────────────────────


def _make_manifest() -> dict:
    wl = [0.5 + 0.01 * i for i in range(51)]
    n = [3.4 + 0.001 * i for i in range(51)]
    k = [1e-4 * i for i in range(51)]
    return {
        "schema": "nk-dataset-manifest/v1",
        "datasets": [
            {
                "material": "Si",
                "source": "si-manifest",
                "citation": "Test",
                "wavelengths": wl,
                "n": n,
                "k": k,
            }
        ],
    }


def _count_rows(db: RamanDatabase, source: str) -> int:
    wl, _, _ = db.get_nk_by_source("Si", source)
    return len(wl)


def test_seed_tabulated_nk_is_idempotent(tmp_path):
    """Re-running seeding clears and re-inserts; row counts stay stable."""
    from seed_db import seed_tabulated_nk

    db_path = tmp_path / "m.db"
    db = RamanDatabase(db_path=db_path)
    db.add_material({"name": "Si", "crystal": "cubic", "bandgap_eV": 1.12, "fR": 0.0})

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_make_manifest()))

    seed_tabulated_nk(db, manifest_path=manifest_path)
    first = _count_rows(db, "si-manifest")

    assert first == 51

    seed_tabulated_nk(db, manifest_path=manifest_path)
    second = _count_rows(db, "si-manifest")

    assert first == second == 51


def test_seed_skips_invalid_datasets(tmp_path):
    """Invalid datasets are logged and skipped, never partially seeded."""
    from seed_db import seed_tabulated_nk

    db_path = tmp_path / "m.db"
    db = RamanDatabase(db_path=db_path)
    db.add_material({"name": "Si", "crystal": "cubic", "bandgap_eV": 1.12, "fR": 0.0})

    good = _valid_entry(material="Si", source="si-good")
    bad = _valid_entry(material="Si", source="si-bad", k=[0.0, 0.0, -1.0, 0.0, 0.0])
    manifest = {"datasets": [good, bad]}

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    seed_tabulated_nk(db, manifest_path=manifest_path)

    assert RamanDatabase(db_path=db_path).list_nk_sources("Si") == ["si-good"]


def test_seed_does_not_touch_sellmeier(tmp_path):
    """Tabulated seeding leaves the sellmeier table intact."""
    from seed_db import seed_tabulated_nk, seed_nk_data

    db_path = tmp_path / "m.db"
    db = RamanDatabase(db_path=db_path)
    db.add_material({"name": "Si", "crystal": "cubic", "bandgap_eV": 1.12, "fR": 0.0})

    seed_nk_data(db)  # seed a Sellmeier for Si
    sellmeier_before = db.get_sellmeier("Si")
    assert sellmeier_before is not None

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_make_manifest()))
    seed_tabulated_nk(db, manifest_path=manifest_path)

    assert db.get_sellmeier("Si") == sellmeier_before
