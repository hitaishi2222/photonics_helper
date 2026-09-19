"""The material catalogue: completeness, columns and the name filter.

The catalogue is the discovery entry point for the bundled data, so it must
list *everything*: every tabulated n/k dataset and every Sellmeier equation,
including rows that have no Raman spec (axis sub-rows such as ``LiNbO3_er`` and
optical materials such as Silicon or Sapphire).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from photonics_helper import (
    MaterialDataset,
    material_catalog,
    print_material_catalog,
)
from photonics_helper.raman import RamanDatabase

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "photonics_helper" / "materials.db"


def _counts() -> tuple[int, int]:
    con = sqlite3.connect(DB_PATH)
    try:
        sellmeier = con.execute("SELECT COUNT(*) FROM sellmeier").fetchone()[0]
        tabulated = con.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT DISTINCT material, source FROM nk_data WHERE source IS NOT NULL"
            ")"
        ).fetchone()[0]
    finally:
        con.close()
    return sellmeier, tabulated


# ─── Completeness ─────────────────────────────────────────────────────────


def test_catalog_lists_every_dataset() -> None:
    sellmeier, tabulated = _counts()
    catalog = material_catalog()
    assert sum(1 for d in catalog if d.kind == "sellmeier") == sellmeier
    assert sum(1 for d in catalog if d.kind == "tabulated") == tabulated
    assert len(catalog) == sellmeier + tabulated


def test_sellmeier_rows_without_raman_spec_are_included() -> None:
    """Regression: iterating raman_specs dropped these datasets."""
    names = {d.material for d in material_catalog()}
    for material in ("Silicon", "Sapphire", "Germanium"):
        assert material in names, f"{material} missing from the catalogue"
    assert "GaP" in names  # tabulated data without a Raman spec


def test_birefringent_axis_rows_are_labelled() -> None:
    linbo3 = [d for d in material_catalog("LiNbO3") if d.kind == "sellmeier"]
    axes = {d.axis for d in linbo3}
    assert "extraordinary" in axes
    assert "ordinary" in axes
    # The base material name is shown, not the raw `_er` / `_or` row name.
    assert all(d.material == "LiNbO3" for d in linbo3)


def test_list_sellmeier_datasets_matches_the_table() -> None:
    sellmeier, _ = _counts()
    assert len(RamanDatabase().list_sellmeier_datasets()) == sellmeier


# ─── Columns the user asked for: name, range, DOI (+ licence) ─────────────


def test_every_dataset_has_a_wavelength_range() -> None:
    for dataset in material_catalog():
        wl_range = dataset.wavelength_range_um
        assert wl_range is not None, f"{dataset.material} has no wavelength range"
        low, high = wl_range
        assert 0 < low <= high, f"{dataset.material}: bad range {wl_range}"


def test_every_dataset_has_a_licence() -> None:
    for dataset in material_catalog():
        assert dataset.license, f"{dataset.material} ({dataset.kind}) has no licence"


def test_doi_is_extracted_from_the_citation() -> None:
    by_source = {d.source: d for d in material_catalog() if d.source_key}
    hagemann = by_source["al2o3-hagemann"]
    assert hagemann.doi == "10.1364/JOSA.65.000742"
    # Provenance rows are backfilled with DOIs, not left NULL.
    assert any(d.doi for d in material_catalog())


def test_kind_values_are_valid() -> None:
    assert {d.kind for d in material_catalog()} == {"tabulated", "sellmeier"}
    assert all(isinstance(d, MaterialDataset) for d in material_catalog())


# ─── Name filter ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("needle", ["sil", "SIL", "Sil"])
def test_filter_is_case_insensitive_substring(needle: str) -> None:
    rows = material_catalog(needle)
    assert rows, f"filter {needle!r} returned nothing"
    assert all(needle.lower() in d.material.lower() for d in rows)


def test_filter_narrows_the_result() -> None:
    assert len(material_catalog("Silica")) < len(material_catalog())
    assert material_catalog("no-such-material-xyz") == []


def test_print_catalog_renders_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    print_material_catalog("sapph")
    out = capsys.readouterr().out
    assert "Sapphire" in out
    assert "Wavelength" in out
    assert "DOI" in out


def test_print_catalog_reports_empty_filter(capsys: pytest.CaptureFixture[str]) -> None:
    print_material_catalog("no-such-material-xyz")
    out = capsys.readouterr().out
    assert "0 dataset(s)" in out
