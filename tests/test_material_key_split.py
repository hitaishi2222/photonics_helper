"""Regression test for fix-material-dataset-key-split.

Every tabulated ``source_key`` returned by ``material_catalog()`` must load
through ``RefractiveIndex.from_material_database``, including keys whose
author field itself contains hyphens. Genuinely unknown keys must still raise
``ValueError`` naming the full key (no silent Sellmeier fallback).
"""

import pytest

from photonics_helper.materials import RefractiveIndex, material_catalog


def test_all_shipped_source_keys_load():
    catalog = [d for d in material_catalog() if d.kind == "tabulated"]
    assert len(catalog) == 30
    failures = []
    for d in catalog:
        try:
            ri = RefractiveIndex.from_material_database(d.source_key)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{d.source_key}: {exc}")
            continue
        wl = ri.wl.as_um
        if len(wl) != d.n_points:
            failures.append(
                f"{d.source_key}: catalog n_points={d.n_points} != loaded {len(wl)}"
            )
    assert not failures, "unresolvable source_keys:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize(
    "key",
    [
        "si3n4-vogt-1-91",  # <mat>-<author>-<suffix> shape
        "gan-lin-wurtzite",  # multi-char author with hyphen
        "gan-lin-zincblende",
        "gaas-franta-300k",
        "si-wang-25c",
        "si-aspnes",  # single-hyphen control
    ],
)
def test_hyphenated_author_key_shapes(key):
    ri = RefractiveIndex.from_material_database(key)
    assert len(ri.wl.as_um) > 0


def test_unknown_key_raises_naming_full_key():
    with pytest.raises(ValueError, match="does-not-exist"):
        RefractiveIndex.from_material_database("does-not-exist")
