"""Catalogue integrity for ``RAMAN_MATERIALS`` (openspec C5).

The catalogue is what `RamanSpec.from_database` returns, so a row is a published
measurement or a derived guess — and the difference has to be visible. These
tests pin the four properties the missing-provenance defect needed:

1. every row states where it comes from;
2. a row that carries a relaxation-time pair (``tau1``/``tau2``) carries both,
   positive, with an ``fR`` in range — and the response builder *uses* the pair
   instead of re-deriving it from shift/linewidth;
3. a row without a pair still yields a usable response (the derived times);
4. every row round-trips through `from_database`, which is the only public door.

Row names are catalogue keys, not formulas — ``AsSe2 (Raman only)`` is an
explicit "no index row" marker, and the pair it carries is one citing paper's
value for a *different* composition.
"""

from __future__ import annotations

import pytest

from photonics_helper.raman.reference import RAMAN_MATERIALS
from photonics_helper.raman.spec import RamanSpec


def test_every_row_states_its_source():
    missing = [
        name
        for name, row in RAMAN_MATERIALS.items()
        if not (isinstance(row.get("references"), str) and row["references"].strip())
    ]
    assert not missing, f"catalogue rows without a source: {missing}"


def test_a_time_pair_is_complete_and_used_by_the_response():
    with_pair = {
        name: row
        for name, row in RAMAN_MATERIALS.items()
        if "tau1" in row or "tau2" in row
    }
    assert with_pair, "no row carries a published (tau1, tau2) pair"

    for name, row in with_pair.items():
        assert {"tau1", "tau2"} <= set(row), f"{name}: half a pair"
        assert row["tau1"] > 0.0 and row["tau2"] > 0.0, name
        fR = row.get("fR")
        if fR is not None:
            assert 0.0 < fR < 1.0, f"{name}: fR={fR}"

        response = RamanSpec.from_database(name).response()
        # The builder used to re-derive the times from shift/linewidth and drop
        # the published pair; a derived pair would not match to round-off.
        assert response.tau1 == pytest.approx(row["tau1"], rel=1e-9), name
        assert response.tau2 == pytest.approx(row["tau2"], rel=1e-9), name


def test_a_row_without_a_pair_still_derives_its_times():
    derived = [
        name
        for name, row in RAMAN_MATERIALS.items()
        if "tau1" not in row and "tau2" not in row and row.get("raman_shift_cm")
    ]
    assert derived, "no row exercises the derived path"
    for name in derived:
        response = RamanSpec.from_database(name).response()
        assert response.tau1 > 0.0 and response.tau2 > 0.0, name


def test_every_row_round_trips_through_from_database():
    for name in RAMAN_MATERIALS:
        spec = RamanSpec.from_database(name)
        assert spec.name, name
        assert spec.raman_shift_cm > 0.0, name
