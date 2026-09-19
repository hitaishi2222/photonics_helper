"""Golden-value regression tests for the bundled material data.

These pin the physics values so a future database regeneration (or a typo in a
seed table) cannot change them silently. Tolerances are stated per value; they
are wide enough for interpolation/coefficient rounding, narrow enough to catch
a wrong dataset or material.

Reference values were captured from the shipped `materials.db` on 2026-09-19.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from photonics_helper.materials import RefractiveIndex
from photonics_helper.phonon import PHONON_MATERIALS
from photonics_helper.raman import RamanSpec

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "photonics_helper" / "materials.db"


def test_row_counts_are_stable() -> None:
    con = sqlite3.connect(DB_PATH)
    try:
        counts = {
            table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("nk_data", "sellmeier", "raman_specs", "phonon_modes")
        }
    finally:
        con.close()
    assert counts["nk_data"] == 24767
    assert counts["sellmeier"] == 39
    assert counts["raman_specs"] == 44
    assert counts["phonon_modes"] == 52


# ─── Refractive index ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("material", "wavelength_um", "expected_n", "tol"),
    [
        ("Silica", 0.633, 1.457012, 1e-5),
        ("Silica", 1.55, 1.444025, 1e-5),
        ("Si3N4-Ligentec", 0.633, 2.039435, 1e-3),
        ("Si3N4-Ligentec", 1.55, 1.996280, 1e-3),
        ("LiNbO3", 0.775, 2.186595, 5e-4),  # extraordinary (default axis)
        ("LiNbO3", 1.55, 2.139469, 5e-4),
    ],
)
def test_refractive_index_golden(
    material: str, wavelength_um: float, expected_n: float, tol: float
) -> None:
    ri = RefractiveIndex.from_material_database(material)
    assert ri.n_func(wavelength_um) == pytest.approx(expected_n, abs=tol)


def test_linbo3_birefringence_golden() -> None:
    ordinary = RefractiveIndex.from_material_database("LiNbO3", axis="ordinary")
    extraordinary = RefractiveIndex.from_material_database("LiNbO3", axis="extraordinary")

    assert ordinary.n_func(1.55) == pytest.approx(2.211111, abs=5e-4)
    assert ordinary.n_func(0.775) == pytest.approx(2.258658, abs=5e-4)
    # n_o - n_e at 1.55 um is ~0.072
    delta = ordinary.n_func(1.55) - extraordinary.n_func(1.55)
    assert delta == pytest.approx(0.072, abs=3e-3)


def test_silicon_tabulated_golden() -> None:
    """Si (Aspnes tabulated dataset) n and k at 0.633 um."""
    si = RefractiveIndex.from_material_database("si-aspnes")
    assert si.n_func(0.633) == pytest.approx(3.881030, abs=1e-4)
    assert si.k_func(0.633) == pytest.approx(0.018819, abs=1e-4)


# ─── Raman specifications ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("material", "shift", "linewidth", "fR"),
    [
        ("Silica", 440.0, 45.0, 0.18),
        ("GeAsSe", 250.0, 50.0, 0.5),
    ],
)
def test_raman_spec_golden(
    material: str, shift: float, linewidth: float, fR: float
) -> None:
    spec = RamanSpec.from_database(material)
    assert spec.raman_shift_cm == pytest.approx(shift, abs=1e-9)
    assert spec.raman_linewidth_cm == pytest.approx(linewidth, abs=1e-9)
    assert spec.fR == pytest.approx(fR, abs=1e-9)


# ─── Phonon modes ─────────────────────────────────────────────────────────


def test_linbo3_phonon_golden() -> None:
    modes = PHONON_MATERIALS["LiNbO3"]
    assert len(modes) == 7
    strongest = max(modes, key=lambda m: m.relative_strength)
    assert strongest.shift_cm.as_1_cm == pytest.approx(254.0, abs=1e-9)
    assert strongest.relative_strength == pytest.approx(1.0, abs=1e-9)
