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
    assert counts["sellmeier"] == 41
    assert counts["raman_specs"] == 45
    assert counts["phonon_modes"] == 59


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
        # chalcogenides + fluoride cladding (G1 rows; see seed_db SELLMEIER_MATERIALS)
        ("GeAsSe", 2.0, 2.627924, 5e-4),  # Ma 2013 Ge11.5As24Se64.5
        ("GeAsS", 2.0, 2.298889, 5e-4),  # Ma 2013 Ge11.5As24S64.5
        ("As2S3", 2.0, 2.426149, 5e-4),  # Rodney 1958
        ("As2Se3", 2.0, 2.808214, 5e-4),  # Thompson 2008
        ("MgF2", 2.0, 1.367835, 5e-4),  # Dodge 1984 o-ray (B=λ_res^2!)
        # Phase M Ge-Sb-Se pair (Wu/Ding/Peng, Appl. Sci. 13, 12923 (2023))
        ("GeSbSe", 2.0, 2.677601, 5e-4),  # Ge12.5Sb15Se72.5 core
    ],
)
def test_refractive_index_golden(
    material: str, wavelength_um: float, expected_n: float, tol: float
) -> None:
    ri = RefractiveIndex.from_material_database(material)
    assert ri.n_func(wavelength_um) == pytest.approx(expected_n, abs=tol)


def test_linbo3_birefringence_golden() -> None:
    ordinary = RefractiveIndex.from_material_database("LiNbO3", axis="ordinary")
    extraordinary = RefractiveIndex.from_material_database(
        "LiNbO3", axis="extraordinary"
    )

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
        ("GeSbSe", 198.1, 9.33, 0.4615),
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


def test_gesbse_phonon_golden() -> None:
    """The 7-mode Ge-Sb-Se fit must stay complete and correctly ordered."""
    modes = PHONON_MATERIALS["GeSbSe"]
    assert len(modes) == 7
    strongest = max(modes, key=lambda m: m.relative_strength)
    assert strongest.shift_cm.as_1_cm == pytest.approx(198.10, abs=1e-9)
    assert strongest.relative_strength == pytest.approx(26.79, abs=1e-9)
    shifts = [m.shift_cm.as_1_cm for m in modes]
    assert shifts == sorted(shifts)


def test_linbo3_phonon_golden() -> None:
    modes = PHONON_MATERIALS["LiNbO3"]
    assert len(modes) == 7
    strongest = max(modes, key=lambda m: m.relative_strength)
    assert strongest.shift_cm.as_1_cm == pytest.approx(254.0, abs=1e-9)
    assert strongest.relative_strength == pytest.approx(1.0, abs=1e-9)
