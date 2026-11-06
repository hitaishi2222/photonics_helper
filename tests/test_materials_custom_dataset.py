"""Registering a custom n(λ) table — and why ``AsSe2`` has no catalogue row.

Naskar et al., *Phys. Scr.* **101** 385502 (2026) need the AsSe₂ core index
(``n(2 µm) = 2.818``), but the paper publishes only the functional form of its
Sellmeier fit (eq. 2) — no coefficients — and cites refs [25, 34] as the source
for the *curves*; neither ref carries an AsSe₂ Sellmeier set either. So the
catalogue row stays absent: a fitted row would be calibration, not data.

What must work instead is the user-supplied path: hold a tabulated ``n(λ)`` you
can cite, wrap it once, and use it like any catalogue material — without a
database write, so a user table can never be mistaken for a shipped entry.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Wavelength
from photonics_helper.core.data import MaterialsDatabase
from photonics_helper.core.materials import Material, OpticalMaterial
from photonics_helper.materials import RefractiveIndex

#: A stand-in for the paper's AsSe₂ curve — shape only, anchored on its one
#: published calibration point. Deliberately *not* offered as Sellmeier data.
ASSE2_WL_UM = np.linspace(1.0, 6.0, 51)
ASSE2_N = 2.818 + 0.05 * (1.0 / ASSE2_WL_UM - 0.5) - 0.01 * (ASSE2_WL_UM - 2.0)


def _index() -> RefractiveIndex:
    return RefractiveIndex.from_samples(ASSE2_WL_UM, ASSE2_N)


class TestFromSamples:
    """``RefractiveIndex.from_samples`` is the registration entry point."""

    def test_anchor_point_and_window_survive_registration(self):
        idx = _index()
        assert idx.n_func(2.0) == pytest.approx(2.818, abs=1e-12)
        assert idx.valid_range_um == (1.0, 6.0)
        # Index-only tables get zero absorption, not a missing array.
        assert np.all(np.asarray(idx.k) == 0.0)

    def test_scalar_k_is_broadcast_and_array_k_is_taken_as_is(self):
        idx = RefractiveIndex.from_samples(ASSE2_WL_UM, ASSE2_N, 1e-4)
        assert idx.k_func(2.0) == pytest.approx(1e-4)
        k = np.linspace(0.0, 1e-3, ASSE2_WL_UM.size)
        idx2 = RefractiveIndex.from_samples(ASSE2_WL_UM, ASSE2_N, k)
        assert np.allclose(np.asarray(idx2.k), k)

    def test_malformed_tables_are_refused(self):
        """A transcription slip must fail loudly, not spline into nonsense."""
        with pytest.raises(ValueError, match="at least 4"):
            RefractiveIndex.from_samples([1.0, 2.0, 3.0], [2.9, 2.8, 2.7])
        with pytest.raises(ValueError, match="strictly increasing"):
            RefractiveIndex.from_samples([1.0, 2.0, 1.5, 3.0], [2.9, 2.8, 2.85, 2.7])
        with pytest.raises(ValueError, match="equal lengths"):
            RefractiveIndex.from_samples([1.0, 2.0, 3.0, 4.0], [2.9, 2.8, 2.7])

    def test_material_wrapper_and_extrapolation_policy(self):
        material = Material(
            name="AsSe2", index=_index(), source="user table (from the paper's ref)"
        )
        assert isinstance(material, OpticalMaterial)
        assert material.n_func(2.0) == pytest.approx(2.818, abs=1e-12)
        assert material.valid_range_um == (1.0, 6.0)
        with pytest.raises(ValueError, match=r"AsSe2: Index valid only between"):
            material.n_func(8.0)
        assert np.isfinite(material.n_func(8.0, extrapolate="allow"))

    def test_custom_table_feeds_the_dispersion_path(self):
        """The point of registering it: n(λ) → D(λ) → β₂ without a DB row."""
        from photonics_helper.fiber import Dispersion

        idx = _index()
        dispersion = Dispersion.from_neff(
            idx.n, idx.wl, Wavelength(2.0, "um"), ignore_fit_error=True
        )
        assert np.all(np.isfinite(dispersion.values))
        assert np.isfinite(dispersion.get_beta2(2000))

    def test_registration_does_not_fabricate_a_catalogue_row(self):
        """The data gap is real: wrapping a table is not a sourced entry."""
        _index()  # a user table exists in this process
        db = MaterialsDatabase()
        assert db.get_sellmeier("AsSe2") is None
        assert "AsSe2" not in db.list_materials()
