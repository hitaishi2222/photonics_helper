"""
Tests for the :class:`photonics_helper.dbr.TMM` implementation.

The tests cover the Fresnel interface matrix, the full transfer matrix, the
spectral response (reflectance and transmittance) and the electric‑field profile.
"""

import numpy as np
import pytest

from photonics_helper.base import Wavelength, WavelengthArray
from photonics_helper.dbr import TMM, Block, Material, Pattren


def _simple_material(name: str, n_val: float, wl_nm: float = 1550.0) -> Material:
    """Create a loss‑less material with a constant refractive index.

    Parameters
    ----------
    name : str
        Identifier for the material.
    n_val : float
        Real part of the refractive index (constant over the wavelength range).
    wl_nm : float, optional
        Central wavelength in nanometres used to construct a
        ``WavelengthArray`` spanning ±200 nm (4 points for cubic spline).
    """
    wl_arr = WavelengthArray(np.linspace(wl_nm - 200, wl_nm + 200, 4), "nm")
    n = np.full(4, n_val)
    k = np.zeros(4)
    return Material(name=name, n=n, k=k, wl=wl_arr)


@pytest.fixture
def simple_pattern() -> Pattren:
    """A two‑layer DBR pattern ``AB``.

    * Layer ``A`` – refractive index 2.0, thickness 200 nm.
    * Layer ``B`` – refractive index 1.0 (air), thickness 200 nm.
    """
    mat_a = _simple_material("mat_a", 2.0)
    mat_b = _simple_material("mat_b", 1.0)

    block_a = Block(length=200e-9, material=mat_a, colour="red")
    block_b = Block(length=200e-9, material=mat_b, colour="blue")

    mapping = {"A": block_a, "B": block_b}
    central = Wavelength(1550, "nm")
    return Pattren(style="AB", mapping=mapping, central_wavelength=central)


def test_interface_matrix_normal_incidence(simple_pattern: Pattren):
    """Validate the Fresnel interface matrix at normal incidence.

    For normal incidence the Fresnel equations reduce to:
    ``r = (n1 - n2) / (n1 + n2)`` and ``t = 2 n1 / (n1 + n2)``.
    The interface matrix is ``(1/t) * [[1, r], [r, 1]]``.
    """
    tmm = TMM(pattern=simple_pattern, angle_of_incidence=0.0, polarisation="TE")
    n1, n2 = 1.0, 2.0
    M = tmm._interface_matrix(n1, n2, angle=0.0, pol="TE")
    r = (n1 - n2) / (n1 + n2)  # -1/3
    t = 2 * n1 / (n1 + n2)  # 2/3
    M_expected = (1 / t) * np.array([[1, r], [r, 1]])
    np.testing.assert_allclose(M, M_expected, rtol=1e-12, atol=1e-12)


def test_transfer_matrix_consistency(simple_pattern: Pattren):
    """Check that the transfer matrix is a 2×2 complex matrix.

    The exact numeric value is not critical for this test; we ensure that the
    method runs without error and returns a matrix with the expected shape and
    data type.
    """
    tmm = TMM(pattern=simple_pattern, angle_of_incidence=0.0, polarisation="TE")
    M = tmm.transfer_matrix(wavelength=Wavelength(1550, "nm"))  # 1550 nm
    assert isinstance(M, np.ndarray)
    assert M.shape == (2, 2)
    assert np.iscomplexobj(M)


def test_spectrum_returns_valid_reflectance_and_transmittance(simple_pattern: Pattren):
    """The spectrum method should return reflectance and transmittance arrays.

    For a loss‑less DBR the sum of reflectance and transmittance should be close
    to unity for each wavelength.
    """
    tmm = TMM(pattern=simple_pattern, angle_of_incidence=0.0, polarisation="TE")
    wl_vals = np.linspace(1500, 1600, 5)  # nanometres
    wl_array = WavelengthArray(wl_vals, "nm")
    R, T = tmm.spectrum(wl_array)
    assert R.shape == wl_vals.shape
    assert T.shape == wl_vals.shape
    assert np.all(R > -1e-12) and np.all(R < 1 + 1e-12)
    assert np.all(T > -1e-12) and np.all(T < 1 + 1e-12)
    np.testing.assert_allclose(R + T, np.ones_like(R), rtol=1e-6, atol=1e-6)


def test_field_profile_length(simple_pattern: Pattren):
    """The field profile should contain one value per layer in the pattern."""
    tmm = TMM(pattern=simple_pattern, angle_of_incidence=0.0, polarisation="TE")
    field = tmm.field_profile(wavelength=Wavelength(1550, "nm"))
    # Pattern "AB" has two layers.
    assert field.shape[0] == 2
    assert np.all(field >= 0)
