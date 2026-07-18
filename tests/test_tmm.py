"""
Tests for the :class:`photonics_helper.dbr.TMM` implementation.

The tests cover the Fresnel interface matrix, the full transfer matrix, the
spectral response (reflectance and transmittance) and the electric‑field profile.
"""

import numpy as np
import pytest

from photonics_helper.base import Wavelength, WavelengthArray, Length
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

    block_a = Block(length=Length(200e-9, "m"), material=mat_a, colour="red")
    block_b = Block(length=Length(200e-9, "m"), material=mat_b, colour="blue")

    mapping = {"A": block_a, "B": block_b}
    central = Wavelength(1550, "nm")
    return Pattren(style="AB", mapping=mapping, central_wavelength=central)


def test_fresnel_reflection_normal_incidence(simple_pattern: Pattren):
    """Validate Fresnel reflection at a single interface at normal incidence.

    A quarter-wave layer of n=2 between air (n=1) on both sides has a
    known analytical reflection coefficient:
        r = (r12 + r23·e^{2iδ}) / (1 + r12·r23·e^{2iδ})
    With r12 = -1/3, r23 = +1/3, δ = π/2 ⇒ e^{2iδ} = -1:
        r = (-1/3 - 1/3) / (1 + 1/9) = -3/5 = -0.6
    """
    wl = Wavelength(1550, "nm")
    n_layer = 2.0
    d = wl.as_m / (4 * n_layer)  # quarter-wave optical thickness (meters)
    mat = _simple_material("layer", n_layer, 1550.0)
    block = Block(length=Length(d, "m"), material=mat, colour="gray")
    pat = Pattren(
        style="Q",
        mapping={"Q": block},
        central_wavelength=Wavelength(1.55, "um"),
    )
    tmm = TMM(pattern=pat, angle_of_incidence=0.0, polarisation="TE")
    r_computed = tmm._reflection_coefficient(wl)

    r_expected = -0.6
    np.testing.assert_allclose(
        np.real(r_computed), r_expected, rtol=1e-10, atol=1e-10
    )


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
