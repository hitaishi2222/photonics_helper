import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from photonics_helper.materials import RefractiveIndex
from photonics_helper.base import WavelengthArray


@pytest.fixture
def sample_refractive_index():
    # Create sample data for testing
    wl = np.linspace(1e-6, 2e-6, 100)  # 1-2 µm range
    n = np.ones_like(wl) * 1.5  # constant n = 1.5
    k = np.zeros_like(wl)  # lossless material
    return RefractiveIndex(n=n, k=k, wl=WavelengthArray(wl, "m"))


def test_refractive_index_init(sample_refractive_index):
    """Test initialization of RefractiveIndex class"""
    assert isinstance(sample_refractive_index.n, np.ndarray)
    assert isinstance(sample_refractive_index.k, np.ndarray)
    assert isinstance(sample_refractive_index.wl, WavelengthArray)
    assert (
        len(sample_refractive_index.n)
        == len(sample_refractive_index.k)
        == len(sample_refractive_index.wl)
    )


def test_from_complex():
    """Test creation from complex refractive index"""
    wl = np.linspace(1e-6, 2e-6, 100)
    nk = 1.5 + 0.1j * np.ones_like(wl)
    ri = RefractiveIndex.from_complex(nk=nk, wl=WavelengthArray(wl, "m"))

    assert_array_almost_equal(ri.n, np.real(nk))
    assert_array_almost_equal(ri.k, np.imag(nk))


def test_nk_property(sample_refractive_index):
    """Test complex refractive index property"""
    expected = sample_refractive_index.n + sample_refractive_index.k
    assert_array_almost_equal(sample_refractive_index.nk, expected)


def test_sellmeier():
    """Test Sellmeier equation implementation"""
    # Example coefficients for fused silica
    A0 = 1
    A = [0.6961663, 0.4079426, 0.8974794]
    B = [0.0684043, 0.1162414, 9.896161]
    wl_range = (0.5e-6, 2e-6)  # 0.5-2 µm

    ri = RefractiveIndex.from_sellmeier(
        A0=A0, A=A, B=B, wl_from_to_in_m=wl_range, n_points=100
    )

    assert isinstance(ri, RefractiveIndex)
    assert len(ri.n) == 100
    assert np.all(ri.k == 0)  # should be lossless


def test_sellmeier_coefficient_mismatch():
    """Test error handling for mismatched Sellmeier coefficients"""
    A = [1, 2, 3]
    B = [1, 2]  # One less than A
    wl_range = (0.5e-6, 2e-6)

    with pytest.raises(ValueError):
        RefractiveIndex.from_sellmeier(A0=1, A=A, B=B, wl_from_to_in_m=wl_range)

    with pytest.raises(ValueError):
        RefractiveIndex.from_alt_sellmeier(A0=1, A=A, B=B, wl_from_to_in_m=wl_range)
