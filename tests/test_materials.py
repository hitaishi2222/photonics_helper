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
    return RefractiveIndex(n=n, k=k, wl=WavelengthArray(wl, "m"))  # type: ignore[arg-type]


def test_refractive_index_init(sample_refractive_index):
    """Test initialization of RefractiveIndex class"""
    assert isinstance(sample_refractive_index.n, np.ndarray)
    assert isinstance(sample_refractive_index.k, np.ndarray)
    assert isinstance(sample_refractive_index.wl, WavelengthArray)
    assert (
        len(sample_refractive_index.n)
        == len(sample_refractive_index.k)
        == len(sample_refractive_index.wl.value)  # type: ignore[arg-type]
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
    expected = sample_refractive_index.n + 1j * sample_refractive_index.k
    assert_array_almost_equal(sample_refractive_index.nk, expected)


def test_sellmeier():
    """Test Sellmeier equation implementation"""
    # Example coefficients for fused silica (Malitson).
    # NOTE: from_sellmeier expects B_i = (resonance wavelength)^2 in um^2.
    A0 = 1
    A = [0.6961663, 0.4079426, 0.8974794]
    B = [0.004679148, 0.013512075, 97.953962]
    wl_range = (0.5, 2)  # 0.5-2 µm

    ri = RefractiveIndex.from_sellmeier(
        A0=A0, A=A, B=B, wl_from_to_in_um=wl_range, n_points=100
    )

    assert isinstance(ri, RefractiveIndex)
    assert len(ri.n) == 100
    assert np.all(ri.k == 0)  # should be lossless


def test_sellmeier_coefficient_mismatch():
    """Test error handling for mismatched Sellmeier coefficients"""
    A = [1.0, 2.0, 3.0]
    B = [1.0, 2.0]  # One less than A
    wl_range = (0.5e-6, 2e-6)

    with pytest.raises(ValueError):
        RefractiveIndex.from_sellmeier(A0=1, A=A, B=B, wl_from_to_in_um=wl_range)

    with pytest.raises(ValueError):
        RefractiveIndex.from_alt_sellmeier(A0=1, A=A, B=B, wl_from_to_in_um=wl_range)


# ---------- group-dispersion tests ----------

def _silica_sellmeier():
    # Malitson fused silica; B_i = (resonance wavelength)^2 in um^2
    A0 = 1
    A = [0.6961663, 0.4079426, 0.8974794]
    B = [0.004679148, 0.013512075, 97.953962]
    return RefractiveIndex.from_sellmeier(
        A0=A0, A=A, B=B, wl_from_to_in_um=(0.5, 2.0), n_points=200
    )


def test_dn_dlambda_tabulated():
    """dn/dλ is finite and negative for normal dispersion (decreasing n)."""
    wl = np.linspace(0.5e-6, 2.0e-6, 200)
    # Silica-like: n decreases with wavelength (B_i = pole^2, um^2)
    A0, A, B = 1, [0.6961663, 0.4079426, 0.8974794], [0.004679148, 0.013512075, 97.953962]
    n_vals = np.sqrt(A0 + np.array(A)[:, None] * wl[None, :] ** 2 /
                     (wl[None, :] ** 2 - np.array(B)[:, None])).sum(axis=0)
    n_vals = n_vals / np.max(n_vals) * 1.45  # scale to realistic range
    ri = RefractiveIndex(n=n_vals, k=np.zeros_like(wl), wl=WavelengthArray(wl, "m"))
    val = ri.dn_dlambda(1.0)
    assert np.isfinite(val)
    assert val < 0  # normal dispersion: n decreases as λ increases


def test_dn_dlambda_sellmeier():
    """dn/dλ works on Sellmeier-constructed material."""
    ri = _silica_sellmeier()
    val = ri.dn_dlambda(1.0)
    assert np.isfinite(val)
    assert val < 0  # silica has normal dispersion at 1 μm


def test_group_index_scalar_valid():
    """group_index returns finite value within valid range."""
    ri = _silica_sellmeier()
    val = ri.group_index(1.0)
    assert np.isfinite(val)
    assert val > 1.0  # normal dispersion in transparent region


def test_group_index_scalar_out_of_range():
    """group_index raises ValueError for out-of-range wavelength."""
    ri = _silica_sellmeier()
    with pytest.raises(ValueError):
        ri.group_index(5.0)


def test_group_index_sellmeier_reasonable():
    """n_g > 1 for silica at 1.55 μm (normal dispersion)."""
    ri = _silica_sellmeier()
    n_g = ri.group_index(1.55)
    assert n_g > 1.0
    assert n_g < 3.0  # physically reasonable bound


def test_group_index_array_shape_and_consistency():
    """Array output matches grid size; midpoint matches scalar call."""
    ri = _silica_sellmeier()
    arr = ri.group_index_array()
    assert len(arr) == 200
    mid_idx = 100
    mid_wl = ri.wl.as_um[mid_idx]
    assert abs(arr[mid_idx] - ri.group_index(mid_wl)) < 1e-10


def test_group_velocity_scalar_subluminal():
    """v_g < c for normal dispersion; out-of-range raises ValueError."""
    ri = _silica_sellmeier()
    v_g = ri.group_velocity(1.0)
    c = 2.99792458e8
    assert 0 < v_g < c
    with pytest.raises(ValueError):
        ri.group_velocity(10.0)


def test_group_velocity_array_shape():
    """v_g array has correct shape and values in physical range."""
    ri = _silica_sellmeier()
    arr = ri.group_velocity_array()
    assert len(arr) == 200
    assert np.all(arr > 1e8) and np.all(arr < 3e8)
