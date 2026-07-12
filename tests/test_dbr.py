"""
Unit tests for the DBR helper classes.

The tests exercise the fundamental data‑model utilities (`Block`, `Pattren`)
and validate that the :class:`~photonics_helper.dbr.TMM` methods operate on
these structures without raising errors.  The focus is on correctness of
indices, pattern construction, simple block manipulation, and TMM physics
(correctness of R/T, energy conservation, polarization, oblique incidence,
and absorbing layers).
"""

import numpy as np
import pytest

from photonics_helper.base import Wavelength, WavelengthArray
from photonics_helper.dbr import TMM, Block, Material, Pattren


@pytest.fixture
def constant_material():
    """Create a loss‑less material with a constant refractive index.

    The wavelength array spans 1000–2000 nm so that ``n_func`` simply
    returns the constant value across the test range.
    """
    wl = WavelengthArray(np.linspace(1000, 2000, 51), "nm")
    n = 2 * np.ones(51)
    k = np.zeros_like(n)
    return Material(name="const", n=n, k=k, wl=wl)


@pytest.fixture
def high_index_material():
    """n = 2.5, k = 0 – high-index layer for DBR."""
    wl = WavelengthArray(np.linspace(1000, 2000, 51), "nm")
    n = 2.5 * np.ones(51)
    k = np.zeros_like(n)
    return Material(name="high", n=n, k=k, wl=wl)


@pytest.fixture
def low_index_material():
    """n = 1.5, k = 0 – low-index layer for DBR."""
    wl = WavelengthArray(np.linspace(1000, 2000, 51), "nm")
    n = 1.5 * np.ones(51)
    k = np.zeros_like(n)
    return Material(name="low", n=n, k=k, wl=wl)


@pytest.fixture
def absorbing_material():
    """n = 2.0, k = 0.1 – lossy material for absorption tests."""
    wl = WavelengthArray(np.linspace(1000, 2000, 51), "nm")
    n = 2.0 * np.ones(51)
    k = 0.1 * np.ones(51)
    return Material(name="absorb", n=n, k=k, wl=wl)


# ── Data model tests ───────────────────────────────────────────────


def test_block_position_setter_and_deleter(constant_material):
    block = Block(length=200e-9, material=constant_material, colour="red")
    # Correct position assignment should succeed
    block.position = (0.0, 200e-9)
    assert block.position == (0.0, 200e-9)

    # Deleting the position resets it to the default (0, length)
    del block.position
    assert block.position == (0.0, 200e-9)

    # Mismatched length should raise a ValueError
    with pytest.raises(ValueError):
        block.position = (0.0, 100e-9)


def test_pattren_construction_and_get_index(constant_material):
    # Two blocks with different lengths
    block_a = Block(length=100e-9, material=constant_material, colour="red")
    block_b = Block(length=150e-9, material=constant_material, colour="blue")
    mapping = {"A": block_a, "B": block_b}
    pat = Pattren(
        style="AB", mapping=mapping, central_wavelength=Wavelength(1.55, "um")
    )

    # Length should be the sum of individual block lengths
    assert pytest.approx(pat.length, rel=1e-12) == 250e-9

    # Positions are set according to the style order
    positions = pat._get_positions()
    assert positions[0] == (0.0, 100e-9)
    assert positions[1] == (100e-9, 250e-9)

    # get_index returns refractive indices for a given wavelength (in microns)
    n_vals, k_vals = pat.get_index(pat.central_wavelength.as_um)
    # Both blocks use the same constant material with n=2.0, k=0.0
    assert pytest.approx(n_vals, rel=1e-6) == [2.0, 2.0]
    assert pytest.approx(k_vals, rel=1e-6) == [0, 0]


def test_pattren_add_and_remove_block(constant_material):
    block_a = Block(length=100e-9, material=constant_material, colour="red")
    pat = Pattren(
        style="A", mapping={"A": block_a}, central_wavelength=Wavelength(1.55, "um")
    )

    # Add a new block B after the first position
    block_b = Block(length=200e-9, material=constant_material, colour="blue")
    pat.add_block(block_b, mapping="B", index=1)
    assert pat.style == "AB"
    assert pat.length == 300e-9

    # Remove block A (index 0) – style should become "B"
    pat.remove_block(index=0)
    assert pat.style == "B"
    assert pat.length == 200e-9


# ── TMM physics tests ──────────────────────────────────────────────


def test_tmm_slab_reflects_at_normal_incidence(constant_material):
    """A slab with n=2 in air reflects via Fresnel interfaces.

    Two blocks with identical material still have entry/exit interfaces.
    The stack acts as a Fabry–Perot etalon: R oscillates with wavelength.
    """
    block_a = Block(length=100e-9, material=constant_material, colour="red")
    block_b = Block(length=100e-9, material=constant_material, colour="blue")
    pat = Pattren(
        style="AB",
        mapping={"A": block_a, "B": block_b},
        central_wavelength=Wavelength(1.55, "um"),
    )
    tmm = TMM(pattern=pat, angle_of_incidence=0.0, polarisation="TE")

    # Scan a wide range
    wl_vals = np.linspace(1000, 2000, 500)  # nm
    wl_arr = WavelengthArray(wl_vals, "nm")
    R, T = tmm.spectrum(wl_arr)

    assert R.shape == wl_vals.shape
    assert T.shape == wl_vals.shape
    # Energy conservation
    assert np.allclose(R + T, 1.0, atol=1e-10)
    # R oscillates: should have distinct maxima and minima
    r_range = np.max(R) - np.min(R)
    assert r_range > 0.1, f"Expected Fabry-Perot oscillations, R range = {r_range}"
    # Single-interface reflectivity for n=2: R₁ = ((2-1)/(2+1))² = 1/9 ≈ 0.111
    # Fabry-Perot max: R_max = 4R₁/(1+R₁)² = 0.36
    assert np.max(R) > 0.3, f"Expected R max near 0.36, got {np.max(R)}"
    assert np.max(R) < 0.5, f"R max should be bounded, got {np.max(R)}"


def test_tmm_energy_conservation_lossless(high_index_material, low_index_material):
    """For a lossless stack: R + T = 1 at every wavelength."""
    # Quarter-wave stack: 5 periods of (H L) centered at 1550 nm
    # λ/4 thickness: d = λ₀ / (4n)
    d_h = 1550e-9 / (4 * 2.5)
    d_l = 1550e-9 / (4 * 1.5)

    blocks = []
    style_chars = []
    mapping = {}
    for i in range(5):
        h_block = Block(length=d_h, material=high_index_material, colour="high")
        l_block = Block(length=d_l, material=low_index_material, colour="low")
        blocks.extend([h_block, l_block])
        style_chars.extend(["H", "L"])
        mapping["H"] = h_block
        mapping["L"] = l_block

    pat = Pattren(
        style="".join(style_chars),
        mapping=mapping,
        central_wavelength=Wavelength(1.55, "um"),
    )
    tmm = TMM(pattern=pat, angle_of_incidence=0.0, polarisation="TE")

    wl_vals = np.linspace(1100, 1900, 200)  # nm — within material valid range (1000–2000 nm)
    wl_arr = WavelengthArray(wl_vals, "nm")
    R, T = tmm.spectrum(wl_arr)

    sum_rt = R + T
    assert np.allclose(sum_rt, 1.0, atol=1e-10), (
        f"Energy conservation violated: max|R+T-1| = {np.max(np.abs(sum_rt - 1.0))}"
    )


def test_tmm_oblique_incidence_blue_shift(high_index_material, low_index_material):
    """Oblique incidence shifts the DBR stopband to shorter wavelengths."""
    # Build a DBR with 8 periods
    d_h = 1550e-9 / (4 * 2.5)
    d_l = 1550e-9 / (4 * 1.5)

    blocks = []
    style_chars = []
    mapping = {}
    for i in range(8):
        h_block = Block(length=d_h, material=high_index_material, colour="high")
        l_block = Block(length=d_l, material=low_index_material, colour="low")
        blocks.extend([h_block, l_block])
        style_chars.extend(["H", "L"])
        mapping["H"] = h_block
        mapping["L"] = l_block

    def make_tmm(angle_deg: float):
        pat = Pattren(
            style="".join(style_chars),
            mapping=mapping,
            central_wavelength=Wavelength(1.55, "um"),
        )
        return TMM(
            pattern=pat,
            angle_of_incidence=np.deg2rad(angle_deg),
            polarisation="TE",
        )

    wl_vals = np.linspace(1100, 1900, 500)  # nm — within material valid range
    wl_arr = WavelengthArray(wl_vals, "nm")

    tmm_0 = make_tmm(0.0)
    tmm_45 = make_tmm(45.0)
    R_0, _ = tmm_0.spectrum(wl_arr)
    R_45, _ = tmm_45.spectrum(wl_arr)

    # Find peak reflectivity wavelength for each angle
    peak_0 = wl_vals[np.argmax(R_0)]
    peak_45 = wl_vals[np.argmax(R_45)]

    # Oblique incidence should blue-shift the stopband
    assert peak_45 < peak_0, (
        f"Expected blue shift at 45°, got peak_0={peak_0:.0f}nm, peak_45={peak_45:.0f}nm"
    )


def test_tmm_te_tm_difference(high_index_material, low_index_material):
    """TE and TM polarizations give different reflectivity at oblique incidence."""
    d_h = 1550e-9 / (4 * 2.5)
    d_l = 1550e-9 / (4 * 1.5)

    blocks = []
    style_chars = []
    mapping = {}
    for i in range(5):
        h_block = Block(length=d_h, material=high_index_material, colour="high")
        l_block = Block(length=d_l, material=low_index_material, colour="low")
        blocks.extend([h_block, l_block])
        style_chars.extend(["H", "L"])
        mapping["H"] = h_block
        mapping["L"] = l_block

    pat_te = Pattren(
        style="".join(style_chars),
        mapping=mapping,
        central_wavelength=Wavelength(1.55, "um"),
    )
    pat_tm = Pattren(
        style="".join(style_chars),
        mapping=mapping,
        central_wavelength=Wavelength(1.55, "um"),
    )

    tmm_te = TMM(pattern=pat_te, angle_of_incidence=np.deg2rad(30.0), polarisation="TE")
    tmm_tm = TMM(pattern=pat_tm, angle_of_incidence=np.deg2rad(30.0), polarisation="TM")

    wl_vals = np.linspace(1200, 1900, 200)  # nm — within material valid range
    wl_arr = WavelengthArray(wl_vals, "nm")
    R_te, _ = tmm_te.spectrum(wl_arr)
    R_tm, _ = tmm_tm.spectrum(wl_arr)

    # TE and TM should differ at 30° incidence (different Fresnel coefficients)
    assert not np.allclose(R_te, R_tm, atol=1e-6), (
        "TE and TM reflectivity should differ at oblique incidence"
    )


def test_tmm_field_profile_shape(constant_material):
    """Field profile returns one value per layer in the pattern."""
    block_a = Block(length=100e-9, material=constant_material, colour="red")
    block_b = Block(length=150e-9, material=constant_material, colour="blue")
    block_c = Block(length=120e-9, material=constant_material, colour="green")
    pat = Pattren(
        style="ABC",
        mapping={"A": block_a, "B": block_b, "C": block_c},
        central_wavelength=Wavelength(1.55, "um"),
    )
    tmm = TMM(pattern=pat, angle_of_incidence=0.0, polarisation="TE")

    field = tmm.field_profile(wavelength=Wavelength(1.55, "um"))

    # One value per block in the pattern
    assert field.shape[0] == 3
    assert np.all(field >= 0), "Field magnitudes must be non-negative"


def test_tmm_absorbing_layer_energy_deficit(absorbing_material):
    """With k > 0, R + T < 1 (energy is absorbed by the layer)."""
    block_a = Block(length=200e-9, material=absorbing_material, colour="red")
    pat = Pattren(
        style="A",
        mapping={"A": block_a},
        central_wavelength=Wavelength(1.55, "um"),
    )
    tmm = TMM(pattern=pat, angle_of_incidence=0.0, polarisation="TE")

    wl_vals = np.array([1500, 1550, 1600])  # nm
    wl_arr = WavelengthArray(wl_vals, "nm")
    R, T = tmm.spectrum(wl_arr)

    # R and T individually in [0, 1]
    assert np.all(R >= 0) and np.all(R <= 1)
    assert np.all(T >= 0) and np.all(T <= 1)
    # Absorption A = 1 - R - T should be positive for k > 0
    absorption = 1.0 - R - T
    assert np.any(absorption > 0), (
        f"Expected absorption for k=0.1, got absorption = {absorption}"
    )
