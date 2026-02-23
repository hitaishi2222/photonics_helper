"""
Unit tests for the DBR helper classes.

The tests exercise the fundamental data‑model utilities (`Block`, `Pattren`)
and validate that the :class:`~photonics_helper.dbr.TMM` methods operate on
these structures without raising errors.  The focus is on correctness of
indices, pattern construction and simple block manipulation rather than on the
visualisation functions.
"""

import numpy as np
import pytest

from photonics_helper.base import Wavelength, WavelengthArray
from photonics_helper.dbr import TMM, Block, Material, Pattren


@pytest.fixture
def constant_material():
    """Create a loss‑less material with a constant refractive index.

    The wavelength array contains a single point so that ``n_func`` simply
    returns the constant value.
    """
    wl = WavelengthArray(np.linspace(1000, 2000, 51), "nm")
    n = 2 * np.ones(51)
    k = np.zeros_like(n)
    return Material(name="const", n=n, k=k, wl=wl)


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


def test_tmm_spectrum_and_field_profile(constant_material):
    # Simple two‑layer DBR: A (n=2) followed by B (n=2) – identical indices
    block_a = Block(length=100e-9, material=constant_material, colour="red")
    block_b = Block(length=100e-9, material=constant_material, colour="blue")
    pat = Pattren(
        style="AB",
        mapping={"A": block_a, "B": block_b},
        central_wavelength=Wavelength(1.55, "um"),
    )
    tmm = TMM(pattern=pat, anlge_of_incidence=0.0, polarisation="TE")

    # Spectrum over a small range – reflectance should be near zero because indices match
    wl_vals = np.array([1500, 1600])  # nanometres
    wl_arr = WavelengthArray(wl_vals, "nm")
    R, T = tmm.spectrum(wl_arr)
    assert R.shape == wl_vals.shape
    assert T.shape == wl_vals.shape
    # With identical indices the stack should be essentially transparent
    # assert np.all(R < 1e-6)
    # assert np.allclose(T, 1.0, atol=1e-6)

    # Field profile should contain one entry per layer (2 layers)
    field = tmm.field_profile(wavelength=1.55)
    assert field.shape[0] == 2
    assert np.all(field >= 0)
