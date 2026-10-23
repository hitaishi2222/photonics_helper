"""Tests for the azimuthal / spin decomposition — the OAM purity metric (GAP A3).

The three control values are the ones the oam_scg audit established as the
metric's self-test, now pinned as a regression:

* circularly polarised analytic ``LG(0, ℓ)``  -> purity ``1.000`` at one ``(σ, ℓ)``
* the same field, linearly polarised          -> purity exactly ``0.500``
* C₄ leakage of a pure ``ℓ`` into ``ℓ ± 4k``  -> ``< 1e-3``

Plus the sign convention, which is load-bearing wherever an export renames
transverse axes (oam_scg GAP B1): swapping the ``(y, z)`` pair must flip ``ℓ``.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.mode_analysis import (
    azimuthal_charge,
    azimuthal_purity,
    azimuthal_spectrum,
    spin_azimuthal_spectrum,
)

N = 121
Y = np.linspace(-2.0, 2.0, N)
Z = np.linspace(-2.0, 2.0, N)
RMAX = 1.2
W0 = 0.30


def _polar():
    yy, zz = np.meshgrid(Y, Z, indexing="ij")
    return np.hypot(yy, zz), np.arctan2(zz, yy)


def lg(p=0, ell=1):
    """Analytic Laguerre–Gaussian (0, ell) on the test grid."""
    r, phi = _polar()
    return (
        (np.sqrt(2.0) * r / W0) ** abs(ell)
        * np.exp(-(r**2) / W0**2)
        * np.exp(1j * ell * phi)
    )


# ---------------------------------------------------------------- the metric
def test_circularly_polarised_vortex_is_pure():
    u = lg(ell=1)
    amps, ls = spin_azimuthal_spectrum(u, 1j * u, Y, Z, rmax=RMAX)
    assert amps.shape == (2, 2 * 6 + 1) or amps.shape[0] == 2
    p, sigma, ell = azimuthal_purity(amps, ls)
    assert p > 0.999, f"purity {p:.6f} for an exact circularly polarised vortex"
    assert ell == 1
    assert np.isclose(np.sum(np.abs(amps) ** 2), 1.0)


def test_real_field_splits_the_spins_exactly_in_half():
    u = lg(ell=1)
    amps, ls = spin_azimuthal_spectrum(u, np.zeros_like(u), Y, Z, rmax=RMAX)
    p, _sigma, ell = azimuthal_purity(amps, ls)
    assert np.isclose(p, 0.5, atol=1e-12), f"real field purity {p:.6f}, must be 0.500"
    assert ell == 1
    # both spins carry the same charge and the same weight — the definition of 0.500
    assert np.isclose(np.abs(amps[0, np.flatnonzero(ls == 1)[0]]), 1.0 / np.sqrt(2.0))


def test_c4_grid_leakage_is_small():
    u = lg(ell=1)
    amps, ls = spin_azimuthal_spectrum(u, 1j * u, Y, Z, rmax=RMAX)
    p = np.abs(amps) ** 2
    leak = sum(
        p[:, i].sum() for i, ell in enumerate(ls) if ell != 1 and (ell - 1) % 4 == 0
    )
    assert leak < 1e-3, f"C₄ leakage {leak:.2e} is a discretisation alarm, not physics"


@pytest.mark.parametrize("ell", [-2, 0, 3])
def test_charge_sign_and_magnitude_are_recovered(ell):
    r, phi = _polar()
    env = np.exp(-(r**2) / 1.0)
    field = env * np.exp(1j * ell * phi)
    got = azimuthal_charge(field, Y, Z, center=(0.0, 0.0))
    assert got == ell, f"charge {got} for a field with ell = {ell}"


def test_swapping_transverse_axes_flips_the_charge():
    """The convention GAP B1 needs pinned: a transposed export means ell -> -ell."""
    u = lg(ell=2)
    direct = azimuthal_charge(u, Y, Z, center=(0.0, 0.0))
    swapped = azimuthal_charge(np.transpose(u), Y, Z, center=(0.0, 0.0))
    assert direct == 2
    assert swapped == -2


def test_spectrum_and_charge_agree():
    """The two measurements in this module must not disagree on the same field."""
    r, phi = _polar()
    field = np.exp(-(r**2) / 1.0) * np.exp(1j * 3 * phi)
    _amps, ls = azimuthal_spectrum(field, Y, Z, rmax=RMAX)
    peak = int(ls[int(np.argmax(np.abs(_amps)))])
    assert peak == azimuthal_charge(field, Y, Z, center=(0.0, 0.0)) == 3


# ------------------------------------------------------------ input handling
def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        azimuthal_spectrum(np.zeros((N, N - 1), dtype=complex), Y, Z)


def test_undersampled_azimuth_raises():
    with pytest.raises(ValueError, match="nphi"):
        azimuthal_spectrum(np.zeros((N, N), dtype=complex), Y, Z, nphi=8, lmax=6)


def test_zero_field_raises():
    z = np.zeros((N, N), dtype=complex)
    with pytest.raises(ValueError, match="zero"):
        spin_azimuthal_spectrum(z, z, Y, Z)


def test_rmax_default_is_inside_the_window():
    """The default must not sample the whole window: power outside it is meaningful."""
    u = lg(ell=1)
    amps_def, ls = spin_azimuthal_spectrum(u, 1j * u, Y, Z)
    amps_wide, _ = spin_azimuthal_spectrum(u, 1j * u, Y, Z, rmax=RMAX)
    assert azimuthal_purity(amps_def, ls)[0] > 0.999
    assert np.isclose(np.sum(np.abs(amps_wide) ** 2), 1.0)
