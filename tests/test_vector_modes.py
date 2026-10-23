"""Tests for the exact circular-fibre vector mode solver (GAP A1/D1).

Three contracts:

* weakly-guiding limit: ``u`` equals the scalar ``lp_roots`` value of the
  matching LP group as Δn → 0 (< 1e-4 at Δn = 1e-4);
* strong-guiding reference: reproduces the published Okamoto solver in
  ``reproductions/xing_2022_multimode_chg`` where that solver returns a root,
  and finds the roots its ±0.5 bracket misses (the documented HE21 NaN);
* labelling and cut-off behaviour, including orders above the fundamental.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.vector_modes import VectorMode, vector_modes
from reproductions.poletti_2008_multimode.overlap import lp_neff, lp_roots
from reproductions.xing_2022_multimode_chg.xing2022 import vector_u as published_u


class _ConstantGlass:
    """Stand-in for the reproduction's Sellmeier glass: fixed n."""

    def __init__(self, n: float):
        self._n = n

    def n(self, lam: float) -> float:
        return self._n


def _radius_for_v(V: float, n_core: float, n_clad: float, lam_um: float) -> float:
    return V * lam_um / (2.0 * np.pi * np.sqrt(n_core**2 - n_clad**2))


# --------------------------------------------------------- weak guidance
@pytest.mark.parametrize("V", [3.0, 5.0, 8.0])
def test_weakly_guiding_reduces_to_lp_roots(V):
    n_core, n_clad, lam = 1.5001, 1.5, 2.0
    radius = _radius_for_v(V, n_core, n_clad, lam)
    lp = lp_roots(V, l_max=5, m_max=3)
    kinds = ["HE11", "HE21", "HE31", "HE41", "EH11", "EH21", "TE01", "TM01"]
    for mode in vector_modes(n_core, n_clad, radius, lam, kinds=kinds):
        target = lp.get((mode.ell, mode.m))
        if target is None:
            continue
        assert abs(mode.u - target) < 1e-4, (
            f"{mode.kind}: u={mode.u:.8f} vs LP={target:.8f} at V={V}"
        )


def test_lp_neff_helper_matches_the_root_and_points_at_the_vector_solver():
    V, n_core, n_clad = 5.0, 1.5001, 1.5
    n_eff, beta = lp_neff(V, n_core, n_clad, ell=1, m=1, wavelength_um=2.0)
    u = lp_roots(V, l_max=2, m_max=2)[(1, 1)]
    expected = np.sqrt(n_core**2 - (n_core**2 - n_clad**2) * (u / V) ** 2)
    assert n_eff == pytest.approx(expected, abs=1e-12)
    assert beta == pytest.approx(2.0 * np.pi / 2e-6 * n_eff, rel=1e-12)
    # beta is unavailable without a wavelength, and the helper says so
    n_eff_only, beta_nan = lp_neff(V, n_core, n_clad, ell=1, m=1)
    assert n_eff_only == pytest.approx(n_eff)
    assert np.isnan(beta_nan)


# ------------------------------------------------------ strong guidance
def test_reproduces_published_solver_where_it_converges():
    n_core, n_clad, lam = 2.65, 1.0, 2.0
    radius = _radius_for_v(5.0, n_core, n_clad, lam)
    core, clad = _ConstantGlass(n_core), _ConstantGlass(n_clad)
    for kind in ("HE11", "TE01", "HE21"):
        ours = vector_modes(n_core, n_clad, radius, lam, kinds=[kind])[0]
        published = published_u(kind, lam, core=core, clad=clad, a=radius)
        assert np.isfinite(published)
        assert ours.u == pytest.approx(published, abs=1e-9)


def test_he21_is_found_where_the_half_scan_returns_nan():
    """The GAP A1 failure mode: the vector root sits >0.5 from the LP guess."""
    n_core, n_clad, lam = 2.65, 1.0, 2.0
    radius = _radius_for_v(3.4, n_core, n_clad, lam)
    ours = vector_modes(n_core, n_clad, radius, lam, kinds=["HE21"])[0]
    assert np.isfinite(ours.u) and np.isfinite(ours.n_eff)
    old = published_u(
        "HE21", lam, core=_ConstantGlass(n_core), clad=_ConstantGlass(n_clad), a=radius
    )
    assert np.isnan(old), "control: the ±0.5 bracket's failure must still be real"


def test_tm01_root_is_found_where_the_half_scan_returns_nan():
    n_core, n_clad, lam = 2.65, 1.0, 2.0
    radius = _radius_for_v(5.0, n_core, n_clad, lam)
    ours = vector_modes(n_core, n_clad, radius, lam, kinds=["TM01"])[0]
    assert np.isfinite(ours.u)


# ------------------------------------------------ labels, cut-off, units
@pytest.mark.parametrize(
    "kind,ell,nu,m",
    [
        ("HE11", 0, 1, 1),
        ("HE21", 1, 2, 1),
        ("HE12", 0, 1, 2),
        ("EH11", 2, 1, 1),
        ("HE31", 2, 3, 1),
        ("EH21", 3, 2, 1),
        ("TE01", 1, 0, 1),
        ("TM01", 1, 0, 1),
    ],
)
def test_labels_follow_the_lp_group(kind, ell, nu, m):
    mode = vector_modes(1.5001, 1.5, 200.0, 2.0, kinds=[kind])[0]
    assert (mode.kind, mode.ell, mode.nu, mode.m) == (kind, ell, nu, m)
    assert isinstance(mode, VectorMode)


def test_cutoff_mode_is_returned_with_nan_not_dropped():
    n_core, n_clad, lam = 1.5001, 1.5, 2.0
    radius = _radius_for_v(2.0, n_core, n_clad, lam)
    mode = vector_modes(n_core, n_clad, radius, lam, kinds=["HE21"])[0]
    assert np.isnan(mode.u) and np.isnan(mode.n_eff) and np.isnan(mode.beta)
    # ... whereas the default enumeration omits it
    kinds = {m.kind for m in vector_modes(n_core, n_clad, radius, lam)}
    assert "HE21" not in kinds


def test_beta_is_si_and_consistent_with_neff():
    mode = vector_modes(2.65, 1.0, 0.5, 2.0, kinds=["HE11"])[0]
    assert mode.beta == pytest.approx(2.0 * np.pi / 2e-6 * mode.n_eff, rel=1e-12)


def test_invalid_geometry_raises():
    with pytest.raises(ValueError, match="n_core > n_clad"):
        vector_modes(1.4, 1.5, 1.0, 2.0)
    with pytest.raises(ValueError, match="radius_um"):
        vector_modes(1.5, 1.4, -1.0, 2.0)
    with pytest.raises(ValueError, match="mode family"):
        vector_modes(1.5, 1.4, 1.0, 2.0, kinds=["LP01"])
