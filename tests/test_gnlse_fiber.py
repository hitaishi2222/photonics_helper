"""Tests for photonics_helper.gnlse.fiber."""

import pytest
from photonics_helper.gnlse import FiberProfile


def test_fiber_profile_construction():
    """FiberProfile can be constructed with required fields."""
    fp = FiberProfile(n2=1.0e-19, alpha=1e-5, A_eff=5e-11, length=1.0)
    assert fp.n2 == 1.0e-19
    assert fp.alpha == 1e-5
    assert fp.A_eff == 5e-11
    assert fp.length == 1.0


def test_fiber_profile_defaults():
    """Optional fields default correctly."""
    fp = FiberProfile(n2=1.0e-19, alpha=1e-5, A_eff=5e-11, length=1.0)
    assert fp.sigma_tpa == 0.0
    assert fp.carrier_lifetime is None
    assert fp.raman_response is None


def test_fiber_profile_optional_fields():
    """Optional fields can be set explicitly."""
    fp = FiberProfile(
        n2=1.0e-19,
        alpha=1e-5,
        A_eff=5e-11,
        length=1.0,
        sigma_tpa=1e-11,
        carrier_lifetime=1e-9,
    )
    assert fp.sigma_tpa == 1e-11
    assert fp.carrier_lifetime == 1e-9
    assert fp.raman_response is None
