"""Tests for photonics_helper.gnlse.fiber."""

from photonics_helper.gnlse import FiberProfile
from photonics_helper.base import Area, Length, Time


def test_fiber_profile_construction():
    """FiberProfile can be constructed with required fields."""
    fp = FiberProfile(n2=1.0e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(1.0, "m"))
    assert fp.n2 == 1.0e-19
    assert fp.alpha == 1e-5
    assert fp.A_eff.as_m2 == 5e-11
    assert fp.length.as_m == 1.0


def test_fiber_profile_defaults():
    """Optional fields default correctly."""
    fp = FiberProfile(n2=1.0e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(1.0, "m"))
    assert fp.sigma_tpa == 0.0
    assert fp.carrier_lifetime is None
    assert fp.raman_response is None


def test_fiber_profile_optional_fields():
    """Optional fields can be set explicitly."""
    fp = FiberProfile(
        n2=1.0e-19,
        alpha=1e-5,
        A_eff=Area(5e-11, "m^2"),
        length=Length(1.0, "m"),
        sigma_tpa=1e-11,
        carrier_lifetime=Time(1e-9, "s"),
    )
    assert fp.sigma_tpa == 1e-11
    assert fp.carrier_lifetime.as_s == 1e-9
    assert fp.raman_response is None
