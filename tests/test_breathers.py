"""Tests for photonics_helper.breathers (exact NLSE soliton-on-background solutions)."""

import numpy as np
import pytest

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.breathers import (
    SolitonOnBackground,
    akhmediev_breather,
    general_sfb,
    kuznetsov_ma,
    peregrine_soliton,
    sfb_peak_ratio,
    sfb_spatial_period,
    sfb_temporal_period,
)
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import TemporalGrid


def _residual(func, xi, tau, h=1e-4):
    """Finite-difference residual of i psi_xi + 0.5 psi_tautau + |psi|^2 psi."""
    psi_xi = (func(xi + h, tau) - func(xi - h, tau)) / (2 * h)
    psi_tt = (func(xi, tau + h) - 2 * func(xi, tau) + func(xi, tau - h)) / h**2
    return abs(1j * psi_xi + 0.5 * psi_tt + abs(func(xi, tau)) ** 2 * func(xi, tau))


# ─── analytic solutions ──────────────────────────────────────────────────────


def test_peregrine_peak_ratio_is_nine():
    assert abs(peregrine_soliton(0.0, 0.0)) ** 2 == pytest.approx(9.0)
    # decays to the background far away
    assert abs(peregrine_soliton(0.0, 50.0)) ** 2 == pytest.approx(1.0, abs=1e-3)


def test_peregrine_solves_nlse():
    f = lambda x, t: peregrine_soliton(x, t)  # noqa: E731
    worst = max(_residual(f, x, t) for x in (-2.0, 0.0, 1.3) for t in (0.0, 0.5, 2.0))
    assert worst < 1e-5


@pytest.mark.parametrize("a", [0.25, 0.66])
def test_general_sfb_solves_nlse(a):
    f = lambda x, t: general_sfb(x, t, a)  # noqa: E731
    worst = max(_residual(f, x, t) for x in (0.1, 0.8, 1.9) for t in (0.0, 0.5, 1.5))
    assert worst < 1e-5


def test_peregrine_is_the_a_to_half_limit():
    for xi, tau in [(0.0, 0.0), (0.7, 0.5), (0.0, 2.0)]:
        limit = general_sfb(xi, tau, 0.5 - 1e-4)
        assert abs(limit - peregrine_soliton(xi, tau)) < 1e-3


def test_peak_ratios_match_exact_extremes():
    assert sfb_peak_ratio(0.25) == pytest.approx(5.828427124746193)
    assert sfb_peak_ratio(0.5) == pytest.approx(9.0)
    assert sfb_peak_ratio(0.66) == pytest.approx(10.875650117230437)


def test_regime_validation():
    with pytest.raises(ValueError):
        akhmediev_breather(0.0, 0.0, 0.66)
    with pytest.raises(ValueError):
        kuznetsov_ma(0.0, 0.0, 0.25)
    with pytest.raises(ValueError):
        general_sfb(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        general_sfb(0.0, 0.0, 1.2)


def test_periods():
    a_km = 0.66
    assert sfb_spatial_period(a_km) == pytest.approx(
        2 * np.pi / np.sqrt(8 * a_km * (2 * a_km - 1))
    )
    a_ab = 0.25
    assert sfb_temporal_period(a_ab) == pytest.approx(
        2 * np.pi / (2 * np.sqrt(1 - 2 * a_ab))
    )
    with pytest.raises(ValueError):
        sfb_spatial_period(0.25)
    with pytest.raises(ValueError):
        sfb_temporal_period(0.66)


# ─── physical mapping ────────────────────────────────────────────────────────


def _km_parameters():
    # Kibler et al., Sci. Rep. 2, 463 (2012): SMF-28, P0 = 0.7 W
    return SolitonOnBackground(beta2=-21.8e-27, gamma=1.3e-3, P0=0.7)


def test_physical_mapping_matches_paper():
    sob = _km_parameters()
    assert sob.L_NL == pytest.approx(1098.90109, rel=1e-6)
    assert sob.T0 * 1e12 == pytest.approx(4.89449, rel=1e-5)
    assert sob.spatial_period_m(0.66) / 1e3 == pytest.approx(5.31186, rel=1e-4)
    assert sob.peak_power(0.66) == pytest.approx(7.612955, rel=1e-5)


def test_requires_anomalous_dispersion():
    with pytest.raises(ValueError):
        SolitonOnBackground(beta2=20e-27, gamma=1.3e-3, P0=0.7)


def test_initial_wave_matches_field():
    sob = _km_parameters()
    grid = TemporalGrid(N=512, Tmax=Time(60e-12, "s"))
    wave = sob.initial_wave(grid, 0.66, z0=0.0)
    expected = np.asarray(sob.field(0.0, grid.t, 0.66), dtype=complex)
    np.testing.assert_allclose(wave.envelope_field, expected)
    assert wave.peak_power() == pytest.approx(sob.peak_power(0.66), rel=1e-6)


def test_breather_propagates_with_gnlse():
    """The exact KM field is preserved by the solver over a fraction of a period."""
    sob = _km_parameters()
    grid = TemporalGrid(N=1024, Tmax=Time(80e-12, "s"))
    wave = sob.initial_wave(grid, 0.66, z0=0.0)
    fiber = FiberProfile.from_gamma(
        gamma=sob.gamma,
        n2=2.6e-20,
        omega0=2 * np.pi * C_MS / 1550e-9,
        length=Length(sob.L_NL, "m"),
    )
    solver = GNLSESolver(
        pulse=wave,
        fiber=fiber,
        betas=np.array([sob.beta2 * 1e24, 0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=2000, nsaves=2)
    A = solver.evolution[-1].envelope_field
    expected = np.asarray(sob.field(sob.L_NL, grid.t, 0.66), dtype=complex)
    intensity = np.abs(A) ** 2
    expected_intensity = np.abs(expected) ** 2
    rel = np.linalg.norm(intensity - expected_intensity) / np.linalg.norm(expected_intensity)
    assert rel < 5e-3


def test_wavelength_argument_is_used():
    sob = _km_parameters()
    grid = TemporalGrid(N=128, Tmax=Time(40e-12, "s"))
    wave = sob.initial_wave(grid, 0.66, wavelength=Wavelength(1060, "nm"))
    assert wave.central_wavelength.as_nm == pytest.approx(1060.0)
