"""Tests for photonics_helper.wave_breaking (normal-dispersion wave breaking)."""

import numpy as np
import pytest

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.wave_breaking import (
    WaveBreaking,
    detect_oscillation_onset,
    detect_steepening_onset,
    dispersion_length,
    edge_steepness,
    gaussian_edge_steepness,
    nonlinear_length,
    wave_breaking_distance,
)

# Representative normal-dispersion silica case used by the reproduction.
_BETA2 = 20e-27  # s^2/m  (+20 ps^2/km)
_GAMMA = 1.5e-3  # 1/(W m)
_P0 = 10.0  # W
_T0 = 10e-12  # s


def test_lengths_and_distance():
    L_D = dispersion_length(_BETA2, _T0)
    L_NL = nonlinear_length(_GAMMA, _P0)
    assert L_D == pytest.approx(5000.0)
    assert L_NL == pytest.approx(66.6667, rel=1e-4)
    z_wb = wave_breaking_distance(_BETA2, _GAMMA, _P0, _T0)
    assert z_wb == pytest.approx(np.exp(0.75) / 2 * np.sqrt(L_D * L_NL))
    assert z_wb == pytest.approx(611.13, rel=1e-3)


def test_requires_normal_dispersion():
    with pytest.raises(ValueError):
        wave_breaking_distance(-_BETA2, _GAMMA, _P0, _T0)
    with pytest.raises(ValueError):
        WaveBreaking(beta2=-_BETA2, gamma=_GAMMA, P0=_P0, T0=_T0)
    with pytest.raises(ValueError):
        dispersion_length(0.0, _T0)


def test_gaussian_edge_steepness():
    t = np.linspace(-5 * _T0, 5 * _T0, 4001)
    intensity = np.exp(-(t**2) / _T0**2)
    value = edge_steepness(intensity=intensity, t=t, T0=_T0)
    assert value == pytest.approx(gaussian_edge_steepness, rel=1e-3)
    assert gaussian_edge_steepness == pytest.approx(0.8577638, rel=1e-6)


def test_edge_steepness_accepts_field():
    t = np.linspace(-5 * _T0, 5 * _T0, 4001)
    field = np.exp(-(t**2) / (2 * _T0**2)).astype(complex)
    assert edge_steepness(field=field, t=t, T0=_T0) == pytest.approx(
        gaussian_edge_steepness, rel=1e-3
    )


def test_detect_steepening_onset():
    z = np.linspace(0, 100, 101)
    steep = np.full_like(z, gaussian_edge_steepness)
    steep[40:] = 1.3 * gaussian_edge_steepness
    assert detect_steepening_onset(z, steep, factor=1.10) == pytest.approx(40.0)
    assert np.isnan(detect_steepening_onset(z, np.full_like(z, 0.1)))


def test_detect_oscillation_onset():
    t = np.linspace(-10, 10, 401)
    single = np.exp(-(t**2)).astype(complex)
    double = (np.exp(-(t**2)) + 0.5 * np.exp(-((t - 3) ** 2))).astype(complex)
    z = np.array([0.0, 1.0])
    assert np.isnan(detect_oscillation_onset(z, [single, single]))
    assert detect_oscillation_onset(z, [single, double]) == pytest.approx(1.0)


def test_wavebreaking_analyze_with_gnlse():
    """Steepening onset appears near z_WB; oscillations follow a few sqrt(L_D L_NL)."""
    wb = WaveBreaking(beta2=_BETA2, gamma=_GAMMA, P0=_P0, T0=_T0)
    grid = TemporalGrid(N=2048, Tmax=Time(140e-12, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=np.sqrt(_P0), pulse_width=Time(_T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1060, "nm"))
    fiber = FiberProfile.from_gamma(
        gamma=_GAMMA,
        n2=2.6e-20,
        omega0=2 * np.pi * C_MS / 1060e-9,
        length=Length(4 * wb.z_WB, "m"),
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([_BETA2 * 1e24, 0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
        step_size=Length(wb.z_WB / 100, "m"),
    )
    solver.propagate(num_steps=400, nsaves=301)
    fields = [w.envelope_field for w in solver.evolution]
    result = wb.analyze(solver.z_array, fields, grid.t)
    assert result["z_WB_m"] == pytest.approx(wb.z_WB)
    # onset lies below the analytic z_WB, steepness grows well above the Gaussian value
    assert 0.2 * wb.z_WB < result["z_onset_m"] < wb.z_WB
    assert result["peak_steepness"] > 1.5 * gaussian_edge_steepness
    # oscillations arrive within a few sqrt(L_D L_NL)
    assert result["z_oscillation_m"] / wb.sqrt_LD_LNL < 4.5
