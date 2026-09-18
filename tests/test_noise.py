"""Tests for photonics_helper.noise (stochastic sources)."""

import numpy as np
import pytest

from photonics_helper.base import Time, Wavelength
from photonics_helper.noise import (
    add_ase_noise,
    add_noise,
    ase_noise_field,
    complex_gaussian_noise,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


def _grid(N=1024):
    return TemporalGrid(N=N, Tmax=Time(20e-12, "s"))


def _cw_wave(grid, power=0.7):
    env = Envelope(
        shape="custom",
        peak_amplitude=float(np.sqrt(power)),
        pulse_width=Time(1e-12, "s"),
        func=lambda t, T0, A0: np.full_like(t, A0, dtype=complex),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550, "nm"))


# ─── complex_gaussian_noise ──────────────────────────────────────────────────


def test_complex_gaussian_noise_rms():
    grid = _grid()
    n = complex_gaussian_noise(grid, 0.01, seed=0)
    assert n.shape == (grid.N,)
    assert np.iscomplexobj(n)
    assert np.std(n) == pytest.approx(0.01, rel=1e-6)
    # unit-variance per quadrature (finite-sample scatter ~1/sqrt(N))
    assert np.std(n.real) == pytest.approx(0.01 / np.sqrt(2), rel=5e-2)
    assert np.std(n.imag) == pytest.approx(0.01 / np.sqrt(2), rel=5e-2)


def test_complex_gaussian_noise_reproducible():
    grid = _grid()
    a = complex_gaussian_noise(grid, 0.02, seed=42)
    b = complex_gaussian_noise(grid, 0.02, seed=42)
    c = complex_gaussian_noise(grid, 0.02, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rng_and_seed_conflict():
    grid = _grid()
    with pytest.raises(ValueError):
        complex_gaussian_noise(grid, 0.01, rng=np.random.default_rng(0), seed=0)


# ─── add_noise ───────────────────────────────────────────────────────────────


def test_add_noise_contrast_and_no_mutation():
    grid = _grid()
    wave = _cw_wave(grid, 0.7)
    field_before = np.array(wave.envelope_field)
    noisy = add_noise(wave, 0.01, seed=1)
    # original untouched
    np.testing.assert_array_equal(wave.envelope_field, field_before)
    # relative amplitude contrast
    field = noisy.envelope_field
    assert np.std(field) / np.abs(field.mean()) == pytest.approx(0.01, rel=1e-2)
    # same seed reproduces
    again = add_noise(wave, 0.01, seed=1)
    np.testing.assert_array_equal(noisy.envelope_field, again.envelope_field)


def test_add_noise_zero_field_raises():
    grid = _grid()
    env = Envelope(
        shape="custom",
        peak_amplitude=0.0,
        pulse_width=Time(1e-12, "s"),
        func=lambda t, T0, A0: np.zeros_like(t, dtype=complex),
    )
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550, "nm"))
    with pytest.raises(ValueError):
        add_noise(wave, 0.01, seed=0)


# ─── ase_noise_field ─────────────────────────────────────────────────────────


def test_ase_noise_level_is_correct():
    """Every non-DC spectral bin sits `level_dB` below the pump DC line."""
    grid = _grid()
    P0, level = 0.7, -50.0
    field = ase_noise_field(grid, P0, level, seed=7)
    spec = np.abs(np.asarray(grid.fft(field)))
    pump_spec = np.abs(
        np.asarray(grid.fft(np.full(grid.N, np.sqrt(P0), dtype=complex)))
    )
    dc = pump_spec[grid.N // 2]
    expected = dc * 10 ** (level / 20.0)
    # DC removed, all other bins at the requested level
    assert spec[grid.N // 2] == pytest.approx(0.0, abs=1e-25)
    others = np.delete(spec, grid.N // 2)
    np.testing.assert_allclose(others, expected, rtol=1e-10)


def test_ase_noise_reproducible_and_phase_random():
    grid = _grid()
    a = ase_noise_field(grid, 0.7, -50.0, seed=3)
    b = ase_noise_field(grid, 0.7, -50.0, seed=3)
    np.testing.assert_array_equal(a, b)
    # random phase: the real/imag parts are comparable in magnitude
    assert abs(abs(a.real).max() - abs(a.imag).max()) / abs(a.real).max() < 0.2


def test_add_ase_noise_does_not_mutate():
    grid = _grid()
    wave = _cw_wave(grid, 0.7)
    before = np.array(wave.envelope_field)
    noisy = add_ase_noise(wave, -50.0, seed=5)
    np.testing.assert_array_equal(wave.envelope_field, before)
    assert not np.array_equal(noisy.envelope_field, before)


def test_add_ase_noise_reference_power_defaults_to_peak():
    grid = _grid()
    wave = _cw_wave(grid, 0.7)
    a = add_ase_noise(wave, -50.0, seed=9)
    b = add_ase_noise(wave, -50.0, reference_power=wave.peak_power(), seed=9)
    np.testing.assert_array_equal(a.envelope_field, b.envelope_field)
