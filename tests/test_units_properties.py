"""Property-based tests for the foundation primitives (Hypothesis).

Unit conversions and the grid FFT are the foundation of every downstream
calculation, so they are checked as *properties* over their whole input domain
rather than on a handful of examples:

- conversion round-trips (λ↔ν↔ω↔E↔wavenumber) within floating-point tolerance;
- the defining physical relations (λν = c, E = hν, ω = 2πν);
- scalar/array consistency;
- grid FFT invertibility and Parseval.

Tolerances are relative and loose enough to be portable across BLAS/scipy
builds, but tight enough to catch sign/scale/convention regressions.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from photonics_helper.base import (
    C_MS,
    H_PLANCK,
    AngularFrequency,
    Energy,
    Frequency,
    Time,
    Wavelength,
    WavelengthArray,
    Wavenumber,
)
from photonics_helper.core.grids import TemporalGrid

REL = 1e-9

# 1 nm .. 1 mm; 3e11 Hz (1 mm) .. 3e18 Hz (1 nm); corresponding photon energies.
WAVELENGTHS_M = st.floats(
    min_value=1e-9, max_value=1e-3, allow_nan=False, allow_infinity=False
)
FREQUENCIES_HZ = st.floats(
    min_value=1e11, max_value=1e18, allow_nan=False, allow_infinity=False
)
ENERGIES_J = st.floats(
    min_value=1e-25, max_value=1e-15, allow_nan=False, allow_infinity=False
)

EVEN_SIZES = st.integers(min_value=16, max_value=128).map(lambda n: 2 * n)


# ─── Unit conversions ─────────────────────────────────────────────────────


@given(WAVELENGTHS_M)
@settings(max_examples=100, deadline=None)
def test_wavelength_frequency_roundtrip(wl_m: float) -> None:
    back = Wavelength(wl_m, "m").to_freq().to_wl()
    assert math.isclose(back.as_m, wl_m, rel_tol=REL)


@given(WAVELENGTHS_M)
@settings(max_examples=100, deadline=None)
def test_wavelength_defining_relations(wl_m: float) -> None:
    wl = Wavelength(wl_m, "m")
    nu = wl.to_freq().as_Hz
    assert math.isclose(wl_m * nu, C_MS, rel_tol=REL)
    assert math.isclose(wl.to_energy().as_J, H_PLANCK * nu, rel_tol=REL)
    assert math.isclose(wl.to_omega().as_rad_s, 2 * math.pi * nu, rel_tol=REL)


@given(WAVELENGTHS_M)
@settings(max_examples=50, deadline=None)
def test_wavelength_energy_and_wavenumber_roundtrips(wl_m: float) -> None:
    wl = Wavelength(wl_m, "m")
    assert math.isclose(wl.to_energy().to_wl().as_m, wl_m, rel_tol=REL)
    assert math.isclose(wl.to_wn().to_wl().as_m, wl_m, rel_tol=REL)
    assert math.isclose(wl.to_omega().to_wl().as_m, wl_m, rel_tol=REL)


@given(FREQUENCIES_HZ)
@settings(max_examples=100, deadline=None)
def test_frequency_roundtrips(nu_hz: float) -> None:
    nu = Frequency(nu_hz, "Hz")
    assert math.isclose(nu.to_wl().to_freq().as_Hz, nu_hz, rel_tol=REL)
    assert math.isclose(nu.to_omega().to_freq().as_Hz, nu_hz, rel_tol=REL)
    assert math.isclose(nu.to_energy().as_J, H_PLANCK * nu_hz, rel_tol=REL)
    assert math.isclose(nu.to_time().to_freq().as_Hz, nu_hz, rel_tol=REL)


@given(ENERGIES_J)
@settings(max_examples=100, deadline=None)
def test_energy_roundtrips(energy_j: float) -> None:
    energy = Energy(energy_j, "J")
    assert math.isclose(energy.to_freq().to_energy().as_J, energy_j, rel_tol=REL)
    assert math.isclose(energy.to_wl().to_energy().as_J, energy_j, rel_tol=REL)
    assert math.isclose(energy.to_omega().to_energy().as_J, energy_j, rel_tol=REL)


@given(WAVELENGTHS_M)
@settings(max_examples=50, deadline=None)
def test_angular_frequency_roundtrip(wl_m: float) -> None:
    omega = Wavelength(wl_m, "m").to_omega()
    assert isinstance(omega, AngularFrequency)
    assert math.isclose(omega.to_wl().as_m, wl_m, rel_tol=REL)


@given(WAVELENGTHS_M)
@settings(max_examples=50, deadline=None)
def test_wavenumber_roundtrip(wl_m: float) -> None:
    wn = Wavelength(wl_m, "m").to_wn()
    assert isinstance(wn, Wavenumber)
    assert math.isclose(wn.to_wl().as_m, wl_m, rel_tol=REL)
    assert math.isclose(wn.to_freq().as_Hz, Wavelength(wl_m, "m").to_freq().as_Hz, rel_tol=REL)


@given(st.lists(WAVELENGTHS_M, min_size=1, max_size=8))
@settings(max_examples=40, deadline=None)
def test_array_scalar_consistency(values: list[float]) -> None:
    array = WavelengthArray(np.array(values), "m")
    from_array = array.to_freq().as_Hz
    for i, value in enumerate(values):
        scalar = Wavelength(value, "m").to_freq().as_Hz
        assert math.isclose(from_array[i], scalar, rel_tol=REL)


# ─── Grid FFT ─────────────────────────────────────────────────────────────


def _random_field(n: int) -> np.ndarray:
    rng = np.random.default_rng(n)
    return rng.normal(size=n) + 1j * rng.normal(size=n)


@given(EVEN_SIZES)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_grid_fft_roundtrip(n: int) -> None:
    grid = TemporalGrid(N=n, Tmax=Time(1.0, "ps"))
    field = _random_field(n)
    assert np.allclose(grid.ifft(grid.fft(field)), field, rtol=1e-9, atol=1e-9)


@given(EVEN_SIZES)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_grid_parseval(n: int) -> None:
    grid = TemporalGrid(N=n, Tmax=Time(1.0, "ps"))
    field = _random_field(n)
    spectral = grid.fft(field)

    energy_time = np.sum(np.abs(field) ** 2) * grid.dt
    energy_freq = np.sum(np.abs(spectral) ** 2) * grid.dw / (2 * math.pi)
    assert math.isclose(energy_freq, energy_time, rel_tol=1e-9)
