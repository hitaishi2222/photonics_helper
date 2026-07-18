"""Tests for photonics_helper.gnlse.effects."""

import numpy as np
import pytest

from photonics_helper.gnlse import FiberProfile, kerr_step, raman_step, self_steepening_step, tpa_step
from photonics_helper.pulse import TemporalGrid
from photonics_helper.base import Area, Length, Time


@pytest.fixture
def fiber():
    return FiberProfile(n2=1e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(1.0, "m"))


@pytest.fixture
def grid():
    return TemporalGrid(N=256, Tmax=Time(10e-12, "s"))


@pytest.fixture
def A():
    t = np.linspace(-5e-12, 5e-12, 256)
    return np.exp(-t**2 / (2 * (1e-12) ** 2))


def test_kerr_step_phase_shift(fiber, grid, A):
    """Kerr step applies correct phase shift."""
    A_new = kerr_step(A, fiber, grid, dz=1e-3, omega0=2e15)
    # Intensity should be preserved
    assert np.allclose(np.abs(A_new), np.abs(A))
    # Phase should be non-zero for non-zero intensity
    phase_shift = np.angle(A_new / A)
    expected = fiber.n2 * 2e15 * np.abs(A) ** 2 * 1e-3 / (299792458.0 * fiber.A_eff.as_m2)
    assert np.allclose(phase_shift, expected, rtol=1e-10)


def test_kerr_step_no_effect_zero_field(fiber, grid):
    """Kerr step is identity for zero field."""
    A = np.zeros_like(grid.t)
    A_new = kerr_step(A, fiber, grid, dz=1e-3, omega0=2e15)
    assert np.allclose(A_new, 0.0)


def test_raman_step_disabled(fiber, grid, A):
    """Raman step returns unchanged field when disabled."""
    A_new = raman_step(A, fiber, grid, dz=1e-3, include_raman=False)
    assert np.allclose(A_new, A)


def test_raman_step_missing_response(fiber, grid, A):
    """Raman step raises when response is missing but enabled."""
    with pytest.raises(ValueError, match="raman_response is None"):
        raman_step(A, fiber, grid, dz=1e-3, include_raman=True)


def test_self_steepening_step_disabled(fiber, grid, A):
    """Self-steepening step returns unchanged field when disabled."""
    A_new = self_steepening_step(A, fiber, grid, dz=1e-3, omega0=2e15, include_self_steepening=False)
    assert np.allclose(A_new, A)


def test_self_steepening_step_amplitude_modulation(fiber, grid, A):
    """Self-steepening applies frequency-domain (1-ω/ω₀) operator via solve_ivp."""
    from scipy.integrate import solve_ivp

    omega0 = 2e15
    dz = 1.0
    A_new = self_steepening_step(A, fiber, grid, dz=dz, omega0=omega0, include_self_steepening=True)

    gamma = fiber.n2 * omega0 / (299792458.0 * fiber.A_eff.as_m2)
    P_NL = np.abs(A) ** 2
    omega_ratio = grid.w / omega0
    factor = 1j * gamma * (1 - omega_ratio)

    def rhs(z, a):
        return grid.ifft(factor * grid.fft(P_NL * a))

    sol = solve_ivp(
        rhs,
        t_span=(0.0, dz),
        y0=A.astype(complex),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )
    A_expected = sol.y[:, -1]

    assert np.allclose(A_new, A_expected)


def test_tpa_step_disabled(fiber, grid, A):
    """TPA step returns unchanged field when disabled."""
    A_new, U_new = tpa_step(A, fiber, grid, dz=1e-3, include_tpa=False)
    assert np.allclose(A_new, A)
    assert U_new == 0.0


def test_tpa_step_attenuation(fiber, grid):
    """TPA step attenuates field when sigma_tpa > 0."""
    fiber.sigma_tpa = 1e-11
    A = np.ones(256) * 1e6  # high intensity
    A_new, U_new = tpa_step(A, fiber, grid, dz=1e-3, include_tpa=True, U=0.0)
    # Field should be attenuated
    assert np.all(np.abs(A_new) <= np.abs(A))
    # Carrier density should increase
    assert U_new > 0
