"""Tests for photonics_helper.gnlse.solver.GNLSESolver."""

import numpy as np
import pytest

from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.base import Wavelength


@pytest.fixture
def setup():
    """Create a basic pulse, fiber, and solver for testing."""
    grid = TemporalGrid(N=256, Tmax=20e-12)
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=1e-12)
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(n2=1e-19, alpha=1e-5, A_eff=5e-11, length=1e-3)
    betas = np.array([0.02])  # normal dispersion, 20 ps²/km
    return pulse, fiber, betas


def test_solver_construction(setup):
    """Solver constructs with all required parameters."""
    pulse, fiber, betas = setup
    solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=betas)
    assert solver.pulse is pulse
    assert solver.fiber is fiber
    assert solver.include_raman is True
    assert solver.include_self_steepening is False
    assert solver.include_tpa is False


def test_solver_construction_with_flags(setup):
    """Solver accepts effect flags."""
    pulse, fiber, betas = setup
    solver = GNLSESolver(
        pulse=pulse, fiber=fiber, betas=betas,
        include_raman=False,
        include_self_steepening=True,
        include_tpa=True,
    )
    assert solver.include_raman is False
    assert solver.include_self_steepening is True
    assert solver.include_tpa is True


def test_solver_propagate_populates_evolution(setup):
    """propagate() populates evolution list."""
    pulse, fiber, betas = setup
    solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=betas, include_raman=False)
    solver.propagate(num_steps=5)
    assert len(solver.evolution) == 6


def test_solver_spectra_vs_z(setup):
    """spectra_vs_z returns correct shape after propagation."""
    pulse, fiber, betas = setup
    solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=betas, include_raman=False)
    solver.propagate(num_steps=10)
    omega, spectra = solver.spectra_vs_z
    assert len(omega) == 256
    assert spectra.shape == (11, 256)


def test_solver_spectra_vs_z_before_propagate(setup):
    """spectra_vs_z raises before propagate()."""
    pulse, fiber, betas = setup
    solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=betas)
    with pytest.raises(RuntimeError, match="Call propagate"):
        _ = solver.spectra_vs_z


def test_solver_energy_conservation(setup):
    """Pure Kerr solver conserves energy."""
    pulse, fiber, betas = setup
    solver = GNLSESolver(
        pulse=pulse, fiber=fiber, betas=betas,
        include_raman=False, include_self_steepening=False, include_tpa=False,
    )
    initial_energy = np.sum(np.abs(pulse.envelope_field) ** 2) * pulse.grid.dt
    solver.propagate(num_steps=20)
    final_field = solver.evolution[-1].envelope_field
    final_energy = np.sum(np.abs(final_field) ** 2) * pulse.grid.dt
    assert np.isclose(final_energy, initial_energy, rtol=1e-6)
