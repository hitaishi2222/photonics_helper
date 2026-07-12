"""Tests for photonics_helper.gnlse.engine.SplitStepEngine."""

import numpy as np
import pytest

from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.base import Wavelength, Frequency


@pytest.fixture
def setup():
    """Create a basic pulse and fiber for testing."""
    grid = TemporalGrid(N=256, Tmax=20e-12)
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=1e-12)
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(n2=1e-19, alpha=1e-5, A_eff=5e-11, length=1e-3)
    betas = np.array([0.02])  # beta2 = +20 ps²/km (= 0.02 ps²/m)
    return pulse, fiber, betas


def test_engine_construction(setup):
    """Engine constructs without error."""
    pulse, fiber, betas = setup
    engine = SplitStepEngine(pulse=pulse, fiber=fiber, betas=betas)
    assert engine.A is not None
    assert len(engine.A) == 256


def test_engine_propagate_returns_evolution(setup):
    """propagate() populates evolution list."""
    pulse, fiber, betas = setup
    engine = SplitStepEngine(pulse=pulse, fiber=fiber, betas=betas)
    engine.propagate(num_steps=5)
    assert len(engine.evolution) == 6  # initial + 5 steps


def test_engine_propagate_produces_spectra(setup):
    """propagate() produces spectra_vs_z."""
    pulse, fiber, betas = setup
    engine = SplitStepEngine(pulse=pulse, fiber=fiber, betas=betas)
    engine.propagate(num_steps=5)
    omega, spectra = engine.spectra_vs_z
    assert omega.shape[0] == 256
    assert spectra.shape == (6, 256)


def test_engine_constant_step_size(setup):
    """Engine uses constant step size when specified."""
    pulse, fiber, betas = setup
    engine = SplitStepEngine(
        pulse=pulse, fiber=fiber, betas=betas, step_size=1e-4
    )
    assert engine.step_size == 1e-4


def test_engine_energy_conservation_pure_kerr(setup):
    """Pure Kerr GNLSE conserves pulse energy."""
    pulse, fiber, betas = setup
    engine = SplitStepEngine(
        pulse=pulse, fiber=fiber, betas=betas,
        include_raman=False, include_self_steepening=False, include_tpa=False,
    )
    initial_energy = np.sum(np.abs(pulse.envelope_field) ** 2) * pulse.grid.dt
    engine.propagate(num_steps=10)
    final_energy = np.sum(np.abs(engine.A) ** 2) * pulse.grid.dt
    assert np.isclose(final_energy, initial_energy, rtol=1e-6)


def test_engine_dispersion_spreads_pulse(setup):
    """Dispersion broadens a Gaussian pulse over propagation."""
    pulse, fiber, betas = setup
    engine = SplitStepEngine(
        pulse=pulse, fiber=fiber, betas=betas,
        include_raman=False, include_self_steepening=False, include_tpa=False,
    )
    initial_width = np.sqrt(np.sum(pulse.grid.t**2 * np.abs(pulse.envelope_field)**2) / np.sum(np.abs(pulse.envelope_field)**2))
    engine.propagate(num_steps=50)
    final_field = engine.A
    final_width = np.sqrt(np.sum(pulse.grid.t**2 * np.abs(final_field)**2) / np.sum(np.abs(final_field)**2))
    # Pulse should broaden due to dispersion
    assert final_width > initial_width
