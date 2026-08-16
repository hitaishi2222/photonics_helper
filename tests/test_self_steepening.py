"""Lightweight tests for GNLSE self-steepening (shock term).

Uses a 128-point grid and short fixed step counts so the suite stays
memory-friendly on modest hardware. Full Dudley-style demos (N≈8192,
thousands of steps) are intentionally not exercised here.
"""

import numpy as np
import pytest

from photonics_helper.gnlse import FiberProfile, GNLSESolver, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.base import Area, Length, Time, Wavelength


@pytest.fixture
def shock_setup():
    """Minimal grid and short fiber — keep memory footprint tiny."""
    grid = TemporalGrid(N=128, Tmax=Time(5e-12, "s"))
    env = Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(1e4),
        pulse_width=Time(100e-15, "s"),
    )
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1064, "nm"),
    )
    fiber = FiberProfile(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80, "um^2"),
        length=Length(1e-3, "m"),
    )
    betas = np.array([-2e-3])
    return pulse, fiber, betas


def test_shock_nonlinear_step_changes_field(shock_setup):
    """Self-steepening alters one nonlinear split-step (no full propagation)."""
    pulse, fiber, betas = shock_setup
    A = np.array(pulse.envelope_field, dtype=complex)
    dz = 1e-5

    engine_off = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
    )
    engine_on = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=True,
    )

    A_off, _ = engine_off._nonlinear_step(A.copy(), dz)
    A_on, _ = engine_on._nonlinear_step(A.copy(), dz)

    assert not np.allclose(A_off, A_on, rtol=1e-6, atol=1e-12)


def test_self_steepening_flag_changes_propagation(shock_setup):
    """include_self_steepening=True vs False yields different output fields."""
    pulse, fiber, betas = shock_setup
    common = dict(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_tpa=False,
    )

    solver_off = GNLSESolver(**common, include_self_steepening=False)
    solver_off.propagate(num_steps=10)

    solver_on = GNLSESolver(**common, include_self_steepening=True)
    solver_on.propagate(num_steps=10)

    A_off = solver_off.evolution[-1].envelope_field
    A_on = solver_on.evolution[-1].envelope_field
    assert not np.allclose(A_off, A_on, rtol=1e-5, atol=1e-10)


def test_nsaves_limits_stored_snapshots(shock_setup):
    """nsaves caps stored evolution length while integration still runs."""
    pulse, fiber, betas = shock_setup
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=50, nsaves=5)
    assert len(solver.evolution) == 5
    assert len(solver.z_array) == 5
    assert np.isclose(solver.z_array[0], 0.0)
    assert np.isclose(solver.z_array[-1], fiber.length.as_m)
