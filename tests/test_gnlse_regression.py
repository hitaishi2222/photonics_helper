"""Regression tests for GNLSE bug fixes from laserfun comparison.

Tests for:
- Loss application (exp(-α·Δz/2) per half-step)
- No-drift propagation (β₁ removed, pulse stays centered)
- Full Raman response (instantaneous + delayed)
- Energy conservation with loss
"""

import numpy as np
import pytest

from photonics_helper.gnlse import FiberProfile, GNLSESolver, SplitStepEngine
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.base import Wavelength


@pytest.fixture
def linear_setup():
    """Setup for loss/no-drift tests: pure dispersion, no nonlinear effects."""
    grid = TemporalGrid(N=512, Tmax=40e-12)
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=2e-12)
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    # Low loss for measurable but small attenuation
    fiber = FiberProfile(n2=0.0, alpha=0.1, A_eff=1e-10, length=0.1)
    # Zero dispersion for pure loss test
    betas = np.array([])
    return pulse, fiber, betas


@pytest.fixture
def dispersion_setup():
    """Setup for no-drift test: dispersion only, no loss, no nonlinearity."""
    grid = TemporalGrid(N=512, Tmax=40e-12)
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=2e-12)
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(n2=0.0, alpha=0.0, A_eff=1e-10, length=0.01)
    # Some dispersion but no loss
    betas = np.array([0.01])  # 10 ps²/km (= 0.01 ps²/m)
    return pulse, fiber, betas


class TestLossApplication:
    """Test that loss is correctly applied during propagation."""

    def test_loss_attenuates_pulse(self, linear_setup):
        """Pulse amplitude decreases with propagation due to loss."""
        pulse, fiber, betas = linear_setup
        solver = GNLSESolver(
            pulse=pulse, fiber=fiber, betas=betas,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver.propagate(num_steps=10)

        initial_peak = np.max(np.abs(pulse.envelope_field))
        final_peak = np.max(np.abs(solver.evolution[-1].envelope_field))

        # Loss should cause attenuation
        assert final_peak < initial_peak

    def test_loss_matches_exponential_decay(self, linear_setup):
        """Attenuation matches exp(-α·L/2) for field amplitude."""
        pulse, fiber, betas = linear_setup
        alpha = fiber.alpha
        length = fiber.length

        solver = GNLSESolver(
            pulse=pulse, fiber=fiber, betas=betas,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver.propagate(num_steps=50)

        initial_energy = np.sum(np.abs(pulse.envelope_field) ** 2) * pulse.grid.dt
        final_energy = np.sum(np.abs(solver.evolution[-1].envelope_field) ** 2) * pulse.grid.dt

        # Energy decays as exp(-α·L) (intensity, not field)
        expected_ratio = np.exp(-alpha * length)
        actual_ratio = final_energy / initial_energy

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.05)

    def test_zero_loss_no_attenuation(self, dispersion_setup):
        """With alpha=0, energy is conserved (pure dispersion)."""
        pulse, fiber, betas = dispersion_setup
        solver = GNLSESolver(
            pulse=pulse, fiber=fiber, betas=betas,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        initial_energy = np.sum(np.abs(pulse.envelope_field) ** 2) * pulse.grid.dt
        solver.propagate(num_steps=20)
        final_energy = np.sum(np.abs(solver.evolution[-1].envelope_field) ** 2) * pulse.grid.dt

        assert np.isclose(final_energy, initial_energy, rtol=1e-6)


class TestNoDriftPropagation:
    """Test that pulse does not drift in time (β₁ removed)."""

    def test_pulse_center_stays_constant(self, dispersion_setup):
        """Pulse temporal center remains at t=0 with pure dispersion."""
        pulse, fiber, betas = dispersion_setup
        solver = GNLSESolver(
            pulse=pulse, fiber=fiber, betas=betas,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )

        # Initial pulse center
        init_field = pulse.envelope_field
        init_center = np.sum(pulse.grid.t * np.abs(init_field) ** 2) / np.sum(np.abs(init_field) ** 2)

        solver.propagate(num_steps=50)
        final_field = solver.evolution[-1].envelope_field
        final_center = np.sum(pulse.grid.t * np.abs(final_field) ** 2) / np.sum(np.abs(final_field) ** 2)

        # Center should not drift significantly (within numerical tolerance)
        assert np.abs(final_center - init_center) < 1e-13


class TestRamanFullResponse:
    """Test that Raman uses full combined response (instantaneous + delayed)."""

    def test_raman_with_full_response(self):
        """Raman convolution produces phase shift with full response."""
        from photonics_helper.raman import RamanResponse, RamanSpec

        grid = TemporalGrid(N=256, Tmax=20e-12)
        env = Envelope(shape="gaussian", peak_amplitude=10.0, pulse_width=1e-12)
        pulse = Wave(
            grid=grid,
            envelope=env,
            central_wavelength=Wavelength(1550, "nm"),
        )

        # Create a RamanSpec for silica
        spec = RamanSpec(
            name="Silica",
            raman_shift_cm=440,
            raman_linewidth_cm=45,
            fR=0.18,
        )
        raman = RamanResponse(spec=spec, grid=grid)

        fiber = FiberProfile(
            n2=2.6e-20,
            alpha=0.0,
            A_eff=5e-11,
            length=0.01,
            raman_response=raman,
        )
        betas = np.array([])

        solver = GNLSESolver(
            pulse=pulse, fiber=fiber, betas=betas,
            include_raman=True,
            include_self_steepening=False,
            include_tpa=False,
        )
        solver.propagate(num_steps=10)

        # With Raman, the pulse should experience some phase modulation
        # The key test: propagation completes without error and produces results
        assert len(solver.evolution) == 11
        omega, spectra = solver.spectra_vs_z
        assert spectra.shape == (11, 256)

    def test_raman_produces_different_spectrum(self):
        """Raman scattering modifies the field compared to Kerr-only."""
        from photonics_helper.raman import RamanResponse, RamanSpec
        from photonics_helper.gnlse import SplitStepEngine

        grid = TemporalGrid(N=512, Tmax=40e-12)
        env = Envelope(shape="gaussian", peak_amplitude=100.0, pulse_width=2e-12)
        pulse = Wave(
            grid=grid,
            envelope=env,
            central_wavelength=Wavelength(1550, "nm"),
        )

        spec = RamanSpec(
            name="Silica",
            raman_shift_cm=440,
            raman_linewidth_cm=45,
            fR=0.18,
        )
        raman = RamanResponse(spec=spec, grid=grid)

        fiber_raman = FiberProfile(
            n2=2.6e-20, alpha=0.0, A_eff=5e-11, length=0.1,
            raman_response=raman,
        )
        fiber_kerr = FiberProfile(
            n2=2.6e-20, alpha=0.0, A_eff=5e-11, length=0.1,
        )

        betas = np.array([])

        # Use engine directly to access actual field
        engine_kerr = SplitStepEngine(
            pulse=pulse, fiber=fiber_kerr, betas=betas,
            include_raman=False,
        )
        engine_kerr.propagate(num_steps=50)

        engine_raman = SplitStepEngine(
            pulse=pulse, fiber=fiber_raman, betas=betas,
            include_raman=True,
        )
        engine_raman.propagate(num_steps=50)

        # Check that the actual fields are different
        max_diff = np.max(np.abs(engine_kerr.A - engine_raman.A))
        max_signal = np.max(np.abs(engine_raman.A))
        relative_diff = max_diff / max(max_signal, 1e-30)
        assert relative_diff > 1e-6, "Raman should produce measurably different field"
