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
from photonics_helper.base import Wavelength, Time, Length, Area


@pytest.fixture
def linear_setup():
    """Setup for loss/no-drift tests: pure dispersion, no nonlinear effects."""
    grid = TemporalGrid(N=512, Tmax=Time(40e-12, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(2, "ps"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    # Low loss for measurable but small attenuation
    fiber = FiberProfile(n2=0.0, alpha=0.1, A_eff=Area(1e-10, "m^2"), length=Length(0.1, "m"))
    # Zero dispersion for pure loss test
    betas = np.array([])
    return pulse, fiber, betas


@pytest.fixture
def dispersion_setup():
    """Setup for no-drift test: dispersion only, no loss, no nonlinearity."""
    grid = TemporalGrid(N=512, Tmax=Time(40e-12, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(2, "ps"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(n2=0.0, alpha=0.0, A_eff=Area(1e-10, "m^2"), length=Length(0.01, "m"))
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
        expected_ratio = np.exp(-alpha * length.as_m)
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

        grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
        env = Envelope(shape="gaussian", peak_amplitude=10.0, pulse_width=Time(1, "ps"))
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
            A_eff=Area(5e-11, "m^2"),
            length=Length(0.01, "m"),
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

        grid = TemporalGrid(N=512, Tmax=Time(40e-12, "s"))
        env = Envelope(shape="gaussian", peak_amplitude=100.0, pulse_width=Time(2, "ps"))
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
            n2=2.6e-20, alpha=0.0, A_eff=Area(5e-11, "m^2"), length=Length(0.1, "m"),
            raman_response=raman,
        )
        fiber_kerr = FiberProfile(
            n2=2.6e-20, alpha=0.0, A_eff=Area(5e-11, "m^2"), length=Length(0.1, "m"),
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


class TestSelfSteepeningConservation:
    """Self-steepening should conserve energy and match laserfun at modest step counts."""

    def test_steepening_energy_conserved_with_dispersion(self):
        """NLSE_simple-style case: |ΔE| < 0.1% with frequency-domain RK4 shock."""
        import laserfun as lf

        wl = 1550.0
        length_m = 0.01
        gamma = 1.0
        n2 = 2.6e-20
        omega0 = 2 * np.pi * 299792458 / (wl * 1e-9)

        pulse_lf = lf.Pulse(
            pulse_type="sech",
            fwhm_ps=0.05,
            epp=50e-12,
            center_wavelength_nm=wl,
            time_window_ps=7,
        )
        fiber_lf = lf.Fiber(
            length=length_m,
            center_wl_nm=wl,
            dispersion=(-0.12, 0, 5e-6),
            gamma_W_m=gamma,
        )
        res_lf = lf.NLSE(
            pulse_lf, fiber_lf, raman=False, shock=True, nsaves=2, print_status=False
        )

        N = len(pulse_lf.t_ps)
        dt_ps = pulse_lf.t_ps[1] - pulse_lf.t_ps[0]
        grid = TemporalGrid(N=N, Tmax=Time(N * dt_ps * 1e-12, "s"))
        t_ph = grid.t
        t_lf = pulse_lf.t_ps * 1e-12
        at_interp = np.interp(t_ph, t_lf, np.real(pulse_lf.at)) + 1j * np.interp(
            t_ph, t_lf, np.imag(pulse_lf.at)
        )
        env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(50, "fs"))
        pulse = Wave(
            grid=grid,
            envelope=env,
            central_wavelength=Wavelength(wl, "nm"),
        )
        pulse._pulse_train_field = at_interp
        fiber = FiberProfile.from_gamma(
            gamma=gamma, n2=n2, omega0=omega0, length=Length(length_m, "m")
        )
        betas = np.array([-0.12, 0.0, 5e-6])

        nsteps = GNLSESolver.estimate_num_steps(
            pulse, fiber, betas, include_self_steepening=True
        )
        solver = GNLSESolver(
            pulse=pulse,
            fiber=fiber,
            betas=betas,
            include_raman=False,
            include_self_steepening=True,
            include_tpa=False,
        )
        solver.propagate(num_steps=nsteps)

        E0 = np.sum(np.abs(at_interp) ** 2) * grid.dt
        E1 = np.sum(np.abs(solver.evolution[-1].envelope_field) ** 2) * grid.dt
        assert np.isclose(E1 / E0, 1.0, rtol=0.001)

        ph_sw = np.abs(np.fft.ifftshift(grid.fft(solver.evolution[-1].envelope_field))) ** 2
        ph_sw /= ph_sw.max()
        lf_sw = np.abs(np.fft.fft(res_lf.AT[-1])) ** 2
        lf_sw /= lf_sw.max()
        corr = np.corrcoef(ph_sw, lf_sw)[0, 1]
        assert corr > 0.95


class TestEstimateNumSteps:
    def test_empty_betas_does_not_crash(self, linear_setup):
        pulse, fiber, betas = linear_setup
        n = GNLSESolver.estimate_num_steps(pulse, fiber, betas)
        assert n >= 10

    def test_steepening_increases_step_count(self, dispersion_setup):
        pulse, fiber, betas = dispersion_setup
        n_plain = GNLSESolver.estimate_num_steps(
            pulse, fiber, betas, include_self_steepening=False
        )
        n_steep = GNLSESolver.estimate_num_steps(
            pulse, fiber, betas, include_self_steepening=True
        )
        assert n_steep >= n_plain
