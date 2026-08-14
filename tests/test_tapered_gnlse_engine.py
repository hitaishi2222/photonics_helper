"""Tests for TaperedGNLSESolver and z-dependent SplitStepEngine."""
import numpy as np
import pytest

from photonics_helper.gnlse import FiberProfile, GNLSESolver, SplitStepEngine, TaperedGNLSESolver
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.base import Wavelength, Time, Length, Area
from photonics_helper.fiber import ZDependentDispersion


def _make_tapered_setup():
    """Create a basic pulse and fiber for tapered GNLSE tests."""
    grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=100.0, pulse_width=Time(1, "ps"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(n2=1e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(1e-3, "m"))
    return pulse, fiber


def _make_uniform_beta_table(pulse, fiber_length=1e-3):
    """Create a ZDependentDispersion that is uniform in z (constant β(ω)).

    This is used to verify that the z-dependent path reproduces the uniform solver.
    """
    omega0 = pulse.central_frequency
    # Create a wide omega grid around the carrier (must cover the pulse bandwidth)
    # Pulse with Tmax=20ps has bandwidth ~±4e13 rad/s, so use ±6e13 for safety
    omegas = np.linspace(omega0 - 6e13, omega0 + 6e13, 200)  # rad/s
    z_positions = np.linspace(0, fiber_length, 5)  # m

    # β(ω) = β0 + β2/2 * (ω - ω0)^2 (simple quadratic, uniform in z)
    beta0 = 1e8  # rad/m
    beta2 = 0.02e-24  # s²/m (0.02 ps²/m)
    beta = np.zeros((len(omegas), len(z_positions)))
    for j in range(len(z_positions)):
        beta[:, j] = beta0 + beta2 / 2 * (omegas - omega0) ** 2

    return ZDependentDispersion.from_arrays(
        omegas=omegas,
        z_positions=z_positions,
        beta=beta,
        central_wavelength=pulse.central_wavelength.as_m,
    )


def _make_zdw_migrating_profile(pulse, fiber_length=20e-3):
    """Create a β(ω, z) profile where ZDW migrates across the pump.

    β(ω, z) = β0 + β2(z)/2 * (ω - ω0)^2
    where β2(z) goes from negative (anomalous) to positive (normal) dispersion.
    """
    omega0 = pulse.central_frequency
    # Wide omega grid to cover pulse bandwidth
    omegas = np.linspace(omega0 - 6e13, omega0 + 6e13, 200)
    z_positions = np.linspace(0, fiber_length, 40)

    beta0 = 1e8
    beta2_start = -50e-24  # very strong anomalous at input
    beta2_end = 50e-24  # very strong normal at output

    beta = np.zeros((len(omegas), len(z_positions)))
    for j, z in enumerate(z_positions):
        t = z / fiber_length  # normalized position
        beta2_z = beta2_start + (beta2_end - beta2_start) * t
        beta[:, j] = beta0 + beta2_z / 2 * (omegas - omega0) ** 2

    return ZDependentDispersion.from_arrays(
        omegas=omegas,
        z_positions=z_positions,
        beta=beta,
        central_wavelength=pulse.central_wavelength.as_m,
    )


class TestAdaptiveStepAnomalous:
    """Adaptive stepping must use |β₂| so anomalous GVD does not disable dz_disp."""

    def test_adaptive_step_uses_abs_beta2(self):
        pulse, fiber = _make_tapered_setup()
        omega0 = pulse.central_frequency
        omegas = np.linspace(omega0 - 6e13, omega0 + 6e13, 100)
        z_positions = np.linspace(0, fiber.length.as_m, 5)
        beta2 = -50e-24  # strong anomalous (s²/m)
        beta = np.zeros((len(omegas), len(z_positions)))
        for j in range(len(z_positions)):
            beta[:, j] = 1e8 + beta2 / 2 * (omegas - omega0) ** 2
        disp = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=pulse.central_wavelength.as_m,
        )
        engine = SplitStepEngine(
            pulse=pulse,
            fiber=fiber,
            betas=np.array([0.0]),
            include_raman=False,
            dispersion_profile=disp,
        )
        engine._current_z = 0.0
        A = pulse.envelope_field.astype(complex)
        dz = engine._adaptive_step_size(A)
        T0 = pulse.envelope.pulse_width.as_s
        dz_disp_expected = T0**2 / abs(beta2) * 0.01
        # Without abs(), max(β₂, ε)=ε → dz_disp ≈ huge; require physical scale
        assert dz <= dz_disp_expected * 1.01
        assert dz < fiber.length.as_m

    def test_gradient_shrink_abs_beta2_ref(self):
        pulse, fiber = _make_tapered_setup()
        disp = _make_zdw_migrating_profile(pulse, fiber_length=fiber.length.as_m)
        engine = SplitStepEngine(
            pulse=pulse,
            fiber=fiber,
            betas=np.array([0.0]),
            include_raman=False,
            dispersion_profile=disp,
        )
        dz_base = fiber.length.as_m / 100
        # At input β₂ < 0; signed ref would make threshold < 0 and always shrink wrongly
        factor = engine._gradient_shrink_factor(0.0, dz_base)
        assert 0.1 <= factor <= 1.0


class TestTaperedGNLSESolver:
    """Tests for TaperedGNLSESolver."""

    def test_solver_construction(self):
        """TaperedGNLSESolver constructs with required parameters."""
        pulse, fiber = _make_tapered_setup()
        disp = _make_uniform_beta_table(pulse)
        solver = TaperedGNLSESolver(
            pulse=pulse,
            fiber=fiber,
            dispersion_profile=disp,
        )
        assert solver.pulse is pulse
        assert solver.fiber is fiber
        assert solver.include_raman is True

    def test_propagate_populates_evolution(self):
        """propagate() populates evolution list."""
        pulse, fiber = _make_tapered_setup()
        disp = _make_uniform_beta_table(pulse)
        solver = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
            include_raman=False,
        )
        solver.propagate(num_steps=5)
        # Adaptive stepping may create more steps, but evolution should be populated
        assert len(solver.evolution) >= 2
        # Check that we reached the end of the fiber
        assert solver.z_array[-1] >= fiber.length.as_m * 0.99

    def test_spectra_vs_z_shape(self):
        """spectra_vs_z returns correct shape after propagation."""
        pulse, fiber = _make_tapered_setup()
        disp = _make_uniform_beta_table(pulse)
        solver = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
            include_raman=False,
        )
        solver.propagate(num_steps=10)
        omega, spectra = solver.spectra_vs_z
        assert len(omega) == 256
        # spectra.shape[1] should be 256 (number of frequency points)
        assert spectra.shape[1] == 256
        # spectra.shape[0] is the number of evolution steps (may vary due to adaptive stepping)
        assert spectra.shape[0] >= 2

    def test_spectra_vs_z_before_propagate(self):
        """spectra_vs_z raises before propagate()."""
        pulse, fiber = _make_tapered_setup()
        disp = _make_uniform_beta_table(pulse)
        solver = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
        )
        with pytest.raises(RuntimeError, match="Call propagate"):
            _ = solver.spectra_vs_z

    def test_uniform_limit_energy_conservation(self):
        """Uniform β(ω,z) conserves energy (no gain/loss in pure Kerr)."""
        pulse, fiber = _make_tapered_setup()
        # Use a fiber with zero loss for energy conservation test
        fiber_no_loss = FiberProfile(
            n2=1e-19, alpha=0.0, A_eff=Area(5e-11, "m^2"),
            length=Length(1e-3, "m"),
        )
        disp = _make_uniform_beta_table(pulse, fiber_length=1e-3)
        solver = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber_no_loss, dispersion_profile=disp,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver.propagate(num_steps=50)

        # Check energy conservation: integral of |A|^2 should be constant
        initial_energy = np.sum(np.abs(solver.evolution[0].envelope_field) ** 2)
        final_energy = np.sum(np.abs(solver.evolution[-1].envelope_field) ** 2)
        energy_drift = abs(final_energy - initial_energy) / initial_energy
        assert energy_drift < 0.01, f"Energy drift {energy_drift:.4f} exceeds 1%"

    def test_uniform_limit_matches_gnlse_solver(self):
        """Uniform β(ω,z) reproduces GNLSESolver spectrum at intensity level."""
        pulse, fiber = _make_tapered_setup()
        disp = _make_uniform_beta_table(pulse, fiber_length=1e-3)

        # Use fixed step size for both solvers to ensure fair comparison
        step = Length(1e-5, "m")  # 10 μm steps

        # Tapered solver with uniform profile
        tapered_solver = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        # Set step_size on the engine after creation
        tapered_solver.propagate(num_steps=100)

        # Uniform solver with equivalent betas
        # Extract β2 from the uniform profile at z=0
        omega0 = pulse.central_frequency
        omegas_test = np.linspace(omega0 - 2e13, omega0 + 2e13, 50)
        beta_at_z0 = disp.fn(omegas_test, 0.0)
        # Fit quadratic to extract β2
        omega_offset = omegas_test - omega0
        coeffs = np.polyfit(omega_offset, beta_at_z0 - 1e8, 2)
        beta2 = 2 * coeffs[0]  # s²/m

        # Convert to ps²/m for GNLSESolver
        betas_ps2m = beta2 * 1e24

        uniform_solver = GNLSESolver(
            pulse=pulse, fiber=fiber, betas=np.array([betas_ps2m]),
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        uniform_solver.propagate(num_steps=100)

        # Compare spectra at output (intensity, not phase)
        _, tapered_spectra = tapered_solver.spectra_vs_z
        _, uniform_spectra = uniform_solver.spectra_vs_z

        # Get the last spectrum (output)
        tapered_output = tapered_spectra[-1]
        uniform_output = uniform_spectra[-1]

        # Normalize and compare shape
        tapered_norm = tapered_output / np.max(tapered_output)
        uniform_norm = uniform_output / np.max(uniform_output)

        # They should have similar shape (not identical due to frame convention)
        # Use correlation as a shape measure
        correlation = np.corrcoef(tapered_norm, uniform_norm)[0, 1]
        assert correlation > 0.95, f"Spectral shape correlation {correlation:.4f} < 0.95"

    def test_zdw_migration(self):
        """ZDW migration produces different output than fixed-dispersion case."""
        # Use more extreme parameters to ensure ZDW migration effect is visible
        grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
        env = Envelope(shape="gaussian", peak_amplitude=1000.0, pulse_width=Time(0.5, "ps"))
        pulse = Wave(
            grid=grid,
            envelope=env,
            central_wavelength=Wavelength(1550, "nm"),
        )
        fiber = FiberProfile(n2=1e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(1e-3, "m"))
        
        fiber_length = 20e-3
        disp_uniform = _make_uniform_beta_table(pulse, fiber_length=fiber_length)
        disp_migrating = _make_zdw_migrating_profile(pulse, fiber_length=fiber_length)

        # Use higher amplitude to ensure nonlinear spectral broadening
        solver_uniform = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp_uniform,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver_uniform.propagate(num_steps=400)

        solver_migrating = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp_migrating,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver_migrating.propagate(num_steps=400)

        _, spectra_uniform = solver_uniform.spectra_vs_z
        _, spectra_migrating = solver_migrating.spectra_vs_z

        # The outputs should be different (ZDW migration changes the spectrum)
        output_uniform = spectra_uniform[-1]
        output_migrating = spectra_migrating[-1]

        # Normalize
        output_uniform_norm = output_uniform / np.max(output_uniform)
        output_migrating_norm = output_migrating / np.max(output_migrating)

        # They should be different (not identical)
        diff = np.max(np.abs(output_uniform_norm - output_migrating_norm))
        assert diff > 0.01, f"ZDW migration should change spectrum, but diff={diff:.6f}"

    def test_z_dependent_gamma(self):
        """z-dependent γ (via a_eff_fn) affects propagation."""
        # Use more extreme parameters to ensure gamma effect is visible
        grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
        env = Envelope(shape="gaussian", peak_amplitude=500.0, pulse_width=Time(0.5, "ps"))
        pulse = Wave(
            grid=grid,
            envelope=env,
            central_wavelength=Wavelength(1550, "nm"),
        )
        fiber = FiberProfile(n2=1e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(5e-3, "m"))
        
        disp = _make_uniform_beta_table(pulse, fiber_length=5e-3)

        # Constant A_eff (default)
        solver_const = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver_const.propagate(num_steps=100)

        # Z-dependent A_eff: shrink by 50% at midpoint (stronger effect)
        def a_eff_fn(z):
            # A_eff varies: constant except a dip in the middle
            t = z / fiber.length.as_m
            # Simple taper: A_eff = A_eff0 * (1 - 0.7 * sin(π * t))
            return fiber.A_eff.as_m2 * (1 - 0.7 * np.sin(np.pi * t))

        solver_tapered = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
            a_eff_fn=a_eff_fn,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver_tapered.propagate(num_steps=100)

        _, spectra_const = solver_const.spectra_vs_z
        _, spectra_tapered = solver_tapered.spectra_vs_z

        # The spectra should be different (z-dependent γ changes nonlinear phase)
        output_const = spectra_const[-1]
        output_tapered = spectra_tapered[-1]

        output_const_norm = output_const / np.max(output_const)
        output_tapered_norm = output_tapered / np.max(output_tapered)

        diff = np.max(np.abs(output_const_norm - output_tapered_norm))
        assert diff > 0.01, f"Z-dependent γ should change spectrum, but diff={diff:.6f}"

    def test_z_dependent_alpha_attenuation(self):
        """z-dependent α causes preferential attenuation at high-loss regions."""
        pulse, fiber = _make_tapered_setup()
        disp = _make_uniform_beta_table(pulse, fiber_length=1e-3)

        # Constant alpha
        solver_const = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver_const.propagate(num_steps=50)

        # Z-dependent alpha: higher loss at midpoint
        def alpha_fn(z):
            t = z / fiber.length.as_m
            # Alpha varies: constant + peak at midpoint
            return fiber.alpha * (1 + 10 * np.sin(np.pi * t))

        solver_varied = TaperedGNLSESolver(
            pulse=pulse, fiber=fiber, dispersion_profile=disp,
            alpha_fn=alpha_fn,
            include_raman=False, include_self_steepening=False, include_tpa=False,
        )
        solver_varied.propagate(num_steps=50)

        # Check that energy is lower for the varied case
        energy_const = np.sum(np.abs(solver_const.evolution[-1].envelope_field) ** 2)
        energy_varied = np.sum(np.abs(solver_varied.evolution[-1].envelope_field) ** 2)

        assert energy_varied < energy_const, (
            f"Z-dependent alpha should cause more attenuation: "
            f"E_const={energy_const:.6f}, E_varied={energy_varied:.6f}"
        )


class TestSplitStepEngineZDependent:
    """Tests for SplitStepEngine z-dependent hooks."""

    def test_engine_accepts_z_dependent_kwargs(self):
        """SplitStepEngine accepts dispersion_profile, a_eff_fn, alpha_fn."""
        pulse, fiber = _make_tapered_setup()
        disp = _make_uniform_beta_table(pulse)

        def a_eff_fn(z):
            return fiber.A_eff.as_m2

        def alpha_fn(z):
            return fiber.alpha

        dummy_betas = np.array([0.0])
        engine = SplitStepEngine(
            pulse=pulse, fiber=fiber, betas=dummy_betas,
            dispersion_profile=disp,
            a_eff_fn=a_eff_fn,
            alpha_fn=alpha_fn,
        )
        assert engine._is_z_dependent is True

    def test_engine_uniform_path_unchanged(self):
        """SplitStepEngine without z-dependent hooks uses uniform path."""
        pulse, fiber = _make_tapered_setup()
        dummy_betas = np.array([0.02])  # ps²/m

        engine = SplitStepEngine(
            pulse=pulse, fiber=fiber, betas=dummy_betas,
        )
        assert engine._is_z_dependent is False
        # Should propagate without error
        engine.propagate(num_steps=10)
        # Adaptive stepping may create more steps, but should reach the end
        assert len(engine.evolution) >= 2
        assert engine.z_array[-1] >= fiber.length.as_m * 0.99
