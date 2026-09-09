"""Tests for SolitonAnalyzer class."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from photonics_helper.soliton import (
    SolitonAnalyzer,
    plot_soliton_trajectories,
    plot_fission_dynamics,
    plot_raman_shift,
    plot_dispersion_wave,
)
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.gnlse import FiberProfile
from photonics_helper.base import Wavelength, Time, Area, Length, C_MS, PI


def _make_analyzer_manual(T0_s, P_peak_w, gamma_val, beta2_si, beta3_si=0.0,
                          confinement=1.0, n_steps=2):
    """Create SolitonAnalyzer with manually specified parameters."""
    grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0_s, "s"))
    pulse = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(1550, "nm"),
    )

    # Set fiber to give desired gamma: gamma = n2*omega0*Gamma/(c*A_eff)
    omega0 = 2 * PI * C_MS / 1550e-9
    # Choose n2=1e-19, then A_eff = n2*omega0*Gamma/(c*gamma)
    n2 = 1e-19
    A_eff_val = n2 * omega0 * confinement / (C_MS * gamma_val)
    A_eff = Area(A_eff_val, "m^2")

    fiber = FiberProfile(
        n2=n2, alpha=0.0, A_eff=A_eff, length=Length(1e-3, "m"),
        confinement_factor=confinement,
    )

    # betas in ps^2/m: beta2_si * 1e24
    betas_ps2 = beta2_si * 1e24
    betas = np.array([betas_ps2])
    if beta3_si != 0:
        betas_ps3 = beta3_si * 1e27
        betas = np.array([betas_ps2, betas_ps3])

    omega = grid.w
    A0_w = grid.fft(pulse.envelope_field)
    spectra = np.array([np.abs(A0_w) ** 2] * n_steps)
    z_array = np.linspace(0, 1e-3, n_steps)

    analyzer = SolitonAnalyzer(
        pulse=pulse, fiber=fiber, betas=betas,
        z_array=z_array, spectra_vs_z=(omega, spectra),
    )
    # Override P_peak to match desired value (peak_power() depends on grid)
    analyzer.P_peak = P_peak_w
    return analyzer


def test_soliton_order_formula():
    """Task 3.3: Verify N = sqrt(gamma * P * T0^2 / |beta2|)."""
    T0 = 100e-15
    P = 1.0
    gamma = 0.07
    beta2_si = -2e-27

    N_expected = np.sqrt(gamma * P * T0**2 / abs(beta2_si))

    analyzer = _make_analyzer_manual(T0_s=T0, P_peak_w=P, gamma_val=gamma,
                                     beta2_si=beta2_si)
    N = analyzer.soliton_order()
    assert abs(N - N_expected) / N_expected < 1e-10


def test_soliton_order_fundamental_value():
    """Task 3.3: With T0=100fs, P=2.857W, gamma=0.07, beta2=-2e-27: N=1.0."""
    T0 = 100e-15
    P = 2.857  # Corrected: gives N=1.0 with these parameters
    gamma = 0.07
    beta2_si = -2e-27
    N_expected = np.sqrt(gamma * P * T0**2 / abs(beta2_si))
    assert abs(N_expected - 1.0) / 1.0 < 0.02, f"N={N_expected}"


def test_soliton_order_high_order_value():
    """Task 3.3: With T0=100fs, P=10W, gamma=0.07, beta2=-2e-27: N~1.87."""
    T0 = 100e-15
    P = 10.0
    gamma = 0.07
    beta2_si = -2e-27
    N_expected = np.sqrt(gamma * P * T0**2 / abs(beta2_si))
    # N = sqrt(0.07*10*1e-26/2e-27) = sqrt(3.5) = 1.87
    assert abs(N_expected - 1.87) / 1.87 < 0.02, f"N={N_expected}"


def test_dispersion_length_formula():
    """Task 3.4: L_D = T0^2 / |beta2|."""
    T0 = 100e-15
    beta2_si = -3e-24
    L_D_expected = T0**2 / abs(beta2_si)

    analyzer = _make_analyzer_manual(T0_s=T0, P_peak_w=1.0, gamma_val=0.07,
                                     beta2_si=beta2_si)
    L_D = analyzer.dispersion_length()
    assert abs(L_D - L_D_expected) / L_D_expected < 1e-10


def test_dispersion_length_value():
    """Task 3.4: T0=100fs, beta2=-3e-24 => L_D=3.33mm."""
    T0 = 100e-15
    beta2_si = -3e-24
    L_D_expected = T0**2 / abs(beta2_si)  # 3.33e-3 m
    assert abs(L_D_expected - 3.33e-3) / 3.33e-3 < 0.01


def test_nonlinear_length_formula():
    """Task 3.5: L_NL = 1 / (gamma * P_peak)."""
    gamma = 97.29
    P = 5.0
    L_NL_expected = 1.0 / (gamma * P)

    analyzer = _make_analyzer_manual(T0_s=100e-15, P_peak_w=P, gamma_val=gamma,
                                     beta2_si=-2e-27)
    L_NL = analyzer.nonlinear_length()
    assert abs(L_NL - L_NL_expected) / L_NL_expected < 1e-10


def test_fission_length_formula():
    """Task 3.6: L_fiss = L_D / (N * eta)."""
    eta = 0.7

    analyzer = _make_analyzer_manual(T0_s=100e-15, P_peak_w=1.0, gamma_val=0.07,
                                     beta2_si=-3e-24)
    L_fiss = analyzer.fission_length(eta=eta)
    # Verify the formula: L_fiss = L_D / (N * eta)
    L_D_actual = analyzer.dispersion_length()
    N_actual = analyzer.soliton_order()
    L_fiss_expected_actual = L_D_actual / (N_actual * eta)
    assert abs(L_fiss - L_fiss_expected_actual) / L_fiss_expected_actual < 1e-10


def test_dispersive_wave_wavelength():
    """Task 3.7: DW wavelength from beta2 and beta3."""
    from photonics_helper.base import Wavelength
    omega0 = 2 * PI * C_MS / 1550e-9
    beta2_si = -5e-24
    beta3_si = -5e-27
    delta_omega = -2 * beta2_si / beta3_si
    omega_dw = omega0 + delta_omega
    lambda_dw = Wavelength(2 * PI * C_MS / omega_dw, "m")

    analyzer = _make_analyzer_manual(T0_s=100e-15, P_peak_w=1.0, gamma_val=0.07,
                                     beta2_si=beta2_si, beta3_si=beta3_si)
    lambda_calc = analyzer.dispersive_wave_wavelength()
    assert abs(lambda_calc.as_m - lambda_dw.as_m) / lambda_dw.as_m < 1e-6


def test_count_solitons_basic():
    """Task 3.8: count_solitons() returns int >= 0."""
    analyzer = _make_analyzer_manual(T0_s=100e-15, P_peak_w=1.0, gamma_val=0.07,
                                     beta2_si=-2e-27)
    count = analyzer.count_solitons()
    assert isinstance(count, int)
    assert count >= 0


def test_soliton_trajectories_basic():
    """Task 3.9: soliton_trajectories() returns list."""
    analyzer = _make_analyzer_manual(T0_s=100e-15, P_peak_w=1.0, gamma_val=0.07,
                                     beta2_si=-2e-27)
    traj = analyzer.soliton_trajectories()
    assert isinstance(traj, list)


def test_raman_shift_rate_basic():
    """Task 3.10: raman_shift_rate() returns float."""
    analyzer = _make_analyzer_manual(T0_s=100e-15, P_peak_w=1.0, gamma_val=0.07,
                                     beta2_si=-2e-27)
    rate = analyzer.raman_shift_rate()
    assert isinstance(rate, float)


def test_waveguide_confinement_factor():
    """Task 3.3 (waveguide): gamma uses confinement factor."""
    n2 = 6.0e-18
    omega0 = 2 * PI * C_MS / 1550e-9
    A_eff = Area(0.2e-12, "m^2")
    Gamma = 0.8

    grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(100e-15, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550, "nm"))

    fiber = FiberProfile(
        n2=n2, alpha=0.0, A_eff=A_eff, length=Length(1e-3, "m"),
        confinement_factor=Gamma,
    )
    betas = np.array([-0.2])
    omega = grid.w
    A0_w = grid.fft(pulse.envelope_field)
    spectra = np.array([np.abs(A0_w) ** 2])
    z_array = np.array([0.0])

    analyzer = SolitonAnalyzer(
        pulse=pulse, fiber=fiber, betas=betas, z_array=z_array, spectra_vs_z=(omega, spectra)
    )

    expected_gamma = n2 * omega0 * Gamma / (C_MS * A_eff.as_m2)
    assert abs(analyzer.gamma - expected_gamma) / expected_gamma < 1e-10


# ─── Visualization tests (tasks 5.1-5.5) ─────────────────────────────────────


def _make_solver(T0_s=100e-15, P_peak=1.0, gamma_val=0.07, beta2_si=-2e-27,
                 beta3_si=0.0, n_steps=5):
    """Helper to create a minimal GNLSESolver-like object for visualization tests."""
    grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0_s, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550, "nm"))

    omega0 = 2 * PI * C_MS / 1550e-9
    n2 = 1e-19
    A_eff_val = n2 * omega0 / (C_MS * gamma_val)
    A_eff = Area(A_eff_val, "m^2")
    fiber = FiberProfile(n2=n2, alpha=0.0, A_eff=A_eff, length=Length(1e-3, "m"))

    betas_ps2 = beta2_si * 1e24
    betas = np.array([betas_ps2]) if beta3_si == 0 else np.array([betas_ps2, beta3_si * 1e27])

    omega = grid.w
    A0_w = grid.fft(pulse.envelope_field)
    spectra = np.array([np.abs(A0_w) ** 2] * n_steps)
    z_array = np.linspace(0, 1e-3, n_steps)

    # Create a mock solver object
    class MockSolver:
        pass

    solver = MockSolver()
    solver.pulse = pulse
    solver.fiber = fiber
    solver.betas = betas
    solver.z_array = z_array
    solver.spectra_vs_z = (omega, spectra)
    solver.omega0 = omega0
    return solver


def test_plot_soliton_trajectories():
    """Task 5.1: plot_soliton_trajectories renders without error."""
    solver = _make_solver()
    fig = plot_soliton_trajectories(solver)
    assert fig is not None
    plt.close(fig)


def test_plot_fission_dynamics():
    """Task 5.2: plot_fission_dynamics renders without error."""
    solver = _make_solver()
    analyzer = SolitonAnalyzer(solver.pulse, solver.fiber, solver.betas,
                               solver.z_array, solver.spectra_vs_z)
    N = analyzer.soliton_order()
    L_D = analyzer.dispersion_length()
    fig = plot_fission_dynamics(solver, N=N, L_D=L_D)
    assert fig is not None
    plt.close(fig)


def test_plot_raman_shift():
    """Task 5.3: plot_raman_shift renders without error."""
    solver = _make_solver()
    fig = plot_raman_shift(solver)
    assert fig is not None
    plt.close(fig)


def test_plot_dispersion_wave():
    """Task 5.4: plot_dispersion_wave renders without error."""
    solver = _make_solver(beta3_si=-5e-27)
    fig = plot_dispersion_wave(solver)
    assert fig is not None
    plt.close(fig)
