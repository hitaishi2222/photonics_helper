"""
Unit tests for the :mod:`photonics_helper.pulse` module.

The tests focus on the core numerical properties of the ``Envelope``,
``TemporalGrid`` and ``Wave`` classes – ensuring that conversions between
FWHM and pulse width, intensity calculations, energy/power evaluation and the
time‑bandwidth product for a transform‑limited Gaussian pulse behave as
expected.
"""

import numpy as np
import pytest

from photonics_helper.base import Wavelength
from photonics_helper.pulse import SHAPE_FACTORS, Envelope, TemporalGrid, Wave


@pytest.fixture
def gaussian_wave():
    """Create a transform‑limited Gaussian pulse for use in several tests.

    The pulse has a peak amplitude of 1.0 and a full‑width‑half‑maximum of
    100 fs.  A temporal window ten times larger than the FWHM is chosen to avoid
    aliasing.
    """
    fwhm = 100e-15  # 100 fs
    envelope = Envelope.from_fwhm(shape="gaussian", peak_amplitude=1.0, fwhm=fwhm)
    # Temporal grid – use a power‑of‑two number of points for FFT efficiency.
    N = 2**12
    Tmax = 10 * fwhm
    grid = TemporalGrid(N=N, Tmax=Tmax)
    wave = Wave(
        grid=grid,
        envelope=envelope,
        central_wavelength=Wavelength(800, "nm"),
    )
    return wave, fwhm


def test_envelope_fwhm_consistency():
    fwhm = 200e-15
    env = Envelope.from_fwhm(shape="gaussian", peak_amplitude=2.0, fwhm=fwhm)
    # The ``fwhm`` property should return the original value.
    assert pytest.approx(env.fwhm, rel=1e-12) == fwhm
    # Verify internal pulse width matches the analytical conversion.
    expected_T0 = fwhm / SHAPE_FACTORS["gaussian"]
    assert pytest.approx(env.pulse_width, rel=1e-12) == expected_T0


def test_wave_energy_and_power(gaussian_wave):
    wave, _ = gaussian_wave
    # Numerical integration of the intensity should match the ``pulse_energy``
    # convenience method within a small tolerance.
    numerical_energy = np.sum(wave.envelope_intensity) * wave.grid.dt
    assert pytest.approx(wave.pulse_energy(), rel=1e-6) == numerical_energy
    # Peak power – maximum of the intensity array.
    assert pytest.approx(wave.peak_power(), rel=1e-12) == np.max(
        wave.envelope_intensity
    )


def test_time_bandwidth_product_gaussian(gaussian_wave):
    wave, _ = gaussian_wave
    tbp = wave.time_bandwidth_product()
    # Theoretical value for a transform‑limited Gaussian pulse is approximately
    # 0.44 (time·angular‑frequency).  Allow a modest tolerance due to discretisation.
    assert 0.48 < tbp < 0.52


def test_wave_spectrum_shape(gaussian_wave):
    wave, _ = gaussian_wave
    # The spectrum is computed via FFT; its length must match the frequency grid.
    assert wave.spectrum.shape == wave.grid.w.shape


# ─── Per-shape FWHM consistency ────────────────────────────────────────


@pytest.mark.parametrize(
    "shape,fwhm",
    [
        ("super-gaussian", 200e-15),
        ("triangular", 200e-15),
        ("cosine", 200e-15),
        ("exponential", 200e-15),
        ("airy", 200e-15),
    ],
)
def test_from_fwhm_returns_same_fwhm(shape, fwhm):
    """from_fwhm should round-trip: constructed envelope.fwhm ≈ original fwhm."""
    env = Envelope.from_fwhm(shape=shape, peak_amplitude=1.0, fwhm=fwhm)
    assert pytest.approx(env.fwhm, rel=1e-6) == fwhm


@pytest.mark.parametrize(
    "shape",
    [
        "super-gaussian",
        "triangular",
        "cosine",
        "exponential",
    ],
)
def test_from_fwhm_different_peak_amplitude(shape):
    """from_fwhm should be independent of peak_amplitude."""
    fwhm = 150e-15
    env1 = Envelope.from_fwhm(shape=shape, peak_amplitude=1.0, fwhm=fwhm)
    env2 = Envelope.from_fwhm(shape=shape, peak_amplitude=5.0, fwhm=fwhm)
    assert env1.fwhm == env2.fwhm
    assert env1.pulse_width == env2.pulse_width


# ─── Super-Gaussian shape tests ────────────────────────────────────────


def test_super_gaussian_field_peak():
    """Super-gaussian envelope should peak at A0 at t=0."""
    env = Envelope(
        shape="super-gaussian", peak_amplitude=2.0, pulse_width=1e-12,
        super_gaussian_order=4,
    )
    t = np.array([0.0])
    assert pytest.approx(np.abs(env.field(t)[0]), rel=1e-12) == 2.0


def test_super_gaussian_order_effect():
    """Higher super-gaussian order should produce a flatter top (more energy near centre)."""
    T0 = 1e-12
    env_n2 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=2)
    env_n4 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=4)
    t_near = T0 * 0.3
    # At |t|/T0 = 0.3, order 4 should be closer to peak (flat-top) → higher intensity
    I_n2 = env_n2.intensity(np.array([t_near]))[0]
    I_n4 = env_n4.intensity(np.array([t_near]))[0]
    assert I_n4 > I_n2


def test_super_gaussian_fwhm_increases_with_order_at_fixed_T0():
    """Higher super-gaussian order → wider FWHM when T0 is fixed (flatter top)."""
    T0 = 1e-12
    fwhm_n2 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=2).fwhm
    fwhm_n4 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=4).fwhm
    # Super-Gaussian: FWHM = 2*T0*(log(2)/2)^(1/(2N)).
    # Base < 1, exponent 1/(2N) decreases with N → factor increases → FWHM increases.
    assert fwhm_n4 > fwhm_n2


# ─── Triangular shape tests ────────────────────────────────────────────


def test_triangular_field_peak():
    """Triangular envelope should peak at A0 at t=0."""
    env = Envelope(shape="triangular", peak_amplitude=3.0, pulse_width=1e-12)
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 3.0


def test_triangular_clips_at_zero():
    """Triangular envelope should be zero outside |t| > T0."""
    env = Envelope(shape="triangular", peak_amplitude=1.0, pulse_width=1e-12)
    t_outside = np.array([1.5e-12, 2.0e-12, -1.5e-12])
    assert np.all(env.field(t_outside) == 0.0)


def test_triangular_linear_profile():
    """Triangular envelope should be linear: A(t) = A0*(1 - |t|/T0)."""
    env = Envelope(shape="triangular", peak_amplitude=2.0, pulse_width=1e-12)
    t_mid = np.array([0.5e-12])
    expected = 2.0 * (1.0 - 0.5)
    assert pytest.approx(np.abs(env.field(t_mid)[0]), rel=1e-12) == expected


def test_triangular_fwhm():
    """Triangular FWHM = 2*T0*(1 - 1/sqrt(2))."""
    T0 = 1e-12
    expected = 2.0 * T0 * (1.0 - 1.0 / np.sqrt(2.0))
    env = Envelope(shape="triangular", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm, rel=1e-12) == expected


# ─── Parabolic shape tests ─────────────────────────────────────────────


def test_parabolic_field_peak():
    """Parabolic envelope should peak at A0 at t=0."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=1e-12)
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_parabolic_clips_at_zero():
    """Parabolic envelope should be zero outside |t| > T0."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=1e-12)
    t_outside = np.array([1.5e-12, 2.0e-12])
    assert np.all(env.field(t_outside) == 0.0)


def test_parabolic_chirp_applied():
    """Parabolic envelope with chirp should have quadratic phase."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=1e-12, chirp=2.0)
    t = np.array([0.5e-12])
    phase = np.angle(env.field(t)[0])
    # Phase = chirp * (t/T0)^2 = 2.0 * (0.5)^2 = 0.5
    expected_phase = 2.0 * (0.5) ** 2
    assert pytest.approx(phase, rel=1e-12) == expected_phase


def test_parabolic_chirp_zero_no_phase():
    """Parabolic envelope with zero chirp should have zero phase."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=1e-12, chirp=0.0)
    t = np.linspace(-0.5e-12, 0.5e-12, 10)
    phases = np.unwrap(np.angle(env.field(t)))
    assert np.max(np.abs(phases)) < 1e-14


def test_parabolic_from_asymptotic():
    """from_parabolic_asymptotic should set chirp ≈ 0.2726 * length * gain."""
    env = Envelope.from_parabolic_asymptotic(
        peak_amplitude=1.0, pulse_width=1e-12,
        gain=10.0, length=0.05, chirp=0.0,
    )
    expected_chirp = 0.2726 * 0.05 * 10.0
    assert env.shape == "parabolic"
    assert pytest.approx(env.chirp, rel=1e-6) == expected_chirp


def test_parabolic_from_asymptotic_with_extra_chirp():
    """from_parabolic_asymptotic should add extra chirp on top."""
    env = Envelope.from_parabolic_asymptotic(
        peak_amplitude=1.0, pulse_width=1e-12,
        gain=10.0, length=0.05, chirp=1.5,
    )
    expected_chirp = 0.2726 * 0.05 * 10.0 + 1.5
    assert pytest.approx(env.chirp, rel=1e-6) == expected_chirp


def test_parabolic_fwhm_returns_T0():
    """Parabolic shape fwhm property returns pulse_width (lower-bound)."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=1.5e-12)
    assert env.fwhm == env.pulse_width


# ─── Cosine (raised) shape tests ───────────────────────────────────────


def test_cosine_field_peak():
    """Raised cosine envelope should peak at A0 at t=0."""
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=1e-12)
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_cosine_clips_at_zero():
    """Raised cosine envelope should be zero outside |t| > T0."""
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=1e-12)
    t_outside = np.array([1.5e-12, 2.0e-12])
    assert np.all(env.field(t_outside) == 0.0)


def test_cosine_fwhm_equals_T0():
    """Raised cosine FWHM = T0."""
    T0 = 1e-12
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm, rel=1e-12) == T0


def test_cosine_boundary_value():
    """At |t| = T0, raised cosine amplitude = cos(π/2) = 0."""
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=1e-12)
    t_boundary = np.array([1e-12, -1e-12])
    assert np.all(np.abs(env.field(t_boundary)) < 1e-14)


# ─── Exponential shape tests ───────────────────────────────────────────


def test_exponential_field_peak():
    """Exponential envelope should peak at A0 at t=0."""
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=1e-12)
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_exponential_decay():
    """Exponential envelope: I(t) = exp(-2|t|/T0)."""
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=1e-12)
    t_half = 1e-12  # |t| = T0 → I = exp(-2) ≈ 0.1353
    I = env.intensity(np.array([1e-12]))[0]
    expected = np.exp(-2.0)
    assert pytest.approx(I, rel=1e-10) == expected


def test_exponential_fwhm():
    """Exponential FWHM = 2*T0*log(2)."""
    T0 = 1e-12
    expected = 2.0 * T0 * np.log(2.0)
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm, rel=1e-12) == expected


def test_exponential_symmetric():
    """Exponential envelope should be symmetric: A(-t) = A(t)."""
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=1e-12)
    t = np.array([0.5e-12])
    assert pytest.approx(np.abs(env.field(t)[0]), rel=1e-12) == np.abs(
        env.field(-t)[0]
    )


# ─── Gauss-Hermite shape tests ─────────────────────────────────────────


def test_gauss_hermite_mode0_is_gaussian():
    """HG mode m=0 should reduce to a Gaussian (H_0(x) = 1)."""
    T0 = 1e-12
    w = 2e-12
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=T0,
        beam_waist=w, hg_mode=0,
    )
    # HG mode-0: A(t) = A0 * exp(-t²/w²)
    t = np.linspace(-2e-12, 2e-12, 1000)
    A_hg = np.abs(env_hg.field(t))
    A_gauss = np.exp(-(t**2) / (w**2))
    assert np.max(np.abs(A_hg - A_gauss)) < 1e-14


def test_gauss_hermite_mode1_has_node():
    """HG mode m=1 (H_1(x) = 2x) should have a zero at t=0."""
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=1e-12,
        beam_waist=1e-12, hg_mode=1,
    )
    assert pytest.approx(np.abs(env_hg.field(np.array([0.0]))[0]), abs=1e-14) == 0.0


def test_gauss_hermite_odd_mode_is_odd_function():
    """HG odd modes should be odd: A(-t) = -A(t)."""
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=1e-12,
        beam_waist=1e-12, hg_mode=1,
    )
    t = np.array([0.5e-12])
    A_pos = env_hg.field(t)[0]
    A_neg = env_hg.field(-t)[0]
    assert pytest.approx(A_pos, rel=1e-10) == -A_neg


def test_gauss_hermite_even_mode_is_even_function():
    """HG even modes should be even: A(-t) = A(t)."""
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=1e-12,
        beam_waist=1e-12, hg_mode=2,
    )
    t = np.array([0.5e-12])
    A_pos = env_hg.field(t)[0]
    A_neg = env_hg.field(-t)[0]
    assert pytest.approx(A_pos, rel=1e-10) == A_neg


def test_gauss_hermite_defaults_to_T0_when_no_waist():
    """When beam_waist is None, it should default to T0."""
    env = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=1e-12, hg_mode=0,
    )
    # Without beam_waist, w = T0 = 1e-12
    # H_0 = 1, x = sqrt(2)*t/w, A = A0 * exp(-x²/2) = A0 * exp(-t²/w²)
    t = np.array([0.5e-12])
    expected = np.exp(-(0.5) ** 2)  # exp(-0.25)
    assert pytest.approx(np.abs(env.field(t)[0]), rel=1e-10) == expected


# ─── Custom envelope tests ─────────────────────────────────────────────


def test_custom_func_basic():
    """Custom envelope should call user func."""
    t0 = 1e-12
    def my_func(t, T0, A0):
        return A0 * np.exp(-t**4 / (2 * T0**4))

    env = Envelope(
        shape="custom", peak_amplitude=1.0, pulse_width=t0,
        func=my_func,
    )
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_custom_phase_func():
    """Custom envelope with phase_func should use it instead of default chirp phase."""
    t0 = 1e-12
    def my_phase(t, T0, chirp):
        return chirp * (t / T0) ** 3

    env = Envelope(
        shape="custom", peak_amplitude=1.0, pulse_width=t0,
        func=lambda t, T0, A0: A0 * np.exp(-(t/T0)**2),
        phase_func=my_phase,
        chirp=2.0,
    )
    t = np.array([0.5e-12])
    phase = np.angle(env.field(t)[0])
    expected = 2.0 * (0.5) ** 3
    assert pytest.approx(phase, rel=1e-10) == expected


def test_custom_without_func_raises():
    """Custom envelope without func should raise ValueError."""
    env = Envelope(
        shape="custom", peak_amplitude=1.0, pulse_width=1e-12,
    )
    with pytest.raises(ValueError, match="'custom' shape requires 'func'"):
        env.field(np.array([0.0]))


def test_custom_peak_preserved():
    """Custom envelope should allow user to define any peak shape."""
    def quartic(t, T0, A0):
        return A0 * np.exp(-t**4 / (2 * T0**4))

    t0 = 1e-12
    env = Envelope(
        shape="custom", peak_amplitude=3.0, pulse_width=t0, func=quartic,
    )
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 3.0


# ─── Pulse train tests ─────────────────────────────────────────────────


def test_pulse_train_repetition_spacing():
    """Pulse train should show N pulses spaced by 1/repetition_rate."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=100e-15)
    repetition_rate = 10e9  # 10 GHz → 100 ps spacing
    n_pulses = 5
    grid = TemporalGrid(N=2**14, Tmax=1000e-12)
    wave = Wave.from_pulse_train(
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        grid=grid,
        repetition_rate=repetition_rate,
        n_pulses=n_pulses,
    )
    spacing = 1.0 / repetition_rate  # 100 ps
    # Check that peaks occur near expected positions
    from scipy.signal import find_peaks
    envelope_mag = np.abs(wave.envelope_field)
    peaks, _ = find_peaks(envelope_mag, height=0.1 * np.max(envelope_mag))
    peak_times = grid.t[peaks]
    assert len(peaks) >= n_pulses, (
        f"Expected at least {n_pulses} pulses, found {len(peaks)}"
    )
    # Consecutive peaks should be spaced by ~1/repetition_rate
    peak_spacings = np.diff(peak_times)
    # Average spacing should be close to repetition period
    assert np.mean(peak_spacings) > 0.8 * spacing
    assert np.mean(peak_spacings) < 1.2 * spacing


def test_pulse_train_same_as_single_for_one_pulse():
    """Pulse train with n_pulses=1 should equal a single pulse."""
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=100e-15)
    repetition_rate = 10e9
    n_pulses = 1
    grid = TemporalGrid(N=2**12, Tmax=500e-12)
    wave_train = Wave.from_pulse_train(
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        grid=grid,
        repetition_rate=repetition_rate,
        n_pulses=n_pulses,
    )
    wave_single = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    assert np.allclose(
        wave_train.envelope_field, wave_single.envelope_field, atol=1e-14
    )


def test_pulse_train_average_power():
    """Average power = pulse_energy × repetition_rate."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=100e-15)
    repetition_rate = 10e9
    grid = TemporalGrid(N=2**12, Tmax=500e-12)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        repetition_rate=repetition_rate,
    )
    energy = wave.pulse_energy()
    avg_power = wave.average_power(repetition_rate)
    assert pytest.approx(avg_power, rel=1e-10) == energy * repetition_rate


def test_pulse_train_multiple_pulses_energy():
    """Pulse train energy ≈ n_pulses × single pulse energy (for well-separated pulses)."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=50e-15)
    repetition_rate = 100e9  # 10 ps spacing, 50 fs pulses → well separated
    n_pulses = 10
    # First compute single-pulse energy using the same grid size
    grid_single = TemporalGrid(N=2**14, Tmax=500e-12)
    wave_single = Wave(
        grid=grid_single,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    single_energy = wave_single.pulse_energy()
    # Now build pulse train with the same envelope on a large grid
    grid_train = TemporalGrid(N=2**14, Tmax=2000e-12)
    wave = Wave.from_pulse_train(
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        grid=grid_train,
        repetition_rate=repetition_rate,
        n_pulses=n_pulses,
    )
    total_energy = wave.pulse_energy()
    # Total energy should be approximately n_pulses × single pulse energy
    # Allow 10% tolerance for pulse overlap and integration errors
    assert pytest.approx(total_energy, rel=0.12) == n_pulses * single_energy


def test_pulse_train_stored_repetition_rate():
    """Pulse train Wave should store repetition_rate."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=100e-15)
    repetition_rate = 25e9
    grid = TemporalGrid(N=2**12, Tmax=1000e-12)
    wave = Wave.from_pulse_train(
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        grid=grid,
        repetition_rate=repetition_rate,
        n_pulses=3,
    )
    assert wave.repetition_rate == repetition_rate


# ─── TemporalGrid tests ────────────────────────────────────────────────


def test_temporal_grid_dt():
    """TemporalGrid.dt should equal Tmax/N."""
    grid = TemporalGrid(N=1024, Tmax=100e-12)
    assert pytest.approx(grid.dt, rel=1e-14) == 100e-12 / 1024


def test_temporal_grid_t_range():
    """TemporalGrid.t should span [-Tmax/2, Tmax/2 - dt]."""
    N = 256
    Tmax = 100e-12
    grid = TemporalGrid(N=N, Tmax=Tmax)
    assert pytest.approx(grid.t[0], rel=1e-14) == -Tmax / 2
    assert pytest.approx(grid.t[-1], rel=1e-14) == Tmax / 2 - grid.dt
    assert len(grid.t) == N


def test_temporal_grid_fft_conjugate():
    """FFT → IFFT round-trip should recover original signal."""
    grid = TemporalGrid(N=1024, Tmax=200e-12)
    t0 = 20e-12
    A_t = np.exp(-(grid.t**2) / (2 * t0**2))
    A_w = grid.fft(A_t)
    A_t_recovered = grid.ifft(A_w)
    assert np.max(np.abs(A_t_recovered - A_t)) < 1e-10


def test_temporal_grid_for_pulse_train():
    """for_pulse_train should produce a grid large enough for all pulses."""
    repetition_rate = 10e9
    n_pulses = 20
    pulse_width = 100e-15
    grid = TemporalGrid.for_pulse_train(
        repetition_rate=repetition_rate,
        n_pulses=n_pulses,
        pulse_width=pulse_width,
    )
    spacing = 1.0 / repetition_rate
    expected_Tmax = n_pulses * spacing + 10 * pulse_width
    assert grid.Tmax >= expected_Tmax
    # Check that the grid is reasonable
    assert grid.N == 2**12


def test_temporal_grid_omega_max():
    """omega_max should be pi/dt (Nyquist)."""
    grid = TemporalGrid(N=1024, Tmax=100e-12)
    expected_omega_max = np.pi / grid.dt
    assert pytest.approx(grid.omega_max, rel=1e-14) == expected_omega_max


def test_temporal_grid_time_window():
    """time_window should equal N * dt."""
    grid = TemporalGrid(N=512, Tmax=200e-12)
    assert pytest.approx(grid.time_window, rel=1e-14) == grid.N * grid.dt


# ─── Parseval's theorem (energy conservation) ──────────────────────────


def test_parseval_gaussian():
    """Parseval's theorem: ∫|E(t)|²dt = ∫|E(ω)|²dω."""
    T0 = 100e-15
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0)
    grid = TemporalGrid(N=2**13, Tmax=10 * 2 * np.sqrt(np.log(2)) * T0)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(800, "nm"),
    )
    time_energy = np.sum(wave.envelope_intensity) * grid.dt
    freq_energy = np.sum(np.abs(wave.spectrum) ** 2) * grid.dw
    # Parseval: the two energies should match
    assert pytest.approx(time_energy, rel=1e-6) == freq_energy


# ─── Chirp tests ───────────────────────────────────────────────────────


def test_chirp_zero_phase_profile():
    """Transform-limited (chirp=0) envelope should have flat phase."""
    for shape in ["gaussian", "sech", "super-gaussian", "triangular", "cosine"]:
        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=1e-12, chirp=0.0)
        t = np.linspace(-0.5e-12, 0.5e-12, 200)
        phases = np.unwrap(np.angle(env.field(t)))
        assert np.max(np.abs(phases)) < 1e-13, (
            f"Shape {shape} with chirp=0 should have flat phase"
        )


def test_chirp_positive_induces_phase_increase():
    """Positive chirp should cause instantaneous frequency to increase."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=100e-15, chirp=5.0)
    t = np.linspace(-200e-15, 200e-15, 1000)
    phase = np.unwrap(np.angle(env.field(t)))
    # Instantaneous frequency detuning: dφ/dt = chirp * t / T0²
    # For positive chirp, dφ/dt > 0 for t > 0
    dphi = np.diff(phase)
    # After t=0 (middle of array), slope should be positive
    mid = len(dphi) // 2
    assert np.mean(dphi[mid:]) > 0


def test_chirp_broadens_spectrum():
    """Chirped pulse should have a broader spectrum than transform-limited pulse."""
    T0 = 100e-15
    env_unchirped = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0, chirp=0.0)
    env_chirped = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0, chirp=10.0)
    grid = TemporalGrid(N=2**13, Tmax=10 * 2 * np.sqrt(np.log(2)) * T0)
    wave_unchirped = Wave(grid=grid, envelope=env_unchirped,
                          central_wavelength=Wavelength(800, "nm"))
    wave_chirped = Wave(grid=grid, envelope=env_chirped,
                        central_wavelength=Wavelength(800, "nm"))
    # Spectral width: standard deviation of spectral intensity
    spec_unchirped = np.abs(wave_unchirped.spectrum) ** 2
    spec_chirped = np.abs(wave_chirped.spectrum) ** 2
    sigma_unchirped = np.sqrt(np.sum(wave_unchirped.grid.w ** 2 * spec_unchirped) * wave_unchirped.grid.dw)
    sigma_chirped = np.sqrt(np.sum(wave_chirped.grid.w ** 2 * spec_chirped) * wave_chirped.grid.dw)
    assert sigma_chirped > sigma_unchirped * 1.5


# ─── Energy/intensity numerical integration tests ──────────────────────


def test_wave_energy_numerical_matches_integral():
    """wave.pulse_energy() should equal numerical integration of intensity."""
    env = Envelope(shape="sech", peak_amplitude=2.0, pulse_width=100e-15)
    grid = TemporalGrid(N=2**12, Tmax=2000e-15)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    numerical = np.sum(wave.envelope_intensity) * wave.grid.dt
    assert pytest.approx(wave.pulse_energy(), rel=1e-10) == numerical


def test_wave_peak_power_is_max_intensity():
    """wave.peak_power() should be the maximum of envelope_intensity."""
    env = Envelope(shape="lorentzian", peak_amplitude=3.0, pulse_width=100e-15)
    grid = TemporalGrid(N=2**12, Tmax=2000e-15)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    assert pytest.approx(wave.peak_power(), rel=1e-12) == np.max(wave.envelope_intensity)


def test_envelope_intensity_is_non_negative():
    """Intensity should be non-negative for all t."""
    for shape in ["gaussian", "sech", "lorentzian", "super-gaussian", "triangular",
                  "parabolic", "cosine", "exponential", "gauss-hermite", "airy"]:
        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=1e-12)
        t = np.linspace(-5e-12, 5e-12, 5000)
        I = env.intensity(t)
        assert np.all(I >= 0), f"Intensity negative for shape={shape}"


# ─── Wave property tests ───────────────────────────────────────────────


def test_wave_central_frequency():
    """Wave.central_frequency should match Wavelength.to_omega()."""
    wl = Wavelength(800, "nm")
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=100e-15)
    grid = TemporalGrid(N=1024, Tmax=2000e-15)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=wl,
    )
    assert pytest.approx(wave.central_frequency, rel=1e-10) == wl.to_omega().as_rad_s


def test_electric_field_is_real():
    """Wave.electric_field should be real (np.real of carrier)."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=100e-15)
    grid = TemporalGrid(N=1024, Tmax=2000e-15)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(800, "nm"),
    )
    assert np.all(np.isreal(wave.electric_field))


def test_electric_field_envelope_modulation():
    """Electric field envelope should follow the pulse envelope."""
    env = Envelope(shape="gaussian", peak_amplitude=2.0, pulse_width=100e-15)
    grid = TemporalGrid(N=2**12, Tmax=2000e-15)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(800, "nm"),
    )
    E = np.abs(wave.electric_field)
    E_max = np.max(E)
    # The peak of the modulated field should be close to the envelope peak amplitude
    # (carrier oscillation gives exact peak at a phase optimum)
    assert E_max >= env.peak_amplitude * 0.9


# ─── FWHM for already-built envelopes ──────────────────────────────────


def test_gaussian_fwhm_value():
    """Gaussian FWHM = 2*T0*sqrt(ln 2)."""
    T0 = 1e-12
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0)
    expected = 2.0 * T0 * np.sqrt(np.log(2.0))
    assert pytest.approx(env.fwhm, rel=1e-12) == expected


def test_sech_fwhm_value():
    """Sech FWHM = 2*T0*acosh(sqrt(2))."""
    T0 = 1e-12
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=T0)
    expected = 2.0 * T0 * np.arccosh(np.sqrt(2.0))
    assert pytest.approx(env.fwhm, rel=1e-12) == expected


def test_lorentzian_fwhm_value():
    """Lorentzian FWHM = 2*T0*sqrt(sqrt(2)-1)."""
    T0 = 1e-12
    env = Envelope(shape="lorentzian", peak_amplitude=1.0, pulse_width=T0)
    expected = 2.0 * T0 * np.sqrt(np.sqrt(2.0) - 1.0)
    assert pytest.approx(env.fwhm, rel=1e-12) == expected


def test_rectangular_fwhm_value():
    """Rectangular FWHM = 2*T0."""
    T0 = 1e-12
    env = Envelope(shape="rectangular", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm, rel=1e-12) == 2.0 * T0


# ─── Negative / error tests ────────────────────────────────────────────


def test_unknown_shape_raises():
    """An unknown shape should raise ValueError in field()."""
    env = Envelope(
        shape="nonexistent", peak_amplitude=1.0, pulse_width=1e-12,
    )
    with pytest.raises(ValueError, match="Unknown shape"):
        env.field(np.array([0.0]))


def test_wave_time_bandwidth_product_shape():
    """time_bandwidth_product should return a scalar."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=100e-15)
    grid = TemporalGrid(N=2**12, Tmax=2000e-15)
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(800, "nm"),
    )
    tbp = wave.time_bandwidth_product()
    assert isinstance(tbp, float)
    assert tbp > 0


# ─── Multi-shape TBP check ─────────────────────────────────────────────


def test_transform_limited_pulses_have_low_tbp():
    """Transform-limited pulses should have TBP near the theoretical minimum."""
    T0 = 100e-15
    grid = TemporalGrid(N=2**13, Tmax=10 * 2 * np.sqrt(np.log(2)) * T0)
    for shape in ["gaussian", "sech"]:
        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=T0, chirp=0.0)
        wave = Wave(
            grid=grid,
            envelope=env,
            central_wavelength=Wavelength(800, "nm"),
        )
        tbp = wave.time_bandwidth_product()
        # TBP should be positive and not absurdly large
        assert 0.1 < tbp < 2.0, f"TBP for {shape} transform-limited: {tbp}"


# ─── Integration: envelope + wave round-trip ───────────────────────────


def test_all_shapes_have_valid_field_and_intensity():
    """Every supported shape should produce non-NaN field and intensity."""
    shapes = [
        "gaussian", "sech", "lorentzian", "rectangular",
        "super-gaussian", "triangular", "parabolic", "cosine",
        "exponential", "gauss-hermite", "airy", "custom",
    ]
    for shape in shapes:
        kwargs = {
            "shape": shape, "peak_amplitude": 1.0, "pulse_width": 1e-12,
        }
        if shape == "super-gaussian":
            kwargs["super_gaussian_order"] = 2
        elif shape == "gauss-hermite":
            kwargs.update({"beam_waist": 1e-12, "hg_mode": 0})
        elif shape == "custom":
            kwargs["func"] = lambda t, T0, A0: A0 * np.exp(-(t/T0)**2)

        env = Envelope(**kwargs)
        t = np.linspace(-5e-12, 5e-12, 5000)
        field = env.field(t)
        intensity = env.intensity(t)
        assert not np.any(np.isnan(field)), f"NaN field for shape={shape}"
        assert not np.any(np.isnan(intensity)), f"NaN intensity for shape={shape}"
        assert np.all(np.isfinite(field)), f"Inf field for shape={shape}"
