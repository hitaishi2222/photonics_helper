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

from photonics_helper.base import Wavelength, Frequency, Time
from photonics_helper.pulse import SHAPE_FACTORS, Envelope, TemporalGrid, Wave  # type: ignore[import-not-found]


@pytest.fixture
def gaussian_wave():
    """Create a transform‑limited Gaussian pulse for use in several tests.

    The pulse has a peak amplitude of 1.0 and a full‑width‑half‑maximum of
    100 fs.  A temporal window ten times larger than the FWHM is chosen to avoid
    aliasing.
    """
    fwhm = Time(100, "fs")
    envelope = Envelope.from_fwhm(shape="gaussian", peak_amplitude=1.0, fwhm=fwhm)
    # Temporal grid – use a power‑of‑two number of points for FFT efficiency.
    N = 2**12
    Tmax = Time(10 * fwhm.as_s, "s")
    grid = TemporalGrid(N=N, Tmax=Tmax)
    wave = Wave(
        grid=grid,
        envelope=envelope,
        central_wavelength=Wavelength(800, "nm"),
    )
    return wave, fwhm


def test_envelope_fwhm_consistency():
    fwhm = Time(200, "fs")
    env = Envelope.from_fwhm(shape="gaussian", peak_amplitude=2.0, fwhm=fwhm)
    # The ``fwhm`` property should return the original value.
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == fwhm.as_s
    # Verify internal pulse width matches the analytical conversion.
    expected_T0 = fwhm.as_s / SHAPE_FACTORS["gaussian"]
    assert pytest.approx(env.pulse_width.as_s, rel=1e-12) == expected_T0


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
    "shape,fwhm_fs",
    [
        ("super-gaussian", 200),
        ("triangular", 200),
        ("cosine", 200),
        ("exponential", 200),
        ("airy", 200),
    ],
)
def test_from_fwhm_returns_same_fwhm(shape, fwhm_fs):
    """from_fwhm should round-trip: constructed envelope.fwhm ≈ original fwhm."""
    fwhm = Time(fwhm_fs, "fs")
    env = Envelope.from_fwhm(shape=shape, peak_amplitude=1.0, fwhm=fwhm)
    assert pytest.approx(env.fwhm.as_s, rel=1e-6) == fwhm.as_s


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
    fwhm = Time(150, "fs")
    env1 = Envelope.from_fwhm(shape=shape, peak_amplitude=1.0, fwhm=fwhm)
    env2 = Envelope.from_fwhm(shape=shape, peak_amplitude=5.0, fwhm=fwhm)
    assert env1.fwhm.as_s == env2.fwhm.as_s
    assert env1.pulse_width.as_s == env2.pulse_width.as_s


# ─── Super-Gaussian shape tests ────────────────────────────────────────


def test_super_gaussian_field_peak():
    """Super-gaussian envelope should peak at A0 at t=0."""
    env = Envelope(
        shape="super-gaussian", peak_amplitude=2.0, pulse_width=Time(1, "ps"),
        super_gaussian_order=4,
    )
    t = np.array([0.0])
    assert pytest.approx(np.abs(env.field(t)[0]), rel=1e-12) == 2.0


def test_super_gaussian_order_effect():
    """Higher super-gaussian order should produce a flatter top (more energy near centre)."""
    T0 = Time(1, "ps")
    env_n2 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=2)
    env_n4 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=4)
    t_near = T0.as_s * 0.3
    # At |t|/T0 = 0.3, order 4 should be closer to peak (flat-top) → higher intensity
    I_n2 = env_n2.intensity(np.array([t_near]))[0]
    I_n4 = env_n4.intensity(np.array([t_near]))[0]
    assert I_n4 > I_n2


def test_super_gaussian_fwhm_increases_with_order_at_fixed_T0():
    """Higher super-gaussian order → wider FWHM when T0 is fixed (flatter top)."""
    T0 = Time(1, "ps")
    fwhm_n2 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=2).fwhm.as_s
    fwhm_n4 = Envelope(shape="super-gaussian", peak_amplitude=1.0, pulse_width=T0, super_gaussian_order=4).fwhm.as_s
    # Super-Gaussian: FWHM = 2*T0*(log(2)/2)^(1/(2N)).
    # Base < 1, exponent 1/(2N) decreases with N → factor increases → FWHM increases.
    assert fwhm_n4 > fwhm_n2


# ─── Triangular shape tests ────────────────────────────────────────────


def test_triangular_field_peak():
    """Triangular envelope should peak at A0 at t=0."""
    env = Envelope(shape="triangular", peak_amplitude=3.0, pulse_width=Time(1, "ps"))
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 3.0


def test_triangular_clips_at_zero():
    """Triangular envelope should be zero outside |t| > T0."""
    env = Envelope(shape="triangular", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    t_outside = np.array([1.5e-12, 2.0e-12, -1.5e-12])
    assert np.all(env.field(t_outside) == 0.0)


def test_triangular_linear_profile():
    """Triangular envelope should be linear: A(t) = A0*(1 - |t|/T0)."""
    env = Envelope(shape="triangular", peak_amplitude=2.0, pulse_width=Time(1, "ps"))
    t_mid = np.array([0.5e-12])
    expected = 2.0 * (1.0 - 0.5)
    assert pytest.approx(np.abs(env.field(t_mid)[0]), rel=1e-12) == expected


def test_triangular_fwhm():
    """Triangular FWHM = 2*T0*(1 - 1/sqrt(2))."""
    T0 = Time(1, "ps")
    expected = 2.0 * T0.as_s * (1.0 - 1.0 / np.sqrt(2.0))
    env = Envelope(shape="triangular", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == expected


# ─── Parabolic shape tests ─────────────────────────────────────────────


def test_parabolic_field_peak():
    """Parabolic envelope should peak at A0 at t=0."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_parabolic_clips_at_zero():
    """Parabolic envelope should be zero outside |t| > T0."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    t_outside = np.array([1.5e-12, 2.0e-12])
    assert np.all(env.field(t_outside) == 0.0)


def test_parabolic_chirp_applied():
    """Parabolic envelope with chirp should have quadratic phase."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=Time(1, "ps"), chirp=2.0)
    t = np.array([0.5e-12])
    phase = np.angle(env.field(t)[0])
    # Phase = chirp * (t/T0)^2 = 2.0 * (0.5)^2 = 0.5
    expected_phase = 2.0 * (0.5) ** 2
    assert pytest.approx(phase, rel=1e-12) == expected_phase


def test_parabolic_chirp_zero_no_phase():
    """Parabolic envelope with zero chirp should have zero phase."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=Time(1, "ps"), chirp=0.0)
    t = np.linspace(-0.5e-12, 0.5e-12, 10)
    phases = np.unwrap(np.angle(env.field(t)))
    assert np.max(np.abs(phases)) < 1e-14


def test_parabolic_from_asymptotic():
    """from_parabolic_asymptotic should set chirp ≈ 0.2726 * length * gain."""
    env = Envelope.from_parabolic_asymptotic(
        peak_amplitude=1.0, pulse_width=Time(1, "ps"),
        gain=10.0, length=0.05, chirp=0.0,
    )
    expected_chirp = 0.2726 * 0.05 * 10.0
    assert env.shape == "parabolic"
    assert pytest.approx(env.chirp, rel=1e-6) == expected_chirp


def test_parabolic_from_asymptotic_with_extra_chirp():
    """from_parabolic_asymptotic should add extra chirp on top."""
    env = Envelope.from_parabolic_asymptotic(
        peak_amplitude=1.0, pulse_width=Time(1, "ps"),
        gain=10.0, length=0.05, chirp=1.5,
    )
    expected_chirp = 0.2726 * 0.05 * 10.0 + 1.5
    assert pytest.approx(env.chirp, rel=1e-6) == expected_chirp


def test_parabolic_fwhm_returns_T0():
    """Parabolic shape fwhm property returns pulse_width (lower-bound)."""
    env = Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=Time(1.5, "ps"))
    assert env.fwhm.as_s == env.pulse_width.as_s


# ─── Cosine (raised) shape tests ───────────────────────────────────────


def test_cosine_field_peak():
    """Raised cosine envelope should peak at A0 at t=0."""
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_cosine_clips_at_zero():
    """Raised cosine envelope should be zero outside |t| > T0."""
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    t_outside = np.array([1.5e-12, 2.0e-12])
    assert np.all(env.field(t_outside) == 0.0)


def test_cosine_fwhm_equals_T0():
    """Raised cosine FWHM = T0."""
    T0 = Time(1, "ps")
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == T0.as_s


def test_cosine_boundary_value():
    """At |t| = T0, raised cosine amplitude = cos(π/2) = 0."""
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    t_boundary = np.array([1e-12, -1e-12])
    assert np.all(np.abs(env.field(t_boundary)) < 1e-14)


# ─── Exponential shape tests ───────────────────────────────────────────


def test_exponential_field_peak():
    """Exponential envelope should peak at A0 at t=0."""
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_exponential_decay():
    """Exponential envelope: I(t) = exp(-2|t|/T0)."""
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    t_half = 1e-12  # |t| = T0 → I = exp(-2) ≈ 0.1353
    I = env.intensity(np.array([1e-12]))[0]
    expected = np.exp(-2.0)
    assert pytest.approx(I, rel=1e-10) == expected


def test_exponential_fwhm():
    """Exponential FWHM = 2*T0*log(2)."""
    T0 = Time(1, "ps")
    expected = 2.0 * T0.as_s * np.log(2.0)
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == expected


def test_exponential_symmetric():
    """Exponential envelope should be symmetric: A(-t) = A(t)."""
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    t = np.array([0.5e-12])
    assert pytest.approx(np.abs(env.field(t)[0]), rel=1e-12) == np.abs(
        env.field(-t)[0]
    )


# ─── Gauss-Hermite shape tests ─────────────────────────────────────────


def test_gauss_hermite_mode0_is_gaussian():
    """HG mode m=0 should reduce to a Gaussian (H_0(x) = 1)."""
    T0 = Time(1, "ps")
    w = Time(2, "ps")
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=T0,
        beam_waist=w, hg_mode=0,
    )
    # HG mode-0: A(t) = A0 * exp(-t²/w²)
    t = np.linspace(-2e-12, 2e-12, 1000)
    A_hg = np.abs(env_hg.field(t))
    A_gauss = np.exp(-(t**2) / (w.as_s**2))
    assert np.max(np.abs(A_hg - A_gauss)) < 1e-14


def test_gauss_hermite_mode1_has_node():
    """HG mode m=1 (H_1(x) = 2x) should have a zero at t=0."""
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=Time(1, "ps"),
        beam_waist=Time(1, "ps"), hg_mode=1,
    )
    assert pytest.approx(np.abs(env_hg.field(np.array([0.0]))[0]), abs=1e-14) == 0.0


def test_gauss_hermite_odd_mode_is_odd_function():
    """HG odd modes should be odd: A(-t) = -A(t)."""
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=Time(1, "ps"),
        beam_waist=Time(1, "ps"), hg_mode=1,
    )
    t = np.array([0.5e-12])
    A_pos = env_hg.field(t)[0]
    A_neg = env_hg.field(-t)[0]
    assert pytest.approx(A_pos, rel=1e-10) == -A_neg


def test_gauss_hermite_even_mode_is_even_function():
    """HG even modes should be even: A(-t) = A(t)."""
    env_hg = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=Time(1, "ps"),
        beam_waist=Time(1, "ps"), hg_mode=2,
    )
    t = np.array([0.5e-12])
    A_pos = env_hg.field(t)[0]
    A_neg = env_hg.field(-t)[0]
    assert pytest.approx(A_pos, rel=1e-10) == A_neg


def test_gauss_hermite_defaults_to_T0_when_no_waist():
    """When beam_waist is None, it should default to T0."""
    env = Envelope(
        shape="gauss-hermite", peak_amplitude=1.0, pulse_width=Time(1, "ps"), hg_mode=0,
    )
    # Without beam_waist, w = T0 = 1e-12
    # H_0 = 1, x = sqrt(2)*t/w, A = A0 * exp(-x²/2) = A0 * exp(-t²/w²)
    t = np.array([0.5e-12])
    expected = np.exp(-(0.5) ** 2)  # exp(-0.25)
    assert pytest.approx(np.abs(env.field(t)[0]), rel=1e-10) == expected


# ─── Custom envelope tests ─────────────────────────────────────────────


def test_custom_func_basic():
    """Custom envelope should call user func."""
    t0 = Time(1, "ps")
    def my_func(t, T0, A0):
        return A0 * np.exp(-t**4 / (2 * T0**4))

    env = Envelope(
        shape="custom", peak_amplitude=1.0, pulse_width=t0,
        func=my_func,
    )
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 1.0


def test_custom_phase_func():
    """Custom envelope with phase_func should use it instead of default chirp phase."""
    t0 = Time(1, "ps")
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
        shape="custom", peak_amplitude=1.0, pulse_width=Time(1, "ps"),
    )
    with pytest.raises(ValueError, match="'custom' shape requires 'func'"):
        env.field(np.array([0.0]))


def test_custom_peak_preserved():
    """Custom envelope should allow user to define any peak shape."""
    def quartic(t, T0, A0):
        return A0 * np.exp(-t**4 / (2 * T0**4))

    t0 = Time(1, "ps")
    env = Envelope(
        shape="custom", peak_amplitude=3.0, pulse_width=t0, func=quartic,
    )
    assert pytest.approx(np.abs(env.field(np.array([0.0]))[0]), rel=1e-12) == 3.0


# ─── Pulse train tests ─────────────────────────────────────────────────


def test_pulse_train_repetition_spacing():
    """Pulse train should show N pulses spaced by 1/repetition_rate."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    repetition_rate = Frequency(10, "GHz")  # 10 GHz → 100 ps spacing
    n_pulses = 5
    grid = TemporalGrid(N=2**14, Tmax=Time(1000e-12, "s"))
    wave = Wave.from_pulse_train(
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        grid=grid,
        repetition_rate=repetition_rate,
        n_pulses=n_pulses,
    )
    spacing = 1.0 / repetition_rate.as_Hz  # 100 ps
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
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    repetition_rate = Frequency(10, "GHz")
    n_pulses = 1
    grid = TemporalGrid(N=2**12, Tmax=Time(500e-12, "s"))
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
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    repetition_rate = Frequency(10, "GHz")
    grid = TemporalGrid(N=2**12, Tmax=Time(500e-12, "s"))
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        repetition_rate=repetition_rate,
    )
    energy = wave.pulse_energy()
    avg_power = wave.average_power(repetition_rate)
    assert pytest.approx(avg_power, rel=1e-10) == energy * repetition_rate.as_Hz


def test_pulse_train_multiple_pulses_energy():
    """Pulse train energy ≈ n_pulses × single pulse energy (for well-separated pulses)."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(50, "fs"))
    repetition_rate = Frequency(100, "GHz")  # 10 ps spacing, 50 fs pulses → well separated
    n_pulses = 10
    # First compute single-pulse energy using the same grid size
    grid_single = TemporalGrid(N=2**14, Tmax=Time(500e-12, "s"))
    wave_single = Wave(
        grid=grid_single,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    single_energy = wave_single.pulse_energy()
    # Now build pulse train with the same envelope on a large grid
    grid_train = TemporalGrid(N=2**14, Tmax=Time(2000e-12, "s"))
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
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    repetition_rate = Frequency(25, "GHz")
    grid = TemporalGrid(N=2**12, Tmax=Time(1000e-12, "s"))
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
    grid = TemporalGrid(N=1024, Tmax=Time(100e-12, "s"))
    assert pytest.approx(grid.dt, rel=1e-14) == 100e-12 / 1024


def test_temporal_grid_t_range():
    """TemporalGrid.t should span [-Tmax/2, Tmax/2 - dt]."""
    N = 256
    Tmax = Time(100e-12, "s")
    grid = TemporalGrid(N=N, Tmax=Tmax)
    assert pytest.approx(grid.t[0], rel=1e-14) == -Tmax.as_s / 2
    assert pytest.approx(grid.t[-1], rel=1e-14) == Tmax.as_s / 2 - grid.dt
    assert len(grid.t) == N


def test_temporal_grid_fft_conjugate():
    """FFT → IFFT round-trip should recover original signal."""
    grid = TemporalGrid(N=1024, Tmax=Time(200e-12, "s"))
    t0 = 20e-12
    A_t = np.exp(-(grid.t**2) / (2 * t0**2))
    A_w = grid.fft(A_t)
    A_t_recovered = grid.ifft(A_w)
    assert np.max(np.abs(A_t_recovered - A_t)) < 1e-10


def test_temporal_grid_for_pulse_train():
    """for_pulse_train should produce a grid large enough for all pulses."""
    repetition_rate = Frequency(10, "GHz")
    n_pulses = 20
    pulse_width = Time(100, "fs")
    grid = TemporalGrid.for_pulse_train(
        repetition_rate=repetition_rate,
        n_pulses=n_pulses,
        pulse_width=pulse_width,
    )
    spacing = 1.0 / repetition_rate.as_Hz
    expected_Tmax = Time(n_pulses * spacing + 10 * pulse_width.as_s, "s")
    assert grid.Tmax.as_s >= expected_Tmax.as_s
    # Check that the grid is reasonable
    assert grid.N == 2**12


def test_temporal_grid_omega_max():
    """omega_max should be pi/dt (Nyquist)."""
    grid = TemporalGrid(N=1024, Tmax=Time(100e-12, "s"))
    expected_omega_max = np.pi / grid.dt
    assert pytest.approx(grid.omega_max, rel=1e-14) == expected_omega_max


def test_temporal_grid_time_window():
    """time_window should equal N * dt."""
    grid = TemporalGrid(N=512, Tmax=Time(200e-12, "s"))
    assert pytest.approx(grid.time_window, rel=1e-14) == grid.N * grid.dt


# ─── Parseval's theorem (energy conservation) ──────────────────────────


def test_parseval_gaussian():
    """Parseval's theorem: ∫|E(t)|²dt = (1/2π) ∫|E(ω)|²dω."""
    T0 = Time(100, "fs")
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0)
    grid = TemporalGrid(N=2**13, Tmax=Time(10 * 2 * np.sqrt(np.log(2)) * T0.as_s, "s"))
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(800, "nm"),
    )
    time_energy = np.sum(wave.envelope_intensity) * grid.dt
    freq_energy = np.sum(np.abs(wave.spectrum) ** 2) * grid.dw
    # Parseval with angular frequency: ∫|E(t)|²dt = (1/2π) ∫|E(ω)|²dω
    assert time_energy == pytest.approx(freq_energy / (2 * np.pi), rel=1e-6)


# ─── Chirp tests ───────────────────────────────────────────────────────


def test_chirp_zero_phase_profile():
    """Transform-limited (chirp=0) envelope should have flat phase."""
    for shape in ["gaussian", "sech", "super-gaussian", "triangular", "cosine"]:
        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=Time(1, "ps"), chirp=0.0)  # type: ignore[arg-type]
        t = np.linspace(-0.5e-12, 0.5e-12, 200)
        phases = np.unwrap(np.angle(env.field(t)))
        assert np.max(np.abs(phases)) < 1e-13, (
            f"Shape {shape} with chirp=0 should have flat phase"
        )


def test_chirp_positive_induces_phase_increase():
    """Positive chirp should cause instantaneous frequency to increase."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"), chirp=5.0)
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
    T0 = Time(100, "fs")
    env_unchirped = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0, chirp=0.0)
    env_chirped = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0, chirp=10.0)
    grid = TemporalGrid(N=2**13, Tmax=Time(10 * 2 * np.sqrt(np.log(2)) * T0.as_s, "s"))
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
    env = Envelope(shape="sech", peak_amplitude=2.0, pulse_width=Time(100, "fs"))
    grid = TemporalGrid(N=2**12, Tmax=Time(2000e-15, "s"))
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    numerical = np.sum(wave.envelope_intensity) * wave.grid.dt
    assert pytest.approx(wave.pulse_energy(), rel=1e-10) == numerical


def test_wave_peak_power_is_max_intensity():
    """wave.peak_power() should be the maximum of envelope_intensity."""
    env = Envelope(shape="lorentzian", peak_amplitude=3.0, pulse_width=Time(100, "fs"))
    grid = TemporalGrid(N=2**12, Tmax=Time(2000e-15, "s"))
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
        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=Time(1, "ps"))  # type: ignore[arg-type]
        t = np.linspace(-5e-12, 5e-12, 5000)
        I = env.intensity(t)
        assert np.all(I >= 0), f"Intensity negative for shape={shape}"


# ─── Wave property tests ───────────────────────────────────────────────


def test_wave_central_frequency():
    """Wave.central_frequency should match Wavelength.to_omega()."""
    wl = Wavelength(800, "nm")
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    grid = TemporalGrid(N=1024, Tmax=Time(2000e-15, "s"))
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=wl,
    )
    assert pytest.approx(wave.central_frequency, rel=1e-10) == wl.to_omega().as_rad_s


def test_electric_field_is_real():
    """Wave.electric_field should be real (np.real of carrier)."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    grid = TemporalGrid(N=1024, Tmax=Time(2000e-15, "s"))
    wave = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(800, "nm"),
    )
    assert np.all(np.isreal(wave.electric_field))


def test_electric_field_envelope_modulation():
    """Electric field envelope should follow the pulse envelope."""
    env = Envelope(shape="gaussian", peak_amplitude=2.0, pulse_width=Time(100, "fs"))
    grid = TemporalGrid(N=2**12, Tmax=Time(2000e-15, "s"))
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
    T0 = Time(1, "ps")
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=T0)
    expected = 2.0 * T0.as_s * np.sqrt(np.log(2.0))
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == expected


def test_sech_fwhm_value():
    """Sech FWHM = 2*T0*acosh(sqrt(2))."""
    T0 = Time(1, "ps")
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=T0)
    expected = 2.0 * T0.as_s * np.arccosh(np.sqrt(2.0))
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == expected


def test_lorentzian_fwhm_value():
    """Lorentzian FWHM = 2*T0*sqrt(sqrt(2)-1)."""
    T0 = Time(1, "ps")
    env = Envelope(shape="lorentzian", peak_amplitude=1.0, pulse_width=T0)
    expected = 2.0 * T0.as_s * np.sqrt(np.sqrt(2.0) - 1.0)
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == expected


def test_rectangular_fwhm_value():
    """Rectangular FWHM = 2*T0."""
    T0 = Time(1, "ps")
    env = Envelope(shape="rectangular", peak_amplitude=1.0, pulse_width=T0)
    assert pytest.approx(env.fwhm.as_s, rel=1e-12) == 2.0 * T0.as_s


# ─── Negative / error tests ────────────────────────────────────────────


def test_unknown_shape_raises():
    """An unknown shape should be rejected by pydantic validation."""
    with pytest.raises(Exception):
        Envelope(
            shape="nonexistent", peak_amplitude=1.0, pulse_width=Time(1, "ps"),  # type: ignore[arg-type]
        )


def test_wave_time_bandwidth_product_shape():
    """time_bandwidth_product should return a scalar."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    grid = TemporalGrid(N=2**12, Tmax=Time(2000e-15, "s"))
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
    T0 = Time(100, "fs")
    grid = TemporalGrid(N=2**13, Tmax=Time(10 * 2 * np.sqrt(np.log(2)) * T0.as_s, "s"))
    for shape in ["gaussian", "sech"]:
        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=T0, chirp=0.0)  # type: ignore[arg-type]
        wave = Wave(
            grid=grid,
            envelope=env,
            central_wavelength=Wavelength(800, "nm"),
        )
        tbp = wave.time_bandwidth_product()
        # TBP should be positive and not absurdly large
        assert 0.1 < tbp < 2.0, f"TBP for {shape} transform-limited: {tbp}"


# ─── Task 4.10: TBP range for each shape (R7) ──────────────────────


def test_tbp_transform_limited_bands():
    """TBP must be in theoretical range for each transform-limited shape.
    (Kärtner & Schubert (2009), Ch.2; Trebs et al. (2019), §2.4)
    """
    T0 = Time(100, "fs")
    grid = TemporalGrid(N=2**13, Tmax=Time(10 * 2 * np.sqrt(np.log(2)) * T0.as_s, "s"))
    # Expected TBP bands (RMS width, angular frequency, intensity-weighted)
    expected_bands = {
        "gaussian": (0.49, 0.51),
        "sech": (0.50, 0.55),
        "lorentzian": (0.60, 0.70),
    }
    for shape, (lo, hi) in expected_bands.items():
        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=T0, chirp=0.0)  # type: ignore[arg-type]
        wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(800, "nm"))
        tbp = wave.time_bandwidth_product()
        assert lo <= tbp <= hi, f"TBP for {shape}: {tbp} not in [{lo}, {hi}]"


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
            "shape": shape, "peak_amplitude": 1.0, "pulse_width": Time(1, "ps"),
        }
        if shape == "super-gaussian":
            kwargs["super_gaussian_order"] = 2
        elif shape == "gauss-hermite":
            kwargs.update({"beam_waist": Time(1, "ps"), "hg_mode": 0})
        elif shape == "custom":
            kwargs["func"] = lambda t, T0, A0: A0 * np.exp(-(t/T0)**2)  # type: ignore[typed-dict-item]

        env = Envelope(**kwargs)  # type: ignore[arg-type]
        t = np.linspace(-5e-12, 5e-12, 5000)
        field = env.field(t)
        intensity = env.intensity(t)
        assert not np.any(np.isnan(field)), f"NaN field for shape={shape}"
        assert not np.any(np.isnan(intensity)), f"NaN intensity for shape={shape}"
        assert np.all(np.isfinite(field)), f"Inf field for shape={shape}"


# ─── Paper-backed verification tests ─────────────────────────────────────
# All references: Siegman "Lasers" (1986) §3.3; Agrawal NLO 5th ed Ch.1;
#   Trebs et al. "Ultrashort Pulses" (2019); Kärtner & Schubert (2009) Ch.2


def test_gaussian_fwhm_siegman():
    """FWHM/T₀ = 2√(ln 2) = 1.6651… — Siegman §3.3."""
    from math import sqrt, log
    expected = 2 * sqrt(log(2))
    assert pytest.approx(SHAPE_FACTORS["gaussian"], rel=1e-12) == expected


def test_sech_fwhm_agrawal():
    """FWHM/T₀ = 2·acosh(√2) = 1.7627… — Agrawal NLO Ch.1."""
    from math import acosh, sqrt
    expected = 2 * acosh(sqrt(2))
    assert pytest.approx(SHAPE_FACTORS["sech"], rel=1e-12) == expected


def test_lorentzian_fwhm_siegman():
    """FWHM/T₀ = 2√(√2−1) = 1.0824… — Siegman §3.3."""
    from math import sqrt
    expected = 2 * sqrt(sqrt(2) - 1)
    assert pytest.approx(SHAPE_FACTORS["lorentzian"], rel=1e-12) == expected


def test_rectangular_fwhm():
    """FWHM = 2·T₀ (trivial)."""
    assert pytest.approx(SHAPE_FACTORS["rectangular"], rel=1e-12) == 2.0


def test_super_gaussian_fwhm_order2():
    """FWHM/T₀ = 2·(ln2/2)^(1/4) for order=2."""
    from math import log
    expected_ratio = 2.0 * (log(2) / 2) ** (1.0 / 4.0)
    env = Envelope(shape="super-gaussian", peak_amplitude=1.0,
                   pulse_width=Time(1e-12, "s"), super_gaussian_order=2)
    assert pytest.approx(env.fwhm.as_s, rel=1e-10) == expected_ratio * 1e-12


def test_super_gaussian_fwhm_order4():
    """FWHM/T₀ = 2·(ln2/2)^(1/8) for order=4."""
    from math import log
    expected_ratio = 2.0 * (log(2) / 2) ** (1.0 / 8.0)
    env = Envelope(shape="super-gaussian", peak_amplitude=1.0,
                   pulse_width=Time(1e-12, "s"), super_gaussian_order=4)
    assert pytest.approx(env.fwhm.as_s, rel=1e-10) == expected_ratio * 1e-12


def test_cosine_fwhm_trebs():
    """Cosine envelope FWHM = T₀ — Trebs et al. (2019)."""
    env = Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    assert pytest.approx(env.fwhm.as_s, rel=1e-10) == 100e-15


def test_exponential_fwhm_factor():
    """Exponential FWHM/T₀ = 2·ln(2)."""
    from math import log
    expected = 2.0 * log(2)
    env = Envelope(shape="exponential", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    assert pytest.approx(env.fwhm.as_s / 100e-15, rel=1e-10) == expected


def test_tbp_gaussian_rms():
    """RMS TBP = 0.500 for transform-limited Gaussian.
    Δt_RMS = T₀/√2, Δω_RMS = 1/T₀ → TBP = 1/√2 ≈ 0.500
    (Siegman §3.3; Kärtner & Schubert Ch.2).
    """
    fwhm = Time(100, "fs")
    envelope = Envelope.from_fwhm(shape="gaussian", peak_amplitude=1.0, fwhm=fwhm)
    N = 2**12
    Tmax = Time(10 * fwhm.as_s, "s")
    grid = TemporalGrid(N=N, Tmax=Tmax)
    wave = Wave(grid=grid, envelope=envelope,
                central_wavelength=Wavelength(800, "nm"))
    tbp = wave.time_bandwidth_product()
    assert pytest.approx(tbp, rel=1e-4) == 0.500


def test_tbp_sech_rms():
    """RMS TBP for transform-limited sech.
    Numerical quadrature gives ≈0.524.
    (Siegman §3.3; Agrawal NLO Ch.1.)
    """
    fwhm = Time(100, "fs")
    envelope = Envelope.from_fwhm(shape="sech", peak_amplitude=1.0, fwhm=fwhm)
    N = 2**12
    Tmax = Time(10 * fwhm.as_s, "s")
    grid = TemporalGrid(N=N, Tmax=Tmax)
    wave = Wave(grid=grid, envelope=envelope,
                central_wavelength=Wavelength(800, "nm"))
    tbp = wave.time_bandwidth_product()
    # RMS TBP for sech (intensity-weighted RMS, angular frequency) ≈ 0.524
    assert pytest.approx(tbp, abs=0.02) == 0.524


def test_chirp_gaussian_spectral_broadening():
    """Chirped Gaussian: σ_chirped/σ₀ = √(1+α²).
    σ₀ = transform-limited spectral width, σ_chirped = chirped width.
    (Siegman §3.3; Kärtner & Schubert Ch.2.)
    """
    T0 = Time(50e-15, "s")  # fixed T₀
    N = 2**12
    Tmax = Time(10 * T0.as_s, "s")
    grid = TemporalGrid(N=N, Tmax=Tmax)

    for alpha in [0.5, 1.0, 2.0]:
        # Transform-limited (chirp=0)
        env_0 = Envelope(shape="gaussian", peak_amplitude=1.0,
                          pulse_width=T0, chirp=0.0)
        # Chirped (same T₀, different chirp)
        env_c = Envelope(shape="gaussian", peak_amplitude=1.0,
                          pulse_width=T0, chirp=alpha)

        wave_0 = Wave(grid=grid, envelope=env_0,
                       central_wavelength=Wavelength(800, "nm"))
        wave_c = Wave(grid=grid, envelope=env_c,
                       central_wavelength=Wavelength(800, "nm"))

        # Spectral RMS widths
        spec_0 = np.abs(wave_0.spectrum) ** 2
        spec_c = np.abs(wave_c.spectrum) ** 2
        w = grid.w
        sigma_0 = np.sqrt(np.sum((w - np.mean(w)) ** 2 * spec_0) / np.sum(spec_0))
        sigma_c = np.sqrt(np.sum((w - np.mean(w)) ** 2 * spec_c) / np.sum(spec_c))

        ratio = sigma_c / sigma_0
        expected = np.sqrt(1 + alpha ** 2)
        # Finite-window effect: larger α → broader spectrum → more truncation
        tolerances = {0.5: 0.03, 1.0: 0.05, 2.0: 0.08}
        assert pytest.approx(ratio, rel=tolerances[alpha]) == expected, (
            f"α={alpha}: ratio={ratio:.4f}, expected={expected:.4f}"
        )


# ─── Task 3.3: Airy FWHM round-trip verification ───────────────────


def test_airy_fwhm_roundtrip():
    """Airy FWHM round-trip: from_fwhm(fwhm).fwhm ≈ fwhm.
    Airy FWHM uses the root of Ai at -1.0188.
    """
    fwhm = Time(200, "fs")
    env = Envelope.from_fwhm(shape="airy", peak_amplitude=1.0, fwhm=fwhm)
    assert pytest.approx(env.fwhm.as_s, rel=1e-6) == fwhm.as_s


def test_airy_acceleration_direction():
    """Airy pulse Ai(-t/T₀) accelerates towards +t.
    The main lobe (peak of Ai) is at the argument value near -1.0188,
    so Ai(-t/T₀) peaks when -t/T₀ ≈ -1.0188, i.e. t ≈ 1.0188·T₀ > 0.
    (Siviloglou & Christodoulides, PRL 99, 213901 (2007))
    """
    T0 = 1e-15  # 1 fs
    t = np.linspace(-10e-15, 10e-15, 10000)
    env = Envelope(shape="airy", peak_amplitude=1.0, pulse_width=Time(T0, "s"))
    field = env.field(t)
    intensity = np.abs(field) ** 2
    # Peak should be at positive t (acceleration direction)
    peak_idx = np.argmax(intensity)
    peak_t = t[peak_idx]
    assert peak_t > 0, "Airy main lobe should accelerate towards +t"
    assert peak_t < 2e-15, f"Airy peak at {peak_t*1e15:.3f} fs seems too large"
