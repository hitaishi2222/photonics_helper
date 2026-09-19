"""Multi-phonon Raman response: normalization, causality and solver integration.

Ground truth: the delayed Raman response must be causal and normalized so that
``integral_0^inf h_R dt = 1`` (Agrawal, *Nonlinear Fiber Optics*, 5th ed.,
Sec. 2.3.2), and the multi-mode form is a superposition of damped oscillators
(Hollenbeck & Cantrell, *JOSA B* 19, 2886 (2002); Stolen et al., *JOSA B* 6,
1159 (1989)). The Stokes sideband must gain (Blow & Wood 1989).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import (
    FiberProfile,
    GNLSESolver,
    SplitStepEngine,
    _delayed_h_R,
    _response_fR,
)
from photonics_helper.phonon import PHONON_MATERIALS, PhononMode, PhononResponse
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

CM_PER_S = 2.99792458e10  # speed of light in cm/s


def _grid() -> TemporalGrid:
    return TemporalGrid(N=2**14, Tmax=Time(20e-12, "s"))


def _silica_like_phonon(fR: float | None = 0.18) -> PhononResponse:
    mode = PhononMode(shift_cm=440.0, linewidth_cm=45.0)
    return PhononResponse(modes=[mode], fR=fR)


# ─── h_R properties ───────────────────────────────────────────────────────


def test_phonon_h_R_is_causal_and_unit_integral() -> None:
    grid = _grid()
    t = grid.t
    for name, modes in PHONON_MATERIALS.items():
        response = PhononResponse(modes=modes, fR=0.18)
        h = response.h_R(t)
        assert np.all(h[t < 0] == 0.0), f"{name}: response is not causal"
        # finite-grid trapezoid; tails and discretisation leave ~1% residual
        assert np.trapezoid(h, t) == pytest.approx(1.0, abs=2e-2), name


def test_empty_response_is_zero() -> None:
    grid = _grid()
    assert np.all(PhononResponse(modes=[]).h_R(grid.t) == 0.0)


def test_single_mode_peaks_at_its_shift_and_decays_with_two_over_gamma() -> None:
    grid = _grid()
    response = _silica_like_phonon()
    h = response.h_R(grid.t)

    # Frequency-domain peak at the mode shift (single Lorentzian).
    spectrum = np.abs(np.fft.rfft(h))
    freqs_cm = np.fft.rfftfreq(len(h), d=grid.dt) / CM_PER_S
    assert freqs_cm[int(np.argmax(spectrum))] == pytest.approx(440.0, abs=5.0)

    # Envelope decay: tau = 2/gamma with gamma = 2*pi*c*linewidth.
    gamma = 2 * np.pi * CM_PER_S * 45.0
    tau = 2.0 / gamma
    envelope = np.abs(hilbert(h))
    t = grid.t
    i0 = int(np.argmax(envelope))
    i_tau = int(np.argmin(np.abs(t - (t[i0] + tau))))
    ratio = envelope[i_tau] / envelope[i0]
    assert ratio == pytest.approx(np.exp(-1.0), rel=0.15)


def test_multi_mode_response_is_not_a_single_lorentzian() -> None:
    grid = _grid()
    response = PhononResponse(modes=PHONON_MATERIALS["YAG"], fR=0.18)
    spectrum = np.abs(np.fft.rfft(response.h_R(grid.t)))
    freqs_cm = np.fft.rfftfreq(len(grid.t), d=grid.dt) / CM_PER_S

    shifts = [float(m.shift_cm.as_1_cm) for m in PHONON_MATERIALS["YAG"]]
    centroid = float(np.sum(freqs_cm * spectrum) / np.sum(spectrum))
    assert min(shifts) - 10 <= centroid <= max(shifts) + 10

    inner = spectrum[1:-1]
    peaks = (inner > spectrum[:-2]) & (inner > spectrum[2:]) & (inner > 0.2 * inner.max())
    assert peaks.sum() >= 2, "multi-mode response should show several lineshapes"


# ─── Response contract used by the solver ─────────────────────────────────


def test_single_mode_response_exposes_the_same_contract() -> None:
    grid = _grid()
    spec = RamanSpec(
        name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
    )
    response = RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)

    assert np.allclose(response.h_R(grid.t), response._h_R(grid.t))
    assert _delayed_h_R(response, grid.t).shape == grid.t.shape
    assert _response_fR(response) == pytest.approx(0.18)


def test_response_without_h_R_is_rejected() -> None:
    with pytest.raises(TypeError, match="h_R"):
        _delayed_h_R(object(), _grid().t)


def test_response_without_fR_is_an_explicit_error() -> None:
    grid = _grid()
    wave = _two_tone_wave(grid)
    fiber = _fiber(wave, _silica_like_phonon(fR=None))
    engine = SplitStepEngine(
        pulse=wave, fiber=fiber, betas=np.array([0.0]), include_raman=True
    )
    with pytest.raises(ValueError, match="fR"):
        engine._nonlinear_step(engine.A, 1e-5)


def test_engine_uses_the_phonon_response_fft() -> None:
    grid = _grid()
    wave = _two_tone_wave(grid)
    response = _silica_like_phonon()
    fiber = _fiber(wave, response)
    engine = SplitStepEngine(
        pulse=wave, fiber=fiber, betas=np.array([0.0]), include_raman=True
    )
    assert np.allclose(engine._get_h_R_fft(), grid.fft(response.h_R(grid.t)))


# ─── Propagation: Stokes gains, anti-Stokes loses ─────────────────────────


def _two_tone_wave(grid: TemporalGrid) -> Wave:
    env = Envelope(
        shape="gaussian", peak_amplitude=np.sqrt(5.0), pulse_width=Time(1.0, "s")
    )
    wave = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(1.55e-6, "m")
    )
    t = grid.t
    omega_r = 2 * np.pi * 13.2e12
    return wave.with_field(np.sqrt(5.0) * (1 + 1e-3 * np.cos(omega_r * t)))


def _fiber(wave: Wave, response: object) -> FiberProfile:
    return FiberProfile.from_gamma(
        gamma=2.0e-3,
        n2=2.6e-20,
        omega0=wave.central_frequency,
        length=Length(2.0, "m"),
        raman_response=response,
    )


def test_phonon_raman_amplifies_stokes_not_anti_stokes() -> None:
    grid = _grid()
    wave = _two_tone_wave(grid)
    field = wave.envelope_field
    omega_r = 2 * np.pi * 13.2e12
    # Library convention: grid.w is the offset Omega from the carrier, so the
    # red (Stokes) sideband sits at Omega = -omega_R and anti-Stokes at +omega_R.
    i_stokes = int(np.argmin(np.abs(grid.w + omega_r)))
    i_anti = int(np.argmin(np.abs(grid.w - omega_r)))

    solver = GNLSESolver(
        pulse=wave,
        fiber=_fiber(wave, _silica_like_phonon()),
        betas=np.array([0.0]),
        include_raman=True,
    )
    solver.propagate(num_steps=500)

    w0 = np.abs(grid.fft(field)) ** 2
    w1 = np.abs(grid.fft(solver.evolution[-1].envelope_field)) ** 2
    assert w1[i_stokes] / w0[i_stokes] > 1.0, "Stokes sideband must gain"
    assert w1[i_anti] / w0[i_anti] < 1.0, "anti-Stokes sideband must lose"
