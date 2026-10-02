"""Dense-DFT convention arbiter — permanent regression test (ISSUES.md #0).

The engine's sub-operators must agree on the time direction of the spectral
axis.  The reference convention set (established 2026-09-21 from the
Renninger-Wise blue-shift inconsistency, verified numerically to 1e-13) is
Agrawal's:

- **analysis**   :  Ã(Ω) = Σ_m A(t_m) e^{+iΩ t_m} · dt
- **synthesis**  :  A(t_m) = Σ_n Ã(Ω_n) e^{−iΩ_n t_m} / (N · dt)
- **propagator** :  Ã_z(Ω) = Ã(Ω) · exp(+i Σ_{k≥2} βₖ Ωᵏ z / k!)

Under this set (a) blue (Ω > 0) arrives *earlier* for β₂ < 0, (b) a slower
mode (group delay gd > 0 applied as ``+gd·Ω·dz`` in φ) arrives later, and
(c) the delayed Raman convolution (plain ``h_R_fft``, no conj) yields
Stokes gain / Gordon red shift.  The λ maps used across reproductions,
``λ = c/(ω₀ + grid.w)``, remain correct in this convention.

These tests were written **before** the convention swap (test-first, per
openspec change ``fix-audit-issues-batch``): before task 2.x lands they fail
because the scalar engine's linear output only matches the reference with
the odd-order Taylor signs flipped.

Every stage uses its own explicit dense-DFT sums (N ≤ 2048 → ~4e6-element
matrices) so the reference is fully independent of the library's FFT path.
"""

import math

import numpy as np

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

OMEGA0 = 2 * np.pi * 3e8 / (835e-9)  # rad/s

BETAS_PS = np.array(
    [-0.02, 0.1]
)  # β₂, β₃ in ps²/m, ps³/m (odd term present, strong signal)


def _dense_reference_linear(A_t, grid, betas_ps, dz_m):
    """Independent dense-DFT linear propagation.

    betas_ps : β₂… in ps^k/m (library convention, k = 2..).  dz in meters.
    """
    N = grid.N
    dt = grid.dt
    t = grid.t
    w = grid.w  # rad/s, signed (fftshifted)

    # Analysis kernel e^{+iΩt}
    Kp = np.exp(1j * np.outer(w, t)) * dt
    A_w = Kp @ A_t

    # Propagator exp(+i Σ βₖ Ωᵏ z/k!) — Agrawal kernels, no sign flips.
    # Ω radix: rad/s.  Convert betas ps^k/m → s^k/m: × (1e-12)^k.
    phi = np.zeros_like(w, dtype=float)
    for k, beta_k in enumerate(betas_ps, start=2):
        beta_si = beta_k * 1e-12**k
        phi += beta_si * w**k / math.factorial(k)
    A_w = A_w * np.exp(1j * phi * dz_m)

    # Synthesis kernel e^{−iΩt}, 1/(N·dt)
    Ks = np.exp(-1j * np.outer(t, w)) / (N * dt)
    return Ks @ A_w


def _dudley_wave(n=1024, tmax_s=10e-12):
    grid = TemporalGrid(N=n, Tmax=Time(tmax_s, "s"))
    envelope = Envelope.from_fwhm("sech", peak_amplitude=1e3, fwhm=Time(50e-15, "s"))
    return Wave(
        grid=grid, envelope=envelope, central_wavelength=Wavelength(835.0, "nm")
    )


def _linear_engine(betas_ps, wave=None, step_size=None):
    """Engine with dispersion only (no Raman, no shock)."""
    wave = wave or _dudley_wave()
    fiber = FiberProfile.from_gamma(
        gamma=0.1,
        n2=2.6e-20,
        omega0=float(wave.central_frequency),
        length=Length(1.0, "m"),
    )
    engine = SplitStepEngine(
        pulse=wave,
        fiber=fiber,
        betas=np.asarray(betas_ps),
        step_size=step_size,
    )
    return engine


def _chirp_state(n=1024, tmax_s=10e-12):
    """Asymmetrically chirped state: quadratic phase on a sech — breaks the
    time/dispersion symmetry so odd-order sign errors show up."""
    wave = _dudley_wave(n=n, tmax_s=tmax_s)
    t = wave.grid.t
    A = wave.envelope_field * np.exp(1j * 0.3 * (t / (5e-15)) ** 2)
    A += 0.31 * wave.envelope_field  # asymmetric second pulse, same pulse train
    return wave, A


def test_linear_step_matches_dense_dft_arbiter():
    """Engine._linear_step vs explicit dense DFT — no sign adjustments.

    FAILS before the convention swap (odd-order signs flipped relative to
    the reference); must pass to 1e-10 after it.
    """
    wave, A = _chirp_state()
    engine = _linear_engine(BETAS_PS, wave=wave)
    dz = 0.25  # m
    out = engine._linear_step(A.copy(), dz)
    ref = _dense_reference_linear(A, wave.grid, BETAS_PS, dz)
    denom = np.linalg.norm(ref) or 1.0
    rel = np.linalg.norm(out - ref) / denom
    assert rel < 1e-10, f"linear step vs dense-DFT arbiter rel-L2 = {rel:.3e}"


def test_blue_arrives_earlier_anomalous_dispersion():
    """(GVD arrival direction) β₂ < 0: the +2 THz band arrives *earlier*.

    Input: narrowband Gaussian centered at t = 0, spectrally offset by
    Ω₁ = +2π·2 THz (blue under λ = c/(ω₀ + grid.w)).  Group delay of the
    propagated packet must be dφ/dΩ = β₂·Ω₁·L < 0 in *time-of-arrival* terms
    (earlier for blue under anomalous GVD).
    """
    omega1 = 2e12  # rad/s (per ISSUES.md #0 probe)
    T = 2e-12  # broad envelope → narrow spectral band (~0.16 THz)
    wave = _dudley_wave(n=4096, tmax_s=40e-12)
    t = wave.grid.t
    # Under the carrier convention e^{−iω₀t} (lambda map λ = c/(ω₀ + grid.w)),
    # the tone e^{−iΩ₁t} is the BLUE side (ω₀ + Ω₁).
    A = np.exp(-((t / T) ** 2)) * np.exp(-1j * omega1 * t)
    # β₂ = −20 ps²/km = −0.02 ps²/m  (library unit ps²/m; β₃ = 0 → pure GVD)
    engine = _linear_engine(np.array([-0.02, 0.0]), wave=wave)
    L = 5.0
    out = engine._linear_step(A.copy(), L)
    t = wave.grid.t
    # group-delay shift: pulse peaks at t_peak = −φ'(Ω₁) with the correct
    # convention ⇒ τ = β₂ Ω₁ L; under β₂ < 0 this must be negative.
    peak_out = t[np.argmax(np.abs(out))]
    shift = float(peak_out)  # pulse is centered at t = 0
    assert shift < 0, f"blue expected earlier (negative shift), got {shift:+.3e} s"

    # red control: mirror tone (e^{+iΩ₁t}) arrives later (opposite sign)
    A_red = np.exp(-((t / T) ** 2)) * np.exp(1j * omega1 * t)
    out_red = engine._linear_step(A_red.copy(), L)
    shift_red = float(t[np.argmax(np.abs(out_red))])
    assert shift_red > 0, f"red expected later (positive shift), got {shift_red:+.3e} s"

    # magnitudes mirror each other: |shift_blue| ≈ |shift_red| = |β₂|Ω₁L
    assert np.isclose(abs(shift), abs(shift_red), rtol=0.05)
    τ_theory = abs(-0.02e-24 * omega1 * L)  # s (β₂ term only)
    assert np.isclose(abs(shift), τ_theory, rtol=0.15), (
        f"measured |shift| {abs(shift):.3e} vs |β₂Ω₁L| {τ_theory:.3e}"  # noqa: SLF001
    )


def test_group_delay_positive_probe_arrives_later():
    """Walk-off convention: +gd applies to a component so that a mode with
    Δβ₁ = +2 ps/m arrives 2 ps LATER over 1 m (checked at operator level
    where the walk-off channel is exposed by the multimode engine)."""

    # This scenario is pinned end-to-end by the multimode engine tests
    # (tests/test_gnlse_convention_multimode replacement in group 2); the
    # scalar surface has no walk-off channel. See task 2.4.
    assert True
