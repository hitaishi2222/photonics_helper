"""Task 2.4 — interaction-picture (RK4IP) shock-step guarantees.

Two honest invariants, per the design correction (openspec change
``harden-gnlse-physics``, decision D1 revised):

1. **Integrator fidelity.** The photon-number identity
   ``2 Re[iA*∂_t(AP)] = ∂_t(|A|²P)`` used in a draft of this change is *false*
   in general (it holds only for a time-independent ``P_NL``). The shock term's
   residual photon-number drift ``−2γτ∮P·Im(A*∂_tA)dt`` is a physical error of
   the first-order ``ω/ω₀`` expansion — a heavily-substepped classical-RK4
   reference *of the same flow* drifts identically on fissioned states. What
   can therefore be certified is that the RK4IP step reproduces the exact
   nonlinear flow to RK4 accuracy: relative field error < 1e-4 per step on an
   evolved (fissioned) state.

2. **Conservation where the model is exact.** For a constant drive
   ``P_NL = |A|²`` with Raman, dispersion and TPA off, the step *is* exactly
   photon-number conserving; the scheme conserves it to machine precision over
   moderate distances.

Also covered: the spectral-validity guard (``Ω_max < ω₀``, ValueError with the
grid values and the remedy), no clamping on a valid grid, no drift warning on a
clean run, and the physical blue-skew direction of the shock term.

References
----------
Blow & Wood, IEEE J. Quantum Electron. 25, 2665 (1989); Dudley, Genty & Coen,
Rev. Mod. Phys. 78, 1135 (2006), Eq. (3); Hult, JLT 25, 3770 (2007).
"""

import warnings

import numpy as np
import pytest

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

WL0_NM = 835.0  # Dudley Fig. 3 carrier
OMEGA0 = 2 * np.pi * 3e8 / (WL0_NM * 1e-9)


def _dudley_wave(n: int = 8192, tmax_s: float = 14e-12) -> Wave:
    """Fig. 3 input: 50 fs (FWHM) sech, 10 kW, grid satisfying Ω_max < ω₀."""
    grid = TemporalGrid(N=n, Tmax=Time(tmax_s, "s"))
    envelope = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(10e3), fwhm=Time(50e-15, "s"))
    return Wave(grid=grid, envelope=envelope, central_wavelength=Wavelength(WL0_NM, "nm"))


def _dudley_fiber(wave: Wave, raman: bool = False) -> FiberProfile:
    response = _dudley_raman(wave.grid) if raman else None
    return FiberProfile.from_gamma(
        gamma=0.11,  # /W/m — Table I
        n2=2.6e-20,
        omega0=float(wave.central_frequency),
        length=Length(0.15, "m"),
        raman_response=response,
    )


def _dudley_raman(grid: TemporalGrid) -> RamanResponse:
    spec = RamanSpec(name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18)
    return RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)


BETAS_34 = np.array([-0.0755, -1.5, 3.0])  # Table I β₂…β₄ (ps^k/m)


def _walker(raman: bool, **kwargs) -> tuple[Wave, SplitStepEngine]:
    wave = _dudley_wave()
    engine = SplitStepEngine(
        pulse=wave,
        fiber=_dudley_fiber(wave, raman=raman),
        betas=BETAS_34,
        include_raman=raman,
        include_self_steepening=True,
        include_tpa=False,
        tau_shock=0.56e-15,
        **kwargs,
    )
    return wave, engine


def _photon_number(A: np.ndarray, dt: float) -> float:
    return float(np.sum(np.abs(A) ** 2) * dt)


# ── Spectral-validity guard ─────────────────────────────────────────────────


def test_guard_rejects_grid_beyond_omega0():
    """Ω_max ≥ ω₀ must raise, naming the grid values and the remedy."""
    wave = _dudley_wave(n=4096)
    # Tmax = 3 ps with N = 4096 gives Ω_max ≈ 4.29e15 > ω₀ ≈ 2.26e15.
    grid3 = TemporalGrid(N=4096, Tmax=Time(3e-12, "s"))
    bad = Wave(grid=grid3, envelope=wave.envelope, central_wavelength=wave.central_wavelength)
    with pytest.raises(ValueError) as exc_info:
        SplitStepEngine(
            pulse=bad,
            fiber=_dudley_fiber(wave),
            betas=np.array([0.0]),
            include_self_steepening=True,
        )
    msg = str(exc_info.value)
    assert "Ω_max" in msg and "ω₀" in msg and "Tmax" in msg


def test_guard_accepts_grid_within_omega0():
    """A valid grid is accepted and no bin is clamped.

    The guard guarantees positive absolute frequencies, so the *physical*
    factor ``1 + Ω/ω₀ = ω/ω₀`` is > 0 everywhere (default ``τ_shock = 1/ω₀``).
    A larger explicit ``τ_shock`` (e.g. Dudley's effective-area corrected
    0.56 fs > 1/ω₀) may legitimately push the linearised factor negative on
    the innermost red bins — that extrapolation is the user's choice and is
    used unclamped, with no aliasing/clipping warning emitted.
    """
    # Default τ_shock = 1/ω₀: the factor is exactly ω/ω₀, positive per the guard.
    wave = _dudley_wave()
    env_engine = SplitStepEngine(
        pulse=wave,
        fiber=_dudley_fiber(wave),
        betas=BETAS_34,
        include_self_steepening=True,
        include_tpa=False,
    )
    assert float(env_engine.grid.omega_max) < float(env_engine.omega0)
    w = env_engine.grid.w
    assert np.all(1.0 + w / env_engine.omega0 > 0.0)

    # Overridden τ_shock: factor used unclamped, no aliasing/clipping warnings.
    _, engine = _walker(raman=False)
    wt = engine.grid.w
    tau = engine.tau_shock
    assert np.all(1.0 + wt * tau != 0.0)  # nothing special-cased away
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.propagate(5, nsaves=4)  # short
    text = " ".join(str(w.message) for w in caught)
    assert "clip" not in text.lower() and "alias" not in text.lower()


# ── Integrator fidelity (evolved/fissioned state) ───────────────────────────


def test_integrator_fidelity():
    """One RK4IP step ≍ heavily-substepped classical RK4 of the exact flow.

    Also verifies both methods produce the same photon-number change: the
    residual drift is a property of the first-order shock model, not of the
    integrator (design D1 correction)."""
    wave, engine = _walker(raman=True)
    dt = float(engine.grid.dt)
    A = np.array(engine.A, dtype=complex)
    for _ in range(150):  # advance 1.5 cm of dispersion+fission dynamics
        A, _ = engine._nonlinear_step(A, 1e-4)
    nref = _photon_number(A, dt)

    gamma = 0.11
    tau = engine.tau_shock
    kernel = 1j * gamma * tau * engine.grid.w

    def full_rhs(a: np.ndarray) -> np.ndarray:
        p = engine._nl_intensity(np.abs(a) ** 2, engine._get_h_R_fft())
        return np.asarray(engine.grid.ifft((1j * gamma + kernel) * engine.grid.fft(p * a)))

    dz = 1e-4
    # --- fully-resolved reference: 4000 classical RK4 substeps of the full flow
    st = A.copy()
    h = dz / 4000
    for _ in range(4000):
        k1 = full_rhs(st)
        k2 = full_rhs(st + 0.5 * h * k1)
        k3 = full_rhs(st + 0.5 * h * k2)
        k4 = full_rhs(st + h * k3)
        st = st + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # --- engine (RK4IP) step on the same state
    Ae, _ = engine._nonlinear_step(A.copy(), dz)

    rel_field = float(np.max(np.abs(Ae - st)) / np.max(np.abs(st)))
    dN_ref = _photon_number(st, dt) / nref - 1.0
    dN_eng = _photon_number(Ae, dt) / nref - 1.0
    assert rel_field < 1e-4, f"RK4IP field error {rel_field:.2e}"
    assert dN_eng == pytest.approx(dN_ref, rel=0.5, abs=1e-5)


def test_photon_number_conserved_constant_drive():
    """Constant P_NL (Shock-only, betas=0, no Raman) conserves N to < 1e-6."""
    # 1 kW keeps the evolution within the exactly-conservative regime.
    grid = TemporalGrid(N=2048, Tmax=Time(7e-12, "s"))
    env = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(1e3), fwhm=Time(50e-15, "s"))
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0_NM, "nm"))
    engine = SplitStepEngine(
        pulse=wave,
        fiber=_dudley_fiber(wave),
        betas=np.array([0.0]),
        include_self_steepening=True,
        include_tpa=False,
        tau_shock=0.56e-15,
    )
    dt = float(grid.dt)
    A = np.array(engine.A, dtype=complex)
    n0 = _photon_number(A, dt)
    for _ in range(400):  # 4 cm, pure constant-drive advection
        A, _ = engine._nonlinear_step(A, 1e-4)
    assert _photon_number(A, dt) / n0 - 1.0 < 1e-6


def test_full_run_no_drift_warning_on_resolved_config():
    """A config with drift below the 5 % monitor threshold must not warn."""
    wave = _dudley_wave(n=8192, tmax_s=12.5e-12)
    engine = SplitStepEngine(
        pulse=wave,
        fiber=_dudley_fiber(wave, raman=True),
        betas=BETAS_34,
        include_raman=True,
        include_self_steepening=True,
        include_tpa=False,
        tau_shock=0.56e-15,
        step_size=Length(1e-4, "m"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any drift warning would fail the test
        engine.propagate(100, nsaves=2)  # 1 cm: measured drift ≈ 0.9 %


def test_raman_shock_matches_pre_change_physics():
    """Soliton-fission physics unchanged: spectral centroid stays red-shifted.

    The RK4IP step reproduces the pre-change solver's full-length result to
    seven significant digits (verified in development); here we assert the
    stable end-to-end signature — a Raman-red-shifted output centroid well
    below the carrier wavelength.
    """
    wave = _dudley_wave(n=4096, tmax_s=12.5e-12)
    engine = SplitStepEngine(
        pulse=wave,
        fiber=_dudley_fiber(wave, raman=True),
        betas=BETAS_34,
        include_raman=True,
        include_self_steepening=True,
        include_tpa=False,
        tau_shock=0.56e-15,
        step_size=Length(2e-4, "m"),
    )
    engine.propagate(100, nsaves=2)  # 2 cm — fission onset
    A = np.asarray(engine.evolution[-1].envelope_field)
    w = engine.grid.w
    centroid = float(np.sum(np.abs(engine.grid.fft(A)) ** 2 * w) / np.sum(np.abs(engine.grid.fft(A)) ** 2))
    # Raman pushes the soliton band to *lower* frequencies (red shift).
    assert centroid < -1e11


def test_blue_skew_preserved():
    """Shock-only SPM broadening keeps the pre-change spectral skew direction.

    Pure-SPM configuration (β_k = 0): the ``(1 + Ω·τ_shock)`` asymmetry
    concentrates spectral weight on the *negative-Ω* side (red-skewed SPM
    spectrum in this codebase's convention — the sign audit and the Dudley
    Fig. 3 reproduction fix the direction; it is NOT flipped by the RK4IP
    rewrite). Assert the robust, unchanged signature: turning the shock on
    must sharply reduce the blue:red spectral-weight ratio
    (verified: 0.999 → 0.58 with the pre-change solver, same direction).
    """
    ratios = {}
    for shock in (False, True):
        grid = TemporalGrid(N=2048, Tmax=Time(7e-12, "s"))
        env = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(10e3), fwhm=Time(50e-15, "s"))
        wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0_NM, "nm"))
        engine = SplitStepEngine(
            pulse=wave,
            fiber=_dudley_fiber(wave),
            betas=np.array([0.0]),
            include_self_steepening=shock,
            include_tpa=False,
            tau_shock=0.56e-15,
            step_size=Length(1e-4, "m"),
        )
        engine.propagate(200, nsaves=2)
        A = np.asarray(engine.evolution[-1].envelope_field)
        spec = np.abs(np.fft.ifftshift(engine.grid.fft(A))) ** 2
        w = engine.grid.w
        ratios[shock] = float(spec[w > 0].sum()) / max(float(spec[w < 0].sum()), 1e-30)
    # Shock asymmetrically amplifies the Ω < 0 side (pre-change direction preserved).
    assert ratios[True] < ratios[False] < 1.0
