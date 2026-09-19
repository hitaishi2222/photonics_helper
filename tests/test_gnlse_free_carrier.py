"""Task 3.3 — time-resolved TPA / free-carrier model (opt-in).

Covers ``photonics_helper.gnlse.free_carrier_step`` and the
``SplitStepEngine(include_free_carriers=True)`` path against the analytic
references named in the ``harden-gnlse-physics`` change:

1. undepleted TPA attenuation ``I(z) = I₀/(1 + β I₀ z)`` — per-sample exact;
2. carrier-lifetime effect — shorter τ_c ⇒ smaller accumulated ``N(t)`` and
   less free-carrier-absorption loss;
3. time resolution — a strongly time-varying pulse gets a peaked, per-sample
   ``N(t)``, and the model output differs from the legacy spatially-averaged
   TPA path;
4. the legacy ``include_tpa`` path is untouched.

References: Soref & Bennett, IEEE JQE 23, 123 (1987); Cowan, Rieger & Young,
Appl. Phys. Lett. 82, 1745 (2003); Yin & Agrawal, Opt. Lett. 32, 2951 (2007).
"""

import numpy as np

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import (
    FiberProfile,
    SplitStepEngine,
    free_carrier_step,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


def _engine(
    *,
    beta_tpa: float = 0.0,
    sigma_fca: float = 0.0,
    tau_c: float | None = None,
    include_free_carriers: bool = True,
    include_tpa: bool = False,
    peak: float = 1e3,
    nsteps: int = 1000,
    length_m: float = 0.1,
) -> tuple[Wave, SplitStepEngine]:
    grid = TemporalGrid(N=2048, Tmax=Time(5e-12, "s"))
    env = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(peak), fwhm=Time(50e-15, "s"))
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550.0, "nm"))
    fiber = FiberProfile.from_gamma(
        gamma=0.05,
        n2=3e-20,
        omega0=float(wave.central_frequency),
        length=Length(length_m, "m"),
        beta_tpa=beta_tpa,
        sigma_fca=sigma_fca,
        carrier_lifetime=Time(tau_c, "s") if tau_c is not None else None,
    )
    engine = SplitStepEngine(
        pulse=wave,
        fiber=fiber,
        betas=np.array([0.0]),
        include_free_carriers=include_free_carriers,
        include_tpa=include_tpa,
        step_size=Length(length_m / nsteps, "m"),
    )
    return wave, engine


# ── Undepleted TPA attenuation ──────────────────────────────────────────────


def test_undepleted_tpa_analytic_attenuation():
    """Peak intensity follows I(z) = I₀/(1 + β I₀ z) (TPA only, σ = 0)."""
    peak = 1e3
    beta = 1e-3
    L = 0.1
    _, engine = _engine(beta_tpa=beta, peak=peak, length_m=L, nsteps=1000)
    engine.propagate(1000, nsaves=2)
    A = np.asarray(engine.evolution[-1].envelope_field)
    i_out = float(np.max(np.abs(A) ** 2))
    i_analytic = peak / (1.0 + beta * peak * L)
    assert np.isclose(i_out, i_analytic, rtol=1e-6)


# ── Carrier lifetime controls accumulation ─────────────────────────────────


def test_carrier_lifetime_limits_accumulation():
    """Shorter τ_c ⇒ smaller N(t) ⇒ less FCA loss for the same input."""
    outcomes = {}
    for tau_c in (1e-12, 1e-6):
        _, eng = _engine(beta_tpa=1.0, sigma_fca=1e-12, tau_c=tau_c,
                         peak=1e3, length_m=0.05, nsteps=500)
        eng.propagate(500, nsaves=2)
        outcomes[tau_c] = (
            float(np.max(eng._N)),
            float(np.sum(np.abs(eng.evolution[-1].envelope_field) ** 2)),
        )
    n_short, e_short = outcomes[1e-12]
    n_long, e_long = outcomes[1e-6]
    assert n_long > 10 * n_short
    # More accumulated carriers (long lifetime) ⇒ more FCA loss.
    assert e_long < e_short


# ── Time resolution matters ────────────────────────────────────────────────


def test_time_resolved_carrier_profile_is_peaked():
    """A pulse-like generation term leaves a peaked N(t) grid state."""
    _, engine = _engine(beta_tpa=1.0, sigma_fca=0.0, peak=1e3, length_m=0.05,
                        nsteps=500)
    engine.propagate(500, nsaves=2)
    n = engine._N
    assert n is not None
    assert n.max() / max(n.mean(), 1e-300) > 3.0


def test_time_resolved_differs_from_legacy_averaged():
    """Time-resolved output differs from the legacy spatially-averaged model
    on the same TPA+σ configuration (the legacy model applies the *mean*
    intensity-driven attenuation uniformly)."""
    def run(**kw):
        _, engine = _engine(beta_tpa=1.0, sigma_fca=1e-12, tau_c=1e-9,
                            peak=1e3, length_m=0.05, nsteps=500, **kw)
        engine.propagate(500, nsaves=2)
        A = np.asarray(engine.evolution[-1].envelope_field)
        return np.asarray(engine.grid.fft(A))
    sp_time = run(include_free_carriers=True, include_tpa=False)
    sp_avg = run(include_free_carriers=False, include_tpa=True)
    rel = float(np.max(np.abs(sp_time - sp_avg)) / np.max(np.abs(sp_avg)))
    assert rel > 1e-3


# ── Legacy path unchanged ──────────────────────────────────────────────────


def test_legacy_tpa_path_untouched():
    """``include_tpa`` still runs the legacy spatially-averaged model."""
    grid = TemporalGrid(N=512, Tmax=Time(2e-12, "s"))
    env = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(1e3), fwhm=Time(50e-15, "s"))
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550.0, "nm"))
    fiber = FiberProfile.from_gamma(
        gamma=0.05, n2=3e-20, omega0=float(wave.central_frequency),
        length=Length(0.05, "m"), sigma_tpa=1e13,
        carrier_lifetime=Time(1e-9, "s"),
    )
    engine = SplitStepEngine(
        pulse=wave, fiber=fiber, betas=np.array([0.0]),
        include_tpa=True, step_size=Length(1e-4, "m"),
    )
    engine.propagate(500, nsaves=2)
    A = np.asarray(engine.evolution[-1].envelope_field)
    assert float(np.max(np.abs(A) ** 2)) < 1e3 * 0.99  # attenuated
    assert engine._U >= 0.0  # scalar carrier state maintained


def test_free_carrier_step_fca_only_limit():
    """FCA-only: with no TPA source and a prescribed N, loss = exp(−σ N z/2)."""
    grid = TemporalGrid(N=256, Tmax=Time(2e-12, "s"))
    env = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(100.0), fwhm=Time(50e-15, "s"))
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550.0, "nm"))
    fiber = FiberProfile.from_gamma(
        gamma=0.05, n2=3e-20, omega0=float(wave.central_frequency),
        length=Length(0.05, "m"), beta_tpa=0.0, sigma_fca=1e-14,
        carrier_lifetime=Time(1e-9, "s"), group_velocity=None,
    )
    N0 = np.full(grid.N, 1e12)
    dz = 1e-3
    A = np.array(wave.envelope_field, dtype=complex)
    A2, _ = free_carrier_step(A.copy(), fiber, grid, dz, N0, float(wave.central_frequency))
    # Exact closed form: N⁺ = N·e^{−dz/L} (no TPA source), loss exp(−σ·N⁺·dz/2).
    tau_c = 1e-9
    L = tau_c * 2.99792458e8  # group_velocity=None → C_MS
    n_mid = 1e12 * np.exp(-dz / L)
    expected = float(np.exp(-0.5 * 1e-14 * n_mid * dz))
    ratio = float(np.max(np.abs(A2) / np.abs(A)))
    assert np.isclose(ratio, expected, rtol=1e-9)
