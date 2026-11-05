"""γ(Ω) composed with self-steepening (``include_self_steepening``).

Ground truth: the shock factor belongs *inside* the γ-weighted operator,

    N(A) = i·√γ(Ω)(1 + τ·Ω)·F{ P_NL(|A|²)·F⁻¹[√γ(Ω)·A] },

because the frequency dependence of the nonlinearity weights the shock source
too. Weighting a scalar-γ source afterwards would double-count it. With
γ(Ω) = const this is the scalar-γ remainder
``iγ·F{(1 + τΩ)·F{P_NL·A}}`` to round-off (the composed form scales once inside
the outer inverse transform, so the test is ``1e-12`` relative, not bitwise).
"""

import warnings

import numpy as np
import pytest

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL0 = Wavelength(835, "nm")
GAMMA0 = 0.11


def _fiber(length_m=2e-3):
    return Fiber.from_gamma(gamma=GAMMA0, alpha=0.0, length=Length(length_m, "m"))


def _wave(grid, field=None):
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(10e3), pulse_width=Time(28.4e-15, "s")
    )
    wave = Wave(grid=grid, envelope=env, central_wavelength=WL0)
    if field is not None:
        wave.with_field(field)
    return wave


def _engine_raw(
    wave, *, gamma_omega=None, naive=False, shock=True, scheme="strang", **kwargs
):
    params = dict(
        pulse=wave,
        fiber=_fiber(),
        betas=np.array([-0.02]),
        include_self_steepening=shock,
        scheme=scheme,
        **kwargs,
    )
    if gamma_omega is not None:
        params["gamma_omega"] = gamma_omega
        params["gamma_omega_naive"] = naive
    return SplitStepEngine(**params)


def _engine(wave, **kwargs):
    """As :func:`_engine_raw`, with the routine UserWarnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _engine_raw(wave, **kwargs)


# ─── 2.2 constant-γ reduction and energy bookkeeping ────────────────────────


def test_constant_gamma_omega_plus_shock_reduces_to_scalar():
    """γ(Ω) ≡ γ with the shock on must equal the scalar-γ remainder to 1e-12."""
    grid = TemporalGrid(N=1024, Tmax=Time(8e-12, "s"))
    wave = _wave(grid)
    state = np.asarray(wave.envelope_field, dtype=complex)
    engine_om = _engine(wave, gamma_omega=np.full(grid.N, GAMMA0))
    engine_sc = _engine(wave)
    n_om = engine_om._nonlinear_rhs(state, GAMMA0)
    n_sc = engine_sc._nonlinear_rhs(state, GAMMA0)
    rel = float(np.max(np.abs(n_om - n_sc)) / np.max(np.abs(n_sc)))
    # Not bit-identical: the composed form scales once inside the outer ifft.
    assert rel <= 1e-12, f"constant-γ composed RHS differs by {rel:.2e}"
    # The shock term is actually present: dropping it changes the answer.
    engine_no_shock = _engine(wave, gamma_omega=np.full(grid.N, GAMMA0), shock=False)
    n_no_shock = engine_no_shock._nonlinear_rhs(state, GAMMA0)
    assert np.max(np.abs(n_om - n_no_shock)) > 1e-3 * np.max(np.abs(n_om))


def test_both_forms_accept_the_shock_flag():
    """The naive diagnostic form composes with the shock too (no refusal)."""
    grid = TemporalGrid(N=256, Tmax=Time(8e-12, "s"))
    wave = _wave(grid)
    g_w = np.full(grid.N, GAMMA0)
    for naive in (False, True):
        engine = _engine(wave, gamma_omega=g_w, naive=naive)
        engine.propagate(num_steps=2, nsaves=None)
        assert np.all(np.isfinite(np.asarray(engine.A)))


def test_shock_grid_guard_applies_to_both_forms():
    """``_validate_shock_grid`` gates the composed and naive paths as it does the
    scalar one: the hard ``Ω_max < ω₀`` refusal and the soft ω₀/3 band warning."""
    hard = TemporalGrid(N=256, Tmax=Time(0.2e-12, "s"))  # Ω_max ≈ 4e15 > ω₀
    for naive in (False, True):
        with pytest.raises(ValueError, match="positive absolute frequencies"):
            _engine(
                _wave(hard),
                gamma_omega=np.full(hard.N, GAMMA0),
                naive=naive,
            )
    # τ·Ω_max ≈ 0.24 sits in the warned band (τ = 1/ω₀, ω₀ = 2.26e15 rad/s).
    soft = TemporalGrid(N=256, Tmax=Time(1.5e-12, "s"))
    for naive in (False, True):
        with pytest.warns(UserWarning, match="ω₀/3"):
            _engine_raw(
                _wave(soft),
                gamma_omega=np.full(soft.N, GAMMA0),
                naive=naive,
            )


def test_photon_number_drift_bounded_against_scalar_path():
    """On a **varying** γ(Ω) deck the composed run's (A5) drift stays comparable.

    The deck must vary γ(Ω), or the two operators are literally the same one
    (constant γ) and the comparison proves nothing. On the ledger-grade scheme
    ETDRK4 and the same 40 steps, measured drifts are 3.7e-8 (composed) and
    4.3e-8 (scalar γ), a ratio of 0.87; the bound below is a factor 2, not a
    tautology. Without the shock both paths drift by 3e-5, i.e. the shock is
    what makes the (A5) bookkeeping tight, so the bound also pins that the
    composed operator carries the shock at all.

    On Strang the comparison is not meaningful: the composed path is an explicit
    midpoint (RK2) advance while the scalar path has a dedicated unitary
    ``exp(iγτΩΔz)`` shock stage, so their drifts differ by the integrator
    (2.2e-3 vs 8e-8 at 50 steps) and converge as the same O(dz²).
    """
    grid = TemporalGrid(N=512, Tmax=Time(8e-12, "s"))
    wave = _wave(grid)
    omega0 = float(WL0.to_omega().as_rad_s)
    # γ(Ω) varying by ±(Ω/ω₀), so the weight is real on this grid.
    g_w = GAMMA0 * (np.maximum(omega0 + np.asarray(grid.w), 0.5 * omega0) / omega0)

    def drift(**kwargs):
        engine = _engine(wave, scheme="etdrk4", **kwargs)
        q_in = engine.photon_number()
        engine.propagate(num_steps=40, nsaves=None)
        return engine.photon_number() / q_in - 1.0

    d_om = drift(gamma_omega=g_w)
    d_sc = drift()
    assert abs(d_om) < 1e-6 and abs(d_sc) < 1e-6, (d_om, d_sc)
    assert abs(d_om) <= 2.0 * abs(d_sc), (
        f"composed drift {d_om:.3e} vs scalar-γ {d_sc:.3e}"
    )


# ─── 2.3 the γ-weighting discriminates (two-tone deck) ─────────────────────


def _two_tone(grid, k1, k2, amplitude=1.0):
    """Two exact-bin tones: ``a·(e^{-iΩ₁t} + e^{-iΩ₂t})``."""
    w = np.asarray(grid.w)
    n = grid.N
    field = amplitude * (
        np.exp(-1j * w[n // 2 + k1] * grid.t) + np.exp(-1j * w[n // 2 + k2] * grid.t)
    )
    return _wave(grid, field), field, n // 2 + k1, n // 2 + k2


def test_weighting_two_tone_gamma_omega_sideband_ratio():
    """Two tones, γ 2× higher at the upper tone: the RHS ratio is predicted exactly.

    With equal amplitudes ``a`` the field's own intensity has a DC part ``2a²``
    and a beat at ``Ω₂−Ω₁``, so at tone ``i`` the composed operator gives
    ``N(Ω_i) = i√γ_i(1+τΩ_i)·a³(2√γ_i + √γ_j)``. The ratio of the two line
    components is therefore a closed form, and the scalar-γ and naive forms have
    their own predictions. All three differ: that is the gate.
    """
    grid = TemporalGrid(N=1024, Tmax=Time(20e-12, "s"))
    wave, field, i1, i2 = _two_tone(grid, 10, 60)
    g_w = np.full(grid.N, GAMMA0)
    g_w[i2] = 2.0 * GAMMA0
    g1, g2 = float(g_w[i1]), float(g_w[i2])

    engine_om = _engine(wave, gamma_omega=g_w)
    tau = engine_om.tau_shock
    w1, w2 = float(grid.w[i1]), float(grid.w[i2])
    shock1, shock2 = 1.0 + tau * w1, 1.0 + tau * w2

    def ratio(engine):
        rhs_w = np.asarray(grid.fft(engine._nonlinear_rhs(field, GAMMA0)))
        return float(np.abs(rhs_w[i2]) / np.abs(rhs_w[i1]))

    expected_sym = (np.sqrt(g2) * (2 * np.sqrt(g2) + np.sqrt(g1)) * shock2) / (
        np.sqrt(g1) * (2 * np.sqrt(g1) + np.sqrt(g2)) * shock1
    )
    expected_scalar = shock2 / shock1
    expected_naive = (g2 / g1) * shock2 / shock1

    r_sym = ratio(engine_om)
    r_scalar = ratio(_engine(wave))
    r_naive = ratio(_engine(wave, gamma_omega=g_w, naive=True))

    assert r_sym == pytest.approx(float(expected_sym), rel=1e-12)
    assert r_scalar == pytest.approx(float(expected_scalar), rel=1e-12)
    assert r_naive == pytest.approx(float(expected_naive), rel=1e-12)
    # The γ weighting discriminates: the composed line ratio is far from both.
    assert abs(r_sym - r_scalar) > 0.5 * r_scalar
    assert abs(r_sym - r_naive) > 0.15 * r_naive


# ─── 2.4 conserving_shock + gamma_omega is refused, not dropped ─────────────


def test_conserving_shock_with_gamma_omega_is_refused():
    grid = TemporalGrid(N=256, Tmax=Time(8e-12, "s"))
    wave = _wave(grid)
    with pytest.raises(ValueError, match="conserving_shock"):
        SplitStepEngine(
            pulse=wave,
            fiber=_fiber(),
            betas=np.array([-0.02]),
            gamma_omega=np.full(grid.N, GAMMA0),
            conserving_shock=True,
        )
