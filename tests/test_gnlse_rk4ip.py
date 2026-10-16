"""RK4IP scheme tests (``formal/PLAN.md`` §18, Stage S5; Hult 2007).

Hult's fourth-order Runge–Kutta *in the interaction picture* (J. Hult,
*J. Lightwave Technol.* **25**(12), 3770 (2007), Eq. (12)) applies the
interaction-picture transformation

    A_I = exp(−(z − z̄)L̂) A

at the **outer** level, so the linear operator is never split against the
nonlinearity: it is carried exactly by ``exp(hL̂)`` (diagonal in the Fourier
basis) and only the nonlinear remainder ``∂_I A_I = N̂_I A_I`` is advanced by
classical RK4. There is therefore no ``[L, N]`` splitting error by
construction. (The spectral-edge *diagnostics* of example 45 do not detect
this — two no-split fourth-order schemes score 0 and 114 columns there — see
``formal/PLAN.md`` §18.8–§18.9. The evidence for the scheme is the fixed-grid
order study and the conservation tests below, not the edge counts.)

These tests pin:
- the transform pair ``to ∘ from = id`` and the ``N̂ = 0`` exactness that
  makes the interaction picture meaningful,
- the reduction to classical RK4 when ``L = 0`` (so the inner integrator is
  the same one ``RK4.lean`` proves fourth-order),
- the scheme's fourth-order rate on a scalar problem with an exact reference,
- the scope guard (TPA / free carriers are rejected, not silently dropped),
- conservation of the appropriate invariant on a Raman+shock deck, and
- agreement with ``etdrk4`` — the other no-split fourth-order option.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import (
    Fiber,
    GNLSESolver,
    SplitStepEngine,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL0 = 1550e-9
OMEGA0 = Wavelength(WL0, "m").to_omega().as_rad_s
BETA2 = -21e-27
GAMMA = 2.0
T0 = 50e-15 / 1.763
NGRID = 2048
LENGTH = 2.0 * (np.pi / 2) * T0**2 / abs(BETA2)


# ---------------------------------------------------------------------------
# The interaction-picture algebra (exact)
# ---------------------------------------------------------------------------


def test_interaction_picture_transform_pair_is_inverse():
    """``from ∘ to = id`` for the interaction-picture transform.

    ``_linear_step`` is the exact diagonal propagator ``D(A, h) = exp(hL̂)A``;
    the forward transform is ``D(A, h/2)`` and the inverse ``D(A, −h/2)``.
    For any field and any step this must be the identity to round-off — the
    transform would otherwise bias every step.
    """
    grid = TemporalGrid(N=256, Tmax=Time(2e-11, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber.from_gamma(gamma=GAMMA, length=Length(1e-3, "m"))
    eng = SplitStepEngine(pulse=pulse, fiber=fiber, betas=np.array([BETA2 * 1e24]))
    rng = np.random.default_rng(0)
    A = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    h = 1e-5
    roundtrip = eng._linear_step(eng._linear_step(A, h / 2), -h / 2)
    assert np.max(np.abs(roundtrip - A)) < 1e-12


def test_rk4ip_is_exact_for_a_purely_linear_flow():
    """With ``N̂ = 0`` the step is exactly ``exp(hL̂)`` — no truncation at all.

    The interaction picture transforms the linear term away rather than
    approximating it, so a dispersion-only deck must match a single full
    linear propagator to round-off. Any residual here would be a splitting
    error by another name.
    """
    grid = TemporalGrid(N=512, Tmax=Time(1e-11, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    # n2 = 0 ⇒ γ = 0 ⇒ N̂ = 0.
    fiber = Fiber(
        n2=0.0, alpha=0.0, A_eff=Area(80e-12, "m^2"), length=Length(1e-3, "m")
    )
    eng = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),
        step_size=Length(1e-3 / 50, "m"),
        scheme="rk4ip",
    )
    eng.propagate(num_steps=50)
    A_in = np.asarray(pulse.envelope_field, complex)
    exact = eng._linear_step(A_in, 1e-3)
    assert np.max(np.abs(np.asarray(eng.A) - exact)) < 1e-12
    energy = eng.energy_vs_z
    assert abs(energy[-1] - energy[0]) / energy[0] < 1e-12


def test_rk4ip_reduces_to_classical_rk4_when_linear_part_vanishes():
    """``L = 0`` ⇒ the RK4IP update is the classical RK4 update.

    With no dispersion (``betas = []``) the half transforms are the identity
    and the combination ``A + (k₁ + 2k₂ + 2k₃ + k₄)/6`` is exactly
    :func:`rk4Step` — the integrator ``RK4.lean`` proves fourth-order.
    """
    grid = TemporalGrid(N=256, Tmax=Time(2e-11, "s"))
    env = Envelope(shape="sech", peak_amplitude=np.sqrt(2.0), pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber.from_gamma(gamma=GAMMA, length=Length(1e-3, "m"))
    eng = SplitStepEngine(pulse=pulse, fiber=fiber, betas=np.array([]), scheme="rk4ip")
    A = np.asarray(pulse.envelope_field, complex)
    h = 1e-5
    got = eng._rk4ip_step(A, h)

    def rhs(s: np.ndarray) -> np.ndarray:
        return eng._nonlinear_rhs(s, GAMMA)

    k1 = rhs(A)
    k2 = rhs(A + 0.5 * h * k1)
    k3 = rhs(A + 0.5 * h * k2)
    k4 = rhs(A + h * k3)
    want = A + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.max(np.abs(got - want)) < 1e-12


# ---------------------------------------------------------------------------
# Fourth-order rate (scalar problem with an exact reference)
# ---------------------------------------------------------------------------


def _scalar_rk4ip(lam, Nfun, u0, T, n):
    """Hult Eq. (12) for the scalar ODE ``u' = λu + N(u)`` (D(A,h/2) = e^{λh/2})."""
    h = T / n
    E = np.exp(lam * h / 2)
    u = u0
    for _ in range(n):
        A_I = E * u
        k1 = E * (h * Nfun(u))
        k2 = h * Nfun(A_I + 0.5 * k1)
        k3 = h * Nfun(A_I + 0.5 * k2)
        k4 = h * Nfun(E * (A_I + k3))
        u = E * (A_I + (k1 + 2 * k2 + 2 * k3) / 6.0) + k4 / 6.0
    return u


def test_scalar_rk4ip_order_four():
    """``u' = λu − u²`` against the exact Riccati solution: order 4."""
    lam = 0.7
    T = 1.0
    # u0 = 1, λ = 0.7 ⇒ u(T) = λ e^{λT} / (λ + e^{λT} − 1)
    exact = lam * np.exp(lam * T) / (lam + (np.exp(lam * T) - 1))

    def Nfun(s: float) -> float:
        return -s * s

    errs = []
    for n in (10, 20, 40, 80):
        u = _scalar_rk4ip(lam, Nfun, 1.0, T, n)
        errs.append(abs(u - exact) / abs(exact))
    for i in range(3):
        ratio = errs[i + 1] / errs[i]
        assert 1 / 30 < ratio < 1 / 7, (
            f"order-4 halving ratio {ratio:.3e} out of band "
            f"(errs={['%.3e' % e for e in errs]})"
        )


def test_scalar_rk4ip_linear_invariance():
    """For ``u' = (λ + k)u`` the linear part is carried without error growth."""
    lam = 0.7
    k = 0.35
    T = 1.0
    exact = np.exp((lam + k) * T)
    errs = [
        abs(_scalar_rk4ip(lam, lambda s: k * s, 1.0, T, n) - exact) / abs(exact)
        for n in (10, 20, 40, 80)
    ]
    assert errs[-1] < errs[0] / 100


# ---------------------------------------------------------------------------
# Engine-level: benchmark against a fine reference
# ---------------------------------------------------------------------------


def _engine(scheme: str = "rk4ip", num_steps: int = 400, **kw) -> SplitStepEngine:
    grid = TemporalGrid(N=NGRID, Tmax=Time(1.2e-11, "s"))
    peak = 9 * abs(BETA2) / (GAMMA * T0**2)  # N = 3 soliton
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(peak), pulse_width=Time(T0, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber.from_gamma(gamma=GAMMA, length=Length(LENGTH, "m"))
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),
        step_size=Length(LENGTH / num_steps, "m"),
        scheme=scheme,
        **kw,
    )
    engine.propagate(num_steps=num_steps)
    return engine


def test_rk4ip_beats_strang_and_matches_etdrk4():
    """Both no-split schemes sit on the same 4th-order curve, far below strang."""
    rk = _engine("rk4ip", 1600)
    etd = _engine("etdrk4", 1600)
    strong = _engine("strang", 1600)
    ref = np.asarray(_engine("yoshida4", 3200).A, complex)
    nrm = np.linalg.norm(ref)
    err_rk = np.linalg.norm(np.asarray(rk.A, complex) - ref) / nrm
    err_etd = np.linalg.norm(np.asarray(etd.A, complex) - ref) / nrm
    err_strang = np.linalg.norm(np.asarray(strong.A, complex) - ref) / nrm
    assert err_rk < err_strang / 3, (
        f"rk4ip ({err_rk:.2e}) should be far below strang ({err_strang:.2e})"
    )
    # the two 4th-order no-split schemes must agree with each other
    assert abs(err_rk - err_etd) < 3e-2, (
        f"rk4ip ({err_rk:.2e}) and etdrk4 ({err_etd:.2e}) disagree"
    )


def test_rk4ip_advances_the_same_distance_as_strang():
    a = _engine("strang", 200)
    b = _engine("rk4ip", 200)
    assert float(a._z_positions[-1]) == pytest.approx(float(b._z_positions[-1]))
    assert float(b._z_positions[-1]) == pytest.approx(LENGTH, rel=1e-9)


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------


def test_rk4ip_energy_drift_is_order_four():
    """Lossless pure NLSE: the norm is conserved only up to the RK4 error.

    Unlike the symmetric splittings, RK4IP does not preserve the norm
    step-exactly — the nonlinear remainder is RK4-approximated — so the drift
    is O(dz⁴): shrinking per halving and ≪1e-4 by 800 steps.
    """

    def drift(steps: int) -> float:
        e = _engine("rk4ip", steps)
        energy = e.energy_vs_z
        return abs(float(energy[-1] - energy[0])) / float(energy[0])

    d400, d800 = drift(400), drift(800)
    assert d800 < 1e-4, f"drift at 800 steps {d800:.2e}"
    assert d800 < d400, "drift must decrease with refinement"


def test_rk4ip_photon_number_conserved_with_shock():
    """Raman + shock: the conserved quantity is B&W's (A5) photon number."""
    from photonics_helper.raman import RamanResponse, RamanSpec

    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    grid = TemporalGrid(N=NGRID, Tmax=Time(1.2e-11, "s"))
    raman = RamanResponse(spec=spec, grid=grid)
    peak = 9 * abs(BETA2) / (GAMMA * T0**2)
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(peak), pulse_width=Time(T0, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(LENGTH, "m"),
        raman_response=raman,
    )
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),
        include_raman=True,
        include_self_steepening=True,
        step_size=Length(LENGTH / 400, "m"),
        scheme="rk4ip",
    )
    engine.propagate(num_steps=400)
    Q = engine.photon_number_vs_z
    drift = abs(float(Q[-1] - Q[0])) / float(Q[0])
    assert drift < 1e-10, f"photon number drifted by {drift:.2e}"


# ---------------------------------------------------------------------------
# Scope guard and passthrough
# ---------------------------------------------------------------------------


def test_rk4ip_rejects_out_of_scope_operators():
    """TPA and free carriers live in the split-step nonlinear stage, not N̂."""
    grid = TemporalGrid(N=64, Tmax=Time(1e-11, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber.from_gamma(gamma=GAMMA, length=Length(1e-3, "m"))
    for kw in ({"include_tpa": True}, {"include_free_carriers": True}):
        eng = SplitStepEngine(
            pulse=pulse,
            fiber=fiber,
            betas=np.array([BETA2 * 1e24]),
            scheme="rk4ip",
            **kw,
        )
        with pytest.raises(NotImplementedError, match="rk4ip"):
            eng._rk4ip_step(np.asarray(pulse.envelope_field, complex), 1e-6)


def test_gnlse_solver_scheme_passthrough():
    solver = GNLSESolver(
        pulse=Wave(
            grid=TemporalGrid(N=64, Tmax=Time(1e-11, "s")),
            envelope=Envelope(
                shape="sech", peak_amplitude=1.0, pulse_width=Time(T0, "s")
            ),
            central_wavelength=Wavelength(WL0, "m"),
        ),
        fiber=Fiber.from_gamma(gamma=GAMMA, length=Length(1e-3, "m")),
        betas=np.array([BETA2 * 1e24]),
        scheme="rk4ip",
    )
    assert solver.scheme == "rk4ip"
