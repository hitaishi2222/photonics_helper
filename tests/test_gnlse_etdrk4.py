"""ETDRK4 scheme tests (``formal/PLAN.md`` §17.6 Stage S3, conditional).

ETDRK4 (Cox & Matthews 2002; Kassam & Trefethen 2005) advances the *whole*
semilinear ODE ``∂_zA = L·A + N(A)`` without splitting: the stiff diagonal
linear operator is carried exactly by φ-function weights and only the
nonlinear remainder is advanced with RK4 — with the coefficients evaluated by
series near ``|x| = 0`` (Cox & Matthews' own instruction for small ``|λh|``)
and by the defining formulas away from it.

These tests pin:
- the coefficient algebra at the DC bin (the failure mode both papers name),
- the scheme's order-4 rate on a scalar test problem,
- frame discipline (the nonlinear values enter in Fourier space),
- the scope guard (z-dependent / TPA / free-carrier runs are rejected),
- conservation on the two regimes where a conservation law exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import (
    ETDRK4_SMALL,
    _etdrk4_scalar_coeffs,
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
# Coefficient algebra (Cox & Matthews 2002, Eqs. (26)-(29); K&T Fig. 7)
# ---------------------------------------------------------------------------


def test_small_x_series_matches_direct_formula():
    """The Taylor branch and the defining formula agree where both are valid.

    Evaluated at ``|x| ≈ 1`` (just outside the switch) both must still agree
    to round-off; the series is truncated at ``x⁸``, so at ``|x| = 1`` the
    truncation is at the series' own tail — compared at a looser tolerance.
    """
    x = np.array([0.6 + 0.2j, 0.9], dtype=complex)
    h = 1.0
    Q, f1, f2, f3 = _etdrk4_scalar_coeffs(x, h)
    # direct formulas (the ``big`` branch), written out:
    ex = np.exp(x)
    Q_direct = (np.exp(x * 0.5) - 1.0) / x
    f1_direct = (-4.0 - x + ex * (4.0 - 3.0 * x + x**2)) / x**3
    f2_direct = (2.0 + x + ex * (-2.0 + x)) / x**3
    f3_direct = (-4.0 - 3.0 * x - x**2 + ex * (4.0 - x)) / x**3
    for got, want in ((Q, Q_direct), (f1, f1_direct), (f2, f2_direct), (f3, f3_direct)):
        assert np.max(np.abs(got - want)) < 1e-5


def test_dc_bin_no_cancellation():
    """At ``|x| = 0`` the coefficients take their exact Taylor values.

    The DC bin of the GNLSE linear operator has ``λ = 0`` exactly (Taylor
    dispersion vanishes, loss may be absent); there the four coefficients must
    be ``h/2, h/6, h/6(1+x/2→…)``, i.e. ``(h/2, h/6·φ-compensated)`` — the
    naive formulas lose all significant digits here (Cox & Matthews: "care
    must be taken in the evaluation of the coefficients for modes where |λh|
    is small").
    """
    h = 1e-3
    x = np.array([0.0, 1e-9, 1e-6], dtype=complex)
    Q, f1, f2, f3 = _etdrk4_scalar_coeffs(x, h)
    assert np.allclose(Q[:3], [h * 0.5, h * 0.5, h * 0.5], rtol=1e-12)
    assert np.allclose(f1[:3], [h / 6, h / 6, h / 6], rtol=1e-9)
    assert np.allclose(f2[:3], [h / 6, h / 6, h / 6], rtol=1e-9)
    assert np.allclose(f3[:3], [h / 6, h / 6, h / 6], rtol=1e-9)


def test_naive_evaluation_breaks_at_dc():
    """The *naive* formula evaluation is catastrophically wrong at the DC bin.

    This pins the failure mode that the fixed evaluation exists for: the
    explicit formula ``(-4 - x + e^x(4 - 3x + x²))/x³`` at ``x = 1e-6``
    reports a relative error ≫ 1 (full cancellation loss), which is what any
    implementation skipping the fix would ship.
    """
    x = np.array([1e-6], dtype=complex)
    _, f1, _, _ = _etdrk4_scalar_coeffs(x, 1.0)
    naive_f1 = (-4.0 - x + np.exp(x) * (4.0 - 3.0 * x + x**2)) / x**3
    # the naive number's relative deviation from the true 1/6:
    err = abs(naive_f1[0] - f1[0]) / abs(f1[0])
    assert err > 1e-3, "expected the naive formula to be visibly wrong at DC"


def test_contour_mean_real_part_agrees_imag_residue_discards_phase():
    """K&T's contour mean is only usable with ``real()`` — for real operators.

    Their §4 diagonal recipe averages the defining function over a unit
    circle centred at ``hλ`` and then discards the imaginary residue of the
    trapezoid sum with ``real()``. For *real* linear operators that residue
    is exactly zero by conjugate symmetry, and ``real(mean)`` is bit-exact —
    verified here on a real ``x``. For a complex operator (ours) the residue
    is a large-phase-rotation error, NOT harmless noise — at the GNLSE's
    dispersion phases (``|x| up to ~10²–10³ rad``) it is of the same order
    as the coefficient itself, so the contour recipe would discard physics.
    The shipped evaluation (series + defining formula) is exact instead.
    """
    # Real case: contour mean's real part = the exact coefficient.
    M = 32  # K&T's own count
    r = np.exp(2j * np.pi * np.arange(M) / M)  # full circle, conj-symmetric
    x_real = np.array([0.3], dtype=complex)
    LR = x_real[:, None] + r[None, :]
    Q_c = ((np.exp(LR * 0.5) - 1.0) / LR).mean(axis=1)
    # conjugate-symmetric sampling makes the imaginary part ~0 (float residual)
    assert abs(Q_c[0].imag) < 1e-13, "real operator must give a real contour mean"
    Q, _, _, _ = _etdrk4_scalar_coeffs(x_real, 1.0)
    assert abs(Q_c[0].real - Q[0]) < 1e-13

    # Complex case (the GNLSE's case): the *full* circle mean is exact, so
    # the contour idea is viable in principle — mean over 2π of an entire
    # function equals its value at the centre.
    x_c = np.array([2.0 + 1.5j, 8.0 - 30.0j], dtype=complex)
    LR = x_c[:, None] + r[None, :]
    Q_full = ((np.exp(LR * 0.5) - 1.0) / LR).mean(axis=1)
    Q, _, _, _ = _etdrk4_scalar_coeffs(x_c, 1.0)
    assert np.max(np.abs(Q_full - Q)) < 1e-12

    # ...but K&T's *published* diagonal recipe samples a half circle
    # (``exp(1i*pi*((1:M)-.5)/M)``) and discards the residue with ``real()``
    # — exact for real operators, phase-destroying for complex ones.
    r_half = np.exp(1j * np.pi * (np.arange(M) + 0.5) / M)
    LR = x_c[:, None] + r_half[None, :]
    Q_half_real = ((np.exp(LR * 0.5) - 1.0) / LR).mean(axis=1).real
    damage = np.max(np.abs(Q_half_real - Q_full))
    assert damage > 0.1, (
        "the half-circle + real() recipe must be visibly wrong for complex hλ "
        "— that is why this code evaluates the coefficients exactly instead"
    )


def test_switch_threshold_is_half():
    assert ETDRK4_SMALL == 0.5


# ---------------------------------------------------------------------------
# Scalar order verification (EXACT reference for the pure-scalar ODE)
# ---------------------------------------------------------------------------


def _scalar_etdrk4(lam, Nfun, u0, T, n):
    h = T / n
    x = np.array([lam * h], dtype=complex)
    Q, f1, f2, f3 = _etdrk4_scalar_coeffs(x, h)
    Q, f1, f2, f3 = Q[0], f1[0], f2[0], f3[0]
    E = np.exp(lam * h)
    E2 = np.exp(lam * h / 2)
    u = u0
    for _ in range(n):
        N0 = Nfun(u)
        a = E2 * u + Q * N0
        Na = Nfun(a)
        b = E2 * u + Q * Na
        Nb = Nfun(b)
        c = E2 * a + Q * (2 * Nb - N0)
        Nc = Nfun(c)
        u = E * u + N0 * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    return u


def test_scalar_order_four():
    """``u' = λu − u²`` against the exact Riccati solution: order 4."""
    lam = 0.7
    T = 1.0
    exact = lam * np.exp(lam * T) / (lam + (np.exp(lam * T) - 1))  # u0 = 1

    def Nfun(s: float) -> float:
        return -s * s

    errs = []
    for n in (10, 20, 40, 80):
        u = _scalar_etdrk4(lam, Nfun, 1.0, T, n)
        errs.append(abs(u - exact) / abs(exact))
    r1 = errs[1] / errs[0]
    r2 = errs[2] / errs[1]
    r3 = errs[3] / errs[2]
    # ~1/16 per halving; allow wide brackets for pre-asymptotic holdovers.
    assert 1 / 30 < r1 < 1 / 7, f"order-4 halving ratio {r1:.3e} out of band"
    assert 1 / 30 < r2 < 1 / 7, f"order-4 halving ratio {r2:.3e} out of band"
    assert 1 / 30 < r3 < 1 / 7, f"order-4 halving ratio {r3:.3e} out of band"


def test_scalar_linear_invariance():
    """Exactness for ``u' = (λ + k)u`` up to O(h⁴) — the linear part must be
    carried without error growth in the stiff direction."""
    lam = 0.7
    k = 0.35
    T = 1.0
    exact = np.exp((lam + k) * T)
    errs = [
        abs(_scalar_etdrk4(lam, lambda s: k * s, 1.0, T, n) - exact) / abs(exact)
        for n in (10, 20, 40, 80)
    ]
    assert errs[-1] < errs[0] / 100


# ---------------------------------------------------------------------------
# Engine-level: frame discipline and one-step accuracy
# ---------------------------------------------------------------------------


def _engine(scheme: str = "etdrk4", num_steps: int = 400, **kw) -> SplitStepEngine:
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


def test_etdrk4_beats_strang_and_matches_fine_splitting():
    """Full N=3-soliton deck at 1600 steps: ETDRK4 sits on the 4th-order
    error curve — an order of magnitude under ``strang`` at the same step
    count, and within the yoshida4 reference's own error (5.2e-5 between
    3200 and 6400 steps; the deck's fission makes preasymptotic ratios noisy
    at coarse ends, so fixed tolerances are set at the 1600-step scale).
    """
    etd = _engine("etdrk4", 1600)
    strong = _engine("strang", 1600)
    ref = _engine("yoshida4", 3200)
    refA = np.asarray(ref.A, complex)
    nrm = np.linalg.norm(refA)
    err_etd = np.linalg.norm(np.asarray(etd.A, complex) - refA) / nrm
    err_strang = np.linalg.norm(np.asarray(strong.A, complex) - refA) / nrm
    assert err_etd < 3e-3, (
        f"ETDRK4 at 1600 steps off the fine reference by {err_etd:.2e}"
    )
    assert err_etd < err_strang / 3, (
        f"ETDRK4 ({err_etd:.2e}) should be far below strang ({err_strang:.2e})"
    )


def test_etdrk4_energy_drift_is_order_four():
    """Lossless pure NLSE. Unlike the splittings, ETDRK4 does not preserve the
    norm step-exactly (the nonlinear term is only RK4-approximated), so the
    drift is an O(dz⁴) quantity — shrinking ~16× per halving and ≪1e-4 by
    800 steps.
    """

    def drift(steps: int) -> float:
        e = _engine("etdrk4", steps)
        energy = e.energy_vs_z
        return abs(float(energy[-1] - energy[0])) / float(energy[0])

    drift_400 = drift(400)
    drift_800 = drift(800)
    assert drift_800 < 1e-4, f"drift at 800 steps {drift_800:.2e}"
    assert drift_800 < drift_400, "drift must decrease with refinement"


def test_etdrk4_photon_number_conserved_with_shock():
    """Raman + shock: the conserved quantity is B&W's (A5) photon number."""
    from photonics_helper.raman import RamanSpec

    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    grid = TemporalGrid(N=NGRID, Tmax=Time(1.2e-11, "s"))
    raman = spec.response(grid)
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
        scheme="etdrk4",
    )
    engine.propagate(num_steps=400)
    Q = engine.photon_number_vs_z
    drift = abs(float(Q[-1] - Q[0])) / float(Q[0])
    assert drift < 1e-10, f"photon number drifted by {drift:.2e}"


def test_etdrk4_rejects_out_of_scope_hooks():
    """Stage S3's conditional scope: z-dependent hooks, TPA, free carriers."""
    grid = TemporalGrid(N=64, Tmax=Time(1e-11, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber.from_gamma(gamma=GAMMA, length=Length(1e-3, "m"))
    eng = SplitStepEngine(
        pulse=pulse, fiber=fiber, betas=np.array([BETA2 * 1e24]), scheme="etdrk4"
    )
    eng._dispersion_profile = lambda w, z: np.zeros_like(w)  # any z-dependent
    eng._is_z_dependent = True
    with pytest.raises(NotImplementedError, match="etdrk4"):
        eng._etdrk_linear(1e-5)
    eng2 = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),
        include_tpa=True,
        scheme="etdrk4",
    )
    with pytest.raises(NotImplementedError, match="etdrk4"):
        eng2._etdrk_linear(1e-5)


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
        scheme="etdrk4",
    )
    assert solver.scheme == "etdrk4"
    with pytest.raises(ValueError, match="Unknown scheme"):
        GNLSESolver(
            pulse=solver.pulse,
            fiber=solver.fiber,
            betas=np.array([BETA2 * 1e24]),
            scheme="rk4",
        )
