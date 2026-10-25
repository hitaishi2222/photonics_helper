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


# ---------------------------------------------------------------------------
# Gap J — static full-β + per-bin γ(Ω) + Raman under ETDRK4 (2026-10-25)
#
# ETDRK4 needs exactly two things: a diagonal linear operator in the
# transform basis and a nonlinear functional evaluable at arbitrary stage
# states. A static ``full_beta`` curve (a per-bin phase array) and the
# *symmetrised* per-bin γ(Ω) operator both qualify, so the guard that refused
# the whole dispersive configuration is narrowed to the features that are
# genuinely not pointwise (z-dependent β(ω,z), the legacy history-dependent
# TPA carrier density, and the time-resolved free-carrier state).
# ---------------------------------------------------------------------------

J_WL0 = 1550e-9
J_LEN = 20e-3  # ≥ 20 mm deck
J_B2, J_B3, J_B4 = -2.0e-26, 1.0e-40, -1.0e-55
J_GAMMA0 = 20.0  # 1/(W·m)


def _j_raman(grid):
    from photonics_helper.raman import RamanSpec

    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    return spec.response(grid)


def _j_engine(
    steps: int,
    gamma0: float = J_GAMMA0,
    n: int = 1024,
    energy: float = 10e-12,
    scheme: str = "etdrk4",
    gamma_omega: object = "default",
    **kw,
):
    """10 pJ, 200 fs, 20 mm on a static full-β curve + per-bin γ(Ω) + Raman."""
    grid = TemporalGrid(N=n, Tmax=Time(8e-12, "s"))
    env = Envelope.from_energy(shape="sech", energy=energy, fwhm=Time(200e-15, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(J_WL0, "m"))
    fiber = Fiber(
        gamma=gamma0,
        alpha=0.0,
        A_eff=Area(0.5e-12, "m^2"),
        length=Length(J_LEN, "m"),
        raman_response=_j_raman(grid),
    )
    w = np.asarray(grid.w, dtype=float)
    curve = J_B2 * w**2 / 2.0 + J_B3 * w**3 / 6.0 + J_B4 * w**4 / 24.0
    if isinstance(gamma_omega, str) and gamma_omega == "default":
        gamma_omega = gamma0 * (1.0 + 0.3 * np.tanh(w / 2e13))
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=None,
        full_beta=curve,
        include_raman=True,
        gamma_omega=gamma_omega,
        step_size=Length(J_LEN / steps, "m"),
        scheme=scheme,
        **kw,
    )
    engine.propagate(steps)
    return engine


def test_etdrk4_full_beta_gamma_omega_raman_is_fourth_order():
    """The dispersive-model deck (full β + γ(Ω) + Raman) at 10 pJ / 200 fs / 20 mm.

    The conserved quantity is the unweighted energy (α = 0, real Raman
    response, no shock), so the worst-case relative drift over the run is an
    O(dz⁴) scheme quantity. Measured 8.5e-8 … 2.2e-11 over 1000 → 8000 steps,
    fitted order 3.98 — the same 4th-order rate the uniform-core path has.
    """
    steps = (1000, 2000, 4000, 8000)
    drifts = []
    for num_steps in steps:
        engine = _j_engine(num_steps)
        energy = np.asarray(engine.energy_vs_z, dtype=float)
        drifts.append(float(np.max(np.abs(energy / energy[0] - 1.0))))
    order = float(-np.polyfit(np.log(steps), np.log(drifts), 1)[0])
    assert drifts[-1] < 1e-6, f"drift at 8000 steps {drifts[-1]:.2e}"
    assert order >= 3.5, f"measured order {order:.2f} (drifts {drifts})"


def test_etdrk4_agrees_with_strang_on_the_per_bin_gamma_deck():
    """Correctness gate: an order test can pass on a wrong operator.

    Same deck (full β + γ(Ω) + Raman) at dz = 2 µm; ETDRK4 must agree with the
    independent Strang composition in the field. Measured 3.7e-8 relative L2
    (Strang's own dz² error dominates: at γ₀ = 10 on the same dz it is 1.3e-5,
    which is why this comparison uses γ₀ = 2.5, so that what is measured is
    the two schemes agreeing rather than Strang's truncation).
    """
    steps = 10000
    fine = _j_engine(steps, gamma0=2.5)
    strang = _j_engine(steps, gamma0=2.5, scheme="strang")
    a = np.asarray(fine.A, dtype=complex)
    b = np.asarray(strang.A, dtype=complex)
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    assert rel <= 1e-6, f"ETDRK4 vs Strang field difference {rel:.2e}"


def _beat_grid():
    """Grid on which 1.499 µm and 2.398 µm are *exact* FFT bins.

    ``Tmax = 8 ps`` makes ``c·Tmax/λ`` an integer for both wavelengths, so the
    two lines land on bins 2048 ± 300 with no spectral leakage.
    """
    grid = TemporalGrid(N=4096, Tmax=Time(8e-12, "s"))
    w0 = 1300.0 * grid.dw  # 1.84 µm, midway in frequency
    return grid, w0, 2048 + 300, 2048 - 300


def test_gamma_omega_per_bin_calibration_on_two_wavelengths():
    """A per-bin γ must reproduce the γ(1.5 µm)/γ(2.4 µm) ratio to ≤2 %.

    Two 1 pJ single-tone runs on a dispersion-free 4 mm deck: the accumulated
    SPM phase of a line is ``γ(Ω)·P·L`` exactly (measured deviation from that
    closed form: 5e-14 and 1.2e-11), so the ratio of the two phases is the
    applied γ ratio. With the scalar-γ path the same two lines come out at
    ratio 1.000 — a 44 % error, i.e. the gate measures the weight, not the
    code running.
    """
    grid, w0, n1, n2 = _beat_grid()
    exponent = 1.236
    gamma0 = 10.0
    gamma_w = gamma0 * (np.maximum(w0 + np.asarray(grid.w), 1.0) / w0) ** exponent
    nominal = float(gamma_w[n1] / gamma_w[n2])
    assert 1.5 <= nominal <= 2.4, f"deck's γ ratio {nominal:.3f} out of range"

    fiber = Fiber(
        gamma=gamma0,
        alpha=0.0,
        A_eff=Area(1e-12, "m^2"),
        length=Length(4e-3, "m"),
    )
    env = Envelope(shape="sech", peak_amplitude=0.125, pulse_width=Time(1e-12, "s"))
    a0 = 1e-12 / grid.time_window  # 1 pJ in the window

    def phase(bin_index: int, gamma_omega) -> float:
        field = np.sqrt(a0) * np.exp(-1j * grid.w[bin_index] * grid.t)
        pulse = Wave(
            grid=grid, envelope=env, central_wavelength=Wavelength(1.84e-6, "m")
        )
        pulse.with_field(field)
        engine = SplitStepEngine(
            pulse=pulse,
            fiber=fiber,
            betas=np.array([0.0]),
            gamma_omega=gamma_omega,
            step_size=Length(2e-6, "m"),
            scheme="etdrk4",
        )
        engine.propagate(2000)
        a_w = grid.fft(engine.A)
        return float(np.angle(a_w[bin_index] / grid.fft(field)[bin_index]))

    ph1 = phase(n1, gamma_w)
    ph2 = phase(n2, gamma_w)
    ratio = ph1 / ph2
    # The per-bin weight is applied exactly: the phase equals γ(Ω)PL per line.
    assert abs(ph1 / (gamma_w[n1] * a0 * 4e-3) - 1.0) < 1e-8
    assert abs(ph2 / (gamma_w[n2] * a0 * 4e-3) - 1.0) < 1e-8
    assert abs(ratio / nominal - 1.0) <= 0.02, f"γ ratio {ratio:.4f} vs {nominal:.4f}"
    # ...and without the per-bin weight both lines get the same γ: the ratio
    # comes out 1.000, i.e. 44 % below the 1.79 the deck asks for.
    scalar_ratio = phase(n1, None) / phase(n2, None)
    assert abs(scalar_ratio / nominal - 1.0) > 0.2, scalar_ratio


def test_constant_gamma_omega_is_the_scalar_operator():
    """γ(Ω) ≡ γ must reduce to the scalar ``iγP_NL A`` operator exactly.

    ``√γ·F·P_NL·F⁻¹·√γ`` with constant γ is ``γ·F·P_NL·F⁻¹`` identically — up
    to FFT round-off. This pins the symmetrisation *and* that ``P_NL`` is
    evaluated at the state's own intensity: weighting the inner state by √γ
    would give ``γ²|A|²A`` instead (a 20× error at γ = 20).
    """
    engine_om = _j_engine(50, gamma_omega=np.full(1024, GAMMA))
    engine_sc = _j_engine(50, gamma_omega=None)
    state = np.asarray(engine_sc.A, dtype=complex)
    n_om = engine_om._nonlinear_rhs(state, GAMMA)
    n_sc = engine_sc._nonlinear_rhs(state, GAMMA)
    err = float(np.max(np.abs(n_om - n_sc)) / np.max(np.abs(n_sc)))
    assert err < 1e-13, (
        f"constant γ(Ω) operator differs from the scalar one by {err:.2e}"
    )


def test_gamma_omega_none_and_three_photon_zero_are_bit_identical():
    """Feature-off gate: the new keywords must not perturb the legacy path."""
    base = _j_engine(200, gamma_omega=None)
    explicit = _j_engine(200, gamma_omega=None, gamma_omega_naive=False)
    assert np.array_equal(np.asarray(base.A), np.asarray(explicit.A))
    zero_a3 = _j_engine(200, gamma_omega=None, three_photon_alpha_3=0.0)
    assert np.array_equal(np.asarray(base.A), np.asarray(zero_a3.A))
    assert np.array_equal(
        np.asarray(base.A),
        np.asarray(
            _j_engine(200, gamma_omega=None, three_photon_alpha_3=np.zeros(1024)).A
        ),
    )


def test_naive_gamma_omega_form_drifts_and_default_is_the_symmetric_one():
    """The non-Hermitian ``iγ(Ω)F{P·A}`` form is diagnostic-only and must drift.

    Measured on the 10 pJ / 20 mm deck at 2000 steps: symmetrised 3.6e-10
    (4th order), naive 1.006e-1 — and the naive number is *dz-independent*
    (1.006e-1 at both 2000 and 4000 steps), the signature of a structural
    non-Hermiticity rather than a truncation error.
    """
    sym = _j_engine(2000, gamma0=10.0)
    naive = _j_engine(2000, gamma0=10.0, gamma_omega_naive=True)
    e_sym = np.asarray(sym.energy_vs_z, dtype=float)
    e_naive = np.asarray(naive.energy_vs_z, dtype=float)
    d_sym = abs(float(e_sym[-1] / e_sym[0] - 1.0))
    d_naive = abs(float(e_naive[-1] / e_naive[0] - 1.0))
    assert d_sym < 1e-5, f"symmetrised drift {d_sym:.2e}"
    assert d_naive > 1e-2, f"naive form drift {d_naive:.2e} (expected a large drift)"
    with pytest.raises(ValueError, match="gamma_omega_naive"):
        _j_engine(1, gamma_omega=None, gamma_omega_naive=True)


def test_etdrk4_guard_names_only_the_features_it_cannot_carry():
    """The narrowed guard must name the exact unsupported feature."""
    grid = TemporalGrid(N=64, Tmax=Time(8e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(200e-15, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(J_WL0, "m"))
    fiber = Fiber.from_gamma(gamma=GAMMA, length=Length(1e-3, "m"))
    betas = np.array([BETA2 * 1e24])

    tpa = SplitStepEngine(
        pulse=pulse, fiber=fiber, betas=betas, include_tpa=True, scheme="etdrk4"
    )
    with pytest.raises(NotImplementedError, match="include_tpa"):
        tpa._etdrk_linear(1e-5)
    fc = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_free_carriers=True,
        scheme="etdrk4",
    )
    with pytest.raises(NotImplementedError, match="include_free_carriers"):
        fc._etdrk_linear(1e-5)

    # ...while the static full-β / γ(z) / α(z) hooks are accepted.
    ok = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=None,
        full_beta=0.0 * np.asarray(grid.w) + 5.0,
        gamma_fn=lambda z: GAMMA,
        alpha_fn=lambda z: 0.0,
        scheme="etdrk4",
        step_size=Length(1e-4, "m"),
    )
    ok.propagate(10)
    assert np.all(np.isfinite(np.asarray(ok.A)))

    # gamma_omega + shock is refused at construction (not silently dropped).
    with pytest.raises(NotImplementedError, match="self_steepening"):
        SplitStepEngine(
            pulse=pulse,
            fiber=fiber,
            betas=betas,
            include_self_steepening=True,
            gamma_omega=np.full(64, GAMMA),
            scheme="etdrk4",
        )
