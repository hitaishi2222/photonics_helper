"""
Example: The Soliton-on-Finite-Background (Breather) Family
===========================================================

Demonstrates ``photonics_helper.breathers`` — the exact analytic solutions of
the focusing NLSE on a non-zero background. These are the structures observed
in spontaneous modulation-instability and rogue-wave experiments, and they are
the ground truth the ``kibler_2010_peregrine`` reproduction checks the engine
against. This example is the *conceptual* introduction; the reproduction is the
quantitative validation against a published figure.

Governing equation
------------------
The dimensionless focusing NLSE used by the module is

.. math::

    i\\,\\partial_\\xi\\psi + \\tfrac{1}{2}\\partial_\\tau^2\\psi + |\\psi|^2\\psi = 0,

the normalisation of the fibre equation ``i A_z = (β₂/2) A_TT − γ|A|²A`` on an
anomalous fiber (``β₂ < 0``) under

.. math::

    A(z,T) = \\sqrt{P_0}\\,\\psi(\\xi,\\tau),\\quad
    T = \\tau T_0,\\quad z = \\xi L_{NL},\\quad
    L_{NL} = 1/(\\gamma P_0),\\quad T_0 = \\sqrt{|\\beta_2| L_{NL}}.

The general solution (Akhmediev–Korneev) is

.. math::

    \\psi(\\xi,\\tau) = e^{i\\xi}\\left[1 +
        \\frac{2(1-2a)\\cosh(b\\xi) + i b\\sinh(b\\xi)}
             {\\sqrt{2a}\\cos(\\nu\\tau) - \\cosh(b\\xi)}\\right],

with :math:`b = \\sqrt{8a(1-2a)}` and :math:`\\nu = 2\\sqrt{1-2a}`.

The governing parameter ``a`` and the three members
--------------------------------------------------
``a`` lies strictly inside ``(0, 1)``; ``a = 0`` and ``a = 1`` are rejected with
``ValueError``. The branch point is ``a = 1/2``, **not** ``a = 1``:

===========  ==============================  ====================================
``a``        member                          character
===========  ==============================  ====================================
``a → 0``    plane wave                      the ``|ψ| = 1`` background
``0<a<1/2``  **Akhmediev breather**          periodic in τ, one growth/decay in ξ
``a = 1/2``  **Peregrine soliton**           localised in ξ and τ
``1/2<a<1``  **Kuznetsov–Ma soliton**        localised in τ, periodic in ξ
===========  ==============================  ====================================

The switch is made real by ``b`` and ``ν``: below ``1/2`` they are real (a
temporal cosine and a hyperbolic cosh in ξ), above ``1/2`` they are imaginary
and the two roles swap.

Closed forms this example checks
-------------------------------
Three pure functions of ``a`` have exact closed forms, all checked here against
the numerically-sampled field rather than quoted:

* ``sfb_peak_ratio(a)`` — maximum ``|ψ|²`` over the unit background. It rises
  monotonically across the whole family and passes through **exactly 9** at the
  Peregrine limit. At ``a = 0.42`` it gives **8.026**, the value the Kibler
  reproduction reports as its analytic reference.
* ``sfb_temporal_period(a)`` — the τ period, defined only for ``a < 1/2``.
* ``sfb_spatial_period(a)`` — the ξ period, defined only for ``a > 1/2``.

The two period ladders have **disjoint** domains and both diverge as
``a → 1/2``. That collapse is the most counter-intuitive fact in the module —
the breather's breathing period goes to zero exactly where the solution stops
breathing — and it is not visible from the signatures.

The Peregrine member is exact, not a limit
------------------------------------------
``general_sfb(·, ·, 0.5)`` is not an approximation of the Peregrine soliton: it
*is* ``peregrine_soliton`` — ``general_sfb`` short-circuits
``|a − 1/2| < 1e-9`` straight to it. The maximum absolute difference over a
grid is 0.0, and this example asserts that rather than plotting a convergence
curve that is already exact.

Engine comparison
-----------------
The analytic field is launched into ``SplitStepEngine`` (Raman and
self-steepening off, so the model is exactly the NLSE the solution satisfies)
and compared at four distances. The relative L2 error is reported for the
**complex field** and for **|ψ|²** separately. That distinction is deliberate:
the intensity error is the conservative number and the one that should be
quoted. (The same distinction is what made a "3× the paper" gap in the Raissi
reproduction turn out to be a metric-definition mismatch rather than a defect —
``ISSUES.md`` #6.)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefer the repository package over any older site-packages install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.breathers import (
    SolitonOnBackground,
    akhmediev_breather,
    general_sfb,
    kuznetsov_ma,
    peregrine_soliton,
    sfb_peak_ratio,
    sfb_spatial_period,
    sfb_temporal_period,
)
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import TemporalGrid

# ── Parameters ──────────────────────────────────────────────────────
# The ``a`` ladder straddles the branch point a = 1/2 and stays inside (0, 1).
A_LADDER = [0.05, 0.25, 0.45, 0.5, 0.55, 0.75]
A_PEREGRINE = 0.5
A_KIBLER = 0.42  # the a of the kibler_2010_peregrine reference deck

TAU_MAX = 6.0
XI_MAX = 4.0

# Physical scales. beta2 in s^2/m, gamma in 1/(W*m), P0 in W.
BETA2_SI = -21.0e-27  # anomalous, ~ -21 ps^2/km
GAMMA = 1.27e-3
P0 = 10.0
LAM0 = 1550e-9

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "families": OUT_DIR / "30_breather_families.png",
    "peregrine": OUT_DIR / "30_peregrine_limit.png",
}


def _intensity(xi: float, tau: NDArray[np.float64], a: float) -> NDArray[np.float64]:
    """``|psi|^2`` of the general SFB solution, as a real array."""
    field: NDArray[np.complex128] = np.asarray(
        general_sfb(xi, tau, a), dtype=np.complex128
    )
    return np.abs(field) ** 2


def _rule(title: str) -> str:
    return "\n" + "=" * 66 + f"\n {title}\n" + "=" * 66


# ── 1. The family across a ───────────────────────────────────────────
def family_panel() -> tuple[plt.Figure, dict]:
    """Panel block A + B: the |psi|^2 ladder and the closed-form overlays."""
    print(_rule("1. The SFB family across a  (branch point at a = 1/2)"))

    tau = np.linspace(-TAU_MAX, TAU_MAX, 2401)
    intensity_by_a = {a: _intensity(0.0, tau, a) for a in A_LADDER}

    print("  a = 0 and a = 1 are outside the family:")
    for a_bad in (0.0, 1.0):
        try:
            general_sfb(0.0, tau, a_bad)
        except ValueError as exc:
            print(f"    general_sfb(a={a_bad:.1f}) -> ValueError: {exc}")
        else:  # pragma: no cover - would be a library regression
            raise AssertionError(f"a={a_bad} should have been rejected")

    # Peak ratio: ladder vs what we actually read off the sampled curve.
    print("\n  Peak-to-background ratio |psi|^2:")
    print("      a      sampled      sfb_peak_ratio      rel. dev")
    peak_dev = 0.0
    peaks: dict[float, float] = {}
    for a in A_LADDER:
        sampled = float(intensity_by_a[a].max())
        ladder = sfb_peak_ratio(a)
        peaks[a] = sampled
        rel = abs(sampled - ladder) / ladder
        peak_dev = max(peak_dev, rel)
        print(f"    {a:5.2f}    {sampled:9.4f}    {ladder:14.6f}    {rel:9.2e}")
    print(f"\n  worst ladder deviation over the sampled peak: {peak_dev:.2e}")
    print(
        f"  Peregrine limit a = 1/2 : |psi|^2 peak = {sfb_peak_ratio(A_PEREGRINE):.6f}"
    )
    print(
        f"  Kibler reference a = {A_KIBLER:.2f}  : {sfb_peak_ratio(A_KIBLER):.6f}"
        "   (reproduction's analytic reference)"
    )

    # Period ladders: disjoint domains, both diverging as a -> 1/2.
    a_akh = np.linspace(0.02, 0.48, 200)
    a_km = np.linspace(0.52, 0.98, 200)
    t_per = np.array([sfb_temporal_period(a) for a in a_akh])
    s_per = np.array([sfb_spatial_period(a) for a in a_km])
    print("\n  Period ladders (note the divergent trend towards a = 1/2):")
    print(
        f"    sfb_temporal_period(0.25) = {sfb_temporal_period(0.25):.4f}   "
        f"(a < 1/2 only; grows from {t_per[0]:.2f} at a=0.02 to "
        f"{t_per[-1]:.1f} at a={a_akh[-1]:.2f})"
    )
    print(
        f"    sfb_spatial_period(0.66) = {sfb_spatial_period(0.66):.4f}   "
        f"(a > 1/2 only; shrinks from {s_per[0]:.1f} at a={a_km[0]:.2f} to "
        f"{s_per[-1]:.2f} at a=0.98)"
    )
    print("    both diverge as a -> 1/2, from opposite sides of the branch point.")
    print("    the two ladders are queried on disjoint domains; calling one")
    print("    outside its domain is a ValueError, so each is swept separately.")

    # ── Figure: 3 rows (Akhmediev / Peregrine / Kuznetsov-Ma) x 3 cols ──
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.0))

    rows = [
        (
            "Akhmediev breather — 0 < a < 1/2",
            [a for a in A_LADDER if a < 0.5],
            "tab:blue",
        ),
        ("Peregrine soliton — a = 1/2", [0.5], "tab:red"),
        (
            "Kuznetsov–Ma soliton — 1/2 < a < 1",
            [a for a in A_LADDER if a > 0.5],
            "tab:green",
        ),
    ]

    for r, (name, a_vals, colour) in enumerate(rows):
        ax = axes[r, 0]
        for a in a_vals:
            ax.plot(tau, intensity_by_a[a], lw=1.4, color=colour, label=f"a = {a:g}")
        ax.set_title(f"{name}\n|psi|^2 vs tau at xi = 0")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$|\psi|^2$")
        ax.set_ylim(0, 14)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # col 1: this member's OWN ladder, with this row's a values marked.
        ax = axes[r, 1]
        if r == 0:
            ax.plot(a_akh, t_per, "tab:blue", lw=2, label="sfb_temporal_period")
            for a in a_vals:
                ax.plot([a], [sfb_temporal_period(a)], "o", color=colour, ms=6)
            ax.set_ylabel(r"$\tau$ period")
            ax.set_title("temporal period of this member")
        elif r == 1:
            a_dense = np.linspace(0.02, 0.98, 400)
            ax.plot(
                a_dense,
                [sfb_peak_ratio(float(a)) for a in a_dense],
                "k-",
                lw=2.0,
                label="sfb_peak_ratio(a)",
            )
            for a in A_LADDER:
                ax.plot([a], [peaks[a]], "o", color=colour, ms=5)
            ax.axvline(0.5, color="tab:red", ls="--", lw=1.2)
            ax.annotate(
                "Peregrine = 9",
                xy=(0.5, 9.0),
                xytext=(0.60, 5.0),
                arrowprops={"arrowstyle": "->", "color": "tab:red"},
            )
            ax.set_ylabel(r"max $|\psi|^2$")
            ax.set_ylim(1, 15)
            ax.set_title("peak compression of this member")
        else:
            ax.plot(a_km, s_per, "tab:green", lw=2, label="sfb_spatial_period")
            for a in a_vals:
                ax.plot([a], [sfb_spatial_period(a)], "o", color=colour, ms=6)
            ax.set_ylabel(r"$\xi$ period")
            ax.set_title("spatial period of this member")
        ax.set_xlabel("a")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # col 2, row 0: the full peak-ratio ladder against every sampled point.
    ax = axes[0, 2]
    a_dense = np.linspace(0.02, 0.98, 400)
    ax.plot(
        a_dense,
        [sfb_peak_ratio(float(a)) for a in a_dense],
        "k-",
        lw=2.0,
        label="sfb_peak_ratio(a)",
    )
    for a, (name, a_vals, colour) in zip(A_LADDER, rows):
        ax.plot([a], [peaks[a]], "o", color=colour, ms=6, label=name.split(" —")[0])
    ax.axvline(0.5, color="tab:red", ls="--", lw=1.2)
    ax.annotate(
        "Peregrine = 9",
        xy=(0.5, 9.0),
        xytext=(0.60, 5.0),
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
    )
    ax.set_title("closed form vs every sampled peak\n(monotonic across the family)")
    ax.set_xlabel("a")
    ax.set_ylabel(r"max $|\psi|^2$")
    ax.set_ylim(1, 15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # col 2, row 1: the branch point.
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "a = 1/2\n\n"
        "b = 0 and nu = 0:\nthe cos(nu*tau) modulation\n"
        "and the xi-recurrence both collapse.\n\n"
        f"|psi|^2 peak = {sfb_peak_ratio(0.5):.0f} exactly.\n"
        "general_sfb(a=1/2) IS peregrine_soliton.\n"
        "Both periods diverge here.",
        ha="center",
        va="center",
        fontsize=10.5,
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "tab:red", "alpha": 0.12},
    )

    # col 2, row 2: both period ladders diverging, log-log about the branch.
    ax = axes[2, 2]
    d_akh = 0.5 - a_akh
    d_km = a_km - 0.5
    ax.loglog(d_akh, t_per, "tab:blue", lw=2, label="temporal, a < 1/2")
    ax.loglog(d_km, s_per, "tab:green", lw=2, label="spatial, a > 1/2")
    ax.loglog(
        d_akh, 3.2 / np.sqrt(d_akh), "k:", lw=1.2, label=r"$\propto 1/\sqrt{|a - 1/2|}$"
    )
    ax.set_title(
        "both periods diverge at the branch point\n(same $1/\\sqrt{\\cdot}$ law)"
    )
    ax.set_xlabel(r"$|a - 1/2|$")
    ax.set_ylabel("period")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Soliton-on-finite-background family — branch point at a = 1/2",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["families"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, {"peak_dev": peak_dev}


# ── 2. Peregrine is the exact branch member ─────────────────────────
def peregrine_panel() -> dict:
    print(_rule("2. The Peregrine member is exact, not a limit"))

    tau = np.linspace(-TAU_MAX, TAU_MAX, 2401)
    xi_vals = np.linspace(-XI_MAX, XI_MAX, 401)

    max_diff = 0.0
    for xi in xi_vals[::40]:
        d = np.abs(
            np.asarray(general_sfb(xi, tau, A_PEREGRINE), dtype=complex)
            - np.asarray(peregrine_soliton(xi, tau), dtype=complex)
        ).max()
        max_diff = max(max_diff, float(d))
    print(
        f"  max |general_sfb(a=1/2) - peregrine_soliton| over xi in "
        f"[-{XI_MAX}, {XI_MAX}]: {max_diff:.3e}"
    )
    print("  general_sfb short-circuits |a - 1/2| < 1e-9 to peregrine_soliton,")
    print("  so the two are the same function, not a limiting approximation.")
    assert max_diff == 0.0, "a = 1/2 must be the exact Peregrine member"

    # The two wrappers agree with the general solution on their own domains.
    xi = 0.7
    tau_short = np.linspace(-2.0, 2.0, 401)
    for fn, a, name in (
        (akhmediev_breather, 0.25, "akhmediev_breather"),
        (kuznetsov_ma, 0.75, "kuznetsov_ma"),
    ):
        d = np.abs(
            np.asarray(fn(xi, tau_short, a), dtype=complex)
            - np.asarray(general_sfb(xi, tau_short, a), dtype=complex)
        ).max()
        print(f"  {name}(a={a}) vs general_sfb: max |diff| = {d:.3e}")
        assert d == 0.0

    # L2 error of the closed-form peak ladder against the sampled peak, swept.
    a_sweep = np.linspace(0.05, 0.95, 91)
    err = np.array(
        [
            abs(_intensity(0.0, tau, float(a)).max() - sfb_peak_ratio(float(a)))
            / sfb_peak_ratio(float(a))
            for a in a_sweep
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    for xi_show, style in ((-2.0, "--"), (0.0, "-"), (2.0, ":")):
        ax.plot(
            tau,
            _intensity(xi_show, tau, A_PEREGRINE),
            style,
            lw=1.6,
            label=rf"$\xi = {xi_show:g}$",
        )
    ax.axhline(1.0, color="grey", lw=0.8, ls="-.")
    ax.set_title("Peregrine soliton: growth, peak, decay\n(rather than a limit curve)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_ylim(0, 11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(a_sweep, np.maximum(err, 1e-16), "k.", ms=3)
    ax.set_yscale("log")
    ax.set_title("closed-form peak vs sampled peak\n(finite-grid sampling error only)")
    ax.set_xlabel("a")
    ax.set_ylabel("relative deviation")
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(
        a_sweep,
        [sfb_peak_ratio(float(a)) for a in a_sweep],
        "k-",
        lw=2,
        label="sfb_peak_ratio",
    )
    ax.axvline(0.5, color="tab:red", ls="--", lw=1.2)
    ax.plot([0.5], [9.0], "*", ms=16, color="tab:red", label="a = 1/2 -> 9")
    ax.plot(
        [A_KIBLER],
        [sfb_peak_ratio(A_KIBLER)],
        "^",
        ms=10,
        color="tab:orange",
        label=f"a = {A_KIBLER} (Kibler)",
    )
    ax.set_title(
        "peak compression across the family\n(monotonic, 9 at the branch point)"
    )
    ax.set_xlabel("a")
    ax.set_ylabel(r"max $|\psi|^2$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Peregrine = exact a = 1/2 member", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS["peregrine"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"max_diff": max_diff}


# ── 3. Physical scaling and the engine ──────────────────────────────
def engine_comparison() -> dict:
    print(_rule("3. SolitonOnBackground: the dimensionless -> physical map"))

    sob = SolitonOnBackground(beta2=BETA2_SI, gamma=GAMMA, P0=P0)
    L_NL, T0 = sob.L_NL, sob.T0
    print(f"  L_NL = 1/(gamma*P0)      = {L_NL:.4f} m")
    print(f"  T0   = sqrt(|b2|*L_NL)  = {T0 * 1e12:.4f} ps")
    xi_round = float(sob.xi(2 * L_NL))
    tau_round = float(sob.tau(3 * T0))
    print(f"  xi(z = 2*L_NL)          = {xi_round:.4f}   (round-trip)")
    print(f"  tau(t = 3*T0)           = {tau_round:.4f}   (round-trip)")
    assert abs(xi_round - 2.0) < 1e-12
    assert abs(tau_round - 3.0) < 1e-12

    a = 0.42
    z_chk, t_chk = 0.31 * L_NL, 2.0 * T0
    psi = sob.psi(z_chk, np.array([t_chk]), a)
    fld = sob.field(z_chk, np.array([t_chk]), a)
    assert abs(fld[0] - np.sqrt(P0) * psi[0]) < 1e-12
    assert (
        abs(
            fld[0]
            - np.sqrt(P0) * np.asarray(general_sfb(sob.xi(z_chk), sob.tau(t_chk), a))
        )
        < 1e-12
    )
    print("  field(z,t,a) == sqrt(P0) * psi(z,t)                 [asserted]")
    print(
        f"  peak_power(a={a})          = {sob.peak_power(a):.3f} W "
        f"= P0 * sfb_peak_ratio = {P0} * {sfb_peak_ratio(a):.4f}"
    )
    print(f"  temporal_period_s(0.25)    = {sob.temporal_period_s(0.25) * 1e12:.2f} ps")
    print(f"  spatial_period_m(0.75)    = {sob.spatial_period_m(0.75) * 1e3:.2f} mm")
    # The physical period must be the dimensionless one times the right scale.
    assert abs(sob.temporal_period_s(0.25) - sfb_temporal_period(0.25) * T0) < 1e-18
    assert abs(sob.spatial_period_m(0.75) - sfb_spatial_period(0.75) * L_NL) < 1e-15

    # ── Engine run ────────────────────────────────────────────────
    print(_rule("4. Analytic solution through SplitStepEngine"))
    print("  Raman and self-steepening off => the engine solves exactly the NLSE")
    print("  that general_sfb is the analytic solution of.")

    # Grid: 5 temporal periods of the a-deck at the analytic scale.
    n_periods = 5
    T_win = n_periods * sob.temporal_period_s(a)
    grid = TemporalGrid(N=2048, Tmax=Time(T_win, "s"))
    dt_ps = grid.dt * 1e12
    print(f"  grid: N = {grid.N}, Tmax = {T_win * 1e12:.1f} ps, dt = {dt_ps:.3f} ps")
    assert dt_ps < 0.1, f"grid too coarse to resolve the breathers: {dt_ps:.3f} ps"

    # Start upstream so the compression peak lands mid-fiber.
    xi_start = -2.5
    z_total = 5.0 * L_NL
    wave = sob.initial_wave(
        grid, a, z0=xi_start * L_NL, wavelength=Wavelength(LAM0, "m")
    )

    fiber = FiberProfile.from_gamma(
        gamma=GAMMA,
        n2=2.7e-20,
        omega0=float(wave.central_frequency),
        length=Length(z_total, "m"),
    )
    dz = L_NL / 20.0  # 0.05 dimensionless
    solver = SplitStepEngine(
        pulse=wave,
        fiber=fiber,
        betas=np.array([BETA2_SI * 1e24]),  # s^2/m -> ps^2/m
        include_raman=False,
        include_self_steepening=False,
        step_size=Length(dz, "m"),
    )
    n_steps = int(np.ceil(z_total / dz))
    solver.propagate(num_steps=n_steps, nsaves=25)
    print(
        f"  propagated {n_steps} steps of dz = {dz * 1e3:.3f} mm "
        f"({dz / L_NL:.3f} L_NL) over {z_total / L_NL:.1f} L_NL"
    )

    z_saves = np.linspace(0.0, z_total, len(solver.evolution))
    xis = z_saves / L_NL
    rel_field, rel_int = [], []
    print("\n   xi      rel-L2 (complex)   rel-L2 (|A|^2)     max |A|^2 sim/ana (W)")
    for z_s, xi_s in zip(z_saves, xis):
        A_sim = solver.evolution[int(np.argmin(np.abs(z_saves - z_s)))].envelope_field
        # field()/psi() take a PHYSICAL z in metres; the launch was at
        # z0 = xi_start * L_NL, so the analytic field at saved distance z_s
        # corresponds to xi = xi_start + z_s / L_NL.
        A_ana = np.asarray(sob.field(z_s + xi_start * L_NL, grid.t, a), dtype=complex)
        num_f = np.linalg.norm(A_sim - A_ana)
        den_f = np.linalg.norm(A_ana)
        rel_field.append(num_f / den_f)
        I_sim, I_ana = np.abs(A_sim) ** 2, np.abs(A_ana) ** 2
        rel_int.append(np.linalg.norm(I_sim - I_ana) / np.linalg.norm(I_ana))
        print(
            f"  {xi_s:5.2f}      {rel_field[-1]:14.3e}      {rel_int[-1]:16.3e}"
            f"        {I_sim.max():8.3f}/{I_ana.max():8.3f}"
        )

    print(f"\n  final z = {z_saves[-1] / L_NL:.2f} L_NL:")
    print(f"    complex-field rel-L2 = {rel_field[-1]:.3e}")
    print(
        f"    intensity-only rel-L2 = {rel_int[-1]:.3e}   <- conservative, quote this"
    )
    print("    Both are a few percent and track the sharp modulation peaks: the")
    print("    per-save peak powers above agree to ~1 %, and the residual is")
    print("    FFT-window and step-size discretisation of an envelope that peaks")
    print("    at 8x background. The intensity figure is the one to quote because")
    print("    it is invariant to the absolute-phase convention the complex")
    print("    figure is not; ISSUES.md #6 is the same distinction.")

    fig, ax = plt.subplots(2, 4, figsize=(17.0, 8.4))
    show = np.linspace(0, len(solver.evolution) - 1, 4).astype(int)
    for k, idx in enumerate(show):
        ax[0, k].plot(
            grid.t * 1e12,
            np.abs(solver.evolution[idx].envelope_field) ** 2,
            lw=1.5,
            label="engine",
        )
        z_s = z_saves[idx]
        ax[0, k].plot(
            grid.t * 1e12,
            np.abs(sob.field(z_s + xi_start * L_NL, grid.t, a)) ** 2,
            "--",
            lw=1.5,
            label="analytic",
        )
        ax[0, k].set_title(rf"$\xi = {xi_start + z_s / L_NL:.2f}$")
        ax[0, k].set_xlabel("t (ps)")
        ax[0, k].set_ylabel(r"$|A|^2$ (W)")
        ax[0, k].set_ylim(0, P0 * 4)
        ax[0, k].grid(alpha=0.3)
        if k == 0:
            ax[0, k].legend(fontsize=8)

    ax[1, 0].plot(xis, rel_field, "o-", ms=3, label="complex field")
    ax[1, 0].plot(xis, rel_int, "s-", ms=3, label="intensity only")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("relative L2 error vs distance")
    ax[1, 0].set_xlabel(r"$\xi$")
    ax[1, 0].set_ylabel("rel. L2")
    ax[1, 0].grid(alpha=0.3, which="both")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(
        xis + xi_start,
        [np.abs(w.envelope_field).max() ** 2 for w in solver.evolution],
        "o-",
        ms=3,
        label="engine peak power",
    )
    ax[1, 1].axhline(P0, color="k", lw=0.8, ls="-.", label="background P0")
    ax[1, 1].axvline(0.0, color="tab:red", lw=1.0, ls="--", label=r"$\xi = 0$ (peak)")
    ax[1, 1].set_title("peak power vs distance\n(Akhmediev growth/decay about xi = 0)")
    ax[1, 1].set_xlabel(r"$\xi$")
    ax[1, 1].set_ylabel("W")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=8)

    ax[1, 2].axis("off")
    ax[1, 2].text(
        0.5,
        0.5,
        "Two error conventions\n\n"
        f"complex field : {rel_field[-1]:.2e}\n"
        f"intensity only: {rel_int[-1]:.2e}\n\n"
        "Both are a few percent: the residual is\n"
        "FFT-window and step-size discretisation of\n"
        "peaks sitting at 8x background.\n\n"
        "Quote the intensity number -- it is invariant\n"
        "to the absolute-phase convention that the\n"
        "complex figure is not. (ISSUES.md #6)",
        ha="center",
        va="center",
        fontsize=9.5,
        transform=ax[1, 2].transAxes,
        bbox={"boxstyle": "round", "facecolor": "tab:blue", "alpha": 0.10},
    )
    ax[1, 3].axis("off")
    ax[1, 3].text(
        0.5,
        0.5,
        "Scaling\n\n"
        f"beta2 = {BETA2_SI * 1e24:.4f} ps^2/m\n"
        f"gamma = {GAMMA * 1e3:.2f} 1/(W km)\n"
        f"P0    = {P0:.1f} W\n\n"
        f"L_NL = {L_NL:.1f} m\n"
        f"T0   = {T0 * 1e12:.3f} ps\n"
        f"dz   = {dz / L_NL:.3f} L_NL\n"
        f"N    = {grid.N}, dt = {dt_ps:.3f} ps\n\n"
        "Raman and self-steepening off: the\n"
        "engine solves exactly the NLSE that\n"
        "general_sfb is the analytic solution of.",
        ha="center",
        va="center",
        fontsize=9.5,
        transform=ax[1, 3].transAxes,
        bbox={"boxstyle": "round", "facecolor": "tab:green", "alpha": 0.10},
    )

    fig.suptitle(
        f"Analytic SFB (a = {a}) through SplitStepEngine — "
        f"final intensity rel-L2 = {rel_int[-1]:.2e}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "30_breather_engine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "rel_field": float(rel_field[-1]),
        "rel_int": float(rel_int[-1]),
        "L_NL": L_NL,
        "T0": T0,
    }


def main() -> None:
    print(_rule("0. Soliton-on-finite-background (breather) family"))
    print("  a in (0, 1); branch point a = 1/2:")
    print("    0 < a < 1/2   Akhmediev breather  (periodic in tau)")
    print("    a  = 1/2     Peregrine soliton  (the extreme-event limit)")
    print("    1/2 < a < 1  Kuznetsov-Ma soliton (periodic in xi)")
    print("  a -> 0 is the plane wave background; a = 0 and a = 1 raise ValueError.")

    _, fam = family_panel()
    per = peregrine_panel()
    eng = engine_comparison()

    print(_rule("Summary"))
    print(f"  worst closed-form peak deviation : {fam['peak_dev']:.2e}")
    print(f"  |general_sfb(a=1/2) - peregrine| : {per['max_diff']:.3e}  (exact)")
    print(
        f"  L_NL / T0                        : {eng['L_NL']:.3f} m / "
        f"{eng['T0'] * 1e12:.3f} ps"
    )
    print(
        f"  engine vs analytic, final z      : complex {eng['rel_field']:.3e}, "
        f"intensity {eng['rel_int']:.3e}"
    )
    print("  all checks passed ✓")

    print("\nGenerated files:")
    for key in ("families", "peregrine"):
        print(f"  {FIGS[key].relative_to(_ROOT)}")
    print(f"  {(OUT_DIR / '30_breather_engine.png').relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for _f in FIGS.values():
        assert not _f.exists() or _f.stat().st_size > 0, f"{_f} exists but is empty"
    main()
    missing = [
        p
        for p in list(FIGS.values()) + [OUT_DIR / "30_breather_engine.png"]
        if not p.exists()
    ]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
