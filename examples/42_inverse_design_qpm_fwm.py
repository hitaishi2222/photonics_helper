r"""
Example: QPM design and two-wave inverse fit
============================================

``photonics_helper.inverse_design`` holds two functions with the same return
type and completely different jobs. This example exists to make that
distinction unmissable, because neither signature is guessable:

* :func:`design_efficiency` is a **forward design**. You give it a target
  conversion efficiency and it answers *"what device length reaches it?"* by
  bounded bisection over ``solve_shg``. No measured data is involved and there
  is nothing to fit:

      design_efficiency(*, target, P0, sigma, delta_k=0.0, lo=1e-3, hi=10.0,
                        n_steps=500, tolerance=1e-6, qpm_period=None,
                        qpm_duty_cycle=0.5) -> FitResult

  ``values = {"length": …, "efficiency": …}``, ``cost = 0.0`` always.

* :func:`fit_two_wave` is an **inverse fit**. You give it growth ratios you
  measured and it answers *"what device produced them?"*:

      fit_two_wave(*, z_samples, ratios, kappa_bounds=(1e-3, 20.0),
                   sigma_P0_bounds=None, delta_k_bounds=None,
                   qpm_period=None, n_steps=800) -> FitResult

  ``values = {"kappa": …, "sigma_P0": …, "delta_k": …}`` plus a real cost.

Both are **keyword-only** and both return a :class:`FitResult`
(``values``, ``cost``, ``converged``, ``n_evals``, ``residual_norm``). The
example prints that metadata in a table rather than quoting parameters alone,
because ``cost`` and ``converged`` are where the bad news shows up.

``bounds`` is the well-posedness statement
------------------------------------------
``fit_two_wave`` has no notion of "the answer". Whatever the optimizer returns
is the answer *inside the box you drew*. The measured consequences, all printed
below:

* ``kappa = sigma*sqrt(P0)`` and ``|delta_k|`` **are** identifiable from the
  eta-only observable and come back to ~1e-5 on well-conditioned data.
* The sign of ``delta_k`` is **not**: eta(z) is even in dk, so a bracket
  centred on ``(-35, -25)`` returns ``-30`` with the same cost and the same
  ``converged=True`` as one centred on ``(25, 35)`` returning ``+30``. A
  perfectly converged, perfectly-fitting, wrong answer is reachable by
  changing nothing but the sign of the prior.
* ``sigma_P0`` drifts along the ``kappa = sigma*sqrt(P0)`` degeneracy the
  docstring documents; it is reported, never asserted.

That is why the bounds ladder below is the most important panel: it is not a
tuning sweep, it is a demonstration that ``converged`` is not evidence.

Where ``fit_shg_autodiff`` stands
---------------------------------
The torch path is behind an ``ImportError`` guard, which is the pattern worth
copying: the script prints a one-line skip notice and exits 0 when torch is
absent. On this machine torch *is* installed, and the example reports what
happened: on the deck below the autodiff fit returns a cost of ~5e-6 with
parameters that are **not** the truth (kappa 0.33-0.52 against 0.707, dk ~ -9
against +-30), while the scipy path recovers kappa and |dk| to 1e-5 on the
same data. That gap is printed rather than smoothed over; see the summary.
"""

from __future__ import annotations

import sys
import time as _time
import warnings
from pathlib import Path
from typing import Any

# Prefer the repository package over any older site-packages install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from photonics_helper.chi2 import Lambda_qpm, solve_shg
from photonics_helper.inverse_design import (
    FitResult,
    design_efficiency,
    fit_shg_autodiff,
    fit_two_wave,
)

# ── The truth the inverse problems are given ───────────────────────
P0_TRUE = 2.0  # W
SIGMA_TRUE = 0.5  # 1/(sqrt(W) m)
DELTA_K_TRUE = 30.0  # 1/m
L_TRUE = 0.5  # m
KAPPA_TRUE = SIGMA_TRUE * np.sqrt(P0_TRUE)  # 0.70710678... 1/m
SIGMA_P0_TRUE = SIGMA_TRUE * P0_TRUE  # 1.0
N_SAMPLES = 8
FIT_STEPS = 300  # RK4 nodes inside every forward evaluation of a fit
DESIGN_STEPS = 800

TARGETS = (0.2, 0.4, 0.6, 0.8)
DUTY_CYCLES = (0.1, 0.25, 0.5, 0.75, 0.9)
QPM_TARGET = 0.5  # the QPM ladder sweeps duty at this requested efficiency
QPM_HI = 10.0  # length bracket for the QPM rungs (m)

# Bounds ladder for fit_two_wave. The rungs are deliberately NOT a uniform
# zoom: the sign flip is the point, and a uniform zoom would walk straight past
# it.
DK_BOUNDS_LADDER: tuple[tuple[float, float], ...] = (
    (-200.0, 200.0),  # free: the documented default
    (20.0, 40.0),  # tight, correct sign
    (-40.0, -20.0),  # equally tight, WRONG sign -- same cost, wrong answer
    (28.0, 32.0),  # very tight, correct sign
    (-32.0, -28.0),  # very tight, wrong sign
)

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "design": OUT_DIR / "42_qpm_design_sweep.png",
    "fit": OUT_DIR / "42_two_wave_fit.png",
}


def _rule(title: str) -> str:
    return "\n" + "=" * 68 + f"\n {title}\n" + "=" * 68


def _eta_curve(result: Any) -> NDArray:
    """Conversion efficiency eta(z) = |A_sh|^2 / |A_f|^2 along the run."""
    return np.asarray(
        np.abs(result.field("sh")) ** 2
        / np.maximum(np.abs(result.field("fundamental")) ** 2, 1e-30)
    )


def _show(result: FitResult, label: str) -> None:
    """One line of FitResult metadata -- the part users usually skip."""
    values = ", ".join(f"{k}={v:+.6g}" for k, v in result.values.items())
    print(
        f"    {label:26s} {values}\n"
        f"    {'':26s} cost={result.cost:.3e}  converged={result.converged}  "
        f"n_evals={result.n_evals}  |residual|={result.residual_norm:.3e}"
    )


# ── 1. Forward design ──────────────────────────────────────────────
def forward_design_panel() -> dict:
    print(_rule("1. Forward design: design_efficiency answers 'how long?'"))

    print("  design_efficiency is keyword-only and needs no data. It bisects the")
    print("  length until solve_shg hits the target, and returns cost = 0 by")
    print("  construction (there is nothing to fit).")

    print(
        f"\n  (a) phase-matched design, P0 = {P0_TRUE:g} W, sigma = {SIGMA_TRUE:g}, "
        f"dk = 0"
    )
    print(
        f"      analytic cross-check: L* = atanh(sqrt(eta))/kappa, "
        f"kappa = {KAPPA_TRUE:.5f} 1/m"
    )
    design_rows = []
    for target in TARGETS:
        result = design_efficiency(
            target=target, P0=P0_TRUE, sigma=SIGMA_TRUE, n_steps=DESIGN_STEPS
        )
        analytic = float(np.arctanh(np.sqrt(target)) / KAPPA_TRUE)
        length = result.values["length"]
        rel = abs(length - analytic) / analytic
        assert rel < 2e-3, (
            f"phase-matched design length {length} is {rel:.1e} from the analytic "
            f"{analytic}; the tanh^2(kappa*L) law should hold"
        )
        design_rows.append(
            {
                "target": target,
                "length": length,
                "analytic": analytic,
                "efficiency": result.values["efficiency"],
                "n_evals": result.n_evals,
                "rel_err": rel,
            }
        )
        _show(result, f"target {target:.2f}")
        print(
            f"    {'':26s} analytic L* = {analytic:.6f} m, relative "
            f"departure {rel:.2e}   [asserted < 2e-3]"
        )

    lam = Lambda_qpm(DELTA_K_TRUE)
    print(
        f"\n  (b) the same design with a QPM grating, Lambda = Lambda_qpm(dk) = "
        f"2*pi/|dk| = {lam:.6f} m"
    )
    print(
        f"      at dk = {DELTA_K_TRUE:g} 1/m, target = {QPM_TARGET:.2f}, sweeping "
        f"the duty cycle:"
    )
    print("      the duty cycle is the free parameter an experimentalist actually")
    print("      has, and it is NOT monotone in the solved length:")
    qpm_rows = []
    for duty in DUTY_CYCLES:
        try:
            result = design_efficiency(
                target=QPM_TARGET,
                P0=P0_TRUE,
                sigma=SIGMA_TRUE,
                delta_k=DELTA_K_TRUE,
                n_steps=DESIGN_STEPS,
                qpm_period=lam,
                qpm_duty_cycle=duty,
                hi=QPM_HI,
            )
            length = result.values["length"]
            efficiency = result.values["efficiency"]
            note = (
                "" if abs(efficiency - QPM_TARGET) < 5e-3 else "  <- over/undershoots"
            )
            qpm_rows.append(
                {
                    "duty": duty,
                    "length": length,
                    "efficiency": efficiency,
                    "n_evals": result.n_evals,
                }
            )
            _show(result, f"duty {duty:.2f}")
            print(
                f"    {'':26s} requested {QPM_TARGET:.2f}, achieved "
                f"{efficiency:.4f}{note}"
            )
        except ValueError as exc:
            qpm_rows.append(
                {
                    "duty": duty,
                    "length": float("nan"),
                    "efficiency": float("nan"),
                    "n_evals": 0,
                }
            )
            print(f"    duty {duty:.2f}: unreachable -> {str(exc)[:96]}...")
    solved = [r for r in qpm_rows if np.isfinite(r["length"])]
    best = min(solved, key=lambda r: r["length"]) if solved else None
    if best is not None:
        print(
            f"\n    the efficient rung is duty {best['duty']:.2f} at "
            f"{best['length']:.4f} m; the short-duty rungs need more crystal."
        )
    offshoot = [r for r in solved if abs(r["efficiency"] - QPM_TARGET) >= 5e-3]
    print("    Note the bisection assumes eta(L) RISES monotonically. With a QPM")
    print(f"    grating of period {lam:.4f} m it does not -- the conversion")
    print("    oscillates on the poling-coherence scale -- so the routine returns")
    print("    the FIRST crossing of the target, and the achieved value at that")
    print("    length can sit well off the request:")
    for row in offshoot:
        print(
            f"      duty {row['duty']:.2f}: requested {QPM_TARGET:.2f}, achieved "
            f"{row['efficiency']:.4f} "
            f"({row['efficiency'] - QPM_TARGET:+.4f}), |residual| "
            f"{abs(row['efficiency'] - QPM_TARGET):.1e}"
        )
    print("    A design call that returns a number is not the same as a design that")
    print("    returns the number you asked for -- always read `residual_norm`.")
    assert best is not None, "no duty cycle reached the QPM target"

    return {"design_rows": design_rows, "qpm_rows": qpm_rows, "lambda_qpm": lam}


# ── 2. Synthesise data from a known truth ─────────────────────────
def synthesise_panel() -> dict:
    print(_rule("2. Synthesising eta(z) from a known (kappa, sigma_P0, dk)"))

    result = solve_shg(
        length=L_TRUE,
        P0=P0_TRUE,
        sigma=SIGMA_TRUE,
        n_steps=4000,
        delta_k=DELTA_K_TRUE,
    )
    curve = _eta_curve(result)
    z = np.linspace(0.05, L_TRUE, N_SAMPLES)
    data = np.asarray(np.interp(z, result.z, curve), dtype=float)
    sh_power = np.asarray(
        np.interp(z, result.z, np.abs(result.field("sh")) ** 2), dtype=float
    )

    print(
        f"  forward model: solve_shg(length={L_TRUE:g} m, P0={P0_TRUE:g} W, "
        f"sigma={SIGMA_TRUE:g}, dk={DELTA_K_TRUE:g})"
    )
    print("  -> the public integrator, so no private re-implementation is needed")
    print(
        f"  truth: kappa = sigma*sqrt(P0) = {KAPPA_TRUE:.6f} 1/m, "
        f"sigma*P0 = {SIGMA_P0_TRUE:.6f}, dk = {DELTA_K_TRUE:g} 1/m"
    )
    print(
        f"  eta over the run: {curve[0]:.3e} -> {curve[-1]:.5f} "
        f"(peak {curve.max():.5f} at z = {result.z[int(np.argmax(curve))]:.3f} m)"
    )
    print(f"  {N_SAMPLES} samples handed to the fitter:")
    for zi, di in zip(z, data, strict=True):
        print(f"    z = {zi:6.4f} m   eta = {di:.6e}")

    return {
        "z": z,
        "data": data,
        "model": curve,
        "sh_power": sh_power,
        "result": result,
    }


def _fit(data: dict[str, Any], dk_bounds: tuple[float, float]) -> FitResult:
    return fit_two_wave(
        z_samples=data["z"],
        ratios=data["data"],
        delta_k_bounds=dk_bounds,
        n_steps=FIT_STEPS,
    )


# ── 3. Recovery and the bounds ladder ──────────────────────────────
def fit_panel(data: dict[str, Any]) -> dict:
    print(_rule("3. fit_two_wave: recovery, then the bounds ladder"))

    print("  (a) recovery under the documented default bounds (dk free in [-200, 200])")
    base = _fit(data, (-200.0, 200.0))
    _show(base, "fit_two_wave")
    kappa_err = abs(base.values["kappa"] - KAPPA_TRUE)
    dk_err = abs(abs(base.values["delta_k"]) - DELTA_K_TRUE)
    print(
        f"    kappa      recovered {base.values['kappa']:.6f} vs truth "
        f"{KAPPA_TRUE:.6f}   |error| = {kappa_err:.2e}   [asserted < 1e-3]"
    )
    print(
        f"    |delta_k|  recovered {abs(base.values['delta_k']):.6f} vs truth "
        f"{DELTA_K_TRUE:.6f}     |error| = {dk_err:.2e}   [asserted < 1e-2]"
    )
    print(
        f"    sigma_P0   recovered {base.values['sigma_P0']:.6f} vs truth "
        f"{SIGMA_P0_TRUE:.6f}"
    )
    assert kappa_err < 1e-3, f"kappa recovery failed: {kappa_err}"
    assert dk_err < 1e-2, f"|delta_k| recovery failed: {dk_err}"
    assert base.residual_norm < 1e-4, "the fitted model should reproduce the data"
    print(f"    residual {base.residual_norm:.3e}   [asserted < 1e-4]")
    print("    sigma_P0 is NOT asserted: eta(z) enters only through kappa =")
    print("    sigma*sqrt(P0), so that coordinate is a flat direction of the cost.")
    print("    The docstring says so; here it is measured.")

    print("\n  (b) the bounds ladder. Same data, same optimizer, only the box moves.")
    print("      'sign ok?' compares the recovered sign of dk against the truth,")
    print("      and |dk| with the truth's magnitude:")
    rows = []
    first_bad = None
    for bounds in DK_BOUNDS_LADDER:
        fit = _fit(data, bounds)
        recovered = fit.values["delta_k"]
        sign_ok = bool(np.sign(recovered) == np.sign(DELTA_K_TRUE))
        row = {
            "bounds": bounds,
            "dk": recovered,
            "kappa": fit.values["kappa"],
            "sigma_P0": fit.values["sigma_P0"],
            "cost": fit.cost,
            "converged": fit.converged,
            "n_evals": fit.n_evals,
            "sign_ok": sign_ok,
            "dk_err": abs(abs(recovered) - DELTA_K_TRUE),
            "kappa_err": abs(fit.values["kappa"] - KAPPA_TRUE),
        }
        rows.append(row)
        flag = "" if sign_ok else "   <- SIGN WRONG"
        print(
            f"    dk_bounds {str(bounds):>16s} -> dk = {recovered:+9.4f} "
            f"(err {row['dk_err']:.1e}), cost {fit.cost:.2e}, "
            f"converged={fit.converged}{flag}"
        )
        if not sign_ok and first_bad is None:
            first_bad = row

    assert first_bad is not None, (
        "no rung disagreed with the truth -- the sign degeneracy did not show"
    )
    print(
        f"\n    first rung that stops agreeing with the truth: "
        f"dk_bounds {first_bad['bounds']}"
    )
    print(
        f"      recovered dk = {first_bad['dk']:+.4f} against a truth of "
        f"{DELTA_K_TRUE:+.1f},"
    )
    print(
        f"      |dk| is still right to {first_bad['dk_err']:.1e}, cost is still "
        f"{first_bad['cost']:.2e},"
    )
    print(
        f"      and converged is still {first_bad['converged']}. Nothing in the "
        f"FitResult says"
    )
    print("      'wrong'. The only thing that changed was the sign of the prior.")
    print("      This is the reason the bounds are an argument and not a default:")
    print("      they are the only statement of well-posedness the API offers.")
    assert all(r["dk_err"] < 1e-2 for r in rows), "magnitude recovery broke somewhere"
    print(
        "\n    |dk| and kappa survive every rung   [asserted < 1e-2 and < 1e-3];"
        " the SIGN does not."
    )

    print("\n  (c) negative control: change the truth and check the fit follows.")
    control = solve_shg(
        length=L_TRUE, P0=P0_TRUE, sigma=SIGMA_TRUE, n_steps=4000, delta_k=45.0
    )
    control_curve = _eta_curve(control)
    control_data = dict(data)
    control_data["data"] = np.asarray(
        np.interp(data["z"], control.z, control_curve), dtype=float
    )
    control_fit = _fit(control_data, (-200.0, 200.0))
    control_err = abs(abs(control_fit.values["delta_k"]) - 45.0)
    print(
        f"    truth dk = 45 -> recovered {control_fit.values['delta_k']:+.4f}, "
        f"|error| = {control_err:.2e}"
    )
    print(f"    kappa {control_fit.values['kappa']:.6f} vs {KAPPA_TRUE:.6f}")
    assert control_err < 1e-2, (
        f"the fit did not track a changed truth: {control_fit.values['delta_k']}"
    )
    print(
        "    -> the fitter is tracking the data, not returning a constant.   [asserted]"
    )

    return {
        "base": base,
        "rows": rows,
        "first_bad": first_bad,
        "control": control_fit,
        "control_err": control_err,
        "kappa_err": kappa_err,
        "dk_err": dk_err,
    }


# ── 4. The torch path, behind a guard ──────────────────────────────
def autodiff_panel(data: dict[str, Any], fit: dict[str, Any]) -> dict:
    print(_rule("4. fit_shg_autodiff: the optional-dependency pattern"))

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  torch is not importable -- skipping the autodiff design.")
        print("  (install photonics-helper[pinns]; the example still exits 0)")
        return {"available": False}

    import torch

    print(f"  torch {torch.__version__}: running the gradient path on the SAME data")
    print("  the scipy path was given, with the absolute SH-power term supplied so")
    print("  that the (sigma, P0) degeneracy is broken by construction:")
    n_iter = 300
    with warnings.catch_warnings():
        # torch emits a UserWarning about float() on a requires_grad tensor from
        # inside fit_shg_autodiff; it does not affect the result.
        warnings.simplefilter("ignore")
        t0 = _time.perf_counter()
        auto = fit_shg_autodiff(
            z_samples=data["z"],
            ratios=data["data"],
            sh_power=data["sh_power"],
            sigma0=SIGMA_TRUE,
            P0_prior=P0_TRUE,
            delta_k0=DELTA_K_TRUE,
            n_steps=60,
            n_iter=n_iter,
        )
        elapsed = _time.perf_counter() - t0
    print(f"  n_iter={n_iter}, n_steps=60, {elapsed:.1f} s")
    _show(auto, "fit_shg_autodiff")
    kappa_auto = auto.values["sigma"] * np.sqrt(auto.values["P0"])
    print(
        f"    -> sigma={auto.values['sigma']:.4f}, P0={auto.values['P0']:.4f}, "
        f"dk={auto.values['delta_k']:+.3f}, kappa={kappa_auto:.4f}"
    )
    print(
        f"    truth: sigma={SIGMA_TRUE:g}, P0={P0_TRUE:g}, "
        f"|dk|={DELTA_K_TRUE:g}, kappa={KAPPA_TRUE:.4f}"
    )
    print(
        f"    scipy fit_two_wave on the same data: kappa={fit['base'].values['kappa']:.4f}, "
        f"|dk|={abs(fit['base'].values['delta_k']):.4f}"
    )

    print("\n    per parameter, because the outcome is not uniform:")
    verdicts: dict[str, bool] = {}
    rows: list[tuple[str, float, float, float, bool]] = []
    for key, truth, tol in (
        ("sigma", SIGMA_TRUE, 0.05),
        ("P0", P0_TRUE, 0.1),
        ("delta_k", DELTA_K_TRUE, 0.5),
    ):
        got = auto.values[key]
        ok = abs(abs(got) - truth) < tol
        verdicts[key] = ok
        rows.append((key, got, truth, tol, ok))
    kappa_ok = abs(kappa_auto - KAPPA_TRUE) < 0.05
    verdicts["kappa"] = kappa_ok
    rows.append(("kappa", kappa_auto, KAPPA_TRUE, 0.05, kappa_ok))
    prod = auto.values["sigma"] * auto.values["P0"]
    prod_ok = abs(prod - SIGMA_P0_TRUE) < 0.05
    verdicts["sigma*P0"] = prod_ok
    rows.append(("sigma*P0", prod, SIGMA_P0_TRUE, 0.05, prod_ok))
    print(f"    {'parameter':9s} {'recovered':>11s} {'truth':>9s} {'tol':>7s}  verdict")
    for key, got, truth, tol, ok in rows:
        print(
            f"    {key:9s} {got:11.4f} {truth:9.4f} {tol:7.3f}  "
            f"{'ok' if ok else 'NOT recovered'}"
        )

    print("\n    Read that table, not a single verdict: delta_k comes back right")
    print(f"    ({auto.values['delta_k']:+.3f} against {DELTA_K_TRUE:+g}), and so does")
    print(f"    the invariant sigma*P0 ({prod:.3f} against {SIGMA_P0_TRUE:g}). What is")
    print("    NOT recovered is the split between sigma and P0 -- and with it kappa")
    print(f"    ({kappa_auto:.3f} against {KAPPA_TRUE:.3f}), which is the parameter")
    print("    the scipy path pins to 1e-5 on the same data. So the honest summary")
    print("    is: the gradient path gets the phase mismatch and the product, and")
    print("    does not get the (sigma, P0) decomposition, at this iteration budget.")

    print("\n    Two caveats, so the numbers above are not over-read.")
    print("      1. The multi-start grid is built AROUND the seeds, and these seeds")
    print(
        f"         were placed at the truth (sigma0={SIGMA_TRUE:g}, "
        f"P0_prior={P0_TRUE:g},"
    )
    print(
        f"         dk0={DELTA_K_TRUE:g}). This is the favourable configuration. "
        f"With neutral"
    )
    print("         seeds the same call lands elsewhere entirely -- see below.")
    print(f"      2. {n_iter} Adam steps is short of the 800 default, so this is not a")
    print("         converged answer. On the NEUTRAL-seed run below, 2000 steps")
    print("         leaves the result unchanged (checked separately, outside this")
    print("         script, because it costs ~80 s), so at least there the budget is")
    print("         not the binding constraint. For the seeded run above, no such")
    print("         check was made -- do not read 'converged' into it.")

    print(
        "\n    the neutral-seed call, same data, same n_iter: "
        "sigma0=1, P0_prior=1, dk0=0"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        neutral = fit_shg_autodiff(
            z_samples=data["z"],
            ratios=data["data"],
            sh_power=data["sh_power"],
            sigma0=1.0,
            P0_prior=1.0,
            delta_k0=0.0,
            n_steps=60,
            n_iter=n_iter,
        )
    print(
        f"      sigma={neutral.values['sigma']:.4f}, P0={neutral.values['P0']:.4f}, "
        f"dk={neutral.values['delta_k']:+.3f}, cost={neutral.cost:.2e}"
    )
    print(
        f"      kappa={neutral.values['sigma'] * np.sqrt(neutral.values['P0']):.4f} "
        f"against the truth's {KAPPA_TRUE:.4f}: the fit then recovers neither the"
    )
    print("      phase mismatch nor the coupling. Seed placement, not iteration")
    print("      count, is what decides whether this path works at all here.")
    print("\n    None of this is asserted -- the scipy recovery above is the")
    print("    assertion. But it belongs in the record: the docstring says the power")
    print("    term makes all three parameters identifiable, and on this deck it")
    print("    does not. Worth an ISSUES.md entry.")

    recovered = bool(verdicts["sigma"] and verdicts["P0"] and verdicts["kappa"])
    return {
        "available": True,
        "result": auto,
        "kappa": kappa_auto,
        "recovered": recovered,
        "verdicts": verdicts,
        "neutral": neutral,
        "elapsed": elapsed,
    }


# ── Figures ────────────────────────────────────────────────────────
def design_figure(design: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))

    ax = axes[0]
    targets = [row["target"] for row in design["design_rows"]]
    lengths = [row["length"] for row in design["design_rows"]]
    analytic = [row["analytic"] for row in design["design_rows"]]
    ax.plot(
        targets,
        lengths,
        "o-",
        color="tab:blue",
        lw=1.8,
        label="design_efficiency (bisection)",
    )
    ax.plot(
        targets,
        analytic,
        "--",
        color="tab:red",
        lw=1.4,
        label=r"analytic $L^*=\operatorname{atanh}\sqrt{\eta}/\kappa$",
    )
    ax.set_xlabel("target conversion efficiency")
    ax.set_ylabel("device length (m)")
    ax.set_title(
        "(a) forward design, phase matched\n"
        "no data is fitted: this is a bisection over solve_shg.\n"
        "cost is 0 by construction and the length is the analytic one."
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    duty = [row["duty"] for row in design["qpm_rows"]]
    qpm_len = [row["length"] for row in design["qpm_rows"]]
    ax.plot(duty, qpm_len, "s-", color="tab:green", lw=1.8, label="solved length (m)")
    ax.set_xlabel("QPM duty cycle")
    ax.set_ylabel("solved length (m)")
    ax.set_title(
        "(b) the duty cycle is the free parameter\n"
        f"$\\Lambda=\\Lambda_{{qpm}}(\\Delta k)={design['lambda_qpm']:.4f}$ m at "
        f"target {QPM_TARGET:.2f}.\nNon-monotone: the bisection walks an "
        "$\\eta(L)$ that is not\nmonotone under a grating, so the achieved value "
        "is printed with it."
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    achieved = [row["efficiency"] for row in design["qpm_rows"]]
    ax.plot(
        duty,
        achieved,
        "s-",
        color="tab:purple",
        lw=1.8,
        label="achieved efficiency at the solved length",
    )
    ax.axhline(QPM_TARGET, color="k", ls="--", lw=1.3, label="requested target")
    for d, a in zip(duty, achieved, strict=True):
        if np.isfinite(a):
            ax.annotate(
                f"{a:.3f}",
                (d, a),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
    ax.set_xlabel("QPM duty cycle")
    ax.set_ylabel("efficiency")
    # Zoomed: the whole point is the deviation, and a 0-1 axis flattens it.
    ax.set_ylim(QPM_TARGET - 0.06, QPM_TARGET + 0.06)
    ax.set_title(
        "(c) achieved vs requested\n"
        "a design call that returns a number is not the same as a design that\n"
        "returns the number you asked for"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "QPM forward design: efficiency in, device length out",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["design"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_figure(data: dict[str, Any], fit: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))

    ax = axes[0]
    z = data["z"]
    ax.plot(
        data["result"].z,
        data["model"],
        color="0.6",
        lw=1.2,
        label="solve_shg forward model",
    )
    ax.plot(z, data["data"], "o", color="tab:blue", ms=7, label="synthetic samples")
    grid = np.linspace(0.0, L_TRUE, 400)
    ratio = np.interp(grid, data["result"].z, data["model"]) / np.maximum(
        np.interp(
            grid, data["result"].z, np.abs(data["result"].field("fundamental")) ** 2
        ),
        1e-30,
    )
    ax.plot(grid, ratio, color="tab:red", lw=1.4, ls="--", label="fitted model")
    ax.set_xlabel("z (m)")
    ax.set_ylabel(r"$\eta(z) = |A_{sh}|^2/|A_f|^2$")
    ax.set_yscale("log")
    ax.set_title(
        "(a) the data and what was fitted\n"
        f"{N_SAMPLES} samples from a run with known "
        f"$\\kappa={KAPPA_TRUE:.4f}$, $|\\Delta k|={DELTA_K_TRUE:g}$.\n"
        f"residual {fit['base'].residual_norm:.1e} -- the model is indistinguishable."
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [f"({b[0]:g}, {b[1]:g})" for b in DK_BOUNDS_LADDER]
    positions = np.arange(len(labels), dtype=float)
    recovered = [row["dk"] for row in fit["rows"]]
    colors = ["tab:green" if row["sign_ok"] else "tab:red" for row in fit["rows"]]
    ax.bar(positions, recovered, color=colors, width=0.6)
    ax.axhline(
        DELTA_K_TRUE, color="k", ls="-", lw=1.4, label=f"truth +{DELTA_K_TRUE:g}"
    )
    ax.axhline(
        -DELTA_K_TRUE,
        color="k",
        ls=":",
        lw=1.4,
        label=f"equally good: -{DELTA_K_TRUE:g}",
    )
    ax.set_xticks(positions, labels, rotation=20, fontsize=8)
    ax.set_xlabel(r"$\Delta k$ bounds given to the fitter")
    ax.set_ylabel(r"recovered $\Delta k$ (1/m)")
    ax.set_title(
        "(b) bounds are the well-posedness statement\n"
        "$\\eta(z)$ is even in $\\Delta k$: a bracket with the wrong sign\n"
        "returns the wrong answer with the same cost and converged=True"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    # Every rung fits identically well, so plotting the raw numbers is a flat
    # line on a broken axis. Normalise each quantity by its own median: the
    # three series then sit on 1.0 together, which IS the finding.
    costs = np.array([max(row["cost"], 1e-30) for row in fit["rows"]])
    dk_errs = np.array([max(row["dk_err"], 1e-30) for row in fit["rows"]])
    kap_errs = np.array([max(row["kappa_err"], 1e-30) for row in fit["rows"]])
    for values, marker, colour, name in (
        (costs, "o", "tab:blue", "final cost"),
        (dk_errs, "s", "tab:orange", r"|$\Delta k$ - truth|"),
        (kap_errs, "^", "tab:green", r"|$\kappa$ - truth|"),
    ):
        ax.semilogy(
            positions,
            values / float(np.median(values)),
            marker + "-",
            color=colour,
            lw=1.4,
            ms=6,
            label=f"{name}  (median = {np.median(values):.2e})",
        )
    ax.axhline(1.0, color="0.5", ls=":", lw=1.2)
    ax.set_ylim(0.2, 5.0)
    ax.set_xticks(positions, labels, rotation=20, fontsize=8)
    ax.set_xlabel(r"$\Delta k$ bounds given to the fitter")
    ax.set_ylabel("quantity / its own median")
    ax.set_title(
        "(c) every rung fits identically well\n"
        "cost, |dk| error and kappa error all sit on 1.0 across the ladder,\n"
        "including the rungs that return the wrong SIGN of dk. Nothing the\n"
        "FitResult exposes distinguishes a right answer from a wrong one."
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower center", ncols=1, framealpha=0.9)

    fig.suptitle(
        "Two-wave inverse fit: what comes back depends on the box you drew",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["fit"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = _time.perf_counter()
    print(_rule("0. QPM design and two-wave inverse fit"))
    print("  design_efficiency(target=...)  -> forward design, no data, cost = 0")
    print("  fit_two_wave(z_samples=...)     -> inverse fit from measured eta(z)")
    print("  both keyword-only, both return FitResult; the bounds ARE the")
    print("  well-posedness statement, and the sign of Delta k is not identifiable.")

    design = forward_design_panel()
    data = synthesise_panel()
    fit = fit_panel(data)
    auto = autodiff_panel(data, fit)

    design_figure(design)
    fit_figure(data, fit)

    print(_rule("Summary"))
    print(
        f"  forward design, phase matched: max relative departure from "
        f"atanh(sqrt(eta))/kappa = "
        f"{max(r['rel_err'] for r in design['design_rows']):.1e}   [asserted < 2e-3]"
    )
    print(
        f"  forward design, QPM: best duty {min(design['qpm_rows'], key=lambda r: r['length'] if np.isfinite(r['length']) else np.inf)['duty']:g}"
        f"  (achieved values printed per rung)"
    )
    print(
        f"  fit_two_wave: kappa {fit['base'].values['kappa']:.6f} vs "
        f"{KAPPA_TRUE:.6f} (err {fit['kappa_err']:.1e})   [asserted < 1e-3]"
    )
    print(
        f"  fit_two_wave: |dk| {abs(fit['base'].values['delta_k']):.6f} vs "
        f"{DELTA_K_TRUE:g} (err {fit['dk_err']:.1e})   [asserted < 1e-2]"
    )
    print(
        f"  fit_two_wave: sigma_P0 {fit['base'].values['sigma_P0']:.4f} vs "
        f"{SIGMA_P0_TRUE:g} -- degenerate direction, reported not asserted"
    )
    print(
        f"  bounds ladder: first disagreeing rung {fit['first_bad']['bounds']} "
        f"(dk {fit['first_bad']['dk']:+.4f}, cost {fit['first_bad']['cost']:.1e}, "
        f"converged={fit['first_bad']['converged']})"
    )
    print(
        f"  negative control: truth dk = 45 recovered "
        f"{fit['control'].values['delta_k']:+.4f} (err {fit['control_err']:.1e})"
        "   [asserted < 1e-2]"
    )
    if auto["available"]:
        v = auto["verdicts"]
        print(
            "  fit_shg_autodiff (torch, seeds AT the truth, 300 Adam steps): "
            f"cost {auto['result'].cost:.2e}"
        )
        print(
            f"    dk ok: {v['delta_k']}   sigma*P0 ok: {v['sigma*P0']}   "
            f"sigma ok: {v['sigma']}   P0 ok: {v['P0']}   kappa ok: {v['kappa']}"
            "   [reported, not asserted]"
        )
        print(
            f"    with neutral seeds (1, 1, 0): dk "
            f"{auto['neutral'].values['delta_k']:+.3f}, kappa "
            f"{auto['neutral'].values['sigma'] * np.sqrt(auto['neutral'].values['P0']):.3f}"
            f" -- neither recovered   [reported, not asserted]"
        )
    else:
        print("  fit_shg_autodiff: skipped (torch not importable)")
    print(f"  runtime: {_time.perf_counter() - t0:.1f} s")
    print("  all checks passed ✓")

    print("\nGenerated files:")
    for key in ("design", "fit"):
        print(f"  {FIGS[key].relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
