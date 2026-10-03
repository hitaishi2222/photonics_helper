"""Example: verifying the GNLSE solver against closed forms and against itself
============================================================================

``photonics_helper.gnlse_validation`` is the library's answer to "how do you
know the engine is right?". This example runs both halves of that answer:

1. **Four cited closed forms.** Each ``check_*`` propagates a pulse with a
   configuration whose exact solution is known and compares a *measured*
   number against that closed form.
2. **A convergence study.** ``convergence_study`` re-runs one configuration
   over a ladder of ``z``-step sizes and reports, per observable, the value at
   each refinement, the relative change between successive refinements, and a
   converged verdict at a stated tolerance (Sinkin, Holzlöhner, Zweck &
   Menyuk, *J. Lightwave Technol.* **21**, 61 (2003)).

And it shows the harness *failing* on purpose, because a validation example
that only ever passes is indistinguishable from a broken one.

The NLSE convention
--------------------
Everything below is the same generalised NLSE, in the retarded-time frame and
with the ``exp(-iwt)`` forward-FFT convention this library uses:

.. math::

    i\\,A_z = -\\beta_2 \\tau \\partial^2_\\tau A - \\gamma |A|^2 A
             + \\text{(Raman, shock, TPA)} ,

so a *bright* soliton requires **anomalous** dispersion (``\\beta_2 < 0``), and
``P(z,t) = |A(z,t)|^2`` is the power in watts. The FFT uses ``e^{-i\\Omega t}``
and spectral bins map to ``\\omega_0 + \\Omega``, with ``\\Omega > 0`` the blue
side — the same sign audit that ``examples/34`` and the MI convention example
rely on.

The four closed forms
---------------------
Each ``check_*`` compares one measured metric against one equation:

``check_spm(gamma, peak_power, t0, wavelength_m, phi_max, grid)``
    Kerr-only, ``\\beta_2 = 0``. The exact field is
    ``A(L,t) = sqrt(P0) e^{-t^2/2T0^2} e^{i phi_max e^{-t^2/T0^2}}`` with
    ``phi_max = gamma P0 L`` (Stolen & Lin, *Phys. Rev. A* **17**, 1448
    (1978)). Two comparisons: the normalised propagated spectrum against the
    analytic ``|FFT(A_exact)|^2`` (max absolute deviation), and the fringe
    count ``N_peaks = floor(phi_max/pi) + 1``.

``check_mi(beta2, gamma, pump_power, wavelength_m, probe_omega, grid)``
    A CW pump plus a weak ``eps cos(Omega t)`` probe; the **power** gain
    ``g(Omega) = |beta2 Omega| sqrt(Omega_c^2 - Omega^2)``,
    ``Omega_c^2 = 4 gamma P / |beta2|`` (Agrawal §5.1.9), is measured as the
    late-time slope of ``ln P_side(z)`` and compared against the closed form
    evaluated at the *resolved* probe bin. Sideband power grows as ``e^{g z}``,
    so the convention-validated gain peak is ``g_max = 2 gamma P`` at
    ``Omega = Omega_c / sqrt(2)``.

``check_soliton(beta2, gamma, t0, wavelength_m, grid, periods)``
    An ``N = 1`` soliton, ``gamma P0 T0^2 / |beta2| = 1``, propagated over
    ``periods`` times the fundamental period ``z_sol = (pi/2) L_D`` (Agrawal
    §5.2). The shape overlap ``|<A_in, A_out>| / sqrt(...)`` is compared
    against 1.

``check_gordon_ssfs(beta2, gamma, peak_power, t0, ..., raman_response, ...)``
    A fundamental soliton with the fibre Raman response; the measured
    red-shift of the spectral peak is compared against Gordon's law
    (Gordon, *Opt. Lett.* **11**, 662 (1986))
    ``dOmega/dz = -8 |beta2| T_R / (15 T0^4)`` with
    ``T_R = f_R \\int t h_R(t) dt``, integrated over the propagation length and
    converted to a wavelength shift. The reported ``ratio`` is
    measured / analytic.

Every tolerance below is printed next to the number it applies to and a note on
what it absorbs. That is deliberate: a tolerance silently tuned until a check
passes is the failure mode this harness exists to prevent.

What ``convergence_study`` expects from you
-------------------------------------------
``build_solver(**kwargs)`` receives the union of ``shared`` and one
``refinement`` dict, and must return a *propagated* object exposing the final
field through ``.envelope_field`` / ``.A`` / ``.field`` / ``.evolution``. A
:class:`SplitStepEngine` after ``.propagate(...)`` satisfies this directly:
``pulse_energy``, ``peak_intensity``, ``rms_bandwidth`` and ``rms_width`` all
read ``.A`` and ``.grid`` off it. ``observables`` is either a sequence of names
into ``DEFAULT_OBSERVABLES`` or a ``{name: callable(result) -> float}`` mapping
— this example passes one of each, so the mapping form is on the page.

Cost note: ``check_gordon_ssfs`` (4000 split steps, N = 8192) and ``check_mi``
(4000 split steps, N = 16384) dominate the runtime; the convergence ladder is
kept at N ≤ 16384 for the same reason. The step counts are *not* lowered below
these values, because the numbers above are the numbers the tolerances were set
against.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Prefer the repository package over any older site-packages install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.gnlse_validation import (
    DEFAULT_OBSERVABLES,
    ConvergenceReport,
    ValidationFailure,
    check_gordon_ssfs,
    check_mi,
    check_soliton,
    check_spm,
    convergence_study,
    gordon_ssfs_rate,
    mi_gain_of,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

WL0 = 1550e-9
OMEGA0 = 2 * np.pi * C_MS / WL0

# ── Reference configurations (one per closed form) ───────────────────────────
SPM = dict(gamma=11.0, peak_power=1e-3, t0=100e-15, wavelength_m=WL0, phi_max=4 * np.pi)
MI = dict(
    beta2=-21e-27,
    gamma=1.0,
    pump_power=1.0,
    wavelength_m=WL0,
    probe_omega=2 * np.pi * 1.0e12,
)
SOLITON = dict(beta2=-21e-27, gamma=1.0, t0=1e-12, wavelength_m=WL0)
SSFS = dict(
    beta2=-7.0e-27, gamma=0.11, t0=50e-15 / 1.763, wavelength_m=835e-9, length=0.5
)

GRID_N, GRID_TMAX = 16384, 8e-12
SSFS_GRID_N, SSFS_GRID_TMAX = 8192, 14e-12

# Convergence ladder: same physical run, step size refined twice.
CONV_SHARED = dict(beta2=-21e-27, gamma=2.0, t0=50e-15 / 1.763, wavelength_m=WL0)
CONV_REFINEMENTS = [
    {"N": 8192, "num_steps": 1000},
    {"N": 16384, "num_steps": 2000},
    {"N": 32768, "num_steps": 2000},
]
CONV_TOLERANCE = 2e-3
BAD_REFINEMENTS = [
    {"N": 128, "num_steps": 8},
    {"N": 256, "num_steps": 16},
    {"N": 256, "num_steps": 32},
]
# A deliberately impossible tolerance for the negative control in task 3.2:
# check_soliton at tolerance 0 must fail on any numerical drift at all.
IMPOSSIBLE_TOLERANCE = 0.0

#: tolerance -> what it is absorbing. Printed by the example (task 2.9).
TOLERANCE_NOTES = {
    "check_spm (5e-3, normalised spectrum)": "grid discretisation of the Gaussian's Fourier integral; not a physics"
    " margin — the analytic spectrum is exact for this input",
    "check_mi (15 %, gain)": "solver step-size error over ~4 e-foldings plus the numerical sideband"
    " floor under the 1 % probe seed",
    "check_soliton (2 %, 1 - overlap)": "accumulated split-step phase error over one soliton period",
    "check_gordon_ssfs (ratio 0.75-1.25)": "Gordon's law is a leading-order law for a fundamental soliton; the"
    " band absorbs the soliton-order drift and the sech-vs-Gordon profile"
    " difference (reproductions/gordon_1986_ssfs measures 1.19)",
    "convergence_study (2e-3, relative change)": "step-size discretisation only; a converged verdict means the last two"
    " refinements agree to 0.2 %, not that the model is exact",
}

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "checks": OUT_DIR / "35_validation_checks.png",
    "convergence": OUT_DIR / "35_convergence_study.png",
}


def _rule(title: str) -> str:
    return "\n" + "=" * 68 + f"\n {title}\n" + "=" * 68


def _tolerance_note(key: str) -> str:
    return TOLERANCE_NOTES[key]


# ── 1. Closed-form checks ────────────────────────────────────────────────────


def run_spm() -> dict:
    """Phase-only rotation vs phi_max, and the fringe count."""
    grid = TemporalGrid(N=GRID_N, Tmax=Time(GRID_TMAX, "s"))
    metrics = check_spm(grid=grid, **SPM)

    # Reproduce the spectrum overlays for the figure.
    from scipy.signal import find_peaks

    t = grid.t
    P0, T0, phi = SPM["peak_power"], SPM["t0"], SPM["phi_max"]
    L = phi / (SPM["gamma"] * P0)
    env = Envelope(
        shape="gaussian", peak_amplitude=np.sqrt(P0), pulse_width=Time(T0, "s")
    )
    pulse = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(SPM["wavelength_m"], "m")
    )
    fiber = FiberProfile.from_gamma(
        gamma=SPM["gamma"], n2=2.6e-20, omega0=OMEGA0, length=Length(L, "m")
    )
    from photonics_helper.gnlse import GNLSESolver

    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(
        num_steps=GNLSESolver.estimate_num_steps(pulse, fiber, np.array([0.0]))
    )
    W_num = np.abs(grid.fft(np.asarray(solver.evolution[-1].envelope_field))) ** 2
    W_num = W_num / W_num.max()
    A_an = (
        np.sqrt(P0)
        * np.exp(-(t**2) / (2 * T0**2))
        * np.exp(1j * phi * np.exp(-(t**2) / T0**2))
    )
    W_an = np.abs(grid.fft(A_an)) ** 2
    W_an = W_an / W_an.max()

    print(_rule("1. SPM — Kerr-only phase rotation vs phi_max (Stolen & Lin 1978)"))
    print(f"  gamma P0 L            = phi_max  : {phi:.4f} rad ({phi / np.pi:.2f} pi)")
    print(f"  measured fringe peaks            : {metrics['n_peaks']}")
    print(f"  closed form floor(phi_max/pi)+1  : {metrics['n_peaks_expected']}")
    print(
        f"  measured max |dW| vs analytic    : {metrics['max_abs_spectrum_diff']:.3e}"
        f"   (tolerance 5e-3)"
    )
    print(f"  -> {_tolerance_note('check_spm (5e-3, normalised spectrum)')}")

    w = np.asarray(grid.w)
    band = np.abs(w) < 8.0 / T0
    W_band = W_num[band]
    peaks, _ = find_peaks(W_band, prominence=0.05)
    return dict(
        metrics=metrics, w=w[band], W_num=W_band, W_an=W_an[band], peak_w=w[band][peaks]
    )


def run_mi() -> dict:
    """Sideband power gain vs g(Omega), plus the analytic gain peak."""
    grid = TemporalGrid(N=GRID_N, Tmax=Time(GRID_TMAX, "s"))
    metrics = check_mi(grid=grid, **MI)

    beta2, gamma, P = MI["beta2"], MI["gamma"], MI["pump_power"]
    omega_c = np.sqrt(-4.0 * gamma * P / beta2)
    omega_peak = omega_c / np.sqrt(2.0)
    g_max = 2.0 * gamma * P
    g_at_peak = mi_gain_of(beta2, gamma, P, omega_peak)

    print(_rule("2. MI — sideband gain vs linear stability (Agrawal 5.1.9)"))
    print(f"  measured gain                   : {metrics['g_measured']:.5g} 1/m")
    print(f"  closed form g(Omega_probe)      : {metrics['g_reference']:.5g} 1/m")
    print(f"  Omega_c                         : {omega_c:.4e} rad/s")
    print(f"  closed-form peak Omega_c/sqrt(2): {omega_peak:.4e} rad/s")
    print(f"  peak frequency (THz)           : {omega_peak / 2 / np.pi * 1e-12:.4f}")
    print(f"  g at the peak                   : {g_at_peak:.5g} 1/m")
    print(f"  closed form g_max = 2 gamma P   : {g_max:.5g} 1/m")
    print(f"  -> {_tolerance_note('check_mi (15 %, gain)')}")
    print("  power-gain convention: sideband *intensity* grows as e^{g z}.")

    w = np.asarray(grid.w)
    band = np.abs(w) < 1.5 * omega_c
    g_curve = np.array([mi_gain_of(beta2, gamma, P, float(x)) for x in w[band]])
    return dict(
        metrics=metrics,
        w=w[band],
        g=g_curve,
        omega_c=omega_c,
        omega_peak=omega_peak,
        g_max=g_max,
    )


def run_soliton() -> dict:
    """Shape retention and energy conservation over one soliton period."""
    grid = TemporalGrid(N=GRID_N, Tmax=Time(GRID_TMAX, "s"))
    metrics = check_soliton(grid=grid, periods=1.0, **SOLITON)

    beta2, gamma, t0 = SOLITON["beta2"], SOLITON["gamma"], SOLITON["t0"]
    P0 = abs(beta2) / (gamma**2 * t0**2)  # N = 1
    print(_rule("3. Fundamental soliton — shape returns after z_sol (Agrawal 5.2)"))
    print(f"  soliton order N = gamma P0 T0^2/|beta2| : 1 (P0 = {P0:.4g} W)")
    print(f"  z_sol = (pi/2) L_D                      : {metrics['z_sol']:.4g} m")
    print(f"  periods propagated                      : {metrics['periods']:g}")
    print(
        f"  measured shape overlap |<A_in,A_out>|   : {metrics['shape_overlap']:.6f}"
        f"   (closed form: 1.0, tolerance 1-0.02)"
    )
    print(
        f"  measured energy ratio                   : {metrics['energy_ratio']:.8f}"
        f"   (closed form: 1.0)"
    )
    print(f"  -> {_tolerance_note('check_soliton (2 %, 1 - overlap)')}")

    # Analytic N = 1 field, plus the propagated field for the figure. This
    # re-runs the same deck check_soliton just validated — check_soliton builds
    # its own solver internally and keeps no state, and re-propagating is
    # cheaper than the closed form being illustrated with a stand-in curve.
    fwhm = 2.0 * np.arccosh(np.sqrt(2.0)) * t0
    env = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(P0), fwhm=Time(fwhm, "s"))
    t = grid.t
    A_in = np.sqrt(P0) / np.cosh(t / t0)  # closed form: N = 1 sech field
    from photonics_helper.gnlse import GNLSESolver

    L = metrics["z_sol"]
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(SOLITON["wavelength_m"], "m"),
    )
    fiber = FiberProfile.from_gamma(
        gamma=gamma, n2=2.6e-20, omega0=OMEGA0, length=Length(L, "m")
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2 * 1e24]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
        step_size=Length(L / 2000.0, "m"),
    )
    solver.propagate(2000, nsaves=3)
    A_out = np.asarray(solver.evolution[-1].envelope_field)

    m = np.abs(t) < 4.0 * t0
    return dict(
        metrics=metrics,
        t=t[m],
        in_shape=np.abs(A_in[m]) ** 2 / np.max(np.abs(A_in) ** 2),
        out_shape=np.abs(A_out[m]) ** 2 / np.max(np.abs(A_in) ** 2),
    )


def run_ssfs() -> dict:
    """Raman SSFS rate vs Gordon's analytic law."""
    grid = TemporalGrid(N=SSFS_GRID_N, Tmax=Time(SSFS_GRID_TMAX, "s"))
    spec = RamanSpec(
        name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
    )
    response = RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)
    beta2, gamma, t0 = SSFS["beta2"], SSFS["gamma"], SSFS["t0"]
    peak = abs(beta2) / (gamma * t0**2)  # N = 1
    metrics = check_gordon_ssfs(
        peak_power=peak,
        raman_response=response,
        grid=grid,
        length=SSFS["length"],
        wavelength_m=SSFS["wavelength_m"],
        t0=t0,
        beta2=beta2,
        gamma=gamma,
    )
    rate = gordon_ssfs_rate(beta2, gamma, peak, t0, metrics["T_R"])

    print(_rule("4. Raman SSFS — measured shift vs Gordon's law (Opt. Lett. 11, 662)"))
    print(f"  T_R = f_R int t h_R(t) dt           : {metrics['T_R'] * 1e15:.4f} fs")
    print(f"  closed form dOmega/dz               : {rate:.5g} rad/s/m")
    print(
        f"  closed-form shift over {SSFS['length']} m      : "
        f"{metrics['analytic_shift_nm']:.5f} nm (Stokes: positive)"
    )
    print(
        f"  measured shift                      : {metrics['measured_shift_nm']:.5f} nm"
    )
    print(
        f"  ratio measured / analytic           : {metrics['ratio']:.4f}"
        f"   (tolerance 0.75 - 1.25)"
    )
    print(f"  -> {_tolerance_note('check_gordon_ssfs (ratio 0.75-1.25)')}")
    return metrics


# ── 2. Convergence study ─────────────────────────────────────────────────────


def build_solver(**overrides):
    """Factory for ``convergence_study``: an N = 3 soliton, fissioning.

    ``convergence_study`` calls ``build_solver(**shared, **refinement)``, so
    every key in ``CONV_SHARED`` and in the refinement dict reaches this
    function. The ladder refines ``num_steps`` (the ``z`` step, since
    ``step_size = L / num_steps``) while holding ``N`` fixed inside each
    refinement entry; the returned :class:`SplitStepEngine` exposes ``.A`` and
    ``.grid``, which is all the default observables need.
    """
    cfg = {**CONV_SHARED, **overrides}
    N = int(cfg["N"])
    num_steps = int(cfg["num_steps"])
    beta2 = float(cfg["beta2"])
    gamma = float(cfg["gamma"])
    t0 = float(cfg["t0"])
    Tmax = 8e-12

    grid = TemporalGrid(N=N, Tmax=Time(Tmax, "s"))
    peak = 9.0 * abs(beta2) / (gamma * t0**2)  # soliton order N = 3
    length = 2.0 * (np.pi / 2) * t0**2 / abs(beta2)
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(peak), pulse_width=Time(t0, "s")
    )
    pulse = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(cfg["wavelength_m"], "m")
    )
    fiber = FiberProfile.from_gamma(
        gamma=gamma, n2=2.6e-20, omega0=OMEGA0, length=Length(length, "m")
    )
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2 * 1e24]),  # s^2/m -> ps^2/m
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
        step_size=Length(length / num_steps, "m"),
    )
    engine.propagate(num_steps=num_steps)
    return engine


def spectral_rms_width(result) -> float:
    """A custom observable: rms spectral width in nm (a mapping, not a name)."""
    A = np.asarray(result.A)
    grid = result.grid
    spec = np.abs(grid.fft(A)) ** 2
    w = np.asarray(grid.w)
    omega0 = float(result.omega0)
    dnu_nm = -(omega0**2) * w / (2 * np.pi * C_MS) * 1e9
    return float(np.sqrt(np.sum(spec * dnu_nm**2) / np.sum(spec)))


#: The *mapping* form of ``observables`` — every default name plus one custom
#: callable, which is how a caller adds an observable of their own. (The other
#: accepted form is a plain sequence of names into DEFAULT_OBSERVABLES, e.g.
#: ``["peak_intensity", "rms_bandwidth"]``; a sequence entry that is not a
#: default name is a KeyError by design.)
OBSERVABLES: dict[str, object] = {
    **DEFAULT_OBSERVABLES,
    "spectral_rms_width_nm": spectral_rms_width,
}


def print_report(report: ConvergenceReport, refinements, title: str) -> None:
    print(_rule(title))
    header = "  observable             " + "".join(
        f"{f'r{i}' if i else '-':>14}" for i in range(len(refinements))
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for obs in report.observables:
        row = f"  {obs.name:<21} " + "".join(f"{v:14.6g}" for v in obs.values)
        print(row)
    print()
    print(f"  {'observable':<21} {'last change':>14} {'tolerance':>10}  verdict")
    for obs in report.observables:
        flag = "converged" if obs.converged else "NOT converged"
        print(
            f"  {obs.name:<21} {obs.change_last:14.3e} {CONV_TOLERANCE:10.1e}  {flag}"
        )
    print(
        "\n  ladder: "
        + ", ".join(f"N={r['N']} dz=L/{r['num_steps']}" for r in refinements)
    )
    print(f"  -> {_tolerance_note('convergence_study (2e-3, relative change)')}")


def run_convergence() -> tuple[ConvergenceReport, ConvergenceReport]:
    good = convergence_study(
        build_solver,
        refinements=CONV_REFINEMENTS,
        observables=OBSERVABLES,
        tolerance=CONV_TOLERANCE,
        shared=CONV_SHARED,
    )
    print_report(
        good, CONV_REFINEMENTS, "5. Convergence — an adequately refined ladder"
    )

    bad = convergence_study(
        build_solver,
        refinements=BAD_REFINEMENTS,
        observables=OBSERVABLES,
        tolerance=CONV_TOLERANCE,
        shared=CONV_SHARED,
    )
    print(_rule("5b. Failure path — an under-resolved ladder does NOT converge"))
    failed = [o.name for o in bad.observables if not o.converged]
    for obs in bad.observables:
        mark = "  " if obs.converged else "**"
        print(
            f" {mark} {obs.name:<21} last change {obs.change_last:.3e}  "
            + ("converged" if obs.converged else "NOT converged")
        )
    print(
        "\n  ladder: "
        + ", ".join(f"N={r['N']} dz=L/{r['num_steps']}" for r in BAD_REFINEMENTS)
    )
    print(
        f"  {len(failed)} of {len(bad.observables)} observables flagged "
        f"converged=False: {', '.join(failed)}"
    )
    assert not bad.converged and failed, "the under-resolved ladder must fail"
    return good, bad


def run_failing_check() -> None:
    """A check_mi run whose closed form is wrong must raise — caught, printed.

    ``beta2`` enters ``check_mi`` twice — once as the dispersion the solver
    propagates, once inside the closed-form reference gain. A wrong ``beta2``
    therefore does *not* fail on its own: the run and its reference are wrong
    together, which is exactly why ``check_mi`` reports both numbers. The
    deviation that shows up here is the residual solver error on the wrong
    deck, so the tolerance is tightened to expose it.
    """
    print(_rule("6. Failure path — a ValidationFailure, caught"))
    grid = TemporalGrid(N=GRID_N, Tmax=Time(GRID_TMAX, "s"))
    wrong_beta2 = 0.6 * MI["beta2"]
    tight = 0.02
    print(
        f"  running check_mi with beta2 = {wrong_beta2:.3e} instead of "
        f"{MI['beta2']:.3e}, rel_tolerance = {tight:g}"
    )
    try:
        check_mi(grid=grid, rel_tolerance=tight, **{**MI, "beta2": wrong_beta2})
    except ValidationFailure as exc:
        print(f"  caught {type(exc).__name__} [case={exc.case}]:")
        for line in str(exc).splitlines():
            print(f"    {line}")
        print("  the harness fails loudly ✓")
    else:
        raise AssertionError(
            f"check_mi passed at rel_tolerance={tight:g} with a deliberately "
            "wrong beta2; the tolerance is not being exercised"
        )


def negative_control() -> None:
    """Impossible tolerance -> the check reports failure (task 3.2)."""
    print(_rule("7. Negative control — the tolerances actually bite"))
    grid = TemporalGrid(N=GRID_N, Tmax=Time(GRID_TMAX, "s"))
    try:
        check_soliton(grid=grid, periods=1.0, tolerance=IMPOSSIBLE_TOLERANCE, **SOLITON)
    except ValidationFailure as exc:
        print(
            f"  check_soliton(tolerance={IMPOSSIBLE_TOLERANCE:g}) -> "
            f"ValidationFailure: {str(exc).splitlines()[0]}"
        )
        print(
            f"  overlap deficit that a real run carries: "
            f"{exc.metrics['shape_overlap']:.6f} vs required 1.0"
        )
    else:
        raise AssertionError(
            f"check_soliton passed at tolerance={IMPOSSIBLE_TOLERANCE:g}; the "
            "tolerance is not being exercised"
        )
    print("  a passing check therefore means something ✓")


# ── Figures ──────────────────────────────────────────────────────────────────


def checks_figure(spm, mi, soliton, ssfs) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.2))

    ax = axes[0]
    nu = spm["w"] / (2 * np.pi) * 1e-12  # THz offset
    ax.plot(nu, spm["W_an"], lw=1.6, label="analytic $\\varphi_{max}$ field")
    ax.plot(nu, spm["W_num"], lw=0.9, ls="--", label="propagated (Kerr only)")
    ax.plot(
        spm["peak_w"] / (2 * np.pi) * 1e-12,
        np.interp(spm["peak_w"], spm["w"], spm["W_num"]),
        "o",
        ms=4,
        color="k",
        label=f"{len(spm['peak_w'])} fringes in view "
        f"(check: {spm['metrics']['n_peaks']} of "
        f"{spm['metrics']['n_peaks_expected']} expected)",
    )
    ax.set_title(
        f"SPM: $\\varphi_{{max}}$ = {SPM['phi_max'] / np.pi:.0f}$\\pi$\n"
        f"max |$\\Delta W$| = {spm['metrics']['max_abs_spectrum_diff']:.1e}"
    )
    ax.set_xlabel("frequency offset (THz)")
    ax.set_ylabel("normalised spectrum")
    ax.legend(fontsize=7)

    ax = axes[1]
    nu = mi["w"] / (2 * np.pi) * 1e-12
    ax.plot(nu, mi["g"], lw=1.6, label="closed form $g(\\Omega)$")
    ax.axvline(
        mi["omega_peak"] / (2 * np.pi) * 1e-12,
        color="grey",
        ls=":",
        label="$\\Omega_c/\\sqrt{2}$",
    )
    ax.axhline(mi["g_max"], color="grey", ls="--", lw=0.8, label="$2\\gamma P$")
    ax.plot(
        [mi["metrics"]["omega_probe"] / (2 * np.pi) * 1e-12],
        [mi["metrics"]["g_measured"]],
        "o",
        ms=7,
        color="crimson",
        label=f"measured {mi['metrics']['g_measured']:.3g} 1/m",
    )
    ax.plot(
        [mi["metrics"]["omega_probe"] / (2 * np.pi) * 1e-12],
        [mi["metrics"]["g_reference"]],
        "x",
        ms=8,
        color="k",
        label=f"reference {mi['metrics']['g_reference']:.3g} 1/m",
    )
    ax.set_title(
        f"MI: power gain vs $\\Omega$\n"
        f"$\\Omega_c$ = {mi['omega_c'] / 2 / np.pi * 1e-12:.2f} THz, "
        f"$g_{{max}}$ = {mi['g_max']:.3g} 1/m"
    )
    ax.set_xlabel("frequency offset (THz)")
    ax.set_ylabel("gain (1/m)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ps = soliton["t"] * 1e15
    ax.plot(ps, soliton["in_shape"], lw=1.8, label="input (analytic sech$^2$)")
    ax.plot(
        ps,
        soliton["out_shape"],
        lw=1.0,
        ls="--",
        label=f"after $z_{{sol}}$ (overlap {soliton['metrics']['shape_overlap']:.5f})",
    )
    ax.set_title(
        f"Soliton: shape returns after $z_{{sol}}$\n"
        f"$z_{{sol}}$ = {soliton['metrics']['z_sol']:.3g} m"
    )
    ax.set_xlabel("retarded time (fs)")
    ax.set_ylabel("normalised power")
    ax.legend(fontsize=7)

    ax = axes[3]
    vals = [ssfs["measured_shift_nm"], ssfs["analytic_shift_nm"]]
    bars = ax.bar(
        ["measured", "Gordon\n1 − 8|β₂|T_R/(15T₀⁴)"],
        vals,
        color=["crimson", "steelblue"],
    )
    for bar, v in zip(bars, vals):
        ax.annotate(
            f"{v:.4f} nm",
            (bar.get_x() + bar.get_width() / 2, v),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Raman SSFS over {SSFS['length']} m\nratio = {ssfs['ratio']:.3f}")
    ax.set_ylabel("peak-wavelength shift (nm)")
    ax.set_ylim(0, max(vals) * 1.35)

    fig.suptitle("Closed-form checks — photonics_helper.gnlse_validation", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS["checks"], dpi=130, bbox_inches="tight")
    plt.close(fig)


def convergence_figure(good, bad) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    names = [o.name for o in good.observables]
    cmap = plt.get_cmap("viridis")

    for ax, report, refinements, title in (
        (axes[0], good, good.refinements, "refined ladder"),
        (axes[1], bad, bad.refinements, "under-resolved ladder"),
    ):
        by_name = {o.name: o for o in report.observables}
        # x = refinement index 2..n: the first change is undefined by design.
        xs = np.arange(2, len(refinements) + 1)
        for k, name in enumerate(names):
            ys = [c if c is not None else np.nan for c in by_name[name].changes][1:]
            ax.plot(
                xs, ys, "o-", color=cmap(k / max(len(names) - 1, 1)), lw=1.4, label=name
            )
        ax.axhspan(1e-18, CONV_TOLERANCE, color="0.86", zorder=0)
        ax.axhline(CONV_TOLERANCE, color="0.35", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                f"r{i}\nN={r['N']}\ndz=L/{r['num_steps']}"
                for i, r in enumerate(refinements, start=1)
            ][1:],
            fontsize=8,
        )
        n_conv = sum(o.converged for o in report.observables)
        ax.set_title(f"{title}: {n_conv}/{len(report.observables)} converged")
        ax.set_xlabel("refinement")
        ax.grid(alpha=0.25, which="both")
        if ax is axes[0]:
            ax.set_ylabel("relative change vs previous refinement")
            ax.legend(fontsize=8, ncol=2)
            ax.annotate(
                f"shaded: change < {CONV_TOLERANCE:g} (converged)",
                (0.02, 0.06),
                xycoords="axes fraction",
                fontsize=8,
            )

    fig.suptitle(
        "convergence_study — the same run, two ladders "
        "(N = 3 soliton over two soliton periods)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(FIGS["convergence"], dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    t0_run = time.perf_counter()
    print(_rule("0. Verifying the GNLSE solver against four closed forms"))
    print("  convention: i A_z = -beta2 tau A_tt - gamma |A|^2 A, exp(-iwt) FFT")
    print("  power = |A|^2 (W); a bright soliton needs beta2 < 0")
    print(f"  known observables: {sorted(DEFAULT_OBSERVABLES)}")

    spm = run_spm()
    mi = run_mi()
    soliton = run_soliton()
    ssfs = run_ssfs()
    good, bad = run_convergence()
    run_failing_check()
    negative_control()

    print(_rule("Summary"))
    print(
        f"  SPM fringe peaks            : {spm['metrics']['n_peaks']} "
        f"(closed form {spm['metrics']['n_peaks_expected']})"
    )
    print(
        f"  MI gain                     : {mi['metrics']['g_measured']:.4g} "
        f"(closed form {mi['metrics']['g_reference']:.4g} 1/m; "
        f"peak 2 gamma P = {mi['g_max']:.4g})"
    )
    print(
        f"  soliton overlap             : "
        f"{soliton['metrics']['shape_overlap']:.6f} (closed form 1.0)"
    )
    print(f"  Gordon SSFS ratio           : {ssfs['ratio']:.4f} (closed form 1.0)")
    print(f"  refined ladder converged    : {good.converged}")
    print(f"  coarse ladder converged     : {bad.converged} (expected False)")
    print("  ValidationFailure caught    : yes")
    print("  every check passed ✓")
    print(
        f"\n  runtime {time.perf_counter() - t0_run:.1f} s "
        f"(check_gordon_ssfs at 4000 split steps, N = {SSFS_GRID_N}, and "
        f"check_mi at 4000 split steps, N = {GRID_N}, dominate)"
    )

    checks_figure(spm, mi, soliton, ssfs)
    convergence_figure(good, bad)

    print("\nGenerated files:")
    for p in FIGS.values():
        print(f"  {p.relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
