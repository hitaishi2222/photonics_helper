"""
Example: Optical Wave Breaking (normal dispersion)
==================================================

Demonstrates ``photonics_helper.wave_breaking`` — where a pulse breaks, and how
to tell from a propagation run that it did.

The regime: normal dispersion, NOT anomalous
--------------------------------------------
This module implements optical wave breaking of a Gaussian in **normal
dispersion** (β₂ > 0). That is the *opposite* regime from the soliton fission
of ``examples/13``, and the library enforces the distinction: both
``wave_breaking_distance`` and ``WaveBreaking`` raise ``ValueError`` for
β₂ ≤ 0. If you arrive here from the soliton examples, discard the soliton
order — it does not exist here, and ``L_D/L_NL`` is **not** the breaking
criterion.

The analytic prediction
-----------------------
For a Gaussian ``A(0,T) = √P₀ exp(-T²/2T₀²)`` (T₀ the 1/e intensity radius),
SPM imprints the chirp

.. math::

    \delta\omega(T) = -\frac{2\gamma P_0 z}{T_0^2}\,T\,e^{-T^2/T_0^2},

and dispersion maps every spectral component to ``T' = T + β₂ z δω(T)``. Wave
breaking is where that map stops being monotonic — the chirp-folding condition
``1 + β₂ z ∂δω/∂T = 0`` — which integrates to

.. math::

    z_{WB} = \frac{T_0}{\sqrt{2\beta_2\gamma P_0}}
           = \frac{e^{3/4}}{2}\sqrt{L_D L_{NL}},
    \qquad L_D = \frac{T_0^2}{\beta_2},\quad L_{NL} = \frac{1}{\gamma P_0}.

So the ordering parameter of this example is the **breaking ratio**

.. math::  r = \frac{L}{z_{WB}},

and breaking is expected for ``r > 1``. Note how little ``r`` depends on the
individual scales: ``L_D`` and ``L_NL`` enter only through their geometric
mean. For the reference case below ``L_D/L_NL = 75``, which would look
"massively supercritical" if mistaken for an ordering parameter — it is not
one, and the example prints it alongside an explicit note.

Measuring it in a run
---------------------
The dimensionless edge steepness

.. math::  S = \frac{\max_T |\partial I/\partial T|\,T_0}{I_{peak}}

starts at the Gaussian value ``gaussian_edge_steepness = √2 e^{-1/2} ≈ 0.8578``
and rises as the edges steepen. Two detectors watch it:

* ``detect_steepening_onset`` fires when ``S > 1.10 × 0.8578 = 0.9435``. That
  10 % margin is a **choice, not physics** — it decides how much of the run you
  have to propagate before the detector fires, so the example prints it and
  then sweeps it.
* ``detect_oscillation_onset`` fires when a profile grows a second local
  maximum with prominence ≥ 1 % of the peak — the flat top splitting.

Both return ``nan`` when nothing happens, and the example asserts that for the
sub-critical cases rather than coercing a number out of a detector that has
none.
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

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.wave_breaking import (
    WaveBreaking,
    detect_oscillation_onset,
    detect_steepening_onset,
    dispersion_length,
    edge_steepness,
    gaussian_edge_steepness,
    nonlinear_length,
    wave_breaking_distance,
)


def _folding_count(field: NDArray, t: NDArray) -> int:
    """Sign changes of d|A|^2/dt -- this example's own "did it fold?" measure.

    A sub-critical Gaussian has exactly one (the peak). A folded profile has
    many: each shock carries oscillations, and the flat top between them is
    where ``detect_oscillation_onset``'s 1 %-of-peak prominence rule looks and
    finds a single peak. This is deliberately NOT a library function; it is here
    to show what the library detectors miss, not to replace them.
    """
    intensity = np.abs(np.asarray(field)) ** 2
    return int(np.sum(np.diff(np.sign(np.gradient(intensity, t))) != 0))


# ── Reference case: standard single-mode silica ────────────────────
BETA2 = 20.0e-27  # s^2/m, NORMAL dispersion (+20 ps^2/km)
GAMMA = 1.5e-3  # 1/(W m)
P0 = 10.0  # W
T0 = 10.0e-12  # s, 1/e intensity radius
LAM0 = 1550e-9

# The breaking-ratio ladder r = L / z_WB. Folding is expected for r > 1.
R_LADDER = [0.5, 0.9, 1.0, 1.5, 3.0, 6.0]
# The only case whose steepening detector stays silent: S never clears 0.9435.
NO_STEEPENING = [0.5]
# Oscillations are not detectable until the ripples exceed the prominence rule.
OSCILLATION_SILENT_MAX_R = 3.0
OSCILLATION_FIRES_AT_R = 6.0
RESOLUTION_R = 1.5
N_STEPS = 300

STEEPENING_FACTORS = [1.02, 1.10, 1.25]
PROMINENCES = [0.001, 0.01, 0.1]

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "theory": OUT_DIR / "34_wave_breaking_theory.png",
    "onset": OUT_DIR / "34_breaking_onset_vs_z.png",
}


def _rule(title: str) -> str:
    return "\n" + "=" * 66 + f"\n {title}\n" + "=" * 66


def _fmt(z: float) -> str:
    """A distance in metres, or an explicit statement that there is none."""
    return "no onset (nan)" if not np.isfinite(z) else f"{z:.2f} m"


def _ratio(z: float, z_wb: float) -> str:
    """A distance expressed in units of z_WB (dimensionless)."""
    return "none" if not np.isfinite(z) else f"{z / z_wb:.2f} z_WB"


# ── 1. The analytic side ───────────────────────────────────────────
def analytic_panel() -> tuple[WaveBreaking, dict]:
    print(_rule("1. The analytic prediction"))

    wb = WaveBreaking(beta2=BETA2, gamma=GAMMA, P0=P0, T0=T0)
    z_wb_fn = wave_breaking_distance(BETA2, GAMMA, P0, T0)

    print("  regime: NORMAL dispersion (beta2 > 0). The anomalous soliton")
    print("  fission picture of examples/13 does NOT apply here.")
    print(f"  beta2  = {BETA2 * 1e24:+.4f} ps^2/m  (normal)")
    print(
        f"  gamma  = {GAMMA * 1e3:.2f} 1/(W km),  P0 = {P0:.1f} W,  "
        f"T0 = {T0 * 1e12:.1f} ps"
    )
    print(f"  L_D    = T0^2/beta2    = {wb.L_D / 1e3:.3f} km")
    print(f"  L_NL   = 1/(gamma*P0) = {wb.L_NL:.3f} m")
    print(f"  sqrt(L_D*L_NL)        = {wb.sqrt_LD_LNL:.3f} m")
    print(f"  z_WB   = (e^(3/4)/2)*sqrt(L_D*L_NL) = {wb.z_WB:.3f} m")
    print(f"    = T0/sqrt(2*beta2*gamma*P0)      = {z_wb_fn:.3f} m")
    print(f"  L_D/L_NL = {wb.L_D / wb.L_NL:.1f}  <- NOT a breaking criterion;")
    print("             it enters only through the geometric mean sqrt(L_D*L_NL).")

    # The free functions must agree with the object's properties.
    assert abs(dispersion_length(BETA2, T0) - wb.L_D) < 1e-9 * wb.L_D
    assert abs(nonlinear_length(GAMMA, P0) - wb.L_NL) < 1e-9 * wb.L_NL
    assert abs(z_wb_fn - wb.z_WB) < 1e-9 * wb.z_WB
    print("\n  dispersion_length / nonlinear_length / wave_breaking_distance")
    print("  agree with the WaveBreaking properties L_D, L_NL, z_WB   [asserted]")

    # Pin the closed form's SCALINGS, not just its value at one point. These two
    # assertions are what catch a wrong exponent in the formula, which a single
    # spot check cannot. (Both were written wrong first and caught by the
    # library: z_WB scales as 1/sqrt(gamma*P0) and 1/sqrt(beta2), not with
    # positive exponents.)
    w_p4 = WaveBreaking(beta2=BETA2, gamma=GAMMA, P0=4.0 * P0, T0=T0)
    w_b2 = WaveBreaking(beta2=2.0 * BETA2, gamma=GAMMA, P0=P0, T0=T0)
    r_p = w_p4.z_WB / wb.z_WB
    r_b2 = w_b2.z_WB / wb.z_WB
    assert abs(r_p - 0.5) < 1e-12, f"z_WB power scaling wrong: {r_p}"
    assert abs(r_b2 - 1.0 / np.sqrt(2.0)) < 1e-12, f"z_WB beta2 scaling wrong: {r_b2}"
    print("\n  the closed form's scalings, which a single spot check cannot pin:")
    print(f"    4x P0 (so L_NL / 4)  -> z_WB x{r_p:.6f} = 1/2         [asserted]")
    print(f"    2x beta2              -> z_WB x{r_b2:.6f} = 1/sqrt(2)  [asserted]")
    print("    i.e. z_WB ~ 1/sqrt(gamma*P0*beta2): strengthening the nonlinearity or")
    print("    the dispersion SHORTENS the distance to breaking.")

    # The sign convention is enforced, not merely documented.
    print("\n  the normal-dispersion requirement is enforced:")
    for bad, label in ((-BETA2, "WaveBreaking"), (0.0, "WaveBreaking")):
        try:
            WaveBreaking(beta2=bad, gamma=GAMMA, P0=P0, T0=T0)
        except ValueError as exc:
            print(f"    {label}(beta2={bad:+.1e}) -> ValueError: {exc}")
    try:
        wave_breaking_distance(-BETA2, GAMMA, P0, T0)
    except ValueError as exc:
        print(f"    wave_breaking_distance(beta2={-BETA2:+.1e}) -> ValueError: {exc}")

    print(
        f"\n  gaussian_edge_steepness = {gaussian_edge_steepness:.7f}  "
        "(analytic, transform-limited Gaussian)"
    )
    print(
        f"  steepening detector fires at S > 1.10 x reference = "
        f"{1.10 * gaussian_edge_steepness:.6f}"
    )
    print(
        "  oscillation detector fires on a 2nd local maximum, prominence >= 1% of peak"
    )
    print("    note: edge_steepness differentiates |A|^2 with np.gradient's central")
    print("    difference, so its Gaussian value matches the analytic reference only")
    print("    to grid resolution (see section 2), not to machine precision.")

    print("\n  breaking ratio ladder (r = L / z_WB; breaking expected for r > 1):")
    for r in R_LADDER:
        verdict = "BREAKS" if r > 1.0 else ("marginal" if r == 1.0 else "no break")
        print(f"    r = {r:4.1f}   L = {r * wb.z_WB:8.2f} m   {verdict}")

    return wb, {"L_D": wb.L_D, "L_NL": wb.L_NL, "z_WB": wb.z_WB}


# ── 2. One propagation ─────────────────────────────────────────────
def _propagate(wb: WaveBreaking, r: float, n_steps: int = N_STEPS) -> dict:
    """Propagate a Gaussian to ``L = r * z_WB`` and run the diagnostics."""
    L = r * wb.z_WB
    grid = TemporalGrid(N=2048, Tmax=Time(12.0 * T0, "s"))
    # Envelope's Gaussian is A0 exp(-t^2/2T0^2), so its intensity has a 1/e
    # radius of exactly T0 -- the same T0 the wave_breaking module uses. The
    # assertion below is what verifies that convention rather than assuming it.
    env = Envelope(
        shape="gaussian", peak_amplitude=float(np.sqrt(P0)), pulse_width=Time(T0, "s")
    )
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(LAM0, "m"))

    fiber = FiberProfile.from_gamma(
        gamma=GAMMA,
        n2=2.7e-20,
        omega0=float(wave.central_frequency),
        length=Length(L, "m"),
    )
    # from_gamma solves A_eff from the target gamma; confirm the engine really
    # runs at GAMMA and not at some gamma implied by a hand-picked A_eff.
    gamma_eff = fiber.n2 * float(wave.central_frequency) / (C_MS * fiber.A_eff.as_m2)
    assert abs(gamma_eff - GAMMA) / GAMMA < 1e-3, (
        f"fiber gamma {gamma_eff} does not match the target {GAMMA}"
    )
    dz = L / n_steps
    solver = SplitStepEngine(
        pulse=wave,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),  # s^2/m -> ps^2/m
        include_raman=False,
        include_self_steepening=False,
        step_size=Length(dz, "m"),
    )
    solver.propagate(num_steps=n_steps, nsaves=60)

    z_array = np.linspace(0.0, L, len(solver.evolution))
    fields = [w.envelope_field for w in solver.evolution]

    # The Gaussian's own steepness must match the reference. It cannot match it
    # to machine precision: edge_steepness differentiates the intensity with
    # np.gradient's central difference, so the residual is finite-grid error.
    s0 = edge_steepness(field=fields[0], t=grid.t, T0=T0)
    s0_err = abs(s0 - gaussian_edge_steepness)
    assert s0_err < 1e-3, (
        f"initial Gaussian steepness {s0} is not the reference "
        f"{gaussian_edge_steepness} (T0 convention mismatch?)"
    )

    result = wb.analyze(z_array, fields, grid.t)
    return {
        "r": r,
        "L": L,
        "grid": grid,
        "z": z_array,
        "fields": fields,
        "steepness": result["steepness"],
        "s0": s0,
        "s0_err": s0_err,
        "gamma_eff": gamma_eff,
        "folds": _folding_count(fields[-1], grid.t),
        "osc_prom": [
            detect_oscillation_onset(z_array, fields, prominence=p) for p in PROMINENCES
        ],
        **result,
    }


def ladder_panel(wb: WaveBreaking) -> dict:
    print(_rule("2. The breaking-ratio ladder — analytic vs measured"))

    runs = []
    for r in R_LADDER:
        run = _propagate(wb, r)
        runs.append(run)
        rel = (
            run["z_onset_m"] / wb.z_WB
            if np.isfinite(run["z_onset_m"])
            else float("nan")
        )
        print(f"\n  r = {r:.1f}   L = {run['L']:.1f} m   (z_WB = {wb.z_WB:.1f} m)")
        if r == R_LADDER[0]:
            print(
                f"    initial S = {run['s0']:.7f} vs analytic "
                f"{gaussian_edge_steepness:.7f} (diff {run['s0_err']:.1e}, "
                f"grid-limited)  [asserted < 1e-3]"
            )
            print(
                f"    fiber gamma = {run['gamma_eff']:.6e} 1/(W m) vs target "
                f"{GAMMA:.6e}                          [asserted < 1e-3 rel]"
            )
        print(
            f"    analyze(): peak steepness {run['peak_steepness']:.4f}"
            f"  (reference {gaussian_edge_steepness:.4f})"
        )
        print(
            f"    steepening onset : {_fmt(run['z_onset_m'])}"
            + (f"   = {rel:.2f} z_WB" if np.isfinite(rel) else "")
        )
        print(f"    oscillation onset: {_fmt(run['z_oscillation_m'])}")

    # ── Assert exactly what the library actually reports ─────────
    # r = 0.5: the steepening detector never clears its 10% threshold.
    for r in NO_STEEPENING:
        run = next(x for x in runs if x["r"] == r)
        assert not np.isfinite(run["z_onset_m"]), (
            f"r = {r} should leave the steepening detector silent, but it fired "
            f"at {run['z_onset_m']}"
        )
        print(f"\n  r = {r}: steepening onset nan, oscillation onset nan.   [asserted]")
        print("    Nothing is coerced out of a detector that saw nothing.")

    # The oscillations stay invisible up to r = 3 -- but the profile HAS folded.
    silent = [x for x in runs if x["r"] <= OSCILLATION_SILENT_MAX_R]
    for run in silent:
        assert not np.isfinite(run["z_oscillation_m"]), (
            f"r = {run['r']} should not yet show detectable oscillations, but the "
            f"detector fired at {run['z_oscillation_m']}"
        )
    print(
        f"\n  oscillation onset is nan for every r <= "
        f"{OSCILLATION_SILENT_MAX_R}   [asserted]"
    )

    far = next(x for x in runs if x["r"] == OSCILLATION_FIRES_AT_R)
    assert np.isfinite(far["z_oscillation_m"]), (
        f"r = {OSCILLATION_FIRES_AT_R} should show detectable oscillations"
    )
    print(
        f"  r = {OSCILLATION_FIRES_AT_R}: oscillation onset fires at "
        f"{_fmt(far['z_oscillation_m'])} "
        f"({far['z_oscillation_m'] / wb.z_WB:.2f} z_WB)   [asserted]"
    )

    # ── What the library detectors do NOT report ────────────────
    print("\n  the detector that is not a wave-breaking detector:")
    r09 = next(x for x in runs if x["r"] == 0.9)
    print(
        f"    r = 0.9 is NOT breaking, yet the steepening detector fires at "
        f"{_fmt(r09['z_onset_m'])}"
    )
    print(
        f"    ({r09['z_onset_m'] / wb.z_WB:.2f} z_WB). Its threshold is only "
        f"S > {1.10 * gaussian_edge_steepness:.4f},"
    )
    print("    a 10% rise in edge steepness, which SPM produces long before the")
    print("    chirp map folds. Every detected onset above is ~0.6 z_WB for the")
    print("    same reason, so quoting one as 'the wave-breaking distance'")
    print("    understates the analytic folding distance by ~40%.")

    print("\n  folding the edges without the oscillation detector noticing:")
    print("      r    folds (dI/dt sign changes)    oscillation onset    peak S")
    for run in runs:
        note = (
            "  <- folded but silent"
            if (
                run["r"] > 1.0
                and run["folds"] > 5
                and not np.isfinite(run["z_oscillation_m"])
            )
            else ""
        )
        print(
            f"    {run['r']:4.1f}        {run['folds']:3d}"
            f"                        {_fmt(run['z_oscillation_m']):>15}"
            f"     {run['peak_steepness']:6.3f}{note}"
        )
    r15 = next(x for x in runs if x["r"] == 1.5)
    assert r15["folds"] > 5, "r = 1.5 should show a folded profile"
    print(f"\n    at r = 1.5 the edges have folded ({r15['folds']} sign changes) but")
    print("    detect_oscillation_onset is still silent: find_peaks needs a second")
    print("    maximum with prominence >= 1% of the peak, and the ripples")
    print("    immediately after folding sit on a flat top far below that.")
    print("    Loosening the prominence recovers it, at the cost of false positives:")
    for prom, z_on in zip(PROMINENCES, r15["osc_prom"]):
        print(f"      prominence {prom:6.3f} -> onset {_fmt(z_on)}")
    assert np.isfinite(r15["osc_prom"][0]), (
        "a 10x looser prominence should recover the already-folded r = 1.5 case"
    )
    assert not np.isfinite(r15["osc_prom"][1]), (
        "the 1% default is expected to miss the freshly folded case"
    )
    print("    only the 10x-looser rule recovers it, at the cost of false positives")
    print("    on runs that have not folded at all.")

    print("\n  and edge_steepness is a growth indicator, not a distance estimate:")
    for run in runs:
        if 1.5 <= run["r"] <= 3.0:
            print(f"    r = {run['r']:.1f}: peak S = {run['peak_steepness']:.4f}")
    print("    the plateau near S = 1.84 between r = 1.5 and 3.0 is the flat top,")
    print("    not convergence.")

    return {"runs": runs}


def theory_figure(wb: WaveBreaking, runs: list[dict]) -> None:
    """Figure 1: profiles at four z for one sub-critical and one super-critical case."""
    sub = next(x for x in runs if x["r"] == 0.5)
    sup = next(x for x in runs if x["r"] == 1.5)
    picks = [sub, sup]
    labels = ["sub-critical  r = 0.5", "super-critical  r = 1.5"]

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.8))
    for row, (run, lab) in enumerate(zip(picks, labels)):
        grid = run["grid"]
        z, fields = run["z"], run["fields"]
        for col, frac in enumerate((0.0, 0.33, 0.66, 1.0)):
            zc = frac * run["L"]
            idx = int(np.argmin(np.abs(z - zc)))
            axes[row, col].plot(grid.t * 1e12, np.abs(fields[idx]) ** 2, lw=1.5)
            axes[row, col].set_title(
                f"z = {zc:.0f} m\n(z / z_WB = {zc / wb.z_WB:.2f})", fontsize=10
            )
            axes[row, col].set_xlabel("t (ps)")
            if col == 0:
                axes[row, col].set_ylabel(r"$|A|^2$ (W)")
            axes[row, col].grid(alpha=0.3)
            axes[row, col].set_ylim(0, P0 * 2.2)
        axes[row, 0].annotate(
            lab,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round",
                "facecolor": "tab:blue" if row else "tab:green",
                "alpha": 0.55,
            },
        )

    fig.suptitle(
        "Normal-dispersion wave breaking: the pulse steepens, then its edges fold",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["theory"], dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. The detection threshold is a choice ─────────────────────────
def threshold_panel(wb: WaveBreaking, run: dict) -> dict:
    print(_rule("3. The steepening threshold is a choice, not physics"))

    z, steep = run["z"], run["steepness"]
    print(f"  reference {gaussian_edge_steepness:.6f}; detector default factor 1.10\n")
    print("   factor   threshold    z_onset        onset / z_WB")
    onsets = {}
    for f in STEEPENING_FACTORS:
        z_on = detect_steepening_onset(z, steep, factor=f)
        onsets[f] = z_on
        print(
            f"   {f:5.2f}    {f * gaussian_edge_steepness:.6f}   "
            f"{_fmt(z_on):>12}   {_ratio(z_on, wb.z_WB):>9}"
        )
    lo, hi = onsets[STEEPENING_FACTORS[0]], onsets[STEEPENING_FACTORS[-1]]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi > lo, "a tighter threshold must fire later, not earlier"
    print("\n  a 2% threshold fires early (during steepening); 25% fires late")
    print("  (near folding). The analytic z_WB is the folding condition, so it")
    print("  sits above all of them — which is why quoting a detected onset as")
    print("  'the wave-breaking distance' understates the prediction by 10-40%.")

    # The detector must also agree with WaveBreaking.analyze at factor 1.10.
    assert abs(detect_steepening_onset(z, steep) - run["z_onset_m"]) < 1e-9
    print("\n  detect_steepening_onset() at its default factor reproduces")
    print("  WaveBreaking.analyze()['z_onset_m']                     [asserted]")
    return onsets


def onset_figure(wb: WaveBreaking, runs: list[dict], resolution: list[dict]) -> None:
    """Figure 2: prediction vs measurement, and what the detectors miss."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.6))
    sup = next(x for x in runs if x["r"] == RESOLUTION_R)

    # (a) The ladder runs are the SAME trajectory stopped at different L, so
    # plotting them all against z/z_WB just draws one curve over itself. Show
    # the longest run's full trajectory once and mark where each r stops.
    ax = axes[0, 0]
    longest = max(runs, key=lambda run: run["r"])
    ax.plot(
        longest["z"] / wb.z_WB,
        longest["steepness"],
        color="0.35",
        lw=2.0,
        label="steepness trajectory (r = 6.0 run)",
    )
    for run in runs:
        ax.axvline(run["r"], color="grey", ls="-", lw=0.7, alpha=0.55)
        ax.annotate(
            f"r={run['r']:g}",
            xy=(run["r"], 0.55),
            fontsize=7,
            rotation=90,
            ha="right",
            va="bottom",
            color="grey",
        )
    for run in runs:
        if np.isfinite(run["z_onset_m"]):
            # Place the marker at the steepness the trajectory had AT the onset,
            # not at the run's peak steepness (which is reached much later).
            s_at = float(np.interp(run["z_onset_m"], run["z"], run["steepness"]))
            ax.plot(
                [run["z_onset_m"] / wb.z_WB],
                [s_at],
                "o",
                color="tab:blue",
                ms=7,
                mec="k",
                mew=0.7,
                label="detected steepening onset" if run is runs[1] else None,
            )
    osc_runs = [run for run in runs if np.isfinite(run["z_oscillation_m"])]
    for run in osc_runs:
        s_at = float(np.interp(run["z_oscillation_m"], run["z"], run["steepness"]))
        ax.plot(
            [run["z_oscillation_m"] / wb.z_WB],
            [s_at],
            "s",
            color="tab:orange",
            ms=8,
            mec="k",
            mew=0.7,
            label="detected oscillation onset" if run is osc_runs[0] else None,
        )
    ax.axhline(
        gaussian_edge_steepness,
        color="tab:green",
        ls="-.",
        lw=1.2,
        label="Gaussian reference",
    )
    ax.axhline(
        1.10 * gaussian_edge_steepness,
        color="tab:red",
        ls="--",
        lw=1.2,
        label="1.10 x reference (steepening detector)",
    )
    ax.axvline(1.0, color="k", ls=":", lw=1.6, label="analytic z_WB (folding)")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 20)
    ax.set_xlabel(r"z / z_{WB}   (vertical grey lines: where each r stops)")
    ax.set_ylabel("edge steepness S (log)")
    ax.set_title(
        "(a) one trajectory, sampled to six lengths:\n"
        "the steepening trigger fires at 0.58 z_WB,\n"
        "folding at 1.00, oscillations at 3.15"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, loc="upper left")

    # (b) folding vs what the oscillation detector reports
    ax = axes[0, 1]
    rs = [run["r"] for run in runs]
    folds = [run["folds"] for run in runs]
    osc = [run["z_oscillation_m"] / wb.z_WB for run in runs]
    ax.plot(
        rs, folds, "o-", lw=1.8, color="tab:blue", label="folding: dI/dt sign changes"
    )
    ax.set_xlabel("breaking ratio r = L / z_WB")
    ax.set_ylabel("sign changes of dI/dt", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(
        rs, osc, "s--", lw=1.4, color="tab:orange", label="oscillation onset / z_WB"
    )
    ax2.set_ylabel(r"oscillation onset / $z_{WB}$", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title("(b) folding happens long before the\noscillation detector reports it")
    ax.grid(alpha=0.3)

    # (c) Both detectors have a threshold that is a CHOICE. Plot each as a
    # multiple of its own default so the two are on one coherent axis.
    ax = axes[1, 0]
    steep_mult = [f / 1.10 for f in STEEPENING_FACTORS]
    steep_onset = [
        detect_steepening_onset(sup["z"], sup["steepness"], factor=f)
        for f in STEEPENING_FACTORS
    ]
    prom_mult = [p / 0.01 for p in PROMINENCES]
    prom_onset = sup["osc_prom"]
    ax.plot(
        steep_mult,
        np.array(steep_onset) / wb.z_WB,
        "o-",
        lw=1.8,
        color="tab:blue",
        label="steepening factor / 1.10  (r = 1.5)",
    )
    ax.plot(
        prom_mult,
        np.array(prom_onset) / wb.z_WB,
        "s--",
        lw=1.8,
        color="tab:orange",
        label="oscillation prominence / 0.01",
    )
    for x, y in zip(steep_mult, np.array(steep_onset) / wb.z_WB):
        if np.isfinite(y):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color="tab:blue",
            )
    ax.axhline(1.0, color="k", ls=":", lw=1.6, label="analytic z_WB")
    ax.axvline(1.0, color="grey", ls="-", lw=1.0)
    ax.annotate(
        "library defaults",
        xy=(1.0, 0.44),
        xytext=(0.72, 0.44),
        fontsize=8,
        color="grey",
    )
    ax.set_xscale("log")
    ax.set_xlabel("threshold as a multiple of the library default")
    ax.set_ylabel(r"reported onset / $z_{WB}$")
    ax.set_ylim(0.4, 1.35)
    ax.set_title(
        "(c) both thresholds are choices, not physics:\n"
        "the reported onset moves with each"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="center right")

    # (d) resolution dependence
    ax = axes[1, 1]
    steps = [res["n_steps"] for res in resolution]
    zs = [res["z_onset_m"] / wb.z_WB for res in resolution]
    ax.plot(steps, zs, "o-", lw=1.8)
    for s, zz in zip(steps, zs):
        ax.annotate(
            f"{zz:.3f}",
            (s, zz),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=8,
            ha="center",
        )
    ax.axhline(1.0, color="k", ls=":", lw=1.4, label="analytic z_WB")
    ax.set_xscale("log")
    ax.set_xlabel("propagation steps over the same L")
    ax.set_ylabel(r"detected steepening onset / $z_{WB}$")
    ax.set_title(
        f"(d) resolution dependence at r = {RESOLUTION_R}\n"
        "(a moving onset means the run is under-resolved)"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Wave breaking: the analytic folding distance, and what the "
        "detectors actually report",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["onset"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(_rule("0. Optical wave breaking — normal dispersion"))
    print("  ordering parameter: r = L / z_WB, breaking for r > 1")
    print("  (this is NOT the anomalous soliton-fission regime of examples/13)")

    wb, scales = analytic_panel()
    lad = ladder_panel(wb)
    runs = lad["runs"]
    sup_run = next(x for x in runs if x["r"] == RESOLUTION_R)
    threshold_panel(wb, sup_run)

    print(_rule("4. Resolution dependence at r = %.1f" % RESOLUTION_R))
    resolution = []
    for n_steps in (N_STEPS, 2 * N_STEPS, 4 * N_STEPS):
        res = _propagate(wb, RESOLUTION_R, n_steps=n_steps)
        resolution.append({**res, "n_steps": n_steps})
        print(
            f"  {n_steps:4d} steps (dz = {res['L'] / n_steps:.3f} m): "
            f"onset {_fmt(res['z_onset_m'])} = "
            f"{res['z_onset_m'] / wb.z_WB:.4f} z_WB, "
            f"oscillation {_fmt(res['z_oscillation_m'])}"
        )
    spread = max(r["z_onset_m"] for r in resolution) - min(
        r["z_onset_m"] for r in resolution
    )
    print(
        f"\n  onset spread across the ladder: {spread:.2f} m "
        f"({spread / scales['z_WB'] * 100:.1f}% of z_WB)"
    )
    if spread > resolution[-1]["L"] / resolution[-1]["n_steps"]:
        print("  -> RESOLUTION-SENSITIVE: the detected onset moves by more than")
        print("     one step, so none of these runs is authoritative; refine first.")
    else:
        print("  -> converged: the onset is stable to within one step.")

    theory_figure(wb, runs)
    onset_figure(wb, runs, resolution)

    print(_rule("Summary"))
    print(
        f"  L_D / L_NL                    : {scales['L_D'] / scales['L_NL']:.1f}"
        "   (not a breaking criterion)"
    )
    print(
        f"  L_D, L_NL                     : {scales['L_D'] / 1e3:.3f} km, "
        f"{scales['L_NL']:.2f} m"
    )
    print(f"  z_WB                          : {scales['z_WB']:.2f} m")
    r09 = next(x for x in runs if x["r"] == 0.9)
    r15 = next(x for x in runs if x["r"] == 1.5)
    r30 = next(x for x in runs if x["r"] == 3.0)
    r60 = next(x for x in runs if x["r"] == OSCILLATION_FIRES_AT_R)
    print("  r = 0.5                       : both detectors nan (nothing to report)")
    print(
        f"  r = 0.9 (not breaking)       : steepening onset "
        f"{_ratio(r09['z_onset_m'], scales['z_WB'])} anyway -- the 10% threshold"
    )
    print(
        f"  r = 1.5 folded                : {r15['folds']:2d} dI/dt sign changes, "
        f"oscillation onset {_fmt(r15['z_oscillation_m'])}"
    )
    print(
        f"  r = 1.5 steepening onset      : {_ratio(r15['z_onset_m'], scales['z_WB'])}"
        f"  ({_fmt(r15['z_onset_m'])})"
    )
    print(
        f"  r = {OSCILLATION_FIRES_AT_R} oscillation onset       : "
        f"{_ratio(r60['z_oscillation_m'], scales['z_WB'])} "
        f"({_fmt(r60['z_oscillation_m'])})"
    )
    print(
        f"  peak S plateau (r = 1.5, 3.0): {r15['peak_steepness']:.4f}, "
        f"{r30['peak_steepness']:.4f}  (flat top, not convergence)"
    )
    print(f"  resolution spread             : {spread:.2f} m")
    print("  all checks passed ✓")

    print("\nGenerated files:")
    for key in ("theory", "onset"):
        print(f"  {FIGS[key].relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
