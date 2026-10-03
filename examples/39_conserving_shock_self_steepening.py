r"""
Example: the photon-conserving shock, and the self-steepening grid guard
======================================================================

``SplitStepEngine`` carries two independent additions for the shock
(self-steepening) term, and this example exists because they fail in
*different* ways and are easy to confuse:

* ``conserving_shock=True`` — the photon-conserving (pcGNLSE) operator. It is
  the same equation with exactly two modifications, both on the **delayed
  (Raman) arm**:

  1. the Raman self-frequency arm carries ``|γ|`` instead of ``γ``;
  2. the SS–Raman dissipative cross term likewise.

  The instantaneous (pure-SPM) arm keeps the signed ``γ``. So the whole flag
  reduces to one substitution, ``g_r = abs(gamma)`` where the standard
  formulation uses ``g_r = gamma``, applied to the delayed arm only.

* ``_validate_shock_grid`` — the construction-time guard. The shock operator is
  ``ω/ω₀ ≈ 1 + Ω·τ_shock``, a *first-order Taylor* form, so it has two validity
  conditions: a hard one (``Ω_max < ω₀``; beyond it the factor describes
  negative absolute frequencies and raises ``ValueError``) and a soft one
  (``τ_shock·Ω_max ≤ _SHOCK_TAYLOR_LIMIT = 0.2``; beyond it the Taylor form is
  inaccurate and the resulting photon-number drift is a *grid artifact*, so it
  raises a ``UserWarning``).

They are independent. ``conserving_shock=True`` is **not** a fix for
under-resolution, and the grid guard is **not** a conservation law. The
example keeps them in separate sections and measures both, because the whole
value of the guard is that it turns an invisible error into a loud one: an
under-resolved shock run does not crash, produces a plausible-looking spectrum,
and inverts a physical conclusion.

What you will actually measure (all numbers printed by the script)
------------------------------------------------------------------
The two claims in the proposal deserve a blunt correction, because what the
code does is narrower and more interesting than the claim:

1. **On an ordinary fiber the flag is a no-op.** For silica ``γ > 0``, so
   ``abs(gamma) == gamma`` and the two formulations are identical to machine
   precision (~1e-14 relative on the output field). There is nothing to see,
   and the script says so rather than hiding the run.
2. **The two differ only when γ carries the opposite sign.** With ``γ < 0`` the
   delayed arm flips sign, the accumulated self-frequency shift changes by a
   factor of ~3, and the output fields differ by O(1). This is the sign bench
   the Huang pcGNLSE reproduction uses, and it is where the flag is meant to
   be read.
3. **Neither formulation fixes an under-resolved grid.** Sweeping
   ``τ_shock·Ω_max`` gives a photon-number drift that grows from ~0.2 % at
   0.073 to several percent at 0.6 — *identically* for both, because the drift
   comes from the truncated ``ω/ω₀`` expansion and not from the ``|γ|``
   substitution.

A consequence worth stating up front, because it constrains every deck below:
``τ_shock = 1/ω₀`` and the soft limit ``τ_shock·Ω_max ≤ 0.2`` together *force*
``dt = π/Ω_max ≥ π/(0.2·ω₀)``. At 1550 nm that is ``dt ≥ 12.9 fs``; at 850 nm,
``dt ≥ 7.1 fs``. A 28.4 fs pulse sampled every 7 fs is four points wide. The
guard is therefore in direct conflict with resolving a short pulse at a short
wavelength — which is exactly why the Hult 2007 reproduction records the shock
as disabled for its 850 nm / 28.4 fs deck.
"""

from __future__ import annotations

import sys
import time as _time
import warnings
from pathlib import Path
from typing import Any, cast

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
from photonics_helper.gnlse import (
    _SHOCK_TAYLOR_LIMIT,
    FiberProfile,
    SplitStepEngine,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

C_LIGHT = 299792458.0

# ── The 850 nm / 28.4 fs / 10 kW supercontinuum deck (Hult 2007 Sec. III.B,
# the same numbers the Hult reproduction carries, and the deck whose shock
# setting it has to record as a deviation) ───────────────────────────
LAM_SCG = 850e-9
GAMMA_SCG = 0.045  # 1/(W m)
P0_SCG = 10000.0  # W
T0_SCG = 28.4e-15  # s
BETA2_SCG = -0.01276e-24  # s^2/m
L_SCG = 0.1  # m

# ── The 1550 nm / 100 fs sign bench on which the two formulations differ ──
LAM_SIGN = 1550e-9
GAMMA_SIGN = 1.5e-3  # 1/(W m)
P0_SIGN = 1333.0  # W
T0_SIGN = 100e-15  # s
BETA2_SIGN = -2.0e-26  # s^2/m
L_SIGN = 10.0  # m

N_POINTS = 8192  # the guard is swept by moving Tmax at fixed N
STEPS_SIGN = 1000
STEPS_SCG = 1200
NSAVES = 101

# τ_shock·Ω_max ladder for the guard and the drift sweep. The first two entries
# are the two values ISSUES.md #1 quotes for this deck; the third is the limit
# itself; the fourth is well past it.
TAU_OMEGA_LADDER = (0.6, 0.29, 0.145, 0.073)

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "comparison": OUT_DIR / "39_conserving_shock_comparison.png",
    "guard": OUT_DIR / "39_shock_taylor_guard.png",
}


def _db(power: NDArray | float) -> NDArray:
    floored = np.maximum(np.asarray(power, dtype=float), 1e-300)
    return np.asarray(10.0 * np.log10(floored))


def _rule(title: str) -> str:
    return "\n" + "=" * 68 + f"\n {title}\n" + "=" * 68


def _silica(grid: TemporalGrid, tau2: float = 236e-15) -> RamanResponse:
    """The Blow & Wood two-exponential silica response used repo-wide."""
    spec = RamanSpec(
        name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
    )
    return RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=tau2, grid=grid)


def _fiber(
    gamma: float,
    omega0: float,
    length_m: float,
    resp: RamanResponse,
    grid: TemporalGrid,
) -> FiberProfile:
    return FiberProfile.from_gamma(
        gamma=gamma,
        n2=2.6e-20 * np.sign(gamma),
        omega0=omega0,
        alpha=0.0,
        length=Length(length_m, "m"),
        raman_response=resp,
    )


def _pulse(grid: TemporalGrid, lam: float, t0: float, p0: float) -> Wave:
    env = Envelope(
        shape="sech", peak_amplitude=float(np.sqrt(p0)), pulse_width=Time(t0, "s")
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(lam, "m"))


def _omega0(lam: float) -> float:
    return 2.0 * np.pi * C_LIGHT / lam


def _tau_omega_max(tmax_s: float, omega0: float, n_points: int = N_POINTS) -> float:
    """``τ_shock·Ω_max`` for the default ``τ_shock = 1/ω₀`` on this grid."""
    omega_max = np.pi * n_points / tmax_s
    return float(omega_max / omega0)


def _tmax_for_tau_omega(
    target: float, omega0: float, n_points: int = N_POINTS
) -> float:
    return float(np.pi * n_points / (target * omega0))


def _catch_shock_warning(build) -> str | None:
    """Run ``build()`` and return the shock-Taylor warning, if it fires.

    Only the *shock* warning is selected: ``conserving_shock=True`` routes the
    delayed-Raman convolution through ``ifft(...).astype(float)`` and numpy
    emits a ``ComplexWarning`` on every step, which would drown a
    ``catch_warnings(record=True)`` block in thousands of unrelated records.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()
    for record in caught:
        if "τ_shock" in str(record.message) and "expansion limit" in str(
            record.message
        ):
            return str(record.message)
    return None


# ── 1. The guard, first ─────────────────────────────────────────────
def guard_panel() -> dict[str, Any]:
    print(_rule("1. The guard: an under-resolved shock grid, caught at construction"))

    omega0 = _omega0(LAM_SCG)
    tau_shock = 1.0 / omega0
    print(
        f"  deck: {LAM_SCG * 1e9:.0f} nm, P0 = {P0_SCG:g} W, T0 = {T0_SCG * 1e15:.1f} fs,"
        f" gamma = {GAMMA_SCG:g} /W/m, L = {L_SCG:g} m"
    )
    print(
        f"  tau_shock = 1/omega0 = {tau_shock * 1e15:.3f} fs; "
        f"soft limit tau*Omega_max <= {_SHOCK_TAYLOR_LIMIT:g}"
    )

    def build(tmax_s: float) -> SplitStepEngine:
        grid = TemporalGrid(N=N_POINTS, Tmax=Time(tmax_s, "s"))
        fiber = _fiber(GAMMA_SCG, omega0, L_SCG, _silica(grid, tau2=32e-15), grid)
        return SplitStepEngine(
            pulse=_pulse(grid, LAM_SCG, T0_SCG, P0_SCG),
            fiber=fiber,
            betas=np.array([BETA2_SCG]),
            betas_unit="s^k/m",
            include_raman=True,
            include_self_steepening=True,
            step_size=Length(L_SCG / 100, "m"),
        )

    # The hard condition first: Ω_max ≥ ω₀ raises, it does not warn.
    hard_tmax = float(np.pi * N_POINTS / (1.2 * omega0))
    print(
        f"\n  (a) the HARD condition, Omega_max < omega0.  A grid with "
        f"Tmax = {hard_tmax * 1e12:.1f} ps"
    )
    print(
        f"      at N = {N_POINTS} has Omega_max/omega0 = "
        f"{np.pi * N_POINTS / hard_tmax / omega0:.2f}:"
    )
    try:
        build(hard_tmax)
    except ValueError as exc:
        print(f"      ValueError: {str(exc)[:150].strip()}...")
        hard_message = str(exc)
    else:  # pragma: no cover - the construction above must raise
        raise AssertionError("Omega_max >= omega0 should raise ValueError")

    # The soft condition: sweep τ·Ω_max and catch the warning.
    print(
        "\n  (b) the SOFT condition, tau*Omega_max <= "
        f"{_SHOCK_TAYLOR_LIMIT:g}.  Sweeping Tmax at fixed N = {N_POINTS}:"
    )
    ladder: list[dict[str, Any]] = []
    for target in TAU_OMEGA_LADDER:
        tmax_s = _tmax_for_tau_omega(target, omega0)
        dt = tmax_s / N_POINTS
        message = _catch_shock_warning(lambda t=tmax_s: build(t))
        fired = message is not None
        ladder.append(
            {
                "target": target,
                "tmax": tmax_s,
                "dt": dt,
                "tau_omega": _tau_omega_max(tmax_s, omega0),
                "warned": fired,
                "message": message,
            }
        )
        verdict = "WARNS" if fired else "passes"
        print(
            f"    Tmax = {tmax_s * 1e12:6.1f} ps  dt = {dt * 1e15:6.2f} fs  "
            f"tau*Omega_max = {_tau_omega_max(tmax_s, omega0):.3f}  -> {verdict}"
        )

    warned = [row for row in ladder if bool(row["warned"])]
    passed = [row for row in ladder if not row["warned"]]
    assert warned, "the ladder never tripped the guard -- it would prove nothing"
    assert passed, "the ladder never satisfied the guard -- nothing was demonstrated"
    print(
        f"\n    {len(warned)} of {len(ladder)} entries warn, {len(passed)} pass  "
        "[asserted both non-empty]"
    )
    for row in warned:
        assert cast(float, row["tau_omega"]) > _SHOCK_TAYLOR_LIMIT, (
            "a warning below the limit?"
        )
    for row in passed:
        assert cast(float, row["tau_omega"]) <= _SHOCK_TAYLOR_LIMIT, (
            "a pass above the limit?"
        )
    print(
        f"    the boundary sits between tau*Omega_max = "
        f"{max(cast(float, r['tau_omega']) for r in passed):.3f} (passes) and "
        f"{min(cast(float, r['tau_omega']) for r in warned):.3f} (warns)"
    )

    first = warned[0]
    print(
        f"\n  the diagnostic verbatim (first entry, Tmax = {first['tmax'] * 1e12:.1f} ps):"
    )
    for line in cast(str, first["message"]).split(". "):
        print(f"    {line.strip().rstrip('.')}.")

    # Refine until it stops firing, the way the warning itself instructs.
    print("\n  refining in a loop until the diagnostic goes quiet:")
    tmax_s = float(np.pi * N_POINTS / (0.5 * omega0))
    for _ in range(12):
        message = _catch_shock_warning(lambda t=tmax_s: build(t))
        print(
            f"    Tmax = {tmax_s * 1e12:7.1f} ps  tau*Omega_max = "
            f"{_tau_omega_max(tmax_s, omega0):.3f}  "
            f"{'-> WARNS' if message else '-> quiet'}"
        )
        if message is None:
            break
        tmax_s *= 1.5
    required = tmax_s
    dt_required = required / N_POINTS
    print(
        f"    required: Tmax >= {required * 1e12:.1f} ps at N = {N_POINTS}, "
        f"i.e. dt >= {dt_required * 1e15:.2f} fs"
    )

    print(
        f"\n  and here is the cost, stated plainly. The guard demands "
        f"dt >= {dt_required * 1e15:.2f} fs;"
    )
    print(
        f"  (that is where the 1.5x refinement loop stopped; the exact limit is "
        f"pi/(0.2*omega0) = {np.pi / (_SHOCK_TAYLOR_LIMIT * omega0) * 1e15:.2f} fs)"
    )
    print(
        f"  this deck's pulse is {T0_SCG * 1e15:.1f} fs wide, so the grid that "
        f"satisfies the guard carries"
    )
    print(
        f"  {T0_SCG / dt_required:.1f} samples across the pulse. A self-steepening run on this"
    )
    print("  codebase cannot resolve a femtosecond-scale pulse: the limit follows")
    print("  from tau = 1/omega0 and the 0.2 cap, not from the integrator. This is")
    print("  why the Hult 2007 reproduction records the shock as disabled for its")
    print("  850 nm / 28.4 fs deck.")

    return {
        "omega0": omega0,
        "tau_shock": tau_shock,
        "ladder": ladder,
        "hard_tmax": hard_tmax,
        "hard_message": hard_message,
        "required_tmax": required,
        "required_dt": dt_required,
    }


# ── 2. The two formulations ─────────────────────────────────────────
def _centroid(grid: TemporalGrid, field: NDArray, band_hz: float = 30.0e12) -> float:
    """Power-weighted spectral centroid (THz) inside ``+-band_hz``.

    Restricted to a band on purpose: the whole grid is padded with numerical
    dust, and an unrestricted centroid is dominated by round-off.
    """
    power = np.abs(grid.fft(np.asarray(field, dtype=complex))) ** 2
    mask = np.abs(grid.w) <= band_hz * 2.0 * np.pi
    return (
        float(np.sum(grid.w[mask] * power[mask]) / np.sum(power[mask]))
        / 2
        / np.pi
        / 1e12
    )


def _run_sign_bench(gamma_sign: float, conserving: bool) -> dict[str, Any]:
    omega0 = _omega0(LAM_SIGN)
    grid = TemporalGrid(N=N_POINTS, Tmax=Time(_tmax_for_tau_omega(0.15, omega0), "s"))
    fiber = _fiber(gamma_sign * GAMMA_SIGN, omega0, L_SIGN, _silica(grid), grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = SplitStepEngine(
            pulse=_pulse(grid, LAM_SIGN, T0_SIGN, P0_SIGN),
            fiber=fiber,
            betas=np.array([BETA2_SIGN]),
            betas_unit="s^k/m",
            include_raman=True,
            include_self_steepening=True,
            conserving_shock=conserving,
            step_size=Length(L_SIGN / STEPS_SIGN, "m"),
        )
        engine.propagate(num_steps=STEPS_SIGN, nsaves=NSAVES)
    z = np.linspace(0.0, L_SIGN, len(engine.evolution))
    spectra = np.stack(
        [
            np.abs(grid.fft(np.asarray(w.envelope_field, dtype=complex))) ** 2
            for w in engine.evolution
        ]
    )
    fields = np.stack(
        [np.asarray(w.envelope_field, dtype=complex) for w in engine.evolution]
    )
    return {
        "grid": grid,
        "engine": engine,
        "z": z,
        "spectra": spectra,
        "fields": fields,
        "centroid": np.array([_centroid(grid, f) for f in fields]),
        "energy": engine.energy_vs_z,
        "final": fields[-1],
        "drift": float(engine.energy_vs_z[-1] / engine.energy_vs_z[0] - 1.0),
    }


def formulation_panel() -> dict[str, Any]:
    print(
        _rule("2. The two formulations: identical at gamma > 0, different at gamma < 0")
    )

    print(
        f"  deck: {LAM_SIGN * 1e9:.0f} nm, T0 = {T0_SIGN * 1e15:.0f} fs, P0 = {P0_SIGN:g} W, "
        f"gamma = +-1.5e-3 /W/m, L = {L_SIGN:g} m, N = {N_POINTS}"
    )

    print(
        "\n  (a) ordinary silica, gamma = +1.5e-3 /W/m.  The pcGNLSE "
        "modification replaces gamma"
    )
    print(
        "      by |gamma| on the delayed arm only, and for a positive gamma "
        "that is the same"
    )
    print(
        "      number. So there is nothing to see, and the run is reported as a "
        "null result:"
    )
    positive = {}
    for conserving in (False, True):
        positive[conserving] = _run_sign_bench(+1.0, conserving)
    rel_diff_pos = float(
        np.linalg.norm(positive[False]["final"] - positive[True]["final"])
        / np.linalg.norm(positive[False]["final"])
    )
    shift_pos = positive[False]["centroid"][-1] - positive[False]["centroid"][0]
    print(
        f"      standard   centroid shift {shift_pos:+.5f} THz, "
        f"energy drift {positive[False]['drift'] * 100:+.4f} %"
    )
    print(
        f"      conserving centroid shift "
        f"{positive[True]['centroid'][-1] - positive[True]['centroid'][0]:+.5f} THz, "
        f"energy drift {positive[True]['drift'] * 100:+.4f} %"
    )
    print(f"      relative field difference: {rel_diff_pos:.2e}   [asserted < 1e-10]")
    assert rel_diff_pos < 1e-10, (
        f"gamma > 0 should make the flag a no-op, but the fields differ by {rel_diff_pos}"
    )
    print("      -> the flag cannot be evaluated on a positive-gamma fiber at all.")

    print(
        "\n  (b) the sign bench, gamma = -1.5e-3 /W/m (the discrimination the "
        "Huang pcGNLSE"
    )
    print("      reproduction uses). Now |gamma| = -gamma and the delayed arm flips:")
    negative = {}
    for conserving in (False, True):
        negative[conserving] = _run_sign_bench(-1.0, conserving)
    std, pc = negative[False], negative[True]
    shift_std = std["centroid"][-1] - std["centroid"][0]
    shift_pc = pc["centroid"][-1] - pc["centroid"][0]
    rel_diff_neg = float(
        np.linalg.norm(std["final"] - pc["final"]) / np.linalg.norm(std["final"])
    )
    print(
        f"      standard   centroid {std['centroid'][0]:+.5f} -> "
        f"{std['centroid'][-1]:+.5f} THz   (shift {shift_std:+.5f} THz)"
    )
    print(
        f"      conserving centroid {pc['centroid'][0]:+.5f} -> "
        f"{pc['centroid'][-1]:+.5f} THz   (shift {shift_pc:+.5f} THz)"
    )
    print(
        f"      the conserving formulation's accumulated shift is "
        f"{abs(shift_std / shift_pc):.1f}x smaller"
    )
    print(
        f"      relative field difference at the exit: {rel_diff_neg:.3f}   "
        "[asserted > 0.1]"
    )
    print(
        f"      energy drift: standard {std['drift'] * 100:+.4f} %, "
        f"conserving {pc['drift'] * 100:+.4f} %"
    )
    assert rel_diff_neg > 0.1, "the sign bench should separate the two formulations"
    assert abs(shift_pc) < abs(shift_std), (
        "the conserving formulation should reduce the accumulated shift"
    )
    print(
        "      -> the difference is a factor, not a sign: the delayed arm "
        "partially cancels"
    )
    print(
        "         its own self-frequency shift, which is the pcGNLSE's "
        "attractor mechanism,"
    )
    print("         rather than reversing it. Reported as measured.")

    int_diff_neg = float(
        np.linalg.norm(np.abs(std["final"]) ** 2 - np.abs(pc["final"]) ** 2)
        / np.linalg.norm(np.abs(std["final"]) ** 2)
    )

    # Exit structure, counted rather than asserted.
    from scipy.signal import find_peaks

    peaks = {}
    for key, run in (("standard", std), ("conserving", pc)):
        intensity = np.abs(run["final"]) ** 2
        found, _ = find_peaks(intensity, prominence=0.05 * float(np.max(intensity)))
        peaks[key] = len(found)
        print(
            f"      {key:11s} temporal peaks at the exit (5 % prominence): {len(found)}"
        )
    print("      and that is the honest picture of the exit state: no fission.")
    print("      It should not be read as a failed demonstration. With gamma < 0")
    print("      the soliton condition is inverted -- the state is an anti-soliton,")
    print("      not a high-order one -- so this deck is a discrimination bench for")
    print("      the |gamma| substitution and nothing else. Fission needs gamma > 0,")
    print("      where the two formulations are identical, which is the null result")
    print("      printed in (a). There is no deck on which both things are visible")
    print("      at once, and the example says so instead of manufacturing one.")

    return {
        "positive": positive,
        "negative": negative,
        "rel_diff_pos": rel_diff_pos,
        "rel_diff_neg": rel_diff_neg,
        "shift_std": shift_std,
        "shift_pc": shift_pc,
        "peaks": peaks,
        "int_diff_neg": int_diff_neg,
    }


# ── 3. Conservation is the discriminator ───────────────────────────
def conservation_panel() -> dict[str, Any]:
    print(_rule("3. Neither formulation fixes under-resolution"))

    omega0 = _omega0(LAM_SCG)
    print(
        f"  deck: {LAM_SCG * 1e9:.0f} nm / {T0_SCG * 1e15:.1f} fs / {P0_SCG:g} W, "
        f"gamma = {GAMMA_SCG:g} /W/m, L = {L_SCG:g} m"
    )
    print("  photon-number drift through the run, against tau*Omega_max:")
    rows: list[dict[str, Any]] = []
    for target in TAU_OMEGA_LADDER:
        tmax_s = _tmax_for_tau_omega(target, omega0)
        entry: dict[str, Any] = {
            "target": target,
            "tmax": tmax_s,
            "tau_omega": _tau_omega_max(tmax_s, omega0),
        }
        for conserving in (False, True):
            grid = TemporalGrid(N=N_POINTS, Tmax=Time(tmax_s, "s"))
            fiber = _fiber(GAMMA_SCG, omega0, L_SCG, _silica(grid, tau2=32e-15), grid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engine = SplitStepEngine(
                    pulse=_pulse(grid, LAM_SCG, T0_SCG, P0_SCG),
                    fiber=fiber,
                    betas=np.array([BETA2_SCG]),
                    betas_unit="s^k/m",
                    include_raman=True,
                    include_self_steepening=True,
                    conserving_shock=conserving,
                    step_size=Length(L_SCG / STEPS_SCG, "m"),
                )
                engine.propagate(num_steps=STEPS_SCG, nsaves=21)
            entry[f"drift_{conserving}"] = float(
                engine.energy_vs_z[-1] / engine.energy_vs_z[0] - 1.0
            )
        entry["warned"] = entry["tau_omega"] > _SHOCK_TAYLOR_LIMIT
        rows.append(entry)
        print(
            f"    tau*Omega_max = {entry['tau_omega']:5.3f} "
            f"({'WARNS' if entry['warned'] else 'passes'})  "
            f"standard {100 * cast(float, entry['drift_False']):+8.3f} %   "
            f"conserving {100 * cast(float, entry['drift_True']):+8.3f} %"
        )

    print("\n  Two things to read off that table.")
    resolved = [r for r in rows if not r["warned"]]
    over = [r for r in rows if r["warned"]]
    assert resolved and over, "the ladder must straddle the limit"
    finest = min(resolved, key=lambda r: float(r["tau_omega"]))
    worst_resolved = max(abs(cast(float, r["drift_False"])) for r in resolved)
    print(
        f"  1. The finest grid here, tau*Omega_max = {finest['tau_omega']:.3f}, drifts "
        f"{100 * cast(float, finest['drift_False']):+.3f} %,"
    )
    print(
        "     which is inside the <=0.3 % ISSUES.md #1 quotes for that point. "
        "[asserted < 0.3 %]"
    )
    assert abs(cast(float, finest["drift_False"])) < 0.003, (
        f"drift at tau*Omega_max = 0.073 is {finest['drift_False']}, not the documented <=0.3 %"
    )
    order = sorted(rows, key=lambda r: float(r["tau_omega"]))
    mags = [abs(cast(float, r["drift_False"])) for r in order]
    assert all(b >= a - 1e-9 for a, b in zip(mags, mags[1:])), (
        f"|drift| must grow with tau*Omega_max, got {mags}"
    )
    print(
        f"     and |drift| grows monotonically with tau*Omega_max "
        f"({', '.join(f'{m * 100:.2f}' for m in mags)} %)   [asserted monotone]"
    )
    print(
        f"     The largest drift INSIDE the limit is "
        f"{worst_resolved * 100:.3f} %, on the {max(resolved, key=lambda r: float(r['tau_omega']))['tau_omega']:.3f} entry."
    )
    print("     Note the signs: ISSUES.md quotes -3.1 % at 0.145 and -5.2 % at 0.29 on")
    print("     a 500 fs deck, and this 850 nm deck gives opposite-signed values of")
    print("     the same order. The magnitudes and their growth are reproduced; the")
    print("     exact signed anchors are deck-specific and are not asserted here.")

    gaps = [
        abs(cast(float, r["drift_False"]) - cast(float, r["drift_True"])) for r in rows
    ]
    print(
        f"  2. The two formulations drift by the SAME amount at every point "
        f"(largest gap {max(gaps) * 100:.4f} %"
    )
    print(
        "     on a drift that reaches "
        f"{max(abs(cast(float, r['drift_False'])) for r in rows) * 100:.2f} %).  "
        "On this deck gamma > 0, so |gamma| ="
    )
    print(
        "     gamma and the flag changes nothing -- which is the point: the "
        "drift belongs to"
    )
    print(
        "     the truncated omega/omega0 expansion and to the grid, not to the "
        "formulation."
    )
    print(
        "     A centrifugal claim of the form 'turn on conserving_shock to fix "
        "energy drift'"
    )
    print("     is false on this evidence, and this is the panel that says so.")
    assert max(gaps) < 1e-3, (
        "the two formulations should agree on a positive-gamma fiber"
    )

    worst = max(rows, key=lambda r: abs(cast(float, r["drift_False"])))
    print(
        f"\n  The worst entry sits at tau*Omega_max = {worst['tau_omega']:.3f}, "
        f"{1.0 / cast(float, worst['tau_omega']) / _SHOCK_TAYLOR_LIMIT:.1f}x past"
    )
    print(
        f"  the limit, and it is {abs(cast(float, worst['drift_False'])) / worst_resolved:.0f}x "
        f"the resolved-grid drift. That is the error the guard"
    )
    print("  is for: it is a grid artifact, and no integrator removes it.")

    return {
        "rows": rows,
        "worst_resolved": worst_resolved,
        "finest_drift": cast(float, finest["drift_False"]),
        "mags": mags,
        "gaps": gaps,
    }


# ── Figures ────────────────────────────────────────────────────────
def guard_figure(guard: dict, cons: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))

    ax = axes[0]
    tm = np.array([cast(float, row["tmax"]) for row in guard["ladder"]]) * 1e12
    for row in guard["ladder"]:
        ax.plot(
            row["tmax"] * 1e12,
            row["tau_omega"],
            "o",
            color="tab:red" if row["warned"] else "tab:green",
            ms=9,
            zorder=3,
        )
        ax.annotate(
            f"{row['dt'] * 1e15:.1f} fs",
            (row["tmax"] * 1e12, row["tau_omega"]),
            textcoords="offset points",
            xytext=(0, 12),
            fontsize=7,
            ha="center",
        )
    tmax_line = np.linspace(tm.min() * 0.8, tm.max() * 1.25, 100)
    curve = np.pi * N_POINTS / (tmax_line * 1e-12) / guard["omega0"]
    ax.plot(
        tmax_line,
        curve,
        color="0.4",
        lw=1.0,
        label=r"$\tau_{shock}\,\Omega_{max}$ at $N$ fixed",
    )
    ax.axhline(
        _SHOCK_TAYLOR_LIMIT,
        color="k",
        ls="--",
        lw=1.4,
        label=f"limit = {_SHOCK_TAYLOR_LIMIT:g}",
    )
    ax.axhline(
        1.0,
        color="tab:red",
        ls=":",
        lw=1.4,
        label=r"hard limit: $\Omega_{max} = \omega_0$ (ValueError)",
    )
    ax.set_xlabel(r"grid time window $T_{max}$ (ps)   [at $N$ = 8192]")
    ax.set_ylabel(r"$\tau_{shock}\,\Omega_{max}$")
    ax.set_title(
        "(a) the guard is a function of the window alone\n"
        f"red: warns, green: silent; labels are the implied dt.\n"
        f"Tmax must reach {guard['required_tmax'] * 1e12:.0f} ps "
        f"(dt >= {guard['required_dt'] * 1e15:.1f} fs) to go quiet"
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    rows = cons["rows"]
    tau = np.array([float(r["tau_omega"]) for r in rows])
    # |drift|, not the signed value: the sign flips between grid steps while the
    # magnitude is what the guard is about, and the printed table carries the signs.
    std = np.abs(np.array([100 * float(r["drift_False"]) for r in rows]))
    pc = np.abs(np.array([100 * float(r["drift_True"]) for r in rows]))
    order = np.argsort(tau)
    ax.plot(
        tau[order], std[order], "o-", color="tab:blue", lw=1.8, label="standard GNLSE"
    )
    ax.plot(
        tau[order],
        pc[order],
        "s--",
        color="tab:orange",
        lw=1.4,
        label="conserving_shock=True",
    )
    ax.set_yscale("log")
    ax.axvline(
        _SHOCK_TAYLOR_LIMIT,
        color="k",
        ls="--",
        lw=1.4,
        label=f"guard limit = {_SHOCK_TAYLOR_LIMIT:g}",
    )
    ax.axvspan(0.3, 1.0, color="tab:red", alpha=0.08)
    ax.text(
        0.55,
        ax.get_ylim()[1],
        " guard fires",
        fontsize=8,
        ha="center",
        va="top",
        color="tab:red",
    )
    ax.set_xlabel(r"$\tau_{shock}\,\Omega_{max}$")
    ax.set_ylabel("|photon-number drift| over the run (%)")
    ax.set_title(
        "(b) the drift is the grid, not the formulation\n"
        "the two curves lie on top of each other: on a positive-$\\gamma$\n"
        "fiber the flag is a no-op, so it cannot absorb the error. The sign\n"
        "flips between steps (see the printed table); the growth is the story"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.axis("off")
    message = str(guard["ladder"][0]["message"])
    ax.text(
        0.0,
        1.0,
        "the diagnostic, verbatim:",
        fontsize=11,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )
    wrapped: list[str] = []
    line = ""
    for word in message.split():
        if len(line) + len(word) + 1 > 62:
            wrapped.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    wrapped.append(line)
    ax.text(
        0.0,
        0.90,
        "\n".join(wrapped[:22]),
        fontsize=8.5,
        va="top",
        family="monospace",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.06,
        "The hard condition raises instead:\n"
        f"  ValueError: Omega_max >= omega0 at Tmax = "
        f"{guard['hard_tmax'] * 1e12:.1f} ps",
        fontsize=8.5,
        va="bottom",
        transform=ax.transAxes,
    )

    fig.suptitle(
        "The self-steepening grid guard: under-resolution is diagnosed, not absorbed",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["guard"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(form: dict) -> None:
    std = form["negative"][False]
    pc = form["negative"][True]
    grid: TemporalGrid = std["grid"]
    offset = grid.w / (2.0 * np.pi) / 1e12
    window = np.abs(offset) <= 15.0
    ref = float(np.max(std["spectra"][0]))

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.2))

    ax = axes[0, 0]
    ax.plot(
        offset[window],
        _db(std["spectra"][-1][window] / ref),
        color="tab:blue",
        lw=1.6,
        label="standard GNLSE, z = L",
    )
    ax.plot(
        offset[window],
        _db(pc["spectra"][-1][window] / ref),
        color="tab:orange",
        lw=1.6,
        ls="--",
        label="conserving_shock=True, z = L",
    )
    ax.plot(
        offset[window],
        _db(std["spectra"][0][window] / ref),
        color="0.5",
        lw=1.0,
        label="input",
    )
    ax.set_xlim(-15, 15)
    ax.set_ylim(-70, 5)
    ax.set_xlabel(r"$\Omega-\Omega_0$ (THz)")
    ax.set_ylabel("power spectrum (dB rel. input peak)")
    ax.set_title(
        "(a) exit spectra, gamma < 0\n"
        f"the conserving run accumulates less shift "
        f"({form['shift_pc']:+.5f} THz against\n"
        f"{form['shift_std']:+.5f} THz); the envelope shapes stay close -- the "
        "difference is\ncarried mostly in the phase, which panel (c) separates out"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[0, 1]
    ax.plot(
        std["z"],
        std["centroid"],
        color="tab:blue",
        lw=1.8,
        label=f"standard (final {form['shift_std']:+.5f} THz)",
    )
    ax.plot(
        pc["z"],
        pc["centroid"],
        color="tab:orange",
        lw=1.8,
        ls="--",
        label=f"conserving (final {form['shift_pc']:+.5f} THz)",
    )
    ax.set_xlabel("z (m)")
    ax.set_ylabel("spectral centroid (THz)")
    ax.set_title(
        "(b) accumulated self-frequency shift, gamma < 0\n"
        f"the conserving delayed arm is "
        f"{abs(form['shift_std'] / form['shift_pc']):.1f}x smaller --\n"
        "it partially cancels its own shift, it does not reverse it"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    t_ps = grid.t * 1e12
    keep = np.abs(t_ps) <= 8.0
    std_intensity = np.abs(std["final"]) ** 2 / P0_SIGN
    pc_intensity = np.abs(pc["final"]) ** 2 / P0_SIGN
    ax.plot(
        t_ps[keep],
        std_intensity[keep],
        color="tab:blue",
        lw=1.6,
        label=f"standard ({form['peaks']['standard']} exit peak)",
    )
    ax.plot(
        t_ps[keep],
        pc_intensity[keep],
        color="tab:orange",
        lw=1.6,
        ls="--",
        label=f"conserving ({form['peaks']['conserving']} exit peak)",
    )
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"$|A|^2$ / $P_0$")
    ax.set_title(
        "(c) exit intensity: same envelope, different phase\n"
        f"relative complex-field difference {form['rel_diff_neg']:.2f}, "
        f"intensity difference {form['int_diff_neg']:.3f}\n"
        "with gamma < 0 the soliton condition is inverted, so this deck is a\n"
        "sign bench and fissions nothing; see the printed note"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = ["gamma > 0\n(ordinary silica)", "gamma < 0\n(sign bench)"]
    values = [max(form["rel_diff_pos"], 1e-16), form["rel_diff_neg"]]
    bars = ax.bar(labels, values, color=["tab:green", "tab:red"], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("relative field difference at the exit")
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1e}",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
        )
    ax.set_title(
        "(d) what the flag can and cannot do\n"
        "on a positive-gamma fiber the flag is a no-op to machine precision;\n"
        "the physics it changes only exists where gamma has the other sign"
    )
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.set_ylim(1e-16, 10)

    fig.suptitle(
        "conserving_shock changes the physics only where gamma does: "
        "the delayed arm carries |gamma|",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["comparison"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = _time.perf_counter()
    print(_rule("0. The photon-conserving shock, and the self-steepening grid guard"))
    print("  two independent mechanisms, two independent failure modes:")
    print("    conserving_shock=True : gamma -> |gamma| on the DELAYED (Raman) arm")
    print("                           only; a no-op when gamma > 0")
    print(
        f"    _SHOCK_TAYLOR_LIMIT   : tau_shock*Omega_max <= "
        f"{_SHOCK_TAYLOR_LIMIT:g}, a UserWarning at construction"
    )
    print("  The flag is NOT a fix for under-resolution. Section 3 measures that.")

    guard = guard_panel()
    form = formulation_panel()
    cons = conservation_panel()

    guard_figure(guard, cons)
    comparison_figure(form)

    print(_rule("Summary"))
    print(
        f"  guard: {sum(1 for r in guard['ladder'] if r['warned'])} of "
        f"{len(guard['ladder'])} ladder entries warn  [asserted non-empty]"
    )
    print(
        f"  guard: quiet only for Tmax >= {guard['required_tmax'] * 1e12:.0f} ps "
        f"(dt >= {guard['required_dt'] * 1e15:.1f} fs) at N = {N_POINTS}"
    )
    print(
        f"  guard: hard ValueError at Tmax = {guard['hard_tmax'] * 1e12:.1f} ps "
        "(Omega_max >= omega0)"
    )
    print(
        f"  gamma > 0: relative field difference {form['rel_diff_pos']:.1e}   "
        "[asserted < 1e-10 -- the flag is a no-op]"
    )
    print(
        f"  gamma < 0: relative field difference {form['rel_diff_neg']:.3f}   "
        "[asserted > 0.1]"
    )
    print(
        f"  gamma < 0: shift standard {form['shift_std']:+.5f} THz vs "
        f"conserving {form['shift_pc']:+.5f} THz"
    )
    print(
        f"  exit peaks: standard {form['peaks']['standard']}, "
        f"conserving {form['peaks']['conserving']}"
    )
    print(
        f"  drift at the finest grid: "
        f"{cons['finest_drift'] * 100:+.3f} %   [asserted < 0.3 %]"
    )
    print(
        f"  |drift| grows monotonically with tau*Omega_max: "
        f"{' -> '.join(f'{v * 100:.2f}' for v in cons['mags'])} %   [asserted]"
    )
    print(
        f"  formulation gap over the whole drift ladder: "
        f"{max(cons['gaps']) * 100:.4f} %   [asserted < 0.1 %]"
    )
    print("  Neither formulation repairs an under-resolved grid.")
    print(f"  runtime: {_time.perf_counter() - t0:.1f} s")
    print("  all checks passed ✓")

    print("\nGenerated files:")
    for key in ("comparison", "guard"):
        print(f"  {FIGS[key].relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
