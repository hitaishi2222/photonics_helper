"""Example: the vector / polarisation GNLSE — three couplings, one flag
========================================================================

``photonics_helper.vector_gnlse`` solves the *coupled* GNLSE on the two
polarisation components ``A = (A_x, A_y)``. The scalar engine models one
linearly-polarised channel; the physics that only exists when the field is
genuinely a vector lives here — per-axis dispersion, differential group delay,
cross-phase modulation, coherent polarisation FWM and Manakov averaging.

This example is about one keyword argument. ``VectorSplitStepEngine(coupling=…)``
selects between three physically different models of the *same* experiment, and
the difference between them is the assumption you make about the fibre's axes.
All three are run on one input pair here, side by side.

The three couplings, in equations
---------------------------------
With ``P_x = |A_x|²`` and ``P_y = |A_y|²``, the nonlinear step is

``"incoherent"`` (the PM/high-birefringence default)
    .. math::  D_x = \\gamma\\,(P_x + \\tfrac{2}{3} P_y), \\quad
               D_y = \\gamma\\,(P_y + \\tfrac{2}{3} P_x)

    ``_XPM_FACTOR = 2/3`` is the XPM anisotropy coefficient of the degenerate
    linearly-polarised mode pair (Agrawal §6.3). Both terms are advanced as
    exact unitary phase rotations — no power moves between the axes. This is
    the right model when the beat length is short enough that the polarisation
    FWM term averages away.

``"coherent"`` (low-birefringence, polarisation FWM retained)
    The same diagonal terms **plus** the non-diagonal mixing
    .. math::  (i/3)\\gamma\\, A_\\perp^2 A^*\\, e^{\\mp 2i\\Delta\\beta z}

    (``_FWM_FACTOR = 1/3``), advanced in frequency space by RK4 substeps
    Strang-split around the exact diagonal phase. The factor ``e^{∓2iΔβz}`` is
    the phase mismatch between the mixed modes and the drive; only when the beat
    length is comparable to or longer than the nonlinear length does it fail to
    rotate away, which is why this mode is *opt-in* and refuses
    ``delta_beta = 0`` without a warning (a fully resonant term exchanges power
    between the axes without bound).

``"manakov"`` (random birefringence, polarisation-averaged)
    .. math::  D_x = D_y = \\tfrac{8}{9}\\,\\gamma\\,(P_x + P_y)

    ``MANAKOV_FACTOR = 8/9`` (Wai & Menyuk, *J. Lightw. Technol.* **14**, 148
    (1996)): a fibre with beat length ≪ nonlinear length behaves on average
    like a scalar NLSE driven at ``8γ/9``. Requires identical per-axis
    dispersion and ``walkoff = 0`` — random axes average the walk-off out, and
    the engine enforces both.

The betas / delta_beta / walkoff contract
-----------------------------------------
These are three *different* quantities and the example never mixes them:

* ``betas`` / ``betas_x`` / ``betas_y`` are the **per-axis Taylor arrays**
  ``[β₂, β₃, …]`` (``betas_y`` defaults to ``betas_x``), in ``ps^k/m`` by
  default. Each axis is advanced by its *own* dispersion.
* ``delta_beta = β_x − β_y`` (rad/m) is the **mean** birefringence, and it is
  used *only* as the ``e^{−2iΔβz}`` mismatch of the coherent FWM term. It is
  not added to either axis's dispersion — that would double-count the
  mismatch. Passing a non-zero ``delta_beta`` with any other coupling raises.
* ``walkoff = β₁_y − β₁_x`` (s/m) is the differential **group delay**, applied
  as a first-order term to the y channel in the retarded frame of x, so the y
  pulse drifts by ``walkoff·z``. Manakov forbids it.

Beat length, not guesswork: the example derives ``delta_beta = 2π/L_beat`` from
a stated datasheet-style beat length, so the number passed to the engine can be
traced back to something a fibre datasheet actually states.

Two negative results worth stating
----------------------------------
* **The Manakov ``8/9`` equivalence is a single-channel statement.** The two
  models agree when ``P_y ≡ 0``: ``manakov`` at ``γ`` gives ``(8/9)γ P_x`` and
  ``incoherent`` at ``γ·8/9`` gives the same. With *both* channels populated
  they do **not** — ``(8/9)γ(P_x+P_y)`` is not ``γ'(P_x + ⅔P_y)`` for any single
  ``γ'`` unless ``P_x = P_y``. Section 2 checks the statement where it holds
  and says so out loud; it is the reason ``MANAKOV_FACTOR`` is exposed as a
  number the caller can use, rather than a hidden rescaling.
* **The ``"coherent"`` flag is not free accuracy.** It adds a term that is
  ``O(γP/3)``; where the mismatch oscillates it out, enabling it changes the
  answer rather than refining it.

Cost note: ``"coherent"`` is the expensive mode (RK4 substeps inside every
split step, capped at 2000 per step), so it runs on the smallest of the three
decks below; ``"incoherent"`` and ``"manakov"`` are exact phase rotations and
cost one ``exp`` per axis per step.
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
from photonics_helper.gnlse import FiberProfile
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.vector_gnlse import (
    MANAKOV_FACTOR,
    RandomBirefringenceEngine,
    VectorSplitStepEngine,
)

# ── Reference deck: PM fibre, two unequal channels ───────────────────────────
WL0 = 1550e-9
OMEGA0 = 2 * np.pi * C_MS / WL0
BETA2 = -21.0e-27  # s^2/m — SMF-28-ish, anomalous
BETA2_PS = BETA2 * 1e24  # ps^2/m

GAMMA = 1.3e-3  # 1/(W m)
LENGTH = 300.0  # m  (walk-off stays inside the Tmax window)
N_SAMPLES = 4096
TMAX = 600e-12  # s  (large enough that the walk-off never wraps the window)
NUM_STEPS = {"incoherent": 400, "coherent": 60, "manakov": 400}

PX = 8.0  # W, x-axis peak power
PY = 3.0  # W, y-axis peak power (unequal: the general case)
T0 = 30e-12  # s, pulse_width parameter of the sech envelope

# Physical birefringence numbers, derived — never guessed.
L_BEAT = 6.0e-3  # m, beat length (6 mm, strongly birefringent PM fibre)
DELTA_BETA = 2.0 * np.pi / L_BEAT  # rad/m  ->  beta_x - beta_y
WALKOFF = 1.0e-12  # s/m  ->  beta_1_y - beta_1_x
PER_AXIS_BETAS_PS = [BETA2_PS, 0.02]  # [beta2, beta3] in ps^k/m, per axis

DELAY_Y = 0.3e-12  # s, the y axis arrives 0.3 ps late (see _wave)
RANDOM_SEED = 20260917
ALT_SEED = 7  # a second realisation, to show the ensemble spread
MANAKOV_TOLERANCE = 1e-3  # the 8/9 check, relative difference of the output field
WRONG_FACTOR = 1.0  # negative control: a factor that is *not* 8/9
POWERS_W = np.array([1.0, 2.0, 4.0, 8.0])  # Manakov scaling sweep
DELTA_BETA_SWEEP = np.array([0.0, 1.0, 5.0, 20.0, 80.0])  # rad/m, coherent
WALKOFF_SWEEP = np.array([0.0, 1.0e-13, 5.0e-13, 1.0e-12])  # s/m

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "couplings": OUT_DIR / "37_vector_couplings.png",
    "manakov": OUT_DIR / "37_manakov_scaling.png",
}


def _rule(title: str) -> str:
    return "\n" + "=" * 68 + f"\n {title}\n" + "=" * 68


def _grid() -> TemporalGrid:
    return TemporalGrid(N=N_SAMPLES, Tmax=Time(TMAX, "s"))


def _fiber(length: float = LENGTH) -> FiberProfile:
    return FiberProfile.from_gamma(
        gamma=GAMMA, n2=2.6e-20, omega0=OMEGA0, length=Length(length, "m")
    )


def _wave(
    grid: TemporalGrid, power: float, pulse_width: float = T0, delay: float = 0.0
) -> Wave:
    """A sech pulse, optionally arriving ``delay`` seconds late.

    The delay is not decoration: with both axes *real and in phase* the
    coherent FWM term ``(i/3)γ A_y² A_x*`` is purely imaginary and moves no
    power at all — the resonance cancels it. Giving the y axis a small arrival
    offset restores the relative phase the term needs, and is the physical
    situation (two axes that do not arrive together) rather than a trick.
    """

    def func(t, T0_s, A0):
        return A0 / np.cosh((t - delay) / T0_s)

    env = Envelope(
        shape="custom" if delay else "sech",
        peak_amplitude=np.sqrt(power),
        pulse_width=Time(pulse_width, "s"),
        func=func if delay else None,
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))


def _input_pair(grid: TemporalGrid, px: float = PX, py: float = PY):
    return _wave(grid, px), _wave(grid, py, delay=DELAY_Y)


def _build(
    grid: TemporalGrid,
    coupling: str,
    *,
    px: float = PX,
    py: float = PY,
    gamma: float = GAMMA,
    length: float = LENGTH,
    delta_beta: float = DELTA_BETA,
    walkoff: float = WALKOFF,
    num_steps: int | None = None,
) -> VectorSplitStepEngine:
    """One engine, three couplings — every knob explicit.

    ``betas_x``/``betas_y`` carry the per-axis dispersion; ``delta_beta`` (only
    meaningful for ``"coherent"``) carries the mean birefringence of the FWM
    term; ``walkoff`` carries the differential group delay. Manakov forces
    ``walkoff = 0`` and identical per-axis betas, which the engine enforces —
    this example passes the walk-off only to the couplings that accept it.
    """
    fiber = FiberProfile.from_gamma(
        gamma=gamma, n2=2.6e-20, omega0=OMEGA0, length=Length(length, "m")
    )
    pulse_x, pulse_y = _input_pair(grid, px, py)
    common = dict(
        include_raman=False,
        step_size=Length(length / (num_steps or NUM_STEPS[coupling]), "m"),
    )
    if coupling == "manakov":
        return VectorSplitStepEngine(
            pulse_x,
            pulse_y,
            fiber,
            betas_x=np.array(PER_AXIS_BETAS_PS),
            betas_y=np.array(PER_AXIS_BETAS_PS),
            coupling="manakov",
            **common,
        )
    if coupling == "coherent":
        return VectorSplitStepEngine(
            pulse_x,
            pulse_y,
            fiber,
            betas_x=np.array(PER_AXIS_BETAS_PS),
            betas_y=np.array(PER_AXIS_BETAS_PS),
            coupling="coherent",
            delta_beta=delta_beta,
            walkoff=walkoff,
            **common,
        )
    return VectorSplitStepEngine(
        pulse_x,
        pulse_y,
        fiber,
        betas_x=np.array(PER_AXIS_BETAS_PS),
        betas_y=np.array(PER_AXIS_BETAS_PS),
        coupling="incoherent",
        walkoff=walkoff,
        **common,
    )


def _mixing(ax: np.ndarray, ay: np.ndarray, grid: TemporalGrid) -> float:
    """Largest fraction of the y-axis energy that ever leaves/joins the axis.

    Measured on the integral, not on a peak: the diagonal terms reshape a pulse
    (SPM + dispersion) without moving power, and a peak-height difference would
    report that as mixing. In a pure rotation model this is identically zero;
    the coherent FWM term is the only thing in the engine that makes it nonzero.
    Taken as the maximum over z, not the value at z = L: the resonance sloshes
    power back and forth, and a single end-snapshot can catch it near a turning
    point.
    """
    ey = np.sum(np.abs(ay) ** 2, axis=1) * grid.dt
    et = float(np.sum(np.abs(ax[0]) ** 2 + np.abs(ay[0]) ** 2) * grid.dt)
    return float(np.max(np.abs(ey - ey[0])) / max(et, 1e-30))


def _centroid(
    ax: np.ndarray, ay: np.ndarray, grid: TemporalGrid
) -> tuple[float, float]:
    t = np.asarray(grid.t)

    def ctr(a: np.ndarray) -> float:
        p = np.abs(a) ** 2
        return float(np.sum(p * t) / np.sum(p))

    return ctr(ax), ctr(ay)


def _walkoff(ay: np.ndarray, grid: TemporalGrid) -> float:
    """Measured y-axis group delay: centroid(z=L) − centroid(z=0), in ps."""
    c_in, c_out = _centroid(ay[0:1], ay[-1:], grid)
    return (c_out - c_in) * 1e12


# ── 1. The three couplings on one input ──────────────────────────────────────


def run_couplings() -> dict:
    grid = _grid()
    runs = {}
    for coupling in ("incoherent", "coherent", "manakov"):
        engine = _build(grid, coupling)
        engine.propagate(NUM_STEPS[coupling], nsaves=NUM_STEPS[coupling] + 1)
        runs[coupling] = engine

    print(_rule("1. One input, three couplings"))
    print(
        f"  deck: P_x = {PX} W, P_y = {PY} W, T0 = {T0 * 1e12:.0f} ps, "
        f"L = {LENGTH:.0f} m, gamma = {GAMMA:g} 1/(W m)"
    )
    print(
        f"  beta2 = {BETA2_PS * 1e3:.1f} ps^2/km (both axes), L_beat = {L_BEAT * 1e3:.0f} mm"
        f" -> delta_beta = {DELTA_BETA:.3g} rad/m"
    )
    print(
        f"  walkoff = {WALKOFF:g} s/m -> y drifts {WALKOFF * LENGTH * 1e12:.3g} ps "
        f"over the fibre\n"
    )
    print(
        f"  {'coupling':<14}{'|A_x|^2 peak':>13}{'|A_y|^2 peak':>13}"
        f"{'x->y mixing':>13}{'y delay (ps)':>14}{'energy drift':>14}"
    )
    for name, eng in runs.items():
        ax, ay = eng.fields_vs_z()
        peak_x = float(np.max(np.abs(ax[-1]) ** 2))
        peak_y = float(np.max(np.abs(ay[-1]) ** 2))
        e = eng.energy_vs_z
        drift = float(abs(e[-1] / e[0] - 1.0))
        print(
            f"  {name:<14}{peak_x:13.5g}{peak_y:13.5g}"
            f"{_mixing(ax, ay, grid):13.3e}{_walkoff(ay, grid):14.4g}{drift:14.2e}"
        )

    notes = {
        "incoherent": "PM fibre, XPM 2/3 (Agrawal 6.3); no power crosses axes",
        "coherent": "polarisation FWM (i/3)gamma A_perp^2 A* e^{+-2i db z}",
        "manakov": f"random axes, MANAKOV_FACTOR = {MANAKOV_FACTOR:.4f}, walk-off averaged out",
    }
    print()
    for name in runs:
        print(f"  {name:<12}: {notes[name]}")

    for name, eng in runs.items():
        e = eng.energy_vs_z
        drift = float(np.max(np.abs(e / e[0] - 1.0)))
        assert drift < 1e-3, f"{name}: total energy drifted {drift:.2e}"
    print("\n  total energy conserved to < 1e-3 (integrator error) in all three ✓")
    return runs, grid


# ── 2. The Manakov 8/9 scaling check ──────────────────────────────────────────


def run_manakov_check() -> tuple[float, float]:
    """manakov@gamma == incoherent@(gamma * 8/9), single-channel."""
    grid = _grid()
    length = 200.0

    def single_channel(coupling: str, gamma: float, steps: int = 400):
        fiber = FiberProfile.from_gamma(
            gamma=gamma, n2=2.6e-20, omega0=OMEGA0, length=Length(length, "m")
        )
        pulse_x = _wave(grid, PX)
        pulse_y = _wave(grid, 0.0)  # A_y == 0: the single-channel limit
        eng = VectorSplitStepEngine(
            pulse_x,
            pulse_y,
            fiber,
            betas_x=np.array(PER_AXIS_BETAS_PS),
            betas_y=np.array(PER_AXIS_BETAS_PS),
            coupling=coupling,
            include_raman=False,
            step_size=Length(length / steps, "m"),
        )  # type: ignore[arg-type]
        eng.propagate(steps, nsaves=3)
        return eng.fields_vs_z()[0][-1]

    a_manakov = single_channel("manakov", GAMMA)
    a_inc = single_channel("incoherent", GAMMA * MANAKOV_FACTOR)
    a_wrong = single_channel("incoherent", GAMMA * WRONG_FACTOR)

    rel = float(np.linalg.norm(a_manakov - a_inc) / np.linalg.norm(a_manakov))
    rel_wrong = float(np.linalg.norm(a_manakov - a_wrong) / np.linalg.norm(a_manakov))

    print(_rule("2. The Manakov 8/9 check"))
    print("  statement: with A_y == 0, coupling='manakov' at gamma equals")
    print("             coupling='incoherent' at gamma * 8/9 (MANAKOV_FACTOR)")
    print(f"  gamma                         : {GAMMA:g} 1/(W m)")
    print(f"  incoherent gamma              : {GAMMA * MANAKOV_FACTOR:.6g} 1/(W m)")
    print(
        f"  ||A_manakov - A_incoherent|| / ||A_manakov|| = {rel:.3e}"
        f"   (tolerance {MANAKOV_TOLERANCE:g})"
    )
    print(
        f"  negative control, incoherent at gamma*{WRONG_FACTOR:g}"
        f" instead of gamma*8/9 : {rel_wrong:.3e}"
    )
    print("  -> MANAKOV_FACTOR = 8/9 is the polarisation-averaged nonlinearity")
    print("     of Wai & Menyuk (1996); with BOTH channels populated the two")
    print("     models are different models (see the module docstring), which")
    print("     is why this equivalence is checked where it actually holds.")
    assert rel < MANAKOV_TOLERANCE, f"Manakov 8/9 check failed: {rel:.3e}"
    assert rel_wrong > 100 * MANAKOV_TOLERANCE, (
        f"negative control only reached {rel_wrong:.3e}; the 8/9 assertion "
        "is not discriminating and would pass for a wrong factor"
    )
    return rel, rel_wrong


def manakov_scaling() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear phase vs power: manakov@gamma vs incoherent@gamma*8/9.

    A CW input (long flat-top envelope) makes the *linear* dispersion phase
    uniform, so what is left in ``arg(A_L/A_0)`` is the nonlinear phase. Both
    curves must be the same straight line through the origin with slope
    ``(8/9)gamma L P``. The phases are unwrapped along the power sweep — the
    measured phase passes π near P = 5 W and a wrapped fit would come out with
    the wrong slope.
    """
    grid = TemporalGrid(N=N_SAMPLES, Tmax=Time(TMAX, "s"))
    length = 500.0
    cw_width = 100e-12  # flat top wide enough that dispersion is negligible

    phases = {"manakov": [], "incoherent@8/9": []}
    for p in POWERS_W:
        for label, coupling, gamma in (
            ("manakov", "manakov", GAMMA),
            ("incoherent@8/9", "incoherent", GAMMA * MANAKOV_FACTOR),
        ):
            fiber = FiberProfile.from_gamma(
                gamma=gamma, n2=2.6e-20, omega0=OMEGA0, length=Length(length, "m")
            )
            eng = VectorSplitStepEngine(
                _wave(grid, p, cw_width),
                _wave(grid, 0.0, cw_width),
                fiber,
                betas_x=np.array([0.0]),
                betas_y=np.array([0.0]),
                coupling=coupling,
                include_raman=False,
                step_size=Length(length / 400, "m"),
            )  # type: ignore[arg-type]
            eng.propagate(400, nsaves=2)
            a0 = np.asarray(eng.evolution_x[0].envelope_field)
            aL = np.asarray(eng.evolution_x[-1].envelope_field)
            i = int(np.argmax(np.abs(a0) ** 2))
            phases[label].append(float(np.angle(aL[i] / a0[i])))

    man = np.unwrap(np.array(phases["manakov"]))
    inc = np.unwrap(np.array(phases["incoherent@8/9"]))
    slope_man = float(np.polyfit(POWERS_W, man, 1)[0])
    slope_inc = float(np.polyfit(POWERS_W, inc, 1)[0])
    predicted = MANAKOV_FACTOR * GAMMA * length
    print(_rule("2b. Manakov scaling: nonlinear phase vs power"))
    print(f"  predicted slope d(phi)/dP = (8/9) gamma L = {predicted:.6g} rad/W")
    print(f"  measured slope, manakov                    = {slope_man:.6g} rad/W")
    print(f"  measured slope, incoherent at (8/9) gamma  = {slope_inc:.6g} rad/W")
    print(f"  ratio manakov / predicted : {slope_man / predicted:.4f}")
    print(
        f"  max |phi_manakov - phi_incoherent| over the power sweep: "
        f"{float(np.max(np.abs(man - inc))):.3e} rad"
    )
    assert abs(slope_man - slope_inc) < 1e-6 * max(abs(slope_man), 1.0), (
        f"Manakov and incoherent 8/9 slopes differ: {slope_man:.6g} vs {slope_inc:.6g}"
    )
    return POWERS_W, man, inc


# ── 3. delta_beta and walkoff, independently ─────────────────────────────────


def delta_beta_sweep() -> dict:
    """Axis mixing vs the birefringence mismatch, with walk-off switched off."""
    grid = TemporalGrid(N=2048, Tmax=Time(TMAX, "s"))
    length = 200.0
    mixing = []
    for db in DELTA_BETA_SWEEP:
        eng = VectorSplitStepEngine(
            *_input_pair(grid),
            _fiber(length),
            betas_x=np.array(PER_AXIS_BETAS_PS),
            betas_y=np.array(PER_AXIS_BETAS_PS),
            coupling="coherent",
            delta_beta=float(db),
            walkoff=0.0,  # isolate the mismatch from the group delay
            include_raman=False,
            step_size=Length(length / 40, "m"),
        )
        eng.propagate(40, nsaves=40)
        ax, ay = eng.fields_vs_z()
        mixing.append(_mixing(ax, ay, grid))
        # the coherent nonlinear step's diagonal drive, for context
        diag = float(GAMMA * (PX + 2.0 / 3.0 * PY) * length)
        print(
            f"  delta_beta = {db:8.3g} rad/m (L_beat = "
            f"{'inf' if db == 0 else f'{2 * np.pi / db * 1e3:.3g} mm'})"
            f" -> peak |x->y| mixing {mixing[-1]:.4%}  [gamma P L = {diag:.3f} rad]"
        )

    print("  -> two things are visible here, and only one of them is naive:")
    print("     the exchange grows once the mismatch leaves the fully resonant")
    print("     point and then falls again towards the PM limit, where the")
    print("     term averages out. At delta_beta = 0 the exchange is ~0: with")
    print("     a fully resonant term and axes that arrive out of phase, the")
    print("     (i/3)gamma A_y^2 A_x* contribution is purely imaginary at z = 0")
    print("     and moves no power. Resonance is not the same as maximum")
    print("     exchange.")
    peak = int(np.argmax(mixing))
    assert peak not in (0, len(mixing) - 1) and mixing[peak] > 5 * mixing[-1], (
        f"mixing did not peak away from resonance ({mixing}); the mismatch is "
        "not being applied"
    )
    return {"delta_beta": DELTA_BETA_SWEEP, "mixing": np.array(mixing)}


def walkoff_sweep() -> dict:
    """Group delay vs differential group velocity, with mixing switched off."""
    grid = TemporalGrid(N=2048, Tmax=Time(TMAX, "s"))
    length = 100.0  # short enough that the largest delay stays inside Tmax
    delays = []
    for w in WALKOFF_SWEEP:
        eng = VectorSplitStepEngine(
            *_input_pair(grid),
            _fiber(length),
            betas_x=np.array(PER_AXIS_BETAS_PS),
            betas_y=np.array(PER_AXIS_BETAS_PS),
            coupling="incoherent",
            walkoff=float(w),
            include_raman=False,
            step_size=Length(length / 200, "m"),
        )
        eng.propagate(200, nsaves=3)
        ax, ay = eng.fields_vs_z()
        delays.append(_walkoff(ay, grid))
        print(
            f"  walkoff = {w:9.3g} s/m -> predicted delay {w * length * 1e12:+.5g} ps,"
            f" measured centroid shift {delays[-1]:+.5g} ps"
        )

    print("  -> the y pulse drifts by exactly walkoff * L; no power crossed axes")
    print("     (coupling='incoherent' has no mixing term).")
    residual = np.max(np.abs(np.array(delays) - WALKOFF_SWEEP * length * 1e12))
    assert residual < 5.0, f"walk-off centroid mismatch {residual:.3g} ps"
    return {
        "walkoff": WALKOFF_SWEEP,
        "delay": np.array(delays),
        "predicted": WALKOFF_SWEEP * length * 1e12,
    }


# ── 4. Random birefringence: Poincaré trajectories ───────────────────────────


def stokes(ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    """Unit-vector Stokes coordinates from a Jones pair: (s1, s2, s3)/s0."""
    s0 = np.abs(ax) ** 2 + np.abs(ay) ** 2
    s1 = 2 * np.real(ax * np.conj(ay))
    s2 = 2 * np.imag(ax * np.conj(ay))
    s3 = np.abs(ax) ** 2 - np.abs(ay) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([s1 / s0, s2 / s0, s3 / s0], axis=-1)


def run_random_birefringence() -> dict:
    """Poincaré trajectories of seeded random-birefringence realisations."""
    grid = TemporalGrid(N=2048, Tmax=Time(120e-12, "s"))
    length = 300.0
    steps = 120  # one random SU(2) frame per split step

    def track(seed: int | None) -> np.ndarray:
        cls = RandomBirefringenceEngine if seed is not None else VectorSplitStepEngine
        eng = cls(
            _wave(grid, PX),
            _wave(grid, PY),
            _fiber(length),
            np.array(PER_AXIS_BETAS_PS),
            include_raman=False,
            step_size=Length(length / steps, "m"),
            **({} if seed is None else {"seed": seed}),
        )
        eng.propagate(steps, nsaves=steps + 1)
        ax, ay = eng.fields_vs_z()
        # Track the state at the pulse peak, not at a grid edge: the Stokes
        # ratios are s/s0 and s0 there is ~1e-8 at the window edge, so a sample
        # index would be reporting noise. The trajectory is a Jones-vector
        # statement about one sample of the field.
        i_peak = int(np.argmax(np.abs(ax[0]) ** 2 + np.abs(ay[0]) ** 2))
        return stokes(ax[:, i_peak], ay[:, i_peak])

    key_a = f"RandomBirefringenceEngine (seed={RANDOM_SEED})"
    key_b = f"RandomBirefringenceEngine (seed={ALT_SEED})"
    key_f = "VectorSplitStepEngine (fixed axes)"
    traj = {key_a: track(RANDOM_SEED), key_b: track(ALT_SEED), key_f: track(None)}
    rb, alt, fx = traj[key_a], traj[key_b], traj[key_f]

    def path_len(s: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))

    print(_rule("4. Random birefringence on the Poincaré sphere"))
    print(
        f"  seeds: {RANDOM_SEED} and {ALT_SEED} — printed so the "
        f"realisations are reproducible"
    )
    print(
        f"  segments: {steps} random SU(2) frames over {length:.0f} m "
        f"(one per split step of {length / steps:.1f} m)"
    )
    print(
        f"  start                      : s1={rb[0, 0]:+.4f} "
        f"s2={rb[0, 1]:+.4f} s3={rb[0, 2]:+.4f}"
    )
    print(
        f"  fixed axes, z=L            : s1={fx[-1, 0]:+.4f} "
        f"s2={fx[-1, 1]:+.4f} s3={fx[-1, 2]:+.4f}"
    )
    print(
        f"  random seed={RANDOM_SEED}, z=L: s1={rb[-1, 0]:+.4f} "
        f"s2={rb[-1, 1]:+.4f} s3={rb[-1, 2]:+.4f}"
    )
    print(
        f"  random seed={ALT_SEED}, z=L    : s1={alt[-1, 0]:+.4f} "
        f"s2={alt[-1, 1]:+.4f} s3={alt[-1, 2]:+.4f}"
    )
    print(
        "  path length on the sphere  : "
        + ", ".join(
            f"{'random' if k is not key_f else 'fixed'} {path_len(v):.3f} rad"
            for k, v in traj.items()
        )
    )
    print("  -> with fixed, non-degenerate axes |A_x| and |A_y| never change (the")
    print("     incoherent step is a pure phase rotation) and the differential")
    print("     nonlinear phase gamma (P_x - P_y) z turns the linear polarisation")
    print("     in place along one smooth arc — the same result every run. The")
    print("     random engine rotates the frame before every nonlinear step, so")
    print("     each realisation walks a different path and lands somewhere")
    print("     else: that ensemble spread is what gets averaged into")
    print(
        f"     MANAKOV_FACTOR = {MANAKOV_FACTOR:.4f} (section 2 checks the "
        f"averaged limit)."
    )

    spread = float(np.linalg.norm(rb[-1] - alt[-1]))
    to_fixed = float(np.linalg.norm(rb[-1] - fx[-1]))
    print(
        f"  |Stokes(seed {RANDOM_SEED}) - Stokes(seed {ALT_SEED})| = "
        f"{spread:.3f} at z = L   (realisation to realisation)"
    )
    print(
        f"  |Stokes(random) - Stokes(fixed axes)|      = {to_fixed:.3f}   "
        f"(randomisation vs the fixed-axis result)"
    )
    print("  -> the two realisations are far closer to each other than either is")
    print("     to the fixed-axis run, and both stay near the input state. That")
    print("     tightness is what makes the ensemble average well defined — and")
    print("     it is why the averaged answer is a single number (8/9 gamma)")
    print("     rather than a range. The randomisation changes the *answer*, not")
    print("     the reproducibility of a given run.")
    assert 1e-3 < spread < to_fixed / 5.0, (
        f"seed-to-seed scatter {spread:.3f} vs random-vs-fixed {to_fixed:.3f}: "
        "the randomisation is either not applied, or dominates the spread"
    )
    assert np.array_equal(track(None), fx), (
        "the fixed-axes run is not reproducible run to run"
    )
    print("  the fixed-axes run reproduces bit-for-bit; the random ones scatter ✓")
    return traj


# ── Figures ───────────────────────────────────────────────────────────────────


def _draw_sphere(ax, pts: np.ndarray | None = None) -> None:
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 80)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="0.88",
        alpha=0.55,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    if pts is None:
        span = 1.0
        centre = np.zeros(3)
    else:
        # Zoom onto where the trajectory actually lives; the state starts
        # nearly linear-polarised, so the full-sphere view is 95% empty space.
        centre = pts.mean(axis=0)
        span = float(np.max(np.abs(pts - centre))) * 1.15 + 0.05
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
    ax.set_xlabel("$s_1$", labelpad=-8)
    ax.set_ylabel("$s_2$", labelpad=-8)
    ax.set_zlabel("$s_3$", labelpad=-8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-58)


def couplings_figure(runs: dict, grid: TemporalGrid, db: dict, wo: dict) -> None:
    fig = plt.figure(figsize=(15.5, 8.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.55)

    colours = {"incoherent": "tab:blue", "coherent": "tab:red", "manakov": "tab:green"}
    labels = {
        "incoherent": "incoherent (PM, XPM 2/3)",
        "coherent": "coherent (+ polarisation FWM)",
        "manakov": "manakov (8/9, random axes)",
    }

    ax = fig.add_subplot(gs[0, 0])
    for name, eng in runs.items():
        axf, ayf = eng.fields_vs_z()
        t = np.asarray(grid.t) * 1e12
        m = np.abs(t) < 5 * T0 * 1e12
        ax.plot(
            t[m],
            np.abs(axf[0][m]) ** 2 / PX,
            lw=1.2,
            color="0.5",
            label="input x" if name == "incoherent" else None,
        )
        ax.plot(
            t[m], np.abs(axf[-1][m]) ** 2 / PX, color=colours[name], label=labels[name]
        )
    ax.set_title("x-axis power (normalised)")
    ax.set_xlabel("retarded time (ps)")
    ax.set_ylabel("|A$_x$|$^2$ / $P_x$")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    for name, eng in runs.items():
        axf, ayf = eng.fields_vs_z()
        t = np.asarray(grid.t) * 1e12
        m = np.abs(t) < 2.0 * LENGTH * abs(WALKOFF) * 1e12
        ax.plot(
            t[m],
            np.abs(ayf[0][m]) ** 2 / PY,
            lw=1.2,
            color="0.5",
            label="input y" if name == "incoherent" else None,
        )
        ax.plot(
            t[m], np.abs(ayf[-1][m]) ** 2 / PY, color=colours[name], label=labels[name]
        )
    ax.set_title("y-axis power — the group delay moves this one", fontsize=10)
    ax.annotate(
        f"window ±2 × walk-off × L = ±{2 * LENGTH * WALKOFF * 1e12:.0f} ps",
        (0.02, 0.92),
        xycoords="axes fraction",
        fontsize=8,
    )
    ax.set_xlabel("retarded time (ps)")

    ax = fig.add_subplot(gs[0, 2])
    for name, eng in runs.items():
        e = eng.energy_vs_z / eng.energy_vs_z[0]
        ax.plot(
            eng.z_array, (e - 1.0) * 1e13, "-", lw=1.3, color=colours[name], label=name
        )
    ax.set_title("total energy — conserved in all three")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("(E(z)/E(0) − 1)  × 10$^{-13}$")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(
        db["delta_beta"][1:] * 1e3, db["mixing"][1:] * 1e2, "o-", color="tab:red"
    )
    ax.set_xlabel("delta_beta (mrad/m)")
    ax.set_ylabel("peak |x→y| mixing (%)")
    peak_i = int(np.argmax(db["mixing"]))
    ax.set_title("delta_beta sweep — axis mixing (walk-off = 0)", fontsize=10)
    ax.annotate(
        f"{db['mixing'][0]:.2%} at resonance, peak "
        f"{db['mixing'][peak_i]:.2%} at "
        f"{DELTA_BETA_SWEEP[peak_i]:.0f} rad/m, "
        f"{db['mixing'][-1]:.2%} at {DELTA_BETA_SWEEP[-1]:.0f}",
        (0.02, 0.03),
        xycoords="axes fraction",
        fontsize=8,
    )
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(
        wo["predicted"],
        wo["delay"],
        "o",
        color="tab:blue",
        label="measured centroid shift",
    )
    lim = [
        min(wo["predicted"].min(), wo["delay"].min()) - 5,
        max(wo["predicted"].max(), wo["delay"].max()) + 5,
    ]
    ax.plot(lim, lim, "-", color="0.4", lw=1.0, label="walkoff × L")
    ax.set_xlabel("walkoff × L (ps)")
    ax.set_ylabel("measured y centroid shift (ps)")
    ax.set_title("walk-off sweep — temporal separation (no mixing)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    names = list(runs)
    mixes = [_mixing(*runs[n].fields_vs_z(), grid) for n in names]
    print(
        "  peak |x->y| mixing across the deck: "
        + ", ".join(f"{n} {m:.2%}" for n, m in zip(names, mixes))
    )
    ax.barh(names, mixes, color=[colours[n] for n in names])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlabel("peak |x→y| power mixing over z")
    ax.set_title("only the coherent FWM term moves power", fontsize=10)
    for i, v in enumerate(mixes):
        ax.annotate(
            f"{v:.2%}",
            (v, i),
            va="center",
            ha="left",
            xytext=(5, 0),
            textcoords="offset points",
        )

    ax = fig.add_subplot(gs[2, :])
    t = np.asarray(grid.t) * 1e12
    m = np.abs(t) < 2.0 * LENGTH * abs(WALKOFF) * 1e12
    for name, eng in runs.items():
        axf, ayf = eng.fields_vs_z()
        ax.plot(
            t[m],
            np.abs(axf[-1][m]) ** 2 / PX,
            color=colours[name],
            label=f"{labels[name]} — x",
        )
        ax.plot(
            t[m],
            np.abs(ayf[-1][m]) ** 2 / PY,
            color=colours[name],
            ls=":",
            label=f"{labels[name]} — y",
        )
    ax.set_title(
        "output on both axes at z = L — solid: x, dotted: y "
        "(the same pulse displaced by the walk-off)",
        fontsize=10,
    )
    ax.set_xlabel("retarded time (ps)")
    ax.set_ylabel("normalised power")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Vector GNLSE — the same input through coupling = "
        "'incoherent' / 'coherent' / 'manakov'",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS["couplings"], dpi=130, bbox_inches="tight")
    plt.close(fig)


def manakov_figure(
    powers: np.ndarray, man: np.ndarray, inc: np.ndarray, traj: dict
) -> None:
    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(powers, man, "o-", color="tab:green", label="coupling='manakov', γ")
    ax.plot(powers, inc, "s--", color="tab:blue", label="coupling='incoherent', γ·8/9")
    pp = np.linspace(0, powers.max() * 1.1, 50)
    ax.plot(
        pp,
        pp * MANAKOV_FACTOR * GAMMA * 500.0,
        "-",
        color="0.4",
        lw=1.0,
        label="(8/9)·γ·L·P",
    )
    ax.set_xlabel("input power P (W)")
    ax.set_ylabel("nonlinear phase at the pulse peak (rad)")
    ax.set_title("the 8/9 rescaling: two models, one line")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for k in (1, 2):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        if k == 1:
            _draw_sphere(
                ax,
                np.concatenate(
                    [
                        traj[f"RandomBirefringenceEngine (seed={RANDOM_SEED})"],
                        traj[f"RandomBirefringenceEngine (seed={ALT_SEED})"],
                    ]
                ),
            )
            for label, colour in (
                (f"RandomBirefringenceEngine (seed={RANDOM_SEED})", "tab:red"),
                (f"RandomBirefringenceEngine (seed={ALT_SEED})", "tab:orange"),
            ):
                s = traj[label]
                ax.plot(
                    s[:, 0],
                    s[:, 1],
                    s[:, 2],
                    color=colour,
                    lw=1.4,
                    label=f"seed {label.split('=')[1][:-1]}",
                )
                ax.scatter(*s[-1], color=colour, s=30)
            ax.plot([], [], " ", label="start (black dot)")
            ax.scatter(
                *traj[f"RandomBirefringenceEngine (seed={RANDOM_SEED})"][0],
                color="k",
                s=30,
            )
            ax.set_title(
                f"RandomBirefringenceEngine\nseeds {RANDOM_SEED} / {ALT_SEED}",
                fontsize=10,
            )
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
        else:
            s = traj["VectorSplitStepEngine (fixed axes)"]
            _draw_sphere(ax, s)
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color="tab:blue", lw=1.6)
            ax.scatter(*s[0], color="k", s=30)
            ax.scatter(*s[-1], color="tab:blue", s=30)
            ax.set_title(
                "VectorSplitStepEngine, fixed axes\none deterministic arc", fontsize=10
            )

    fig.suptitle(
        "Manakov scaling check and random-birefringence polarisation trajectories",
        y=1.02,
    )
    fig.savefig(FIGS["manakov"], dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.perf_counter()
    print(_rule("0. The vector GNLSE — one flag, three models"))
    print(f"  coupling: MANAKOV_FACTOR = {MANAKOV_FACTOR:.6f} (Wai & Menyuk 1996)")
    print(
        f"  delta_beta = 2 pi / L_beat = 2 pi / {L_BEAT * 1e3:.0f} mm "
        f"= {DELTA_BETA:.4g} rad/m"
    )
    print(f"  walkoff   = beta1_y - beta1_x = {WALKOFF:g} s/m")

    runs, grid = run_couplings()
    rel_ok, rel_wrong = run_manakov_check()
    powers, man, inc = manakov_scaling()
    print(_rule("3. delta_beta and walk-off, independently"))
    db = delta_beta_sweep()
    wo = walkoff_sweep()
    traj = run_random_birefringence()

    print(_rule("Summary"))
    print("  coupling='incoherent'  : PM fibre, XPM 2/3, no axis exchange")
    print("  coupling='coherent'    : + polarisation FWM, mismatched by delta_beta")
    print(
        f"  coupling='manakov'     : polarisation-averaged, factor {MANAKOV_FACTOR:.4f}"
    )
    print(
        f"  Manakov 8/9 check       : passed (relative error {rel_ok:.3e}, "
        f"tolerance {MANAKOV_TOLERANCE:g})"
    )
    print(
        f"  negative control        : wrong factor {WRONG_FACTOR:g} gives "
        f"{rel_wrong:.3e} — the check bites ✓"
    )
    print(
        f"  delta_beta mixing       : {db['mixing'][0]:.3%} at resonance, "
        f"peak {db['mixing'].max():.3%} at "
        f"{DELTA_BETA_SWEEP[int(np.argmax(db['mixing']))]:.0f} rad/m, "
        f"{db['mixing'][-1]:.3%} at {DELTA_BETA_SWEEP[-1]:.0f} rad/m"
    )
    print(
        f"  walk-off centroid shift : matches walk-off x L to "
        f"{float(np.max(np.abs(wo['delay'] - wo['predicted']))):.2e} ps"
    )
    print("  all checks passed ✓")
    print(
        f"\n  runtime {time.perf_counter() - t0:.1f} s (the coherent runs, "
        f"which RK4-substep the FWM term inside every split step, dominate)"
    )

    couplings_figure(runs, grid, db, wo)
    manakov_figure(powers, man, inc, traj)
    print("\nGenerated files:")
    for p in FIGS.values():
        print(f"  {p.relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
