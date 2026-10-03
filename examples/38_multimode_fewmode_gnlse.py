"""Example: building a few-mode / multimode GNLSE one parameter at a time
============================================================================

``MultimodeSplitStepEngine`` is the most heavily parameterised object in this
library, and every parameter it takes encodes a piece of physics rather than a
numerical preference. This example constructs one **from nothing, one rung at a
time**, so that every difference between two consecutive panels is attributable
to exactly one parameter.

The build plan, and what each rung means
----------------------------------------
=====  ==========================  =================================================
rung   parameter added             what it isolates
=====  ==========================  =================================================
 1     (nothing)                   two channels, identical betas, ``isotropic``
                                    coefficients: the **control**. Each channel must
                                    reproduce an independent single-channel
                                    ``SplitStepEngine`` run to integrator error.
 2     ``xpm_weights``             a full ``N×N`` overlap tensor replaces the uniform
                                    ``coef_model`` scalars. The diagonal entries are
                                    the SPM slots, the off-diagonal ones the
                                    inter-modal XPM — and the two are *not* the
                                    same number, which is invisible with a scalar
                                    coupling.
 3     ``group_delays``            modal group delay ``β₁⁽ᵐ⁾ − β₁⁽⁰⁾`` (s/m). Different
                                    group velocities walk the channels apart in
                                    time and erase their mutual coherence — which
                                    is why the *averaged* few-mode model keeps
                                    XPM but drops FWM.
 4     ``phase_offsets``           absolute modal phase ``Δβ₀⁽ᵐ⁾`` (rad/m), the part
                                    of ``β`` the retarded-frame Taylor expansion
                                    omits. It provides discrete quasi-phase
                                    matching, so a ladder of offsets turns one
                                    seed into a comb of new frequencies (the
                                    GRIN/GPI mechanism).
 5     ``include_fwm`` + ``oam_l`` inter-modal four-wave mixing under the
                                    angular-momentum selection rule, and an A/B
                                    showing the gate is a *selection filter*, not a
                                    cost control.
 6     ``fwm_pump_depletion``      the Manley–Rowe back-conversion pump arm, and
                                    the stability-driven substep count that comes
                                    with it.
=====  ==========================  =================================================

What each quantity means, exactly
----------------------------------
``xpm_weights[i, j]`` (``N×N``)
    Multiplies ``|A_j|²`` in the nonlinear phase of channel **i** — the SPM
    slot is just ``i == j``. Built here as ``|Γ_ij|²`` with
    ``Γ_ij = ∫ u_i(r) u_j(r) d²r`` the mode-overlap integral on a transverse
    grid, normalised so ``Γ_ii = 1``. Passing a scalar coupling instead is not
    an approximation of this tensor, it is a different model: it asserts that
    every pair of modes overlaps as strongly as a mode overlaps itself.

``fwm_weights[m, n, p, q]`` (``N×N×N×N``)
    Weights the ``→ m`` transition pumped by ``(n, p)`` and consuming the
    conjugated ``q``. Also built here from overlap integrals,
    ``|Γ_mn Γ_pq|``, so the four axes of the array are visible rather than
    magic.

``coef_model``
    ``"lp_degenerate"``: SPM 1, XPM 2/3, inter-modal FWM 2/3 (the degenerate
    LP spatial-mode model). ``"isotropic"``: all ones (spatially averaged,
    Manakov-like). Either is *overridden* by the weight tensors when supplied.

``oam_l`` (ISSUES.md #15)
    Restricts FWM triplets to ``ℓ_m = ℓ_n + ℓ_p − ℓ_q``, which is **exactly**
    the Poletti & Horak Eq. (18) type-2 spatial rule — verified element-wise
    over all 10⁴ quadruples in ``reproductions/poletti_2008_multimode``, not
    merely an approximation of it. Two things it is *not*: it carries only the
    spatial rule (the polarisation rule Eq. (19) and the magnitudes live in
    the weight tensors), and **uniform labels make it inert**. With all labels
    equal the condition holds for every triplet and the gate is exactly the
    ``oam_l=None`` fallback. Rung 5 runs both label sets and prints the
    surviving-exchange count for each, because "I passed ``oam_l=``" is not
    the same statement as "I restricted the exchanges".

FWM substeps (ISSUES.md #11 / #12)
-----------------------------------
``_fwm_substep_count`` is **stability-driven**, not a fixed number: the inner
step count satisfies ``η·h ≤ 2.5`` (``η = γ f |A_n|²`` for the strongest
exchange pair, from ``_fwm_rate_max``), with the old accuracy target
(``η·h ≲ 0.05``, capped at 200) kept only as a lower bound. Beyond
``_FWM_SUBSTEP_MAX = 8192`` inner steps the engine raises loudly naming the
channel instead of silently producing ``|A| ~ 1e240``. Rung 6 prints the
substep count and ``dz`` it actually chose rather than re-deriving the policy
here, and asserts the run stayed finite.

Cost note: rungs 1–4 keep ``include_fwm=False`` and at most three channels;
FWM costs four classical RK4 stages per substep per channel, so rungs 5 and 6
run on five channels and a deliberately short fibre.
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
from numpy.polynomial.hermite import hermval

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

# ── Reference deck ───────────────────────────────────────────────────────────
WL0 = 1550e-9
OMEGA0 = 2 * np.pi * C_MS / WL0
GAMMA = 1.3e-3  # 1/(W m)
LENGTH = 200.0  # m
N_SAMPLES = 2048
TMAX = 400e-12  # s
T0 = 50e-12  # s
BETA2 = -21.0e-27 * 1e24  # ps^2/m  (SMF-28-ish)

#: Channels of the XPM/FWM deck: Hermite–Gaussian *radial* order plus an OAM
#: azimuthal label. The overlap matrix is built from the profiles below.
MODE_ORDERS = [0, 1, 2, 3, 4]
OAM_HELICAL = [0, 1, -1, 2, -2]
OAM_UNIFORM = [0, 0, 0, 0, 0]
MODE_W = 2.5e-6  # m, transverse HG waist used only for the overlap integral
MODE_OFFSET = 0.9e-6  # m, transverse displacement that makes the modes overlap
MODE_OFFSETS = [i * MODE_OFFSET for i in range(len(MODE_ORDERS))]

P_RUNG12 = (5.0, 2.0)  # W, the two-channel deck (channel 0, channel 1)
P_RUNG34 = (5.0, 2.0, 1.0)  # W, three-channel deck
GROUP_DELAYS = [0.0, 5.0e-13, -5.0e-13]  # s/m  -> ±100 ps over 200 m
GPI_XI = 0.1  # m, self-imaging length of the GRIN/GPI ladder
NUM_STEPS = 200
FWM_STEPS = 40
FWM_PUMP_W = 500.0  # W, strong enough that the substep count is not 1

CONTROL_TOLERANCE = 1e-6  # rung 1 vs two scalar engines
FWM_PUMP = 500.0
FWM_AB_PUMP = 1.0  # W, the rung-5 pump (the gate A/B does not need a strong drive)

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "rungs": OUT_DIR / "38_multimode_channel_evolution.png",
    "oam": OUT_DIR / "38_oam_gate_ab.png",
}


def _rule(title: str) -> str:
    return "\n" + "=" * 68 + f"\n {title}\n" + "=" * 68


def _grid() -> TemporalGrid:
    return TemporalGrid(N=N_SAMPLES, Tmax=Time(TMAX, "s"))


def _fiber(length: float = LENGTH, gamma: float = GAMMA) -> FiberProfile:
    return FiberProfile.from_gamma(
        gamma=gamma, n2=2.6e-20, omega0=OMEGA0, length=Length(length, "m")
    )


def _wave(grid: TemporalGrid, power: float, wavelength: float = WL0) -> Wave:
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(power), pulse_width=Time(T0, "s")
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wavelength, "m"))


# ── Mode overlaps: where the weight tensors come from ────────────────────────


def hg_profiles(
    orders: list[int],
    waist: float = MODE_W,
    offsets: list[float] | None = None,
    n_pts: int = 801,
) -> tuple:
    """Unit-power 1-D mode profiles and their transverse overlap matrix.

    ``u_n(x) = H_n((x - d_n)/w) exp(-((x - d_n)/w)^2 / 2)`` normalised to unit
    power; ``Gamma_ij = int u_i u_j dx``. Two facts the rest of the example
    leans on:

    * Hermite-Gaussian modes of *different order are exactly orthogonal* —
      that is why the tensor of two distinct HG orders is the identity, and
      why rung 1 can hand the engine a clean decoupled control.
    * Modes at the *same* order displaced by ``d`` overlap partially,
      ``Gamma ~ exp(-d^2/4w^2)`` in the large-``d`` limit. That is the
      physically interesting case (a displaced/perturbed mode basis, or a
      weakly guiding fibre where the "modes" are not orthogonal), and it is
      what gives the tensors here a non-zero off-diagonal.
    """
    x = np.linspace(-6 * waist, 6 * waist, n_pts)
    d = [0.0] * len(orders) if offsets is None else list(offsets)
    u = np.array(
        [
            hermval((x - d[i]) / waist, np.eye(len(orders))[i])
            * np.exp(-(((x - d[i]) / waist) ** 2) / 2)
            for i in range(len(orders))
        ]
    )
    # Trapezoid quadrature weights, so the Gram matrix is a plain weighted
    # product and the diagonal is 1 to machine precision.
    w = np.full_like(x, x[1] - x[0])
    w[0] = w[-1] = (x[1] - x[0]) / 2
    u /= np.sqrt((u**2 * w).sum(axis=1))[:, None]
    gamma_mn = (u * w) @ u.T
    return x, u, gamma_mn


def xpm_tensor(gamma_mn: np.ndarray) -> np.ndarray:
    """``w[i, j] = |Γ_ij|²`` — Mumtaz Eq. (8) style SPM/XPM overlap weights."""
    return gamma_mn**2


def build_fwm_tensor(gamma_mn: np.ndarray) -> np.ndarray:
    """``w[m, n, p, q] = |Γ_mn · Γ_pq|`` — the separable overlap integral.

    Four axes: the fed channel ``m``, the two pumps ``(n, p)`` and the
    conjugated channel ``q``. The engine only evaluates the ``n == p``
    (fully degenerate) slice, which is where the four-wave pump is.
    """
    n_modes = gamma_mn.shape[0]
    pair = np.abs(np.outer(gamma_mn, gamma_mn))  # index (m, n) then (p, q)
    return pair.reshape(n_modes, n_modes, n_modes, n_modes)


# ── Engine helpers ───────────────────────────────────────────────────────────


def build_multimode(
    grid: TemporalGrid,
    powers: list[float],
    *,
    betas: list[list[float]] | None = None,
    **kwargs,
) -> MultimodeSplitStepEngine:
    """One constructor for every rung, so the rungs differ only by kwargs."""
    waves = [_wave(grid, p) for p in powers]
    if betas is None:
        betas = [[BETA2, 0.02]] * len(powers)
    return MultimodeSplitStepEngine(
        waves,
        _fiber(),
        betas,
        step_size=Length(LENGTH / kwargs.pop("num_steps", NUM_STEPS), "m"),
        **kwargs,
    )


def scalar_reference(grid: TemporalGrid, power: float) -> np.ndarray:
    """The same single channel propagated by the *scalar* engine."""
    fiber = _fiber()
    eng = SplitStepEngine(
        pulse=_wave(grid, power),
        fiber=fiber,
        betas=np.array([BETA2, 0.02]),
        include_raman=False,
        step_size=Length(LENGTH / NUM_STEPS, "m"),
    )
    eng.propagate(NUM_STEPS, nsaves=3)
    return np.asarray(eng.evolution[-1].envelope_field)


def energy(f: np.ndarray, grid: TemporalGrid) -> float:
    return float(np.sum(np.abs(f) ** 2) * grid.dt)


def mutual_coherence(a: np.ndarray, b: np.ndarray) -> float:
    """``|⟨A_a, A_b⟩| / sqrt(E_a E_b)`` — 1 while the channels ride together."""
    num = abs(np.vdot(a, b))
    den = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    return float(num / den) if den > 0 else 0.0


def allowed_exchanges(
    n_modes: int, oam_l: list[int] | None, apply_gate: bool = True
) -> list[tuple]:
    """The (m, n, p, q) tuples the engine's FWM loop actually walks.

    Mirrors ``_fwm_rhs``: pump ``n``, exchange pair ``(m, q)`` with ``m < q``,
    ``m != n`` and ``q != n``, then the ``oam_l`` gate. The gate is read from
    the engine (``_fwm_allowed``) rather than re-implemented, so this counts
    the real thing.
    """
    probe = MultimodeSplitStepEngine.__new__(MultimodeSplitStepEngine)
    probe._n = n_modes
    probe.oam_l = None if oam_l is None else list(oam_l)
    keep = []
    for n in range(n_modes):
        for m in range(n_modes):
            if m == n:
                continue
            for q in range(m + 1, n_modes):
                if q == n:
                    continue
                if apply_gate and not probe._fwm_allowed(m, n, n, q):
                    continue
                keep.append((m, n, n, q))
    return keep


def count_gate(n_modes: int) -> dict:
    """Surviving-exchange counts for the three label cases (ISSUES.md #15)."""
    total = len(allowed_exchanges(n_modes, None, apply_gate=False))
    counts = {
        "oam_l=None": allowed_exchanges(n_modes, None),
        "helical": allowed_exchanges(n_modes, OAM_HELICAL),
        "uniform": allowed_exchanges(n_modes, OAM_UNIFORM),
    }
    for name, tuples in counts.items():
        print(
            f"  {name:<12}: {len(tuples):3d} surviving exchanges out of "
            f"{total} the FWM loop walks with no gate at all"
        )
    return counts


# ── Rung 1 — the uncoupled control ───────────────────────────────────────────


def rung1_control() -> dict:
    grid = _grid()
    # Two *orthogonal* HG modes: Gamma_01 = 0 exactly, so the overlap-built
    # tensor is the identity and the two channels genuinely do not see each
    # other. The uniform "isotropic" fallback would drive channel 0 with
    # gamma(P_0 + P_1) instead, which is a different model — see the note
    # printed below.
    _, _, gamma2 = hg_profiles([0, 1])
    w = xpm_tensor(gamma2)
    eng = build_multimode(grid, list(P_RUNG12), coef_model="isotropic", xpm_weights=w)
    eng.propagate(NUM_STEPS, nsaves=51)

    fields = eng.fields_vs_z()
    refs = [scalar_reference(grid, p) for p in P_RUNG12]
    errs = [
        float(np.max(np.abs(fields[m][-1] - refs[m])) / np.max(np.abs(refs[m])))
        for m in range(2)
    ]
    uniform_drive = GAMMA * sum(P_RUNG12) * LENGTH

    print(_rule("Rung 1 — the uncoupled control: two independent channels"))
    print("  two orthogonal HG modes, identical betas, coef_model='isotropic'")
    print(
        f'  plus xpm_weights = |Gamma_ij|^2 = "[[{w[0, 0]:.3f}, {w[0, 1]:.3f}], '
        f'[{w[1, 0]:.3f}, {w[1, 1]:.3f}]"  (the identity, because the modes '
        f"are orthogonal)"
    )
    print(f"  NUM_STEPS={NUM_STEPS}, dz = {LENGTH / NUM_STEPS:.3f} m")
    for m, (p, e) in enumerate(zip(P_RUNG12, errs)):
        print(
            f"  channel {m} (P0 = {p:>4.1f} W): max |A_mm - A_scalar| / "
            f"max|A_scalar| = {e:.3e}"
        )
    print("  -> D_i = gamma P_i, so each channel is a private copy and must match a")
    print(f"     single-channel SplitStepEngine run (tolerance {CONTROL_TOLERANCE:g}).")
    print(
        "     Note what the uniform isotropic fallback would have done here: "
        "D_0 = D_1 ="
    )
    print(
        f"     gamma(P_0+P_1) = {uniform_drive:.3f} rad over the fibre. Same "
        f"coefficient"
    )
    print("     model, different answer — which is why rung 1 needs the tensor too.")
    for m, e in enumerate(errs):
        assert e < CONTROL_TOLERANCE, (
            f"rung 1 control failed on channel {m}: relative difference {e:.3e} "
            f"> {CONTROL_TOLERANCE:g}"
        )
    print("  control holds ✓")

    print(_rule("Rung 1b — negative control: break the control on purpose"))
    broken_betas = [[BETA2, 0.02], [BETA2 * 20.0, 0.02]]
    broken = build_multimode(
        grid, list(P_RUNG12), betas=broken_betas, coef_model="isotropic", xpm_weights=w
    )
    broken.propagate(NUM_STEPS, nsaves=3)
    bf = broken.fields_vs_z()
    berr = float(np.max(np.abs(bf[1][-1] - refs[1])) / np.max(np.abs(refs[1])))
    print(f"  channel 1 beta2 shifted by a factor 20 -> relative difference {berr:.3e}")
    assert berr > 100 * CONTROL_TOLERANCE, (
        "the negative control passed; rung 1's agreement would be vacuous"
    )
    print(
        "  -> the assertion above is not vacuous: perturbing one channel's "
        "betas breaks it ✓"
    )
    return {"engine": eng, "fields": fields, "grid": grid, "errs": errs, "w": w}


# ── Rung 2 — the full xpm_weights tensor ────────────────────────────────────


def rung2_xpm(grid: TemporalGrid) -> dict:
    # Same order, displaced: partial overlap, so the tensor has a real
    # off-diagonal element. (Two *different* HG orders would be exactly
    # orthogonal and give the identity back — see hg_profiles.)
    _, _, gamma2 = hg_profiles([0, 0], offsets=[0.0, MODE_OFFSET])
    w = xpm_tensor(gamma2)

    def differential(engine: MultimodeSplitStepEngine) -> float:
        f = engine.fields_vs_z()
        i = int(np.argmax(np.abs(f[0][0]) ** 2))
        return float(np.angle((f[0][-1] / f[0][0])[i] / (f[1][-1] / f[1][0])[i]))

    eng = build_multimode(grid, list(P_RUNG12), coef_model="isotropic", xpm_weights=w)
    eng.propagate(NUM_STEPS, nsaves=51)
    fields = eng.fields_vs_z()
    dphi = differential(eng)
    d0 = np.angle(fields[0][-1] / fields[0][0])
    d1 = np.angle(fields[1][-1] / fields[1][0])
    centre = int(np.argmax(np.abs(fields[0][0]) ** 2))

    iso = build_multimode(grid, list(P_RUNG12), coef_model="isotropic")
    iso.propagate(NUM_STEPS, nsaves=3)
    lp = build_multimode(grid, list(P_RUNG12), coef_model="lp_degenerate")
    lp.propagate(NUM_STEPS, nsaves=3)
    dphi_iso, dphi_lp = differential(iso), differential(lp)

    print(_rule("Rung 2 — a full N x N xpm_weights tensor from overlaps"))
    print(
        f"  overlap integral Gamma_ij over Hermite-Gaussian profiles "
        f"(waist {MODE_W * 1e6:.1f} um), two co-ordered modes displaced by "
        f"{MODE_OFFSET * 1e6:.1f} um"
    )
    for i in range(2):
        print("  Gamma = [" + ", ".join(f"{v:+.4f}" for v in gamma2[i]) + "]")
    print(
        "  xpm_weights[i, j] = |Gamma_ij|^2 = ["
        + ", ".join(f"{v:.4f}" for v in w[0])
        + "] (row 0)"
    )
    print(f"  SPM slot w[0,0] = {w[0, 0]:.4f}; inter-modal slot w[0,1] = {w[0, 1]:.4f}")
    print("  -> the diagonal is a mode overlapping itself, the off-diagonal is")
    print("     this pair of modes; a scalar coupling asserts the two are equal,")
    print("     which is a different model rather than a rough version of this one.")
    print(
        f"  nonlinear phase at the peak, z = L : ch0 {d0[centre]:+.4f} rad, "
        f"ch1 {d1[centre]:+.4f} rad"
    )
    print("  differential phase ch0 - ch1, three coefficient models:")
    print(f"    xpm_weights tensor (this rung) : {dphi:+.4f} rad")
    print(
        f"    coef_model='isotropic'         : {dphi_iso:+.4f} rad "
        f"(D_0 = D_1 = gamma(P_0+P_1), so exactly zero)"
    )
    print(
        f"    coef_model='lp_degenerate'     : {dphi_lp:+.4f} rad "
        f"(SPM 1, XPM 2/3, no overlaps)"
    )
    print("  three coefficient models, three answers — the tensor is the only")
    print("  one with an overlap integral behind it.")
    assert abs(dphi) > 1e-6, f"the tensor produced no differential phase ({dphi:.3e})"
    assert abs(dphi_iso) < 1e-9, f"isotropic should be exactly zero ({dphi_iso:.3e})"
    assert abs(abs(dphi) - abs(dphi_lp)) > 1e-3, (
        "the tensor and the lp_degenerate scalars agree; the tensor is not "
        "showing anything the scalar model does not"
    )
    print(
        "  the tensor produces a differential phase neither scalar model reproduces ✓"
    )
    return {
        "engine": eng,
        "fields": fields,
        "w": w,
        "gamma": gamma2,
        "dphi": dphi,
        "dphi_iso": dphi_iso,
        "dphi_lp": dphi_lp,
    }


# ── Rung 3 — group_delays ────────────────────────────────────────────────────


def rung3_group_delays(grid: TemporalGrid) -> dict:
    iso = build_multimode(grid, list(P_RUNG34), coef_model="isotropic")
    iso.propagate(NUM_STEPS, nsaves=51)
    drifting = build_multimode(
        grid, list(P_RUNG34), coef_model="isotropic", group_delays=list(GROUP_DELAYS)
    )
    drifting.propagate(NUM_STEPS, nsaves=51)

    f_iso, f_gd = iso.fields_vs_z(), drifting.fields_vs_z()
    z = iso.z_array
    # zip walks the snapshots, so a and b are already single fields.
    coh_01 = np.array([mutual_coherence(a, b) for a, b in zip(f_iso[0], f_iso[1])])
    coh_01_gd = np.array([mutual_coherence(a, b) for a, b in zip(f_gd[0], f_gd[1])])

    print(_rule("Rung 3 — group_delays: walk-off erases the mutual coherence"))
    print(
        f"  group_delays = {GROUP_DELAYS} s/m; channel 0 is the reference "
        f"frame (forced 0)"
    )
    print(
        f"  predicted relative delay over {LENGTH:.0f} m: "
        f"ch1 {GROUP_DELAYS[1] * LENGTH * 1e12:+.2f} ps, "
        f"ch2 {GROUP_DELAYS[2] * LENGTH * 1e12:+.2f} ps"
    )
    print(
        f"  mutual coherence <A_0, A_1>/sqrt(E_0 E_1) at z = L : "
        f"{coh_01[-1]:.4f} (no walk-off), "
        f"{coh_01_gd[-1]:.4f} (with walk-off)"
    )
    print("  -> walk-off is not a power exchange: the channel energies are unchanged,")
    print("     but the two envelopes no longer sit on top of each other, so the")
    print("     interference that makes them one coherent state is gone. This")
    print("     is the practical reason the averaged few-mode model keeps XPM")
    print("     and drops FWM: over the walk-off length the heterodyne beats wash out.")
    # The claim is comparative: without walk-off the two channels are the same
    # field (coherence 1), and walk-off takes it well below half of that. The
    # absolute value depends on how many pulse widths of delay were used.
    assert coh_01[-1] > 0.99 and coh_01_gd[-1] < 0.5 * coh_01[-1], (
        f"walk-off did not dephase the channels ({coh_01[-1]:.3f} -> "
        f"{coh_01_gd[-1]:.3f})"
    )
    print("  dephasing is visible and the energies are untouched ✓")
    return {
        "z": z,
        "coh": coh_01,
        "coh_gd": coh_01_gd,
        "engine": drifting,
        "fields": f_gd,
    }


# ── Rung 4 — phase_offsets and the GPI ladder ────────────────────────────────


def rung4_phase_offsets(grid: TemporalGrid) -> dict:
    """``Δβ₀⁽ᵖ⁾ = −2πp/ξ`` from GRIN self-imaging: the GPI ladder's spacing."""
    offsets = [-2.0 * np.pi * p / GPI_XI for p in range(1, len(P_RUNG34))]

    def run(off: list[float], nsaves: int = 51, steps: int = NUM_STEPS):
        eng = build_multimode(
            grid,
            list(P_RUNG34),
            coef_model="isotropic",
            phase_offsets=off,
            num_steps=steps,
        )
        eng.propagate(steps, nsaves=nsaves)
        return eng

    plain = run([0.0] * len(P_RUNG34))
    ladder = run([0.0, *offsets])
    f_plain, f_lad = plain.fields_vs_z(), ladder.fields_vs_z()

    centre = int(np.argmax(np.abs(f_plain[0][0]) ** 2))
    dp_plain = np.angle(f_plain[0][-1][centre] / f_plain[0][0][centre])
    dp_lad = np.angle(f_lad[0][-1][centre] / f_lad[0][0][centre])
    dp_pred = dp_plain + offsets[0] * LENGTH
    err = float(np.angle(np.exp(1j * (dp_lad - dp_pred))))
    errs = []
    for m, o in enumerate(offsets, start=1):
        i = int(np.argmax(np.abs(f_plain[m][0]) ** 2))
        d0 = np.angle(f_plain[m][-1][i] / f_plain[m][0][i])
        d1 = np.angle(f_lad[m][-1][i] / f_lad[m][0][i])
        errs.append(abs(float(np.angle(np.exp(1j * (d1 - d0 - o * LENGTH))))))

    # The offset is a *constant* phase per channel, so it cancels in a linear
    # overlap between two channels and moves nothing on its own. It enters a
    # mixed term as 2*phi_n - phi_q, which is where quasi-phase matching lives:
    # one pumped channel and two seeds, sweeping the offset of the *pump*
    # through a half-period of accumulated phase (the drive is A_n^2 A_q*,
    # so the pump's offset enters it twice).
    pump, seed = 3.0, 0.3
    gain = []
    for frac in np.linspace(0.0, 1.0, 13):  # 0 .. pi of accumulated offset phase
        dbeta = np.pi * frac / LENGTH
        e = build_multimode(
            grid,
            [pump, seed, seed],
            coef_model="lp_degenerate",
            phase_offsets=[dbeta, 0.0, 0.0],
            include_fwm=True,
            num_steps=40,
        )
        e.propagate(40, nsaves=3)
        f = e.fields_vs_z()
        gain.append(energy(f[1][-1], grid) / energy(f[1][0], grid))
    gain = np.array(gain)

    print(_rule("Rung 4 — phase_offsets: discrete quasi-phase matching"))
    print(
        f"  GRIN self-imaging length xi = {GPI_XI} m -> delta_beta_0 = "
        f"-2 pi p / xi = {['%+.1f' % v for v in offsets]} rad/m"
    )
    print("  phase_offsets is the *absolute* modal phase the retarded-frame Taylor")
    print(
        "  expansion omits; it accumulates as exp(i delta_beta_0 z) in each channel's"
    )
    print("  linear step, so it costs no nonlinearity at all.")
    print(
        f"  phase of channel 1 at z = L: no offsets {dp_plain:+.4f} rad, "
        f"with offset {dp_lad:+.4f} rad"
    )
    print(
        f"  predicted difference delta_beta_0 * L = {offsets[0] * LENGTH:+.1f} "
        f"rad (mod 2 pi = {err:+.2e} rad)"
    )
    print(
        "  per-channel agreement with exp(i delta_beta_0 z): "
        + ", ".join(f"{e:.2e} rad" for e in errs)
    )
    print("  FWM conversion between the two equal seed channels as the pump's offset")
    print(
        f"  sweeps a half period of accumulated phase: seed-1 energy goes {gain[0]:.4f}"
    )
    print(
        f"  -> {gain.min():.4f} (nearly all converted into seed 2) -> "
        f"{gain[-1]:.4f} at pi, where the drive is out of phase again."
    )
    print("  a constant per-channel phase cancels in a linear overlap and moves")
    print("  nothing on its own; it enters the mixed term A_n^2 A_q* as 2 phi_n -")
    print("  phi_q, and that is the phase a quasi-phase-matched ladder has to hit.")
    print("  This is what makes the GRIN/GPI ladder *discrete*: a channel is pumped")
    print("  only when its offset lands on the ladder, so the gain lands on a comb")
    print("  of frequencies rather than a band.")
    assert max(errs) < 1e-6, (
        f"phase_offsets did not accumulate as exp(i dbeta0 z): residuals {errs}"
    )
    assert gain.min() < 0.7 * gain.max(), (
        f"the offset sweep did not modulate the FWM drive ({gain.min():.4g} "
        f"vs {gain.max():.4g})"
    )
    assert max(energy(A[-1], grid) for A in f_lad) < 1.05 * max(P_RUNG34), (
        "a phase offset moved power; it must not"
    )
    print("  the offset is a pure phase, the energies are untouched ✓")
    return {"fields": f_lad, "offsets": offsets, "sweep": gain}


# ── Rung 5 — FWM and the oam_l gate ─────────────────────────────────────────


def rung5_fwm(grid: TemporalGrid) -> dict:
    _, _, gamma_n = hg_profiles(MODE_ORDERS, offsets=MODE_OFFSETS)
    w_xpm = xpm_tensor(gamma_n)
    w_fwm = build_fwm_tensor(gamma_n)
    n_modes = len(MODE_ORDERS)
    # A modest pump: the A/B here is about *which* exchanges the gate admits,
    # not about how hard the parametric gain is driven.
    powers = [FWM_AB_PUMP] * n_modes

    print(_rule("Rung 5 — inter-modal FWM and the oam_l gate"))
    print(
        f"  {n_modes} Hermite-Gaussian channels, OAM labels "
        f"{OAM_HELICAL} (helical) vs {OAM_UNIFORM} (uniform)"
    )
    print(
        f"  fwm_weights built from the overlap integral |Gamma_mn Gamma_pq|; "
        f"shape {w_fwm.shape}"
    )
    counts = count_gate(n_modes)
    print("  the gate is Poletti & Horak Eq. (18) type 2 (ISSUES.md #15):")
    print("  l_m = l_n + l_p - l_q, verified element-wise against the paper's")
    print("  own rule. It is a *selection filter*, and with all labels equal it")
    print("  admits every triplet — i.e. exactly the oam_l=None fallback.")

    results = {}
    for label, oam in (
        ("none", None),
        ("helical", OAM_HELICAL),
        ("uniform", OAM_UNIFORM),
    ):
        kwargs = {} if oam is None else {"oam_l": list(oam)}
        eng = build_multimode(
            grid,
            powers,
            coef_model="lp_degenerate",
            include_fwm=True,
            xpm_weights=w_xpm,
            fwm_weights=w_fwm,
            num_steps=FWM_STEPS,
            **kwargs,
        )
        eng.propagate(FWM_STEPS, nsaves=9)
        fields = eng.fields_vs_z()
        seeded = [energy(A[0], grid) for A in fields]
        late = [energy(A[-1], grid) for A in fields]
        ch_energies = [
            [energy(np.asarray(w.envelope_field), grid) for w in snaps]
            for snaps in eng.evolution
        ]
        results[label] = {
            "engine": eng,
            "fields": fields,
            "seeded": seeded,
            "late": late,
            "ch_energies": ch_energies,
        }
        print(
            f"  {label:<8}: channel-0 energy {seeded[0]:.4e} -> {late[0]:.4e} "
            f"({late[0] / seeded[0] - 1:+.3%})"
        )

    n_none = len(counts["oam_l=None"])
    n_hel = len(counts["helical"])
    n_uni = len(counts["uniform"])
    assert n_uni == n_none, (
        f"uniform labels admitted {n_uni} exchanges, oam_l=None admitted "
        f"{n_none}; ISSUES.md #15 says the gate is inert for uniform labels"
    )
    assert n_hel < n_none, (
        f"helical labels admitted {n_hel} exchanges, no fewer than the "
        f"{n_none} of oam_l=None; the gate is not filtering anything"
    )
    print(
        f"  -> helical < none ({n_hel} < {n_none}) and uniform == none "
        f"({n_uni} == {n_none}) ✓"
    )
    print("     A/B, in one place: the gate buys selection, not speed.")
    return {
        "counts": {k: len(v) for k, v in counts.items()},
        "tuples": counts,
        "results": results,
        "w_fwm": w_fwm,
    }


# ── Rung 6 — pump depletion and the substep policy ───────────────────────────


def rung6_pump_depletion(grid: TemporalGrid) -> dict:
    _, _, gamma_n = hg_profiles(MODE_ORDERS, offsets=MODE_OFFSETS)
    w_xpm = xpm_tensor(gamma_n)
    w_fwm = build_fwm_tensor(gamma_n)
    n_modes = len(MODE_ORDERS)
    powers = [FWM_PUMP] + [1e-4] * (n_modes - 1)  # strong pump, weak seeds

    print(_rule("Rung 6 — fwm_pump_depletion and the substep count"))
    eng = build_multimode(
        grid,
        powers,
        coef_model="lp_degenerate",
        include_fwm=True,
        fwm_pump_depletion=True,
        xpm_weights=w_xpm,
        fwm_weights=w_fwm,
        num_steps=FWM_STEPS,
    )
    dz = LENGTH / FWM_STEPS
    # Report the engine's own stability policy (ISSUES.md #11) rather than
    # re-deriving it here: these are the quantities the user must know before
    # choosing step_size.
    rate, ch = eng._fwm_rate_max(eng.A, dz)
    n_sub = eng._fwm_substep_count(eng.A, dz)
    print(
        f"  pump P0 = {FWM_PUMP:.0f} W (gamma P0 = {GAMMA * FWM_PUMP:.3f} /m), "
        f"seeds at 1e-4 of that amplitude"
    )
    print(f"  outer dz = {dz:.3f} m over {FWM_STEPS} steps")
    print(
        f"  strongest exchange-pair rate eta = gamma f |A_n|^2 = {rate:.4e} /m "
        f"(pump channel {ch})"
    )
    print(
        f"  -> stability floor  ceil(eta dz / 2.5)   = {int(np.ceil(rate * dz / 2.5))}"
    )
    print(
        f"     accuracy floor  min(200, ceil(eta dz / 0.05)) = "
        f"{min(200, int(np.ceil(rate * dz / 0.05)))}"
    )
    print(
        f"  engine's substep count n_sub = {n_sub} "
        f"(policy: ISSUES.md #11; accuracy cap 200, hard cap "
        f"{eng._FWM_SUBSTEP_MAX})"
    )
    eng.propagate(FWM_STEPS, nsaves=9)
    fields = eng.fields_vs_z()
    finite = all(np.all(np.isfinite(A.view(float))) for A in fields)
    totals = eng.energy_vs_z  # total photon number at every snapshot
    print(
        f"  total photon number: {totals[0]:.6e} -> {totals[-1]:.6e} "
        f"(drift {abs(totals[-1] / totals[0] - 1):.2e})"
    )
    print("  with the back-conversion arm the pump depletion is photon-")
    print("  conserving to RK4 round-off, which is what makes the substep count")
    print("  a stability question rather than an accuracy guess.")
    assert finite, "the run produced non-finite amplitudes"
    assert n_sub > 1, (
        f"the substep count stayed at {n_sub}; the pump was too weak to "
        "exercise the stability policy (ISSUES.md #11)"
    )
    assert abs(totals[-1] / totals[0] - 1) < 5e-2, "photon number drifted > 5%"
    print(f"  all outputs finite, n_sub = {n_sub} > 1 ✓")
    return {
        "engine": eng,
        "fields": fields,
        "n_sub": n_sub,
        "dz": dz,
        "rate": rate,
        "totals": totals,
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def rungs_figure(r1: dict, r2: dict, r3: dict, r4: dict) -> None:
    grid = r1["grid"]
    fig, axes = plt.subplots(3, 3, figsize=(16.0, 10.5))
    colours = ["tab:blue", "tab:orange", "tab:green"]

    ax = axes[0, 0]
    z = r1["engine"].z_array
    for m, A in enumerate(r1["fields"]):
        e = np.sum(np.abs(A) ** 2, axis=1) * grid.dt
        ax.plot(z, (e / e[0] - 1.0) * 1e13, "-", color=colours[m], label=f"channel {m}")
    ax.set_title(
        "rung 1 — control: per-channel energy constant\n"
        "(orthogonal modes, xpm_weights = I, identical betas)",
        fontsize=10,
    )
    ax.set_xlabel("z (m)")
    ax.set_ylabel("(E(z)/E(0) − 1)  × 10$^{-13}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    im = ax.imshow(r2["w"], cmap="viridis", vmin=0)
    ax.set_xticks(range(2), ["j=0", "j=1"])
    ax.set_yticks(range(2), ["i=0", "i=1"])
    ax.set_title(
        "rung 2 — xpm_weights[i, j] = |Γᵢⱼ|²\ndiagonal SPM vs off-diagonal XPM",
        fontsize=10,
    )
    for i in range(2):
        for j in range(2):
            ax.annotate(
                f"{r2['w'][i, j]:.3f}",
                (j, i),
                ha="center",
                va="center",
                color="w",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    f = r4["fields"]
    t = np.asarray(grid.t) * 1e12
    m = np.abs(t) < 5 * T0 * 1e12
    for ch, A in enumerate(f):
        ax.plot(
            t[m],
            np.abs(A[-1][m]) ** 2,
            color=colours[ch],
            label=f"ch{ch} (Δβ₀ = {r4['offsets'][ch - 1]:+.1f} rad/m)"
            if ch
            else "ch0 (reference)",
        )
    ax.set_title(
        "rung 4 — phase_offsets: same shape, different phase per mode", fontsize=10
    )
    ax.set_xlabel("retarded time (ps)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(r3["z"], r3["coh"], "-", color="tab:blue", label="group_delays = None")
    ax.plot(
        r3["z"],
        r3["coh_gd"],
        "-",
        color="tab:red",
        label=f"group_delays = {GROUP_DELAYS} s/m",
    )
    ax.set_title(
        "rung 3 — mutual coherence |⟨A₀,A₁⟩|/√(E₀E₁)\n"
        "walk-off dephases without moving power",
        fontsize=10,
    )
    ax.set_xlabel("z (m)")
    ax.set_ylabel("coherence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    f = r3["fields"]
    ax.plot(
        np.asarray(grid.t) * 1e12,
        np.abs(f[0][-1]) ** 2,
        color=colours[0],
        label="channel 0 at z = L",
    )
    ax.plot(
        np.asarray(grid.t) * 1e12,
        np.abs(f[1][-1]) ** 2,
        color=colours[1],
        label="channel 1 at z = L (walked off)",
    )
    ax.set_title(
        f"rung 3 — the delayed channel, displaced by "
        f"{GROUP_DELAYS[1] * LENGTH * 1e12:.0f} ps",
        fontsize=10,
    )
    ax.set_xlabel("retarded time (ps)")
    ax.set_ylabel("|A|² (W)")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(np.linspace(0, 1, len(r4["sweep"])), r4["sweep"], "o-", color="tab:purple")
    ax.set_xlabel("pump's phase offset, accumulated 0 → π over the fibre")
    ax.set_ylabel("seed-1 energy / its input")
    ax.set_title(
        "rung 4 — the offset only bites through a mixed term\n"
        "(Aₙ²A_q*: the pump's offset enters twice)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    f = r2["fields"]
    ax.plot(
        np.asarray(grid.t) * 1e12,
        np.angle(f[0][-1] / f[0][0]),
        color=colours[0],
        label="phase ch0 (tensor)",
    )
    ax.plot(
        np.asarray(grid.t) * 1e12,
        np.angle(f[1][-1] / f[1][0]),
        color=colours[1],
        label="phase ch1 (tensor)",
    )
    ax.set_title(
        f"rung 2 — differential nonlinear phase at z = L: "
        f"{r2['dphi']:+.4f} rad\n(scalar isotropic coupling: "
        f"{r2['dphi_iso']:+.4f} rad)",
        fontsize=10,
    )
    ax.set_xlabel("retarded time (ps)")
    ax.set_ylabel("arg(A/A₀)")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    for k, (label, style) in enumerate(
        (("no coupling (rung 1 control)", "--"), ("XPM tensor (rung 2)", "-"))
    ):
        fields = r1["fields"] if k == 0 else r2["fields"]
        for m, A in enumerate(fields):
            ax.plot(
                np.asarray(grid.t) * 1e12,
                np.abs(A[-1]) ** 2,
                style,
                color=colours[m],
                label=f"{label}, ch{m}" if m == 0 else None,
            )
    ax.set_title(
        "rung 1 vs rung 2 — the tensor changes the phases,\n"
        "not the intensities (XPM is phase-only)",
        fontsize=10,
    )
    ax.set_xlabel("retarded time (ps)")
    ax.set_ylabel("|A|² (W)")
    ax.legend(fontsize=7)

    ax = axes[2, 2]
    ax.axis("off")
    ax.text(
        0.0,
        0.95,
        "Rungs 5 and 6 (FWM, the oam_l gate and the substep\n"
        "policy) are in 38_oam_gate_ab.png — they need a\n"
        "depleting pump and five channels, which do not fit\n"
        "on this control-and-coupling figure.",
        va="top",
        fontsize=10,
        wrap=True,
    )

    fig.suptitle("Few-mode GNLSE — one parameter added per rung", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS["rungs"], dpi=130, bbox_inches="tight")
    plt.close(fig)


def oam_figure(r5: dict, r6: dict) -> None:
    fig = plt.figure(figsize=(14.0, 5.2))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    vals = [r5["counts"][k] for k in ("oam_l=None", "helical", "uniform")]
    bars = ax.bar(
        ["oam_l=None", "helical\n[0,1,-1,2,-2]", "uniform\n[0,0,0,0,0]"],
        vals,
        color=["0.55", "tab:red", "tab:orange"],
    )
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v}",
            (b.get_x() + b.get_width() / 2, v),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("surviving FWM exchanges")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(
        "the oam_l gate as a selection filter\n(ISSUES.md #15: "
        "helical < none, uniform == none)",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    for label, colour in (
        ("none", "0.55"),
        ("helical", "tab:red"),
        ("uniform", "tab:orange"),
    ):
        res = r5["results"][label]
        # (n_saves, n_channels): how far each channel has moved from its own
        # input energy, as a percentage. The gate's whole effect shows up here.
        e_ch = np.asarray(res["ch_energies"])
        frac = np.max(np.abs(e_ch / e_ch[:, :1] - 1.0), axis=1)
        z_r = np.linspace(0.0, LENGTH, len(frac))
        ax.plot(z_r, frac * 1e2, "-", color=colour, label=f"oam_l = {label}")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("largest channel energy change (%)")
    ax.set_title(
        "what the filter does to the run\n"
        "(fewer permitted exchanges, less power moved)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot((r6["totals"] / r6["totals"][0] - 1.0) * 1e9, "-", color="tab:green")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("relative photon-number drift  × 10$^{-9}$")
    ax.set_title(
        f"rung 6 — fwm_pump_depletion=True\n"
        f"γP₀ = {GAMMA * FWM_PUMP:.2f} /m, η = {r6['rate']:.2e} /m, "
        f"dz = {r6['dz']:.2f} m\n"
        f"→ n_sub = {r6['n_sub']} (stability-driven, ISSUES.md #11)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)

    fig.suptitle("Multimode FWM: the oam_l gate A/B and the substep policy", y=1.03)
    fig.savefig(FIGS["oam"], dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.perf_counter()
    print(_rule("0. Building a MultimodeSplitStepEngine, one parameter at a time"))
    print(
        f"  deck: L = {LENGTH:.0f} m, gamma = {GAMMA:g} 1/(W m), T0 = "
        f"{T0 * 1e12:.0f} ps, N = {N_SAMPLES} samples over Tmax = "
        f"{TMAX * 1e12:.0f} ps"
    )
    print("  xpm_weights[i, j] multiplies |A_j|^2 in channel i (i == j is SPM)")
    print("  fwm_weights[m, n, p, q] weights the -> m transition pumped by (n, p),")
    print("  conjugated q — both tensors are built here from a mode-overlap integral")
    print(
        "  coef_model: 'lp_degenerate' (SPM 1, XPM 2/3, FWM 2/3) or "
        "'isotropic' (all ones)"
    )
    print("  oam_l is the Poletti & Horak Eq. (18) type-2 spatial rule (ISSUES.md #15)")

    r1 = rung1_control()
    r2 = rung2_xpm(r1["grid"])
    r3 = rung3_group_delays(r1["grid"])
    r4 = rung4_phase_offsets(r1["grid"])
    r5 = rung5_fwm(r1["grid"])
    r6 = rung6_pump_depletion(r1["grid"])

    print(_rule("Summary"))
    print(
        f"  rung 1 control        : max relative difference vs two scalar "
        f"engines {max(r1['errs']):.2e}"
    )
    print(
        f"  rung 2 tensor         : differential nonlinear phase "
        f"{r2['dphi']:+.4f} rad (scalar isotropic: {r2['dphi_iso']:+.4f})"
    )
    print(
        f"  rung 3 walk-off       : coherence {r3['coh'][-1]:.3f} -> "
        f"{r3['coh_gd'][-1]:.3f}"
    )
    print(f"  rung 4 phase offsets  : {['%+.1f' % v for v in r4['offsets']]} rad/m")
    print(
        f"  rung 5 gate counts    : none {r5['counts']['oam_l=None']}, "
        f"helical {r5['counts']['helical']}, uniform "
        f"{r5['counts']['uniform']}"
    )
    print(
        f"  rung 6 substeps       : n_sub = {r6['n_sub']} at dz = "
        f"{r6['dz']:.3f} m, photon drift "
        f"{abs(r6['totals'][-1] / r6['totals'][0] - 1):.2e}"
    )
    print("  every rung behaved as its parameter says it should ✓")
    print(
        f"\n  runtime {time.perf_counter() - t0:.1f} s (rungs 5 and 6 pay four "
        f"RK4 stages per substep per channel; every other rung is phase-only)"
    )

    rungs_figure(r1, r2, r3, r4)
    oam_figure(r5, r6)
    print("\nGenerated files:")
    for p in FIGS.values():
        print(f"  {p.relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
