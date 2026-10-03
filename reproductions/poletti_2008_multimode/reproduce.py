"""Reproduction: the multimode GNLSE overlap coefficients and their symmetry
properties (Poletti & Horak 2008).

Reference
---------
F. Poletti and P. Horak, "Description of ultrashort pulse propagation in
multimode optical fibers", *J. Opt. Soc. Am. B* **25**(10), 1645 (2008),
doi:10.1364/JOSAB.25.001645.  Local PDF ``JOSAB.25.001645.pdf``;
``parameters.json`` carries the Fig. 1 fibre and the extracted numbers.

What is reproduced
------------------
The paper's Section 4.A analysis of the symmetry properties of the nonlinear
coupling coefficients

    Q^(1)_plmn = C/(N_p N_l N_m N_n) INT [F_p* . F_l ][F_m . F_n*] dx dy     (Eq. 7)
    Q^(2)_plmn = C/(N_p N_l N_m N_n) INT [F_p* . F_l*][F_m . F_n ] dx dy

for the *first ten modes* of the paper's step-index fibre (6 um core radius,
NA = 0.17 at 1.5 um, V = 4.2726: LP01, LP11 and LP21, 2 + 4 + 4 = 10 modes,
2 x 10^4 coefficients over the two types).

The coupling coefficients are evaluated **from the transverse fields** by
quadrature.  Nothing about the selection rules is hard-coded: Eq. (18) and
Eq. (19) are *predictions* checked against what the integral produces.

CHECKS (assert-carrying)
------------------------
0. Mode solver: the LP radial roots of the step-index eigenvalue equation and
   the unit-power normalisation of every profile.
1. Eq. (18), spatial selection rules, both types.  1360 of 10^4 quadruples
   survive per type; every forbidden entry is at round-off.
2. Eq. (19), polarisation selection rules: 1360 -> 340 survivors per type, the
   discarded 1020 quadruples carrying |Q| up to 1.0, i.e. the rule does real
   work.  Cross-sigma FWM coefficients such as Q^(2)_{0,1,1,0} are exactly
   zero.
3. Eq. (16), the permutation / complex-conjugation identities: all eleven
   printed identities hold to machine precision.
4. Sec. 3, "for real-valued mode functions we have Q^(1) = Q^(2)": exact in
   the real (cos/sin) LP basis, and *not* an identity in the helical Eq. (17)
   basis -- both halves of the statement are checked.
5. Fig. 1 shape: the log-log magnitude distribution of |Q| with the
   spatial-rule and polarisation-rule sub-populations, and the 1 %-of-maximum
   band the paper reads its 936 "small but non-zero" coefficients off.
6. Sec. 5 complexity: the surviving-coefficient count versus M against the
   unconstrained M^4, i.e. the factor by which the selection rules cut the
   dominant term of the algorithm.
7. Engine: the polarisation closure the paper states in Sec. 3.B / Eq. (14) --
   a launch in a single circular polarisation cannot couple into the
   orthogonal one, so the opposite-sigma channels stay at machine zero and the
   simulation may be restricted to a single-mode GNLSE.  A control run with
   isotropic weights (which ignore Eq. 19) must leak, so the check is not
   vacuous.  Energy conservation of Eq. (15) in both runs.

CAVEAT (recorded, not asserted)
-------------------------------
The paper's Fig. 1 counts (17872 of 2 x 10^4 coefficients vanishing
identically, a further 936 below 1 % of the maximum) come from the *exact,
complex-valued vectorial* mode functions of the step-index fibre.  The
weakly-guiding scalar LP model implemented here reproduces the selection
rules exactly but gives different counts (8640 zeros per type from the spatial
rule alone, and no surviving coefficient below 1 % of the maximum).  Closing
that gap needs a full vectorial HE/EH/TE/TM mode solver, which is out of scope
for this deck; see the folder README.

Usage
-----
    python reproductions/poletti_2008_multimode/reproduce.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photonics_helper.base import (  # noqa: E402
    C_MS,
    Area,
    Length,
    Time,
    Wavelength,
)
from photonics_helper.gnlse import FiberProfile  # noqa: E402
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine  # noqa: E402
from photonics_helper.pulse import Envelope, TemporalGrid, Wave  # noqa: E402

import overlap as ov  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "poletti_2008_overlaps.png"

# ---------------------------------------------------------------------------
# deck constants (all from parameters.json / the paper)
# ---------------------------------------------------------------------------
A_CORE_UM = 6.0
NA = 0.17
LAMBDA_UM = 1.5
N_MODES = 10

# engine deck: short enough that the dispersed pulse stays inside the periodic
# window (over L the pulse broadens by |beta2| L / T0; a long span wraps around
# and the apparent power then wobbles at the 1e-3 level -- a window artefact,
# not a conservation failure)
LAMBDA0_NM = LAMBDA_UM * 1e3
N_GRID = 4096
WINDOW = 60e-12  # s
T0 = 2e-12  # s
P_LAUNCH = 1.0e3  # W per launched channel
L_FIBER = 0.5  # m
DZ = 1e-3  # m
BETA2 = -21.0  # ps^2/m (standard SMF at 1.5 um; the polarisation closure is
# linear-operator independent -- beta2 only sets the walk-off scale)
GAMMA_TARGET = 4.0e-3  # 1/(W m), giving gamma P L = 2

# tolerances
TOL_RULE = 1e-12  # selection-rule round-off
TOL_IDENT = 1e-12  # Eq. (16) machine precision
TOL_ENERGY = 1e-6  # Eq. (15) relative drift (see check_engine for the measured
# value and its 1/N scaling -- it is the engine's split-step round-off)
TOL_LEAK = 1e-12  # opposite-sigma power fraction


# ---------------------------------------------------------------------------
# 0. mode solver
# ---------------------------------------------------------------------------
def check_mode_solver() -> dict:
    """LP roots of the step-index eigenproblem + unit-power normalisation."""
    V = ov.v_number(A_CORE_UM, NA, LAMBDA_UM)
    roots = ov.lp_roots(V)
    found = {f"LP{ell}{m}": u for (ell, m), u in sorted(roots.items())}
    assert set(found) == {"LP01", "LP11", "LP21"}, found
    # LP01 cuts in below the first zero of J_1 and LP21 just above j_{2,1};
    # the ordering by u is the mode ordering of the paper.
    assert found["LP01"] < found["LP11"] < found["LP21"]

    fib_h = ov.mode_set_10(V, basis="helical")
    fib_r = ov.mode_set_10(V, basis="real")
    assert len(fib_h) == N_MODES == len(fib_r)
    err_h = max(fib_h.unit_power_error(), fib_r.unit_power_error())
    assert err_h < 1e-12, f"unit-power normalisation off by {err_h:.2e}"
    # the two bases must describe the same ten modes, in the same order
    assert [m.name for m in fib_h.modes] == [m.name for m in fib_r.modes]
    return {
        "V": V,
        "roots": found,
        "n_modes": len(fib_h),
        "mode_names": [m.name for m in fib_h.modes],
        "unit_power_error": err_h,
    }


# ---------------------------------------------------------------------------
# 1-2. selection rules, Eq. (18) and Eq. (19)
# ---------------------------------------------------------------------------
def check_selection_rules(fib_h: ov.StepIndexFiber) -> dict:
    """Eq. (18) and Eq. (19) as emergent zeros of the quadrature."""
    q_spatial = {t: ov.overlap_tensor(fib_h, t, apply_pol=False) for t in (1, 2)}
    measured = ov.measure_selection_rules(fib_h, q_spatial)

    for typ, rec in measured.items():
        assert rec["matched_spatial"], f"{typ}: spatial rule mismatch {rec}"
        assert rec["n_surviving_numeric"] == rec["expected_spatial"], rec
        assert rec["max_abs_violating_spatial"] < TOL_RULE, (
            f"{typ}: |Q| = {rec['max_abs_violating_spatial']:.2e} survives Eq. (18)"
        )
        # the polarisation rule must remove a real population of non-negligible
        # coefficients -- otherwise the closure of Sec. 3.B would be trivial
        assert rec["n_dropped_by_pol_rule"] > 0, rec
        assert rec["max_abs_dropped_by_pol_rule"] > 1e-3, (
            f"{typ}: Eq. (19) discards nothing (max |Q| dropped "
            f"{rec['max_abs_dropped_by_pol_rule']:.2e})"
        )

    # with Eq. (19) applied the survivor count matches the combinatorial
    # prediction of the combined rule, and the tensor is *exactly* zero outside
    q_full = {t: ov.overlap_tensor(fib_h, t) for t in (1, 2)}
    masks = ov.selection_rule_masks(fib_h)
    for typ, q in q_full.items():
        allowed = masks[f"spatial_{typ}"] & masks[f"pol_{typ}"]
        assert int((np.abs(q) > TOL_RULE).sum()) == int(allowed.sum()), (
            f"type {typ}: masked survivor count does not match Eq. (18)+(19)"
        )
        # the polarisation mask is an exact identity of the circular basis, so
        # those zeros are *exactly* zero; the spatial zeros only reach
        # round-off, because that rule emerges from the quadrature
        assert np.abs(q[~masks[f"pol_{typ}"]]).max() == 0.0
        assert np.abs(q[~allowed]).max() < TOL_RULE
    s = fib_h.sigmas
    cross = [
        (p, ell, m, n)
        for p in range(N_MODES)
        for ell in range(N_MODES)
        for m in range(N_MODES)
        for n in range(N_MODES)
        if (s[p] != s[ell]) or (s[m] != s[n])
    ]
    max_cross = max(abs(q_full[1][t]) for t in cross)
    assert max_cross == 0.0, f"cross-polarisation FWM not zero: {max_cross:.2e}"
    return {
        "measured": measured,
        "expected": {f"type_{t}": ov.expected_survivors(fib_h, t) for t in (1, 2)},
        "masked_survivors": {
            f"type_{t}": int((np.abs(ov.overlap_tensor(fib_h, t)) > TOL_RULE).sum())
            for t in (1, 2)
        },
        "n_cross_polarisation_quadruples": len(cross),
        "max_abs_cross_polarisation_Q2": max_cross,
    }


# ---------------------------------------------------------------------------
# 3-4. Eq. (16) and the real-basis equality
# ---------------------------------------------------------------------------
def check_symmetries(fib_h: ov.StepIndexFiber, fib_r: ov.StepIndexFiber) -> dict:
    q_h = {t: ov.overlap_tensor(fib_h, t) for t in (1, 2)}
    q_r = {t: ov.overlap_tensor(fib_r, t) for t in (1, 2)}

    errs = ov.symmetry_identity_errors(q_h)
    worst = max(e["max_err"] for e in errs)
    assert worst < TOL_IDENT, f"Eq. (16) identity off by {worst:.2e}: {errs}"

    eq = ov.real_basis_equality(q_r[1], q_r[2])
    assert eq["max_abs_diff"] < TOL_IDENT, f"Q1 != Q2 in the real basis: {eq}"
    # the complementary half of the paper's statement: in the helical basis the
    # two tensors are *not* equal (their survivor sets differ), so the equality
    # is a property of the real-valued mode functions, not of the definitions
    helical_diff = float(np.abs(q_h[1] - q_h[2]).max())
    assert helical_diff > 1e-3, (
        "Q1 == Q2 also in the helical basis -- the basis distinction of "
        "Sec. 3 would be untested"
    )
    return {
        "eq16": errs,
        "eq16_worst": worst,
        "real_basis_equality": eq,
        "helical_max_abs_diff": helical_diff,
    }


# ---------------------------------------------------------------------------
# 5-6. Fig. 1 distribution and the Sec. 5 complexity argument
# ---------------------------------------------------------------------------
def check_fig1(fib_h: ov.StepIndexFiber, fib_r: ov.StepIndexFiber) -> dict:
    q_h1 = ov.overlap_tensor(fib_h, 1, apply_pol=False)
    q_h2 = ov.overlap_tensor(fib_h, 2, apply_pol=False)
    q_h1p = ov.overlap_tensor(fib_h, 1)
    q_h2p = ov.overlap_tensor(fib_h, 2)
    q_r1 = ov.overlap_tensor(fib_r, 1)

    masks = ov.selection_rule_masks(fib_h)
    allc = np.concatenate([np.abs(q_h1).ravel(), np.abs(q_h2).ravel()])
    after18 = np.concatenate(
        [
            np.abs(q_h1)[masks["spatial_1"]].ravel(),
            np.abs(q_h2)[masks["spatial_2"]].ravel(),
        ]
    )
    after19 = np.concatenate([np.abs(q_h1p).ravel(), np.abs(q_h2p).ravel()])
    real = np.abs(q_r1).ravel()

    stats = {
        "helical_all": ov.magnitude_distribution(allc),
        "helical_after_rule18": ov.magnitude_distribution(after18),
        "helical_after_rule18_and_19": ov.magnitude_distribution(after19),
        "real_basis": ov.magnitude_distribution(real),
    }
    # the paper's three Fig. 1 curves are nested populations: 2 x 10^4 raw
    # coefficients, the Eq. (18) survivors, and the Eq. (18)+(19) survivors
    n_all = stats["helical_all"]["n_total"]
    nz_all = (
        stats["helical_all"]["n_nonzero"]
        if "n_nonzero" in stats["helical_all"]
        else stats["helical_all"]["n_total"] - stats["helical_all"]["n_zero"]
    )
    nz_18 = (
        stats["helical_after_rule18"]["n_total"]
        - stats["helical_after_rule18"]["n_zero"]
    )
    nz_19 = (
        stats["helical_after_rule18_and_19"]["n_total"]
        - stats["helical_after_rule18_and_19"]["n_zero"]
    )
    z_all = stats["helical_all"]["n_zero"]
    z_19 = stats["helical_after_rule18_and_19"]["n_zero"]
    assert n_all == 2 * N_MODES**4 == 20000
    assert nz_18 == nz_all, (nz_18, nz_all)
    assert nz_19 == 2 * 340 == 680, nz_19
    assert 0 < nz_19 < nz_18 < n_all
    assert z_all < z_19 < n_all, (z_all, z_19, n_all)
    # the 1 %-band: the weakly-guiding model keeps every surviving coefficient
    # above it (the paper's 936 small terms are a vectorial effect)
    frac_19 = stats["helical_after_rule18_and_19"]["n_below_1pct_nonzero"] / nz_19
    assert frac_19 == 0.0, frac_19
    return {
        "n_total": {
            "all": n_all,
            "after_rule_18": nz_18,
            "after_rule_18_19": n_all,
        },
        "n_nonzero": {"all": nz_all, "after_rule_18": nz_18, "after_rule_18_19": nz_19},
        "n_zero": {"all": z_all, "after_rule_18": 0, "after_rule_18_19": z_19},
        "stats": stats,
        "frac_below_1pct_after_both_rules": frac_19,
        "paper_reference": {
            "total": 20000,
            "zero_by_rule_18": 17872,
            "below_1pct": 936,
        },
    }


def check_complexity() -> dict:
    """Sec. 5: the M^4 term of Eq. (6) against the surviving coefficient count."""
    v_values = [4.2725660088821185, 6.0, 8.0, 10.0, 12.0]
    n_mode_list = [6, 8, 10, 14, 18]
    sweep = ov.complexity_sweep(v_values, n_mode_list)
    # with the rules the count still grows close to M^4 (the paper's Fig. 3(a)
    # measures 3.43 for the A_l A_m A_n* term) and the reduction factor is the
    # saving the paper's Sec. 5 argument is about
    assert sweep["fit_exponent"] > 3.0, sweep["fit_exponent"]
    assert sweep["reduction_factor"].min() > 20.0, sweep["reduction_factor"]
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sweep.items()
    }


# ---------------------------------------------------------------------------
# 7. engine deck
# ---------------------------------------------------------------------------
def _make_grid() -> TemporalGrid:
    return TemporalGrid(N=N_GRID, Tmax=Time(WINDOW, "s"))


def _channels(grid: TemporalGrid, powers: dict[int, float]) -> list[Wave]:
    """One Gaussian channel per entry of ``powers`` (0 W = empty channel).

    The empty channels matter: the polarisation-closure check of Eq. (14) is
    only meaningful if the orthogonal polarisation is *present* in the
    simulation and stays empty, rather than absent from it.
    """
    waves = []
    for i, pwr in powers.items():
        wv = Wave(
            grid=grid,
            envelope=Envelope(
                shape="gaussian", peak_amplitude=1.0, pulse_width=Time(T0, "s")
            ),
            central_wavelength=Wavelength(LAMBDA0_NM, "nm"),
        )
        wv._pulse_train_field = np.sqrt(pwr) * np.exp(-(grid.t**2) / (2 * T0**2))
        waves.append(wv)
    return waves


def _launch(idx: list[int], live: list[int]) -> dict[int, float]:
    """Launch ``P_LAUNCH`` in every channel of ``live``, 0 elsewhere in ``idx``."""
    return {i: (P_LAUNCH if i in live else 0.0) for i in idx}


def _make_engine(waves, weights, *, include_fwm: bool, oam_l=None) -> object:
    fiber = FiberProfile(
        n2=1.0, alpha=0.0, A_eff=Area(1.0, "m^2"), length=Length(L_FIBER, "m")
    )
    eng = MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[BETA2] * len(waves),
        betas_unit="ps^k/m",
        coef_model="isotropic",
        include_fwm=include_fwm,
        oam_l=oam_l,
        xpm_weights=weights["xpm"],
        fwm_weights=weights.get("fwm"),
        fwm_pump_depletion=include_fwm,
        step_size=Length(DZ, "m"),
    )
    # engine gamma = n2 * omega0 / (c A_eff); pick n2 so that gamma == GAMMA_TARGET
    eng.fiber.n2 = GAMMA_TARGET * C_MS / eng.omega0 / Area(1.0, "m^2").as_m2
    return eng


def _power(eng) -> list[float]:
    """Total power of every channel at the last saved snapshot."""
    return [
        float(np.trapezoid(np.abs(w.envelope_field) ** 2, eng.grid.t))
        for w in eng.evolution[-1]
    ]


def _leak_fraction(eng, pos_sigma: list[int], neg_sigma: list[int]) -> float:
    p = np.array(_power(eng))
    tot = p[pos_sigma].sum()
    return float(p[neg_sigma].sum() / tot)


def check_engine(fib_r: ov.StepIndexFiber, fib_h: ov.StepIndexFiber) -> dict:
    """Eq. (19)/Eq. (14) closure on the engine, with an isotropic control."""
    w = ov.engine_weights(fib_r)
    xpm_q, fwm_q = w["xpm_weights"], w["fwm_weights"]
    sig = fib_r.sigmas
    pos = [i for i, s in enumerate(sig) if s == "+"]
    neg = [
        i for i, s in enumerate(sig) if s == "-"
    ]  # Eq. (14): the two circular states of ONE spatial mode see each other in
    # full (xpm = 1), which is why |A_1|^2 + |A_2|^2 appears in Eq. (14)
    assert xpm_q[0, 1] == 1.0 and xpm_q[1, 0] == 1.0
    # the sigma = + and sigma = - copies of a spatial mode are physically
    # identical, so the coupling matrix must not depend on which state is used
    # (the mode ordering interleaves them: 0/1, 2/3, ... are the two states of
    # one spatial shape)
    assert np.allclose(xpm_q[0::2, 0::2], xpm_q[1::2, 1::2], atol=1e-12)
    # Eq. (19): every FWM term mixing e_+ and e_- vanishes identically
    for m in range(N_MODES):
        for n in range(N_MODES):
            for q in range(N_MODES):
                if sig[m] != sig[q] or sig[m] != sig[n]:
                    assert fwm_q[m, n, n, q] == 0.0, (m, n, q)
    # The engine's `oam_l` gate IS the Poletti & Horak Eq. (18) type-2 spatial
    # rule.  Index map: the paper's Q^(2) term in Eq. (6) is
    #   Q^(2)_plmn * A_l* A_m A_n  ->  channel p,  with L the CONJUGATED field,
    # so transverse-photon-momentum balance reads m_m + m_n = m_p + m_l, which is
    # exactly the printed rule -m_p - m_l + m_m + m_n = 0.  The engine's
    # degenerate term is A_n A_p A_q* -> channel m with q conjugated, i.e. the
    # map (p, l, m, n)_paper = (m, q, n, p)_engine, under which the paper's rule
    # becomes l_m + l_q = l_n + l_p -- with p = n the engine's own gate
    # l_m = l_n + l_p - l_q.  Verified here on all 10^4 quadruples.
    labels = np.array(fib_h.labels)
    q2 = ov.overlap_tensor(fib_h, 2, apply_pol=False)
    engine_gate = (
        labels[:, None, None, None]
        == labels[None, :, None, None]
        + labels[None, None, :, None]
        - labels[None, None, None, :]
    )
    eq18 = (
        labels[None, None, None, :] + labels[:, None, None, None]
        == labels[None, :, None, None] + labels[None, None, :, None]
    )
    gate_ok = bool((engine_gate == eq18).all())
    assert gate_ok, "engine oam_l gate disagrees with Eq. (18) type 2"

    # what the gate does NOT cover: it is the *spatial* rule only.  The
    # polarisation rule Eq. (19) and the exact magnitudes live in the weight
    # tensor, which is what the engine is driven with below.
    idx = np.arange(N_MODES)
    degenerate = q2[
        idx[:, None, None, None],
        idx[None, :, None, None],
        idx[None, :, None, None],
        idx[None, None, None, :],
    ]
    nonzero = np.abs(degenerate) > 1e-12
    gate_stats = {
        "n_quadruples": int(q2.size),
        "n_gate_eq_eq18_mismatches": int((engine_gate != eq18).sum()),
        "n_spatially_allowed": int(eq18.sum()),
        "n_nonzero_on_degenerate_slice": int(nonzero.sum()),
        "note": "gate == Eq. (18) type 2 exactly; Eq. (19) and the magnitudes "
        "are carried by fwm_weights, not by the gate",
    }

    grid = _make_grid()
    all_idx = list(range(N_MODES))
    n_steps = int(round(L_FIBER / DZ))

    def input_power(waves) -> NDArray:
        return np.array(
            [float(np.trapezoid(np.abs(w.envelope_field) ** 2, grid.t)) for w in waves]
        )

    # --- run A: the full 10-mode set, XPM only, sigma = + launch -----------
    # all ten channels are in the engine; the five sigma = - ones start empty
    wa = _channels(grid, _launch(all_idx, pos))
    p0a = input_power(wa)
    assert p0a[neg].max() == 0.0
    eng_a = _make_engine(wa, {"xpm": xpm_q}, include_fwm=False)
    eng_a.propagate(n_steps, nsaves=41)
    leak_a = _leak_fraction(eng_a, pos, neg)
    drift_a = float(abs(np.array(_power(eng_a)).sum() / p0a.sum() - 1.0))
    assert leak_a < TOL_LEAK, f"run A leaked into sigma-: {leak_a:.2e}"
    assert drift_a < TOL_ENERGY, f"run A energy drift {drift_a:.2e}"

    # --- run B: 4 channels with FWM + pump depletion, same launch ---------
    idx4 = [0, 1, 2, 3]  # LP01+/-, LP11 cos+/-
    wb = _channels(grid, _launch(idx4, [0, 2]))
    w4 = {
        "xpm": np.ascontiguousarray(xpm_q[np.ix_(idx4, idx4)]),
        "fwm": np.ascontiguousarray(np.real(fwm_q[np.ix_(idx4, idx4, idx4, idx4)])),
    }
    p0b = input_power(wb)
    eng_b = _make_engine(wb, w4, include_fwm=True)
    eng_b.propagate(n_steps, nsaves=41)
    leak_b = _leak_fraction(eng_b, [0, 2], [1, 3])
    drift_b = float(abs(np.array(_power(eng_b)).sum() / p0b.sum() - 1.0))
    assert leak_b < TOL_LEAK, f"run B leaked into sigma-: {leak_b:.2e}"
    assert drift_b < TOL_ENERGY, f"run B energy drift {drift_b:.2e}"

    # --- control: isotropic weights ignore Eq. (19) and must leak ---------
    wc = _channels(grid, _launch(idx4, [0, 2]))
    iso = {
        "xpm": np.ones((4, 4)),
        "fwm": np.ones((4, 4, 4, 4)),
    }
    eng_c = _make_engine(wc, iso, include_fwm=True)
    eng_c.propagate(n_steps, nsaves=41)
    leak_c = _leak_fraction(eng_c, [0, 2], [1, 3])
    assert leak_c > 1e-6, (
        f"isotropic control did not leak ({leak_c:.2e}) -- the sigma-closure "
        "check would be vacuous"
    )
    return {
        "xpm_weights": xpm_q.tolist(),
        "oam_gate_equals_eq18_type2": gate_ok,
        "oam_gate": gate_stats,
        "run_a_10modes_xpm": {
            "n_modes": len(wa),
            "leak_fraction": leak_a,
            "energy_drift": drift_a,
        },
        "run_b_4modes_fwm": {
            "n_modes": len(wb),
            "leak_fraction": leak_b,
            "energy_drift": drift_b,
        },
        "control_isotropic": {"leak_fraction": leak_c},
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(
    fig1: dict,
    complexity: dict,
    engine: dict,
    fib_r: ov.StepIndexFiber,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    keys = (
        ("helical_all", "all $2\\times10^4$ coefficients", "k", "-"),
        ("helical_after_rule18", "after Eq. (18)", "tab:blue", "--"),
        ("helical_after_rule18_and_19", "after Eq. (18)+(19)", "tab:red", "-."),
        ("real_basis", "real (cos/sin) LP basis", "tab:green", ":"),
    )
    for key, label, c, ls in keys:
        st = fig1["stats"][key]
        ax.plot(st["centres"], st["counts"], ls, color=c, lw=1.0, label=label)
    ax.annotate(
        "the 'all' and 'Eq. (18)' curves coincide:\n"
        f"all {fig1['n_zero']['all']} zeros of the 2 x 10^4\n"
        "coefficients are already the spatial rule",
        xy=(0.42, 0.30),
        xycoords="axes fraction",
        fontsize=7,
        color="k",
    )
    ax.axvline(0.01, color="gray", lw=0.8)
    ax.text(0.011, ax.get_ylim()[1] * 0.4, "1 % of max", fontsize=7, color="gray")
    ax.set(xscale="log", yscale="log", xlabel="$|Q| / \\max|Q|$", ylabel="count")
    ax.set_title("Fig. 1: magnitude distribution of the coupling coefficients")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    xpm = np.array(engine["xpm_weights"])
    im = ax.imshow(xpm, cmap="viridis", vmin=0, vmax=1)
    ax.set(
        xticks=range(N_MODES),
        yticks=range(N_MODES),
        xticklabels=[fib_r.modes[i].name for i in range(N_MODES)],
        yticklabels=[fib_r.modes[i].name for i in range(N_MODES)],
        title="$x_{ij} = Q^{(1)}_{iijj}$ (real LP basis)",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    m = np.array(complexity["M"])
    ax.loglog(m, complexity["n_all"], "k--", lw=0.9, label="$2M^4$ (unconstrained)")
    ax.loglog(
        m, complexity["n_nonzero"], "o-", color="tab:blue", lw=1.0, label="nonzero"
    )
    ax.loglog(
        m,
        m**4 * complexity["n_nonzero"][0] / m[0] ** 4,
        ":",
        color="gray",
        lw=0.9,
        label=f"$\\propto M^4$ (fit {complexity['fit_exponent']:.2f})",
    )
    for mm, rf in zip(m, complexity["reduction_factor"], strict=True):
        ax.annotate(
            f"{rf:.0f}x",
            (mm, complexity["n_nonzero"][list(m).index(mm)]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=7,
        )
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18])
    ax.set_xticklabels(["6", "8", "10", "12", "14", "16", "18"])
    ax.set(xlabel="number of modes $M$", ylabel="coupling coefficients")
    ax.set_title("Sec. 5: the selection rules cut the $O(M^4N)$ term")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ra, rb, rc = (
        engine["run_a_10modes_xpm"],
        engine["run_b_4modes_fwm"],
        engine["control_isotropic"],
    )
    labels = ["10 modes, XPM only", "4 modes + FWM", "isotropic control"]
    leaks = [ra["leak_fraction"], rb["leak_fraction"], rc["leak_fraction"]]
    ax.barh(
        labels,
        [max(v, 1e-18) for v in leaks],
        color=["tab:blue", "tab:blue", "tab:red"],
    )
    ax.axvline(TOL_LEAK, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set(
        xlabel="power fraction in the opposite circular polarisation", xlim=(1e-18, 1)
    )
    ax.set_title("Eq. (14): single-polarisation launch stays closed")
    for i, v in enumerate(leaks):
        ax.annotate(
            f"{v:.1e}",
            (max(v, 1e-18), i),
            textcoords="offset points",
            xytext=(5, -3),
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def validate(make_plot: bool = True) -> dict:
    fib_h = ov.mode_set_10(ov.v_number(A_CORE_UM, NA, LAMBDA_UM), basis="helical")
    fib_r = ov.mode_set_10(ov.v_number(A_CORE_UM, NA, LAMBDA_UM), basis="real")

    out = {
        "mode_solver": check_mode_solver(),
        "selection_rules": check_selection_rules(fib_h),
        "symmetries": check_symmetries(fib_h, fib_r),
        "fig1_distribution": check_fig1(fib_h, fib_r),
        "complexity_sec5": check_complexity(),
        "engine": check_engine(fib_r, fib_h),
    }
    if make_plot:
        make_figure(
            out["fig1_distribution"], out["complexity_sec5"], out["engine"], fib_r
        )
        out["figure"] = str(OUT_PNG)
    return out


if __name__ == "__main__":
    result = validate(make_plot=True)
    print(json.dumps(result, indent=2, default=str))
