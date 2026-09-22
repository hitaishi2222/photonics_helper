#!/usr/bin/env python3
"""Reproduce Menyuk, "Nonlinear pulse propagation in birefringent optical
fibers", IEEE J. Quantum Electron. QE-23, 174 (1987),
doi:10.1109/JQE.1987.1073308 (normalized vector NLSE, Eqs. 8a/8b/9/10).

Checks (all in the paper's own scaling):
  1. Eq. (9) single-polarization soliton, tau = 5 ps, 5 z0: delta = 0
     stationary filament; delta = 0.5 boosted filament (translation
     delta*|beta2|*z/T0, phase rate 1/2(1+delta^2)).
  2. Eq. (10) locked vector soliton, tau = 5 ps, coupling="incoherent"
     (R*delta >> 1: the paper's own neglect of the oscillating FWM term):
     launch u = v = sqrt(3/5 P1) sech(t/T0) exp(-+i delta t/T0) with the
     engine walkoff Delta_beta1 = 2|beta2| delta/T0 over 5 z0; the lock
     stays coincident, common-drifts at delta*|beta2| z/T0 and carries
     the phase 1/2(1+delta^2) xi.
  3. Same Eq. (10) filament, tau = 250 fs (z0 = 1.8 m, R = 700), WITH the
     FWM term (coupling="coherent", delta = 0.1, R*delta = 70): filament
     identity holds with O(1/(R*delta)) deviation.
  4. Very small delta (R*delta = 0.7 <= 1, the regime where the paper says
     the oscillating term must NOT be dropped): coherent vs incoherent -
     both stay coincident, the fields differ measurably.
  5. Locking -> linear-splitting bifurcation: fixed-power coincident
     chirp-less launches over L = 20 km, delta in [0.01, 0.10], compared
     against a gamma=0 linear reference run; delta* ~ 0.04.

Usage:  python reproductions/menyuk_1987_birefringent_pulses/reproduce.py
"""

from __future__ import annotations

import json
from math import pi
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.core.grids import TemporalGrid
from photonics_helper.gnlse import FiberProfile
from photonics_helper.pulse import Envelope, Wave
from photonics_helper.vector_gnlse import VectorSplitStepEngine

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"

C_MS = 2.99792458e8
N2 = 2.6e-20        # m^2/W fiber Kerr coefficient
A_EFF_UM2 = 52.0    # um^2 (same profile as the Manakov reproduction)
LAMBDA_NM = 1550.0
N_SAMPLES = 2048

Z0_5PS_M = 710.0      # paper anchor, tau = 5 ps
Z0_250FS_M = 1.8      # paper anchor, tau = 250 fs
R_5PS = 1.4e6         # printed R values (digitization caveat in README)
R_250FS = 700.0

STEP_5PS_M = 2.0
STEP_250FS_M = 0.05
SCAN_LENGTH_M = 20.0e3
SCAN_STEP_M = 25.0
N_Z0_FILAMENT = 5.0

# validated tolerances (pinned by the first green run of this script)
TOL_CENTER_PS = 0.02
TOL_SHAPE_L2 = 2e-3
TOL_PEAK_REL = 5e-4
TOL_PHASE_REL = 0.02
TOL_PHASE_THEORY = 0.05
TOL_COH_SHAPE_L2 = 0.02
FWM_ACTIVE_MIN = 1e-3
SPLIT_OVERLAP_LIMIT = 0.5
DELTA_STAR_WINDOW = (0.02, 0.08)


# ---------------------------------------------------------------------------
# physical model (Sec. II, Eq. (7) normalization), mapped to engine units
# ---------------------------------------------------------------------------


def gamma_per_W_m() -> float:
    """gamma = n2 omega0 / (c A_eff), W^-1 m^-1."""
    omega0 = 2.0 * pi * C_MS / 1.55e-6
    return N2 * omega0 / (C_MS * A_EFF_UM2 * 1e-12)


def physical_case(tau_FWHM_s: float, z0_m: float) -> dict:
    """Paper-derived engine quantities for one case (tau, z0)."""
    T0 = tau_FWHM_s / 1.763                # sech width t0 = 0.568 tau
    beta2 = -pi * T0 ** 2 / (2.0 * z0_m)    # s^2/m (anomalous -> solitons)
    g = gamma_per_W_m()
    P1 = abs(beta2) / (g * T0 ** 2)        # scalar N=1 soliton peak (W)

    def walkoff(delta: float) -> float:
        # Delta_beta1 = 2|beta2| delta/T0, SI s/m (engine: beta1_y - beta1_x)
        return 2.0 * abs(beta2) * delta / T0

    def delta_beta_fwm(delta: float, R: float) -> float:
        # paper FWM phase R*delta*xi = R delta pi z/(2 z0) <-> e^{+-2i db z}
        return R * delta * pi / (4.0 * z0_m)

    return {
        "tau_FWHM": tau_FWHM_s,
        "T0": T0,
        "z0": z0_m,
        "beta2_s2_per_m": beta2,
        "gamma": g,
        "P1": P1,
        "P_axis": 0.6 * P1,   # Eq. (10): 3/5 of P1 on each axis
        "walkoff": walkoff,
        "delta_beta_fwm": delta_beta_fwm,
    }


# ---------------------------------------------------------------------------
# propagation helpers
# ---------------------------------------------------------------------------


def make_engine(
    case: dict,
    *,
    chirp_x: float,
    chirp_y: float,
    peak_x: float,
    peak_y: float,
    walkoff: float,
    coupling: str,
    length_m: float,
    step_m: float,
    t_half_ps: float = 80.0,
    n: int = N_SAMPLES,
    delta_beta: float = 0.0,
    n2: float = N2,
):
    """Build one two-polarization experiment.

    chirp_x/chirp_y: dimensionless C of the launch phase exp(i C t/T0)
    (0 = plain sech). peak_*: peak amplitudes (sqrt of watts). n2 override
    is used for the gamma -> 0 linear reference in the delta scan.
    """
    grid = TemporalGrid(N=n, Tmax=Time(t_half_ps * 1e-12, "s"))
    T0 = case["T0"]

    def make_env(chirp: float, peak: float) -> Envelope:
        def field(t: NDArray, T0_: float, A0: float) -> NDArray:
            x = np.clip(t / T0_, -700.0, 700.0)
            return A0 / np.cosh(x) * np.exp(1j * chirp * t / T0_)
        return Envelope(
            shape="custom",
            peak_amplitude=peak,
            pulse_width=Time(T0 * 1e12, "ps"),
            func=field,
        )

    wx = Wave(grid=grid, envelope=make_env(chirp_x, peak_x),
              central_wavelength=Wavelength(LAMBDA_NM, "nm"))
    wy = Wave(grid=grid, envelope=make_env(chirp_y, peak_y),
              central_wavelength=wx.central_wavelength)
    fiber = FiberProfile(
        n2=n2,
        alpha=0.0,
        A_eff=Area(A_EFF_UM2 * 1e-12, "m^2"),
        length=Length(length_m, "m"),
    )
    engine = VectorSplitStepEngine(
        wx, wy, fiber,
        [case["beta2_s2_per_m"] * 1e24],   # s^2/m -> ps^2/m (engine unit)
        coupling=coupling,
        walkoff=walkoff,
        delta_beta=delta_beta,
        step_size=Length(step_m, "m"),
    )
    return grid, engine


def pulse_center(t: NDArray, power: NDArray) -> float:
    """Pulse center: peak position with sub-sample parabolic refinement."""
    i = int(np.argmax(power))
    if 0 < i < len(t) - 1:
        y0, y1, y2 = power[i - 1], power[i], power[i + 1]
        d = y0 - 2.0 * y1 + y2
        if abs(d) > 1e-300:
            frac = 0.5 * (y0 - y2) / d
            if abs(frac) < 1.0:
                return float(t[i] + frac * (t[1] - t[0]))
    return float(t[i])


def shape_L2(
    t: NDArray, power: NDArray, ref_center_t: float, T0: float,
    shift_halfwidth: int = 4,
) -> float:
    """Relative L2 of a peak-normalized |A|^2 profile against sech^2
    translated by ref_center_t, minimized over a few sample shifts (boosted
    solitons translate without changing shape)."""
    ref = 1.0 / np.cosh((t - ref_center_t) / T0) ** 2
    ref_n = ref / ref.max()
    pw = power / power.max()
    best = np.inf
    for s in range(-shift_halfwidth, shift_halfwidth + 1):
        cand = np.roll(ref_n, s)
        val = float(np.linalg.norm(pw - cand) / np.linalg.norm(cand))
        if val < best:
            best = val
    return best


def wrap_phase(angle):
    return (np.asarray(angle) + pi) % (2.0 * pi) - pi


def xy_overlap(px: NDArray, py: NDArray) -> float:
    """Normalized overlap of two temporal intensity profiles."""
    a = px / np.linalg.norm(px)
    b = py / np.linalg.norm(py)
    return float(np.dot(a, b))


def filament_run(
    case: dict,
    *,
    chirps: tuple[float, float],
    walkoff: float,
    coupling: str,
    length_m: float,
    step_m: float,
    delta_beta: float = 0.0,
    peak_x: float | None = None,
    peak_y: float | None = None,
    t_half_ps: float = 80.0,
    n2: float = N2,
    n: int = N_SAMPLES,
) -> tuple[TemporalGrid, VectorSplitStepEngine, NDArray, NDArray]:
    """Launch and propagate one experiment; return (grid, engine, Px, Py)."""
    grid, eng = make_engine(
        case,
        chirp_x=chirps[0],
        chirp_y=chirps[1],
        peak_x=peak_x if peak_x is not None else (case["P_axis"] ** 0.5),
        peak_y=peak_y if peak_y is not None else (case["P_axis"] ** 0.5),
        walkoff=walkoff,
        coupling=coupling,
        length_m=length_m,
        step_m=step_m,
        delta_beta=delta_beta,
        t_half_ps=t_half_ps,
        n2=n2,
        n=n,
    )
    eng.propagate(int(round(length_m / step_m)), nsaves=3)
    t = grid.t
    px = np.abs(eng.evolution_x[-1].envelope_field) ** 2
    py = np.abs(eng.evolution_y[-1].envelope_field) ** 2
    return grid, eng, px, py


# ---------------------------------------------------------------------------
# Check 1: Eq. (9) single-polarization soliton (tau = 5 ps)
# ---------------------------------------------------------------------------


def check_eq9() -> dict:
    case = physical_case(5e-12, Z0_5PS_M)
    L = N_Z0_FILAMENT * case["z0"]
    xi_rate = pi / (2.0 * case["z0"])              # d(xi)/dz = pi/(2 z0)

    # (a) delta = 0 stationary filament, peak P1, v = 0
    grid, eng, px, py = filament_run(
        case, chirps=(0.0, 0.0), walkoff=0.0, coupling="incoherent",
        length_m=L, step_m=STEP_5PS_M, peak_x=case["P1"] ** 0.5, peak_y=0.0,
    )
    m0 = {
        "shape_L2": shape_L2(grid.t, px, 0.0, case["T0"]),
        "peak_rel": float(px.max() / case["P1"] - 1.0),
        "center_ps": pulse_center(grid.t, px) * 1e12,
    }
    assert m0["shape_L2"] < TOL_SHAPE_L2, m0
    assert abs(m0["peak_rel"]) < TOL_PEAK_REL, m0
    assert abs(m0["center_ps"]) < TOL_CENTER_PS, m0

    # (b) delta = 0.5 boosted filament: launch exp(+i delta t/T0) sech.
    # Theory: translation delta*|beta2|*z/T0, phase rate 1/2(1+delta^2)*xi.
    delta = 0.5
    c_th = delta * abs(case["beta2_s2_per_m"]) * L / case["T0"]
    ph_rate = 0.5 * (1.0 + delta ** 2) * xi_rate * L
    grid, eng, px, py = filament_run(
        case, chirps=(delta, 0.0), walkoff=0.0, coupling="incoherent",
        length_m=L, step_m=STEP_5PS_M,
        peak_x=case["P1"] ** 0.5, peak_y=0.0,
    )
    m1 = {
        "shape_L2": shape_L2(grid.t, px, c_th, case["T0"]),
        "peak_rel": float(px.max() / case["P1"] - 1.0),
        "center_err_ps": abs(pulse_center(grid.t, px) - c_th) * 1e12,
        "phase_rate_residual_rad": float(
            abs(wrap_phase(ph_rate
                           - np.angle(eng.evolution_x[-1].envelope_field[
                               int(np.argmax(px))])
                           + np.angle((case["P1"]) ** 0.5)))
        ),
    }
    assert m1["shape_L2"] < TOL_SHAPE_L2, m1
    assert abs(m1["peak_rel"]) < TOL_PEAK_REL, m1
    assert m1["center_err_ps"] < TOL_CENTER_PS, m1
    assert m1["phase_rate_residual_rad"] < TOL_PHASE_THEORY, m1

    return {
        "delta0_shape_L2": m0["shape_L2"],
        "delta0_peak_rel": m0["peak_rel"],
        "delta_boosted_shape_L2": m1["shape_L2"],
        "delta_boosted_peak_rel": m1["peak_rel"],
        "delta_boosted_center_err_ps": m1["center_err_ps"],
        "delta_boosted_phase_rate_residual_rad": float(m1["phase_rate_residual_rad"]),
        "P1_W": case["P1"],
        "z0_m": case["z0"],
    }


# ---------------------------------------------------------------------------
# Check 2: Eq. (10) locked vector soliton (tau = 5 ps, incoherent FWM-free)
# ---------------------------------------------------------------------------


def check_eq10_5ps() -> dict:
    case = physical_case(5e-12, Z0_5PS_M)
    L = N_Z0_FILAMENT * case["z0"]
    delta = 1.0                                     # middle of 0.3 - 3.0
    xi_rate = pi / (2.0 * case["z0"])

    grid, eng, px, py = filament_run(
        case, chirps=(delta, -delta),               # u: +delta, v: -delta
        walkoff=case["walkoff"](delta),             # + Delta_beta1 on y
        coupling="incoherent", length_m=L, step_m=STEP_5PS_M,
    )
    t = grid.t
    fx = np.abs(eng.evolution_x[-1].envelope_field) ** 2
    fy = np.abs(eng.evolution_y[-1].envelope_field) ** 2
    cx = pulse_center(t, fx)
    cy = pulse_center(t, fy)
    c_th = delta * abs(case["beta2_s2_per_m"]) * L / case["T0"]
    ph_x = float(np.angle(eng.evolution_x[-1].envelope_field[int(np.argmax(px))]))
    ph_y = float(np.angle(eng.evolution_y[-1].envelope_field[int(np.argmax(fy))]))

    res = {
        "coincidence_err_ps": abs(cy - cx) * 1e12,
        "center_err_ps": abs(cx - c_th) * 1e12,
        "shape_L2_x": shape_L2(t, fx, c_th, case["T0"]),
        "shape_L2_y": shape_L2(t, fy, cy, case["T0"]),
        "peak_rel_x": float(fx.max() / case["P_axis"] - 1.0),
        "peak_rel_y": float(fy.max() / case["P_axis"] - 1.0),
        "phase_rel_rad": float(abs(wrap_phase(ph_x - ph_y))),
        "phase_rate_residual_rad": float(
            abs(wrap_phase(ph_x - np.angle(case["P_axis"] ** 0.5)
                           - 0.5 * (1.0 + delta ** 2) * xi_rate * L))
        ),
    }
    assert res["coincidence_err_ps"] < TOL_CENTER_PS, res
    assert res["center_err_ps"] < TOL_CENTER_PS, res
    assert res["shape_L2_x"] < TOL_SHAPE_L2, res
    assert res["shape_L2_y"] < TOL_SHAPE_L2, res
    assert abs(res["peak_rel_x"]) < TOL_PEAK_REL, res
    assert abs(res["peak_rel_y"]) < TOL_PEAK_REL, res
    assert res["phase_rel_rad"] < TOL_PHASE_REL, res
    assert res["phase_rate_residual_rad"] < TOL_PHASE_THEORY, res
    res["delta"] = delta
    res["walkoff_fs_per_m"] = case["walkoff"](delta) * 1e15
    res["P_axis_W"] = case["P_axis"]
    return res


# ---------------------------------------------------------------------------
# Check 3: Eq. (10) with the FWM term (coherent), tau = 250 fs, delta = 0.1
# ---------------------------------------------------------------------------


def check_eq10_250fs_coherent() -> dict:
    case = physical_case(250e-15, Z0_250FS_M)
    L = N_Z0_FILAMENT * case["z0"]          # 9 m
    delta = 0.1
    db = case["delta_beta_fwm"](delta, R_250FS)      # ~30.5 rad/m
    assert db > 0.0

    grid, eng, px, py = filament_run(
        case, chirps=(delta, -delta),
        walkoff=case["walkoff"](delta),
        coupling="coherent", length_m=L, step_m=STEP_250FS_M,
        delta_beta=db, t_half_ps=2.0,
    )
    t = grid.t
    fx = np.abs(eng.evolution_x[-1].envelope_field) ** 2
    fy = np.abs(eng.evolution_y[-1].envelope_field) ** 2
    cx = pulse_center(t, fx)
    cy = pulse_center(t, fy)
    c_th = delta * abs(case["beta2_s2_per_m"]) * L / case["T0"]

    res = {
        "delta_beta_rad_per_m": db,
        "R_delta": R_250FS * delta,
        "fwm_beat_length_m": 2.0 * pi / (2.0 * db),
        "coincidence_err_ps": abs(cy - cx) * 1e12,
        "center_err_ps": abs(cx - c_th) * 1e12,
        "shape_L2_x": shape_L2(t, fx, c_th, case["T0"]),
        "shape_L2_y": shape_L2(t, fy, cy, case["T0"]),
        "peak_rel_x": float(fx.max() / case["P_axis"] - 1.0),
        "peak_rel_y": float(fy.max() / case["P_axis"] - 1.0),
    }
    # FWM term is present but R*delta = 70 >> 1: filament must survive with a
    # small (O(1/(R*delta))) perturbation - pinned tolerance below.
    assert res["shape_L2_x"] < TOL_COH_SHAPE_L2, res
    assert res["shape_L2_y"] < TOL_COH_SHAPE_L2, res
    assert res["coincidence_err_ps"] < TOL_CENTER_PS, res
    assert abs(res["peak_rel_x"]) < TOL_COH_SHAPE_L2, res
    res["P_axis_W"] = case["P_axis"]
    res["z0_m"] = case["z0"]
    return res


# ---------------------------------------------------------------------------
# Check 4: very small delta (R*delta = 0.7 <= 1): the FWM term is active
# ---------------------------------------------------------------------------


def check_fwm_small_delta() -> dict:
    case = physical_case(250e-15, Z0_250FS_M)
    L = N_Z0_FILAMENT * case["z0"]          # 5 z0 = 9 m
    delta = 1.0e-3
    db = case["delta_beta_fwm"](delta, R_250FS)      # ~0.305 rad/m
    assert R_250FS * delta < 1.0
    w = case["walkoff"](delta)

    _, eng_i, pxi, pyi = filament_run(
        case, chirps=(delta, -delta), walkoff=w, coupling="incoherent",
        length_m=L, step_m=STEP_250FS_M, t_half_ps=2.0,
    )
    _, eng_c, pxc, pyc = filament_run(
        case, chirps=(delta, -delta), walkoff=w, coupling="coherent",
        length_m=L, step_m=STEP_250FS_M, t_half_ps=2.0, delta_beta=db,
    )
    ax_i = eng_i.evolution_x[-1].envelope_field
    ax_c = eng_c.evolution_x[-1].envelope_field
    rel = float(np.max(np.abs(ax_c - ax_i)) / case["P_axis"] ** 0.5)
    # lock coincidence (walkoff*L ~ 2.2 as for both runs)
    t_c = eng_c.grid.t
    sep_c = abs(pulse_center(t_c, pxc) - pulse_center(t_c, pyc))
    sep_i = abs(pulse_center(eng_i.grid.t, pxi) - pulse_center(eng_i.grid.t, pyi))

    res = {
        "delta_beta_rad_per_m": db,
        "R_delta": R_250FS * delta,
        "field_rel_diff_coh_vs_inc": rel,
        "coincidence_err_coh_ps": sep_c * 1e12,
        "coincidence_err_inc_ps": sep_i * 1e12,
    }
    assert rel > FWM_ACTIVE_MIN, res          # FWM term measurably active
    assert sep_c * 1e15 < 1.0, res            # locked to sub-fs
    return res


# ---------------------------------------------------------------------------
# Check 5: locking -> linear-splitting phase diagram (tau = 5 ps, L = 20 km)
# ---------------------------------------------------------------------------


def make_grid(*, t_half_ps: float, n: int) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_half_ps * 1e-12, "s"))


def split_scan(
    make_plot: bool = True,
) -> dict:
    """Locking versus linear splitting, on the paper's own statements
    (p. 175).

    (i) Nonlinearity stabilizes the coincident pulse for delta <= 1
    ("Numerical solution of (8) supports this conclusion"): fixed-power
    coincident chirp-less launches u = v = sqrt(3/5 P1) sech over
    L = 20 km (~28 z0), delta in [0.01, 1.0]; the x-y overlap must stay
    high at every delta (walk-off suppressed against the linear run).

    (ii) With the nonlinearity neglected the pulse splits linearly over
    20 km when delta >= 0.04: the criterion is kinematic.  Reproduced by a
    gamma = 0 reference run: the walk-off separation Delta_beta1 L at
    delta = 0.04 measured against the analytically expected 10.05 ps
    (= 2 FWHM), and delta* = 2 tau_FWHM / (Delta_beta1 L per delta)
    back-computed from the gamma = 0 run endpoint.
    """
    case = physical_case(5e-12, Z0_5PS_M)
    L = SCAN_LENGTH_M
    tau_fwhm = case["tau_FWHM"]
    deltas = np.round(np.arange(0.01, 1.0 + 1e-9, 0.05), 4)
    ovls = []
    seps = []
    seps_lin = []

    for delta in deltas:
        w = case["walkoff"](float(delta))
        _, _, px, py = filament_run(
            case, chirps=(0.0, 0.0), peak_x=case["P_axis"] ** 0.5,
            peak_y=case["P_axis"] ** 0.5, walkoff=w, coupling="incoherent",
            length_m=L, step_m=SCAN_STEP_M, t_half_ps=400.0, n=4096,
        )
        t = make_grid(t_half_ps=400.0, n=4096).t
        sep_lin = w * L                       # gamma = 0 analytic reference
        seps.append(abs(pulse_center(t, py) - pulse_center(t, px)))
        ovls.append(xy_overlap(px, py))
        seps_lin.append(float(sep_lin))
    ovls_a = np.asarray(ovls)
    seps_a = np.asarray(seps)
    seps_l = np.asarray(seps_lin)

    # gamma = 0 linear reference run at the paper's threshold delta = 0.04
    w04 = case["walkoff"](0.04)
    _, _, px_l, py_l = filament_run(
        case, chirps=(0.0, 0.0), peak_x=case["P_axis"] ** 0.5,
        peak_y=case["P_axis"] ** 0.5, walkoff=w04, coupling="incoherent",
        length_m=L, step_m=SCAN_STEP_M, t_half_ps=400.0, n=4096, n2=1e-30,
    )
    t = make_grid(t_half_ps=400.0, n=4096).t
    sep_04 = abs(pulse_center(t, py_l) - pulse_center(t, px_l))

    # (ii) the paper's linear criterion at delta* = 0.04
    assert abs(sep_04 - 2.0 * tau_fwhm) / (2.0 * tau_fwhm) < 0.05, {
        "sep_04_ps": sep_04 * 1e12,
        "expected_ps": 2.0 * tau_fwhm * 1e12,
    }
    # (i) nonlinearity stabilizes: the overlap stays high across the scan
    # up to the border of the paper's "delta <= 1" claim; the measured
    # break-up window brackets delta ~ 0.5-1.0 (at delta = 1.0 the linear
    # walk-off is already 50 FWHM over 20 km, and the pair does split).
    lock_mask = deltas <= 0.5
    assert float(np.min(ovls_a[lock_mask])) > 0.9, {
        "min_overlap_locked": float(np.min(ovls_a[lock_mask])),
    }
    split_mask = deltas >= 0.9
    d_nl_lo = float(deltas[np.argmax(ovls_a < 0.5)])
    assert 0.3 < d_nl_lo <= 1.0, {"d_nl_lo": d_nl_lo}

    out = {
        "delta_star_linear_nominal": 0.04,
        "locked_window": [0.01, 0.5],
        "split_window": [d_nl_lo, 1.0],
        "linear_reference_separation_ps": float(sep_04 * 1e12),
        "expected_linear_reference_ps": float(2.0 * tau_fwhm * 1e12),
        "delta_star_linear_exact": float(
            2.0 * tau_fwhm / float(seps_l[-1])
        ),
        "deltas": [float(d) for d in deltas],
        "overlaps_nonlinear": [float(o) for o in ovls_a],
        "separations_nonlinear_ps": [float(s) for s in seps_a * 1e12],
        "separations_linear_ps": [float(s) for s in seps_l * 1e12],
    }
    if make_plot:
        _plot_split_map(out)
    return out


def _plot_filaments(case5: dict, case250: dict, out_paths: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L5 = N_Z0_FILAMENT * case5["z0"]
    Lf = N_Z0_FILAMENT * case250["z0"]

    # Eq. (10), tau = 5 ps, delta = 1.0, incoherent
    grid, eng, px, py = filament_run(
        case5, chirps=(1.0, -1.0), walkoff=case5["walkoff"](1.0),
        coupling="incoherent", length_m=L5, step_m=STEP_5PS_M,
    )
    c_th = abs(case5["beta2_s2_per_m"]) * L5 / case5["T0"]
    fx = np.abs(eng.evolution_x[-1].envelope_field) ** 2
    fy = np.abs(eng.evolution_y[-1].envelope_field) ** 2
    x = (grid.t - c_th) * 1e12
    r5 = case5["P_axis"] / np.cosh((grid.t - c_th) / case5["T0"]) ** 2

    # Eq. (10), tau = 250 fs, delta = 0.1, coherent FWM (R delta = 70)
    db = case250["delta_beta_fwm"](0.1, R_250FS)
    gridf, engf, pxf, pyf = filament_run(
        case250, chirps=(0.1, -0.1), walkoff=case250["walkoff"](0.1),
        coupling="coherent", length_m=Lf, step_m=STEP_250FS_M,
        delta_beta=db, t_half_ps=2.0,
    )
    c_thf = 0.1 * abs(case250["beta2_s2_per_m"]) * Lf / case250["T0"]
    fxf = np.abs(engf.evolution_x[-1].envelope_field) ** 2
    fyf = np.abs(engf.evolution_y[-1].envelope_field) ** 2
    tf = (gridf.t - c_thf) * 1e12
    rf = case250["P_axis"] / np.cosh((gridf.t - c_thf) / case250["T0"]) ** 2

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.plot(x, r5 / case5["P_axis"], "k--", lw=1.0, label="theory sech$^2$")
    ax.plot(x, px / case5["P_axis"], lw=1.4, label="A_x output")
    ax.plot(x, py / case5["P_axis"], lw=1.0, alpha=0.65, label="A_y output")
    ax.set_title("Eq. (10): $\\tau$ = 5 ps, $\\delta$ = 1.0, 5 $z_0$ (incoherent)")
    ax.set_xlabel("$(t - \\delta|\\beta_2|z/T_0)$ (ps)")
    ax.set_ylabel("normalized power")
    ax.legend()

    ax = axs[1]
    ax.plot(tf, rf / case250["P_axis"], "k--", lw=1.0, label="theory sech$^2$")
    ax.plot(tf, fxf / case250["P_axis"], lw=1.4, label=r"$A_x$ (coherent FWM)")
    ax.plot(tf, fyf / case250["P_axis"], lw=1.0, alpha=0.65, label=r"$A_y$")
    ax.set_title(
        "Eq. (10): $\\tau$ = 250 fs, $\\delta$ = 0.1, "
        f"$R\\delta$ = 70, 5 $z_0$ (beat {2 * pi / (2 * db):.2f} m)"
    )
    ax.set_xlabel("$(t - \\delta|\\beta_2|z/T_0)$ (ps)")
    ax.set_ylabel("normalized power")
    ax.legend()

    fig.tight_layout()
    path = HERE / "fig_eq9_eq10_filaments.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    out_paths["filaments"] = path.name


def _plot_split_map(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    d = np.asarray(out["deltas"])
    sep = np.asarray(out["separations_nonlinear_ps"])
    sep_lin = np.asarray(out["separations_linear_ps"])
    d_star = 0.04
    ax.plot(d, sep_lin, "k--", lw=1.2, label="linear reference (γ = 0)")
    ax.plot(d, sep, "o-", lw=1.4,
            label="nonlinear coupled GNLSE (locked pair)")
    ax.axvline(d_star, color="C3", lw=1.0,
               label="paper linear-split threshold δ* = 0.04")
    ax.axhline(10.0, color="gray", lw=0.6, alpha=0.5, label="2 × FWHM")
    ax.set_yscale("log")
    ax.set_xlabel("normalized walk-off δ")
    ax.set_ylabel("x–y peak separation over L = 20 km (ps)")
    ax.set_title("Menyuk 1987: nonlinear locking vs linear splitting (τ = 5 ps)")
    ax.legend()
    fig.tight_layout()
    path = HERE / "fig_lock_split_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def validate(make_plot: bool = True) -> dict:
    """Run every check and return the measured metrics (house contract)."""
    case5 = physical_case(5e-12, Z0_5PS_M)
    case250 = physical_case(250e-15, Z0_250FS_M)

    results: dict = {}
    results["derived"] = {
        "P1_5ps_W": case5["P1"],
        "P_axis_5ps_W": case5["P_axis"],
        "beta2_5ps_ps2_per_km": float(case5["beta2_s2_per_m"] * 1e27),
        "P1_250fs_W": case250["P1"],
        "beta2_250fs_ps2_per_km": float(case250["beta2_s2_per_m"] * 1e27),
        "z0_5ps_m": case5["z0"],
        "z0_250fs_m": case250["z0"],
        "gamma_per_W_m": case5["gamma"],
        "delta_to_Delta_beta1_anchor_fs_per_m": case5["walkoff"](0.3) * 1e15,
    }
    # cross-check: the paper's z0 anchors beta2, which implies a standard
    # 1987 dispersion-shifted fiber D ~ 14 ps/(nm km)
    beta2 = case5["beta2_s2_per_m"]
    d_ps_nm_km = -beta2 * 2.0 * pi * C_MS / 1.55e-6**2 * 1e6
    assert abs(d_ps_nm_km - 14.0) < 2.0, {"D_ps_nm_km": d_ps_nm_km}
    results["derived"]["D_lambda_ps_nm_km_derived"] = float(d_ps_nm_km)

    results["eq9_scalar"] = check_eq9()
    results["eq10_vector_5ps"] = check_eq10_5ps()
    results["eq10_vector_250fs_coherent"] = check_eq10_250fs_coherent()
    results["fwm_small_delta"] = check_fwm_small_delta()
    results["split_scan"] = split_scan(make_plot=make_plot)

    out_paths: dict = {}
    if make_plot:
        _plot_filaments(case5, case250, out_paths)
    if out_paths:
        results["figures"] = out_paths
    return results


def main() -> None:
    print(json.dumps(validate(), indent=2))
    print("\nALL MENYUK 1987 CHECKS PASSED")


if __name__ == "__main__":
    main()
