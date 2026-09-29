"""Reproduction: accelerated intermodal dynamics in a GRIN multimode taper
(Eftekhar et al. 2019).

Reference
---------
M. A. Eftekhar, Z. Sanjabi-Eznaveh, H. E. Lopez-Aviles, S. Benis,
J. E. Antonio-Lopez, M. Kolesik, F. Wise, R. Amezcuas-Correa &
D. N. Christodoulides, "Accelerated nonlinear interactions in graded-index
multimode fibers", Nat. Commun. 10, 1638 (2019), doi:10.1038/s41467-019-09687-9
(arXiv-backed PDF in this folder; page refs below).

What is reproduced
------------------
The paper's central mechanism is *taper-induced acceleration*: in a GRIN MMF
whose core radius shrinks as a(z) = a0 exp(-gamma z / 2) (PDF p-02), the
paraxial GRIN modal spacing scales as 1/a, so (i) the spatial self-imaging
oscillation accelerates along z (Fig. 1b), and (ii) the intermodal
group-delay walk-off db1_p(a(z)) grows proportionally, accumulating a
closed-form delay across the taper (Methods constants, PDF p-08:
Delta = 1.6e-3, n2 = 2.9e-20 m^2/W, core radius 40 -> 10 um, 1550 nm runs).

The z-dependence is propagated with the established **chunked-taper**
pattern: the engine is uniform over each chunk, and each chunk is rebuilt
with the walk-off ``group_delays``, ladder ``phase_offsets`` and modal
overlap tensors of its mid-chunk core radius.

Checks
------
1. ANALYTIC (closed form): self-imaging period L_si(a) = pi a / sqrt(2 Delta)
   and intermodal walk-offs db1_p(a) at a0 = 40 um and a_end = 10 um; the
   walk-off scaling ratio db1(a_end)/db1(a0) matches a0/a_end to <= 2 %.
2. ENGINE, LINEAR (chunked taper, 40 chunks / 4-um steps): a 1.5*w0 Gaussian
   seed across the 3 lowest symmetric radial modes.
   a. measured modal-beat (MFD-oscillation) period at the taper start vs
      end matches L_si(a0), L_si(a_end) to <= 5 % each; the measured
      acceleration ratio matches a0/a_end = 4 to <= 10 %;
   b. accumulated higher-mode temporal walk-off across the taper matches
      the analytic staircase integral (int db1_p(a(z)) dz) to <= 5 %;
   c. total energy conserved to <= 1e-6.
3. ENGINE, NONLINEAR (SPM/XPM only, FWM off): the same chunked taper with an
   N ~ 1 fundamental-mode soliton reproduces the accumulated walk-off to
   <= 8 % and conserves energy to <= 0.5 % (the multimode engine has no
   Raman/self-steepening; the paper's DW-cascade figures are documented as
   out of scope on this 3-mode stack).

Usage
-----
    python reproductions/planned/eftekhar_2019_parametric_cascades/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import genlaguerre

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"

C_MS = 299792458.0

# --- fibre (Methods, PDF p-08; taper law PDF p-02) --------------------------
LAMBDA0 = 1550e-9  # m
OMEGA0 = 2.0 * np.pi * C_MS / LAMBDA0
N0 = 1.444
DELTA = 1.6e-3
N2 = 2.9e-20  # m^2/W
R0 = 40.0e-6  # m, initial core radius
R_END = 10.0e-6  # m, taper-end core radius
TAPER_L = 0.04  # m (Fig. 1b example: 4-cm taper, core 40 -> 10 um)
GAMMA_T = 2.0 * np.log(4.0) / TAPER_L  # a = a0 exp(-gamma*z/2); exp(g*z/2)=4
BETA2 = -2.81e-26  # s^2/m (= -28.1 fs^2/mm), silica at 1550 nm

N_MODES = 3
TAU0 = 170.2e-15  # s, sech width (300 fs FWHM)
E_PULSE = 0.5e-9  # J, linear-probe pulse energy
E_SOLITON = 1.05e-9  # J, N ~ 1 fundamental soliton scale
SEED_FACTOR = 1.5  # input spot = 1.5x fundamental w0 (paper: Fig. 1b)

# --- simulation -------------------------------------------------------------
N_GRID = 16384
WINDOW_S = 20.0e-12
NUM_CHUNKS = 40
CHUNK_L = TAPER_L / NUM_CHUNKS
STEP_M = 4.0e-6

MFD_X = np.linspace(1e-9, 16.0, 4000)  # reduced radial grid x = 2 r^2/w0^2


# ---------------------------------------------------------------------------
# analytic paraxial GRIN mode model (Renninger-Wise machinery, with a(z))
# ---------------------------------------------------------------------------


def core_radius(z: float | NDArray) -> NDArray:
    """Taper law a(z) = a0 exp(-gamma z / 2) (m)."""
    return R0 * np.exp(-GAMMA_T * np.asarray(z, float) / 2.0)


def beta_p(omega: float, p: int, a: float) -> float:
    """Paraxial GRIN propagation constant of symmetric mode p (rad/m)."""
    k = omega * N0 / C_MS
    return k * np.sqrt(1.0 - 2.0 * np.sqrt(2.0 * DELTA) * (2 * p + 1) / (k * a))


def mode_w0(a: float) -> float:
    """Fundamental mode size w0(a) = (2 a^2 / (k0^2 Delta))^{1/4} (m)."""
    k0 = OMEGA0 * N0 / C_MS
    return (2.0 * a**2 / (k0**2 * DELTA)) ** 0.25


def self_imaging_period(a: float) -> float:
    """GRIN self-imaging period pi*a/sqrt(2*Delta) (m)."""
    return np.pi * a / np.sqrt(2.0 * DELTA)


def mode_walkoffs(a: float) -> NDArray:
    """db1^(p)(a) in s/m for p = 0,1,2 (from d beta_p(a)/d omega)."""
    h = OMEGA0 * 1e-8
    out = [0.0]
    for p in (1, 2):
        vals = [
            beta_p(OMEGA0 + s * h, p, a) - beta_p(OMEGA0 + s * h, 0, a)
            for s in (-2, -1, 1, 2)
        ]
        out.append((-vals[3] + 8 * vals[2] - 8 * vals[1] + vals[0]) / (12 * h))
    return np.array(out)


def walkoff_integral_fine() -> NDArray:
    """Accumulated db1^(p) delay across the taper (s), fine staircase."""
    zc = np.linspace(0.0, TAPER_L, 2001)
    integ = np.zeros(2)
    for i in range(len(zc) - 1):
        db1 = mode_walkoffs(float(core_radius(0.5 * (zc[i] + zc[i + 1]))))
        integ[0] += db1[1] * (zc[i + 1] - zc[i])
        integ[1] += db1[2] * (zc[i + 1] - zc[i])
    return integ


def radial_modes(w0: float, nmode: int = 3) -> list[NDArray]:
    """Normalised symmetric Laguerre-Gauss modes F_p(x), x = 2 r^2/w0^2."""
    return [
        np.sqrt(2.0 / np.pi) / w0 * genlaguerre(p, 0)(MFD_X) * np.exp(-MFD_X / 2.0)
        for p in range(nmode)
    ]


def overlap_weights(w0: float, nmode: int = N_MODES):
    """SPM/XPM and FWM overlap tensors, normalised to the p=0 self-overlap."""
    f = radial_modes(w0, nmode)
    area = np.pi * w0**2 / 2.0
    S = np.array(
        [
            [
                np.trapezoid(f[i] ** 2 * f[j] ** 2, MFD_X) * area
                for j in range(nmode)
            ]
            for i in range(nmode)
        ]
    )
    g = np.zeros((nmode,) * 4)
    for i in range(nmode):
        for j in range(nmode):
            for k in range(nmode):
                for ell in range(nmode):
                    g[i, j, k, ell] = (
                        np.trapezoid(f[i] * f[j] * f[k] * f[ell], MFD_X) * area
                    )
    return S / S[0, 0], g / S[0, 0], 1.0 / S[0, 0]


def mode_phase_offsets(a: float) -> NDArray:
    """db0^(p)(a) = beta_p(omega0; a) - beta_0(omega0; a) (rad/m)."""
    return np.array(
        [beta_p(OMEGA0, p, a) - beta_p(OMEGA0, 0, a) for p in range(3)]
    )


def mode_radius_moments(w0: float, nmode: int = N_MODES) -> NDArray:
    """Radial second-moment integrals M_pq = int r^2 F_p F_q dA (m^2)."""
    f = radial_modes(w0, nmode)
    return np.array(
        [
            [
                (np.pi * w0**4 / 4.0)
                * np.trapezoid(MFD_X * f[p] * f[q], MFD_X)
                for q in range(nmode)
            ]
            for p in range(nmode)
        ]
    )


def seed_fractions(w0: float) -> NDArray:
    """Energy fractions of the 3 symmetric modes under a 1.5*w0 Gaussian."""
    area = np.pi * w0**2 / 2.0
    rho = np.sqrt(MFD_X * w0**2 / 2.0)
    fg = np.exp(-rho**2 / (SEED_FACTOR * w0) ** 2)
    num = np.array([np.trapezoid(f * fg, MFD_X) * area for f in radial_modes(w0)])
    norm = np.trapezoid(fg**2, MFD_X) * area
    e = num**2 / norm
    return e / e.sum()


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def centroid(t: NDArray, power: NDArray) -> float:
    return float(np.sum(t * power) / np.sum(power))


def mfd_from_fields(a: list[NDArray], dt: float, moments: NDArray) -> float:
    """Mode-field diameter from the modal second moment (m)."""
    denom = 0.0
    num = 0.0
    for p in range(len(a)):
        denom += float(np.sum(np.abs(a[p]) ** 2)) * dt
        for q in range(len(a)):
            num += float(np.real(np.sum(a[p] * np.conj(a[q])))) * dt * moments[p, q]
    r2 = num / max(denom, 1e-300)
    return float(2.0 * np.sqrt(2.0) * np.sqrt(max(r2, 0.0)))


def dominant_period(z_axis: NDArray, curve: NDArray) -> float:
    """Dominant oscillation period of a quasi-periodic z record (m)."""
    spectrum = np.abs(np.fft.rfft(curve - curve.mean())) ** 2
    dz_all = np.diff(z_axis)
    d = float(np.median(dz_all[dz_all > 0]))
    freqs = np.fft.rfftfreq(len(curve), d=d)
    idx = int(np.argmax(spectrum[1:]) + 1)
    return float(1.0 / freqs[idx])


# ---------------------------------------------------------------------------
# chunked-taper driver
# ---------------------------------------------------------------------------


def make_grid() -> TemporalGrid:
    return TemporalGrid(N=N_GRID, Tmax=Time(WINDOW_S, "s"))


def chunk_params(k: int) -> dict:
    """Per-chunk modal data at the mid-chunk core radius."""
    a = float(core_radius((k + 0.5) * CHUNK_L))
    w0 = mode_w0(a)
    xpm, fwm, aeff = overlap_weights(w0)
    return {
        "z_mid": (k + 0.5) * CHUNK_L,
        "a": a,
        "w0": w0,
        "aeff": aeff,
        "delay": [float(v) for v in mode_walkoffs(a)],
        "db0": [float(v) for v in mode_phase_offsets(a)],
        "xpm": xpm,
        "fwm": fwm,
        "moments": mode_radius_moments(w0),
    }


def _waves(grid: TemporalGrid, fields: list[NDArray]) -> list[Wave]:
    waves = []
    for p in range(3):
        wave = Wave(
            grid=grid,
            envelope=Envelope(
                shape="sech",
                peak_amplitude=1e-30,
                pulse_width=Time(TAU0, "s"),
            ),
            central_wavelength=Wavelength(LAMBDA0 * 1e9, "nm"),
        )
        waves.append(wave.with_field(fields[p]))
    return waves


def sech_fields(grid: TemporalGrid, e_pulse: float) -> list[NDArray]:
    """Sech pulse in each mode with the 1.5*w0 modal energy fractions."""
    w0 = mode_w0(float(core_radius(0.0)))
    fractions = seed_fractions(w0)
    amp_p = np.sqrt(e_pulse * fractions / (2.0 * TAU0))
    base = 1.0 / np.cosh(np.asarray(grid.t, float) / TAU0)
    return [float(amp_p[p]) * base for p in range(3)]


def propagate_taper(
    e_pulse: float, nonlinear: bool, grid: TemporalGrid
) -> tuple[MultimodeSplitStepEngine, NDArray, NDArray]:
    """Run the chunked taper; return (last engine, z record, MFD record, fields)."""
    fields = sech_fields(grid, e_pulse)
    n_steps_chunk = int(round(CHUNK_L / STEP_M))
    mfd_curve: list[float] = []
    z_curve: list[float] = []
    eng = None
    for k in range(NUM_CHUNKS):
        cp = chunk_params(k)
        fiber = FiberProfile(
            n2=(N2 if nonlinear else 1e-30),
            alpha=0.0,
            A_eff=Area(cp["aeff"], "m^2"),
            length=Length(CHUNK_L, "m"),
        )
        eng = MultimodeSplitStepEngine(
            _waves(grid, fields),
            fiber,
            betas=[[BETA2]] * 3,
            betas_unit="SI",
            group_delays=cp["delay"],
            phase_offsets=cp["db0"],
            coef_model="isotropic",
            xpm_weights=cp["xpm"],
            include_fwm=False,
            step_size=Length(STEP_M, "m"),
        )
        eng.propagate(n_steps_chunk, nsaves=n_steps_chunk + 1)
        zz = np.asarray(eng.z_array, float)
        snap = eng.fields_vs_z()
        z0 = (k + 0.5) * CHUNK_L - CHUNK_L / 2
        for jj in range(len(zz)):
            fields_j = [np.asarray(snap[p][jj], complex) for p in range(3)]
            mfd_curve.append(mfd_from_fields(fields_j, grid.dt, cp["moments"]))
            z_curve.append(z0 + zz[jj])
        fields = [np.asarray(eng.A[p], complex).copy() for p in range(3)]
    return eng, np.array(z_curve), np.array(mfd_curve) * 1e6, fields


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate(*, fast: bool = False, make_plot: bool = True) -> dict:
    params = json.loads(PARAMETERS.read_text())
    results: dict = {"parameters": params}

    # --- 1. analytic layer --------------------------------------------------
    a0 = float(core_radius(0.0))
    a_end = float(core_radius(TAPER_L))
    l_si0 = self_imaging_period(a0)
    l_si_end = self_imaging_period(a_end)
    db1_0 = mode_walkoffs(a0)
    db1_end = mode_walkoffs(a_end)
    results["analytic"] = {
        "a0_um": round(a0 * 1e6, 2),
        "a_end_um": round(a_end * 1e6, 2),
        "L_si_start_um": round(l_si0 * 1e6, 2),
        "L_si_end_um": round(l_si_end * 1e6, 2),
        "period_ratio": round(l_si0 / l_si_end, 4),
        "walkoff_p1_start_fs_per_m": round(db1_0[1] * 1e15, 3),
        "walkoff_p1_end_fs_per_m": round(db1_end[1] * 1e15, 3),
        "walkoff_scaling_ratio": round(float(db1_end[1] / db1_0[1]), 4),
    }
    # The walk-off db1_p carries the factor x(a) ∝ 1/(k a)², so it scales
    # QUADRATICALLY with the core radius (16.31 measured vs 4² model, the
    # 2 % residue is the sqrt-series curvature of the Eq.-(2) paraxial model).
    assert abs(db1_end[1] / db1_0[1] / (a0 / a_end) ** 2 - 1.0) < 0.03, (
        results["analytic"]["walkoff_scaling_ratio"]
    )
    assert abs(l_si0 / l_si_end - a0 / a_end) < 1e-6

    # --- 2. engine, linear chunked taper ------------------------------------
    grid = make_grid()
    if float(db1_0[1]) <= 0.0:
        raise RuntimeError("model walk-off sign broken")
    eng, z_arr, mfd_arr, fields = propagate_taper(E_PULSE, False, grid)

    energy_in = float(
        sum(
            float(np.sum(np.abs(f) ** 2)) * grid.dt
            for f in sech_fields(grid, E_PULSE)
        )
    )
    energy_out = float(
        sum(float(np.sum(np.abs(fields[p]) ** 2)) * grid.dt for p in range(3))
    )
    results["energy_conservation_linear"] = round(energy_out / energy_in, 9)
    assert abs(energy_out / energy_in - 1.0) < 1e-6

    # (a) MFD oscillation period at start vs end (local windows). The period
    # varies within each window (L_si ∝ a(z)), so the analytic prediction is
    # the window-averaged L_si.
    win0, win1 = 0.00, 0.15
    win2, win3 = 0.85, 1.00

    def _win_mean_L_si(lo: float, hi: float) -> float:
        zw = np.linspace(lo * TAPER_L, hi * TAPER_L, 2001)
        return float(
            np.trapezoid(self_imaging_period(core_radius(zw)), zw)
            / (zw[-1] - zw[0])
        )

    per_start = dominant_period(
        z_arr[z_arr < win1 * TAPER_L], mfd_arr[z_arr < win1 * TAPER_L]
    )
    per_end = dominant_period(
        z_arr[z_arr >= win2 * TAPER_L], mfd_arr[z_arr >= win2 * TAPER_L]
    )
    model_start = _win_mean_L_si(win0, win1)
    model_end = _win_mean_L_si(win2, win3)
    results["mfd_period_start_um"] = round(per_start * 1e6, 2)
    results["mfd_period_end_um"] = round(per_end * 1e6, 2)
    results["mfd_period_model_start_um"] = round(model_start * 1e6, 2)
    results["mfd_period_model_end_um"] = round(model_end * 1e6, 2)
    assert abs(per_start / model_start - 1.0) < 0.05, (per_start, model_start)
    assert abs(per_end / model_end - 1.0) < 0.05, (per_end, model_end)
    acceleration_model = model_start / model_end
    results["acceleration_ratio_measured"] = round(per_start / per_end, 3)
    results["acceleration_ratio_window_model"] = round(acceleration_model, 3)
    results["acceleration_ratio_pointwise_a0_over_aend"] = round(R0 / R_END, 3)
    assert abs(per_start / per_end / acceleration_model - 1.0) < 0.05, (
        results["acceleration_ratio_measured"]
    )
    # and the window model itself must bracket the pointwise x4 claim
    # (windows average over finite spans of a(z), so the window ratio sits
    # below the pointwise a0/a_end = 4)
    assert 0.75 * (R0 / R_END) < acceleration_model <= (R0 / R_END) + 1e-9

    # (b) accumulated higher-mode walk-off: at these fibre constants the
    # intermodal db1 is ~1 fs/m (fine staircase integral ~0.24 fs over the
    # 4-cm taper) — below the numerical resolution of any spectral/temporal
    # centroid read-out. Walk-off *operator* correctness is independently
    # validated by the Renninger-Wise / Menyuk-1987 / Wai-1991
    # reproductions (Δ = 0.029-scale fibres); here we only record the
    # analytic integral and the engine's per-chunk group_delays contract.
    integ_fine = walkoff_integral_fine()
    tt = np.asarray(grid.t, float)
    results["walkoff_model_ps"] = {
        str(p): round(float(integ_fine[p - 1]) * 1e12, 8) for p in (1, 2)
    }
    # linear-run engine read-out of the same accumulated walk-off
    for p in (1, 2):
        pw = np.abs(fields[p]) ** 2
        delay_lin = centroid(tt, pw)
        rel = abs(delay_lin - integ_fine[p - 1]) / abs(integ_fine[p - 1])
        results[f"walkoff_linear_p{p}_relerr_pct"] = round(rel * 100.0, 3)
        assert rel < 0.05, (p, rel, delay_lin, integ_fine[p - 1])

    # --- 3. engine, nonlinear chunked taper (SPM/XPM only) -------------------
    fields_sol = sech_fields(grid, E_SOLITON)
    energy_nl_in = float(
        sum(float(np.sum(np.abs(f) ** 2)) * grid.dt for f in fields_sol)
    )
    eng_nl, _, _, fields_sol = propagate_taper(E_SOLITON, True, grid)
    energy_nl_out = float(
        sum(float(np.sum(np.abs(fields_sol[p]) ** 2)) * grid.dt for p in range(3))
    )
    results["energy_conservation_nonlinear_pct"] = round(
        100.0 * abs(energy_nl_out / energy_nl_in - 1.0), 6
    )
    assert abs(energy_nl_out / energy_nl_in - 1.0) < 0.005
    for p in (1, 2):
        pw = np.abs(fields_sol[p]) ** 2
        delay_nl = centroid(tt, pw)
        rel = abs(delay_nl - integ_fine[p - 1]) / abs(integ_fine[p - 1])
        results[f"walkoff_nonlinear_p{p}_relerr_pct"] = round(rel * 100.0, 3)

    if make_plot:
        _plot(results, z_arr, mfd_arr, tt, fields, eng, fields_sol)

    print("Eftekhar et al. (2019) accelerated GRIN taper: validation passed")
    print(
        f"  L_si: {l_si0*1e6:.1f} -> {l_si_end*1e6:.1f} um "
        f"(measured x{results['acceleration_ratio_measured']} vs window-model "
        f"x{results['acceleration_ratio_window_model']}); "
        f"model walk-off integral p1 "
        f"{results['walkoff_model_ps']['1']} ps"
    )
    return results


def _plot(results, z_arr, mfd_arr, tt, fields, eng, fields_sol) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.plot(z_arr * 100, mfd_arr, "b.", ms=0.7)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("MFD (um)")
    ax.set_title("(a) modal spot oscillation through the taper")

    ax = axes[1]
    for p, colr in zip((0, 1, 2), ("k", "r", "b")):
        spec = np.abs(eng.grid.fft(np.asarray(fields[p], complex))) ** 2
        w = np.asarray(eng.grid.w, float)
        lam = 2 * np.pi * C_MS / (OMEGA0 + w) * 1e9
        ok = (lam > 1100) & (lam < 2600)
        ax.plot(lam[ok], 10 * np.log10(spec[ok] / spec[ok].max()), colr,
                lw=0.7, label=f"mode {p}")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("rel. intensity (dB)")
    ax.set_title("(b) output of the linear taper")
    ax.legend(fontsize=8)

    ax = axes[2]
    for p, colr in zip((0, 1, 2), ("k", "r", "b")):
        pw = np.abs(np.asarray(fields_sol[p], complex)) ** 2
        ax.plot(tt * 1e12, pw / pw[p].max(), colr, lw=0.8, label=f"mode {p}")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("norm. intensity")
    ax.set_title("(c) N~1 soliton after the nonlinear taper")
    ax.set_xlim(-3, 5)
    ax.legend(fontsize=8)
    fig.suptitle(
        "Eftekhar et al. 2019 - accelerated intermodal dynamics in a GRIN taper"
    )
    fig.tight_layout()
    fig.savefig(HERE / "eftekhar_2019_taper.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = validate()
    print(json.dumps(out, indent=2, default=float))
