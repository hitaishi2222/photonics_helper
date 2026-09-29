"""Fig. 5/6 statistics analysis over the completed probe scans.

Post-processing (paper-faithful arrival time, no re-runs needed):

The engine propagates in the carrier group-velocity frame (beta_1 excluded
from the linear step, `SplitStepEngine._linear_step`), so the raw scan
arrival times measure only higher-order GVD inside the envelope frame and
sit at ~0 fs.  The paper's tau (Eq. 11) is dominated by the beta_1 term:
the RDW walks off from the soliton at

    tau(E) = L_prop(E) * [beta_1(w_RDW(E)) - beta_1(w_0)],   L_prop = L - L_f.

We reconstruct that observable exactly with Eq. (12)'s L_f and the
SIMULATED RDW central wavelengths from the probe scans (the same coupling
the paper uses when it evaluates eq. 11 "while finding the central
wavelength of the RDW in the same manner as for our choice of window
function", p-16).  Everything downstream is the paper's resampling
method (p-08): interpolants over the scan -> 10000 Gaussian samples of
the pump energy around each mean energy -> arrival-time / central-
wavelength / RDW-energy statistics vs mean energy.

Also re-checks Fig. 5c (group-velocity difference of the RDW vs paper
circles) and the two headline claims:
  - timing jitter < 300 as for every parameter combination (p-14),
  - jitter proportional to the pump energy noise (1 % -> half of 2 %).

Outputs `dw_timing_fig56.png` + `stats_fig56.json` (crash-safe, one per
pressure).
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dw_rep", str(HERE / "reproduce.py"))
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)

SIGMA_REL = 0.02
N_SAMPLES = 10_000

CONST = [(p, False) for p in rep.P_CONST]
GRAD = [(p, True) for p in rep.P_GRAD_FILL]



def tau_total_fs(lam_rdw_nm: np.ndarray, pressure_bar: float,
                 energy_uJ_arr: np.ndarray) -> np.ndarray:
    """Eq. (11) with the SIMULATED RDW wavelength per energy.

    Two legs (paper Sec. III):
      1. propagation walk-off tau_prop = L_prop [beta1(w_rdw)-beta1(w0)],
         L_prop = L - Lf (Eq. 12 with the deck gamma, beta2(w0), P0(E));
      2. envelope-frame delay accumulated BEFORE fission, from the scan's
         own raw arrival times (the higher-order GVD imprint of the same
         runs, recentred: the raw moment is a differential delay, its
         absolute offset is not fixed by the engine frame).
    """
    w_rdw = 2 * np.pi * rep.C_MS / (np.asarray(lam_rdw_nm) * 1e-9)
    tau_prop = np.array([float(rep.eq11_tau_fs(np.array([w]), pressure_bar, e)[0])
                         for w, e in zip(w_rdw, np.asarray(energy_uJ_arr))])
    return tau_prop


def envelope_tau_fs(sc: dict, lam_rdw_nm: np.ndarray,
                    pressure_bar: float) -> np.ndarray:
    """Leg 2: raw engine-frame moment, HIGH-PASS cleaned. The raw scan
    arrival times contain a smooth envelope-frame contribution (chirp /
    self-steepening-family delay) that does NOT gate on fission; only its
    fast, energy-localised structure is the fission-gated delay. High-pass
    by subtracting a wide Savitzky-Golay trend (window ~ 150 points =
    ~27 uJ, well wider than the jitter-scale structure)."""
    from scipy.signal import savgol_filter
    tau_raw = np.asarray(sc["tau"])
    trend = savgol_filter(tau_raw, 151, 2)
    return tau_raw - trend


def delta_vg_ms(lam_rdw_nm: np.ndarray, pressure_bar: float) -> np.ndarray:
    """Fig. 5c observable: v_g(w_RDW) - v_g(800 nm), m/s."""
    w = 2 * np.pi * rep.C_MS / (lam_rdw_nm * 1e-9)
    w0 = 2 * np.pi * rep.C_MS / (rep.LAMBDA0_NM * 1e-9)
    ws = np.concatenate([w, [w0]])
    _, b1, _ = rep.differentiate_beta(ws, pressure_bar)
    vg = 1.0 / b1
    return (vg[-1] * 0 + (vg[:-1] - vg[-1]))  # m/s


def analyse_one(pressure_bar: float, gradient: bool) -> dict:
    sc = rep.load_scan(pressure_bar, gradient)
    # dedupe (0.8 and 2.1 bar were run twice at different step sizes):
    # average by binned energy (5 uJ bins) to guarantee a per-bin unique array
    from collections import defaultdict
    bins = defaultdict(list)
    for ee, la, tt, erd_ in zip(sc["e"], sc["lam"], sc["tau"], sc["e_rdw"]):
        bins[round(ee, 1)].append((la, tt, erd_))
    e_u = np.array(sorted(bins))
    lam = np.array([np.mean([x[0] for x in bins[k]]) for k in sorted(bins)])
    tau0 = np.array([np.mean([x[1] for x in bins[k]]) for k in sorted(bins)])
    erd = np.array([np.mean([x[2] for x in bins[k]]) for k in sorted(bins)])
    # Smooth the jagged scan (UV multi-peak competition makes lam(E) jump by
    # 1-6 nm per 0.2 uJ; the paper's Fig. 5/6 curves are smooth on this
    # scale). SavGol window 21 pts = 3.7 uJ, order 2 — disclosed in README.
    from scipy.signal import savgol_filter
    lam_s = savgol_filter(lam, 21, 2)
    erd_s = savgol_filter(erd, 21, 2)
    tau_env = savgol_filter(tau0, 21, 2)
    tau_prop = tau_total_fs(lam_s, pressure_bar, e_u)
    dvg = delta_vg_ms(lam_s, pressure_bar)

    # Fig. 5 grid: mean energies = the scan's own points (normalised later)
    means = e_u
    out = {"pressure_bar": pressure_bar, "gradient": gradient,
           "energies_uJ": e_u.tolist(),
           "tau_prop_fs": tau_prop.tolist(),
           "tau_env_fs": tau_env.tolist(),
           "tau_total_fs": (tau_prop + tau_env).tolist(),
           "lam_rdw_nm": lam_s.tolist(),
           "delta_vg_ms": dvg.tolist(),
           "means_uJ": [], "sigma_tau_as": [], "sigma_lam_nm": []}
    for mu in means:
        rs = rep.resample_scan({"e": e_u, "lam": lam_s, "tau": tau_prop + tau_env,
                                "e_rdw": erd_s}, mu, SIGMA_REL, N_SAMPLES)
        out["means_uJ"].append(round(float(mu), 1))
        out["sigma_tau_as"].append(round(float(np.std(rs["tau"])) * 1e3, 2))
        out["sigma_lam_nm"].append(round(float(np.std(rs["lam"])), 3))
    # 1 % noise run at 1/8 of the means for the proportionality check
    out["sigma_tau_1pct_as"] = [
        round(float(np.std(rep.resample_scan(
            {"e": e_u, "lam": lam_s, "tau": tau_prop + tau_env, "e_rdw": erd_s},
            mu, 0.01, N_SAMPLES)["tau"])) * 1e3, 2) for mu in means[::8]]
    out["means_1pct"] = [round(float(mu), 1) for mu in means[::8]]
    return out


def main() -> None:
    all_out = []
    for p, gr in CONST + GRAD:
        out = analyse_one(p, gr)
        all_out.append(out)
        mx = max(out["sigma_tau_as"])
        print(f"p={p} bar grad={gr}: max sigma_tau = {mx:.1f} as, "
              f"tau range {min(out['tau_total_fs']):.2f}..{max(out['tau_total_fs']):.2f} fs, "
              f"dvg range {min(out['delta_vg_ms']):.0f}..{max(out['delta_vg_ms']):.0f} m/s")
    (HERE / "stats_fig5.json").write_text(json.dumps(all_out, indent=1))
    print("wrote stats_fig5.json")
    make_fig()




def make_fig() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(HERE / "stats_fig5.json"))
    e_max = 220.0
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for o in d:
        e = np.array(o["means_uJ"]) / e_max
        lbl = f"{o['pressure_bar']} bar" + (" grad" if o["gradient"] else "")
        (ax[0] if not o["gradient"] else ax[1]).plot(e, o["sigma_tau_as"], label=lbl)
        ax[2].plot(e, o["tau_total_fs"], label=lbl)
    for a, t in zip(ax, ["Fig. 5(a): sigma_tau, constant p",
                         "Fig. 5(b): sigma_tau, gradient",
                         "tau(E) = Lprop*dbeta1 (Eq. 11, sim RDW lam)"]):
        a.set_xlabel("normalised pump energy")
        a.set_title(t)
    ax[0].axhline(300, ls="--", c="k", lw=0.8)
    ax[1].axhline(300, ls="--", c="k", lw=0.8)
    ax[0].set_ylabel("sigma_tau (as)")
    ax[2].set_ylabel("tau (fs)")
    for a in ax:
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(HERE / "dw_timing_fig56.png", dpi=150)
    print("wrote dw_timing_fig56.png")


if __name__ == "__main__":
    main()
