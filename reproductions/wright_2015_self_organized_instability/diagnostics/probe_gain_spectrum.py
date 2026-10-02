"""Task 6.4 — measured-vs-analytic gain spectrum (no log noise floor).

For each detuning f (THz) on a dense sweep around the corrected STMI root,
propagate the CW pump + coherent conjugate-tone probe pair through the
engine and read the growth exponent directly from the seeded sibling
channel:

    g_meas(f) = asinh(|b2(L)| / a_sig) / L        (exact for dbar = d(f))

compared against the analytic reduced-2x2 gain

    g_ana(f)  = sqrt(c^2 - dbar(f)^2),
    dbar(f)   = 0.5 * sym(f) - N*kappa - gamma*P0/3,   c = (2/3)*gamma*P0.

No spectral peak picking: everything is a coherent time-domain projection,
so a mostly-zero engine floor does not corrupt the readout.

Output: JSONL appended per frequency point (crash-safe incremental write)
plus a final JSON summary. Results consumed by plot_gain_spectrum.py.
Run from the repo root:
    python reproductions/planned/wright_2015_self_organized_instability/\
diagnostics/probe_gain_spectrum.py [--orders 1 2] [--n-grid 16384]
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT_JSONL = HERE / "gain_spectrum_points.jsonl"
OUT_SUM = HERE / "gain_spectrum_summary.json"

spec = importlib.util.spec_from_file_location("rep", str(HERE.parent / "reproduce.py"))
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def probe_point(n_ord: int, f_thz: float, L: float, n_grid: int) -> dict:
    """One frequency point via rep.probe_gain (two engine runs)."""
    rep.N_GRID = n_grid  # probe_gain reads the module global
    res = rep.probe_gain(n_ord, f_thz=f_thz, L=L)
    a1, a2 = res["cols_a1"]  # (|b1|/a_sig, |b2|/a_sig), seed a1 only
    b1b, b2b = res["cols_a2"]  # seed a2 only (conjugate symmetry probe)
    a_sig = rep.PROBE_RMS * float(np.sqrt(rep.P0_W))
    # recovery exponent from the sibling channel (sinh growth) and own
    # channel (cosh growth); average the two independent estimates.
    g_sinh = float(np.arcsinh(abs(a2)) / L)
    g_sinh_b = float(np.arcsinh(abs(b1b) / 1.0) / L)
    g_cosh = float(np.arccosh(max(abs(a1), 1.0 + 1e-12)) / L)
    c = (2.0 / 3.0) * (2.0 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF) * rep.P0_W
    om = 2.0 * np.pi * f_thz * 1e12  # rad/s
    sym = (
        rep.beta0(rep.PUMP_THZ + f_thz)
        + rep.beta0(rep.PUMP_THZ - f_thz)
        - 2.0 * rep.beta0(rep.PUMP_THZ)
    )
    dbar = 0.5 * sym - n_ord * rep.KAPPA - c  # 0.5 sym - N k - gP0/3
    g_ana = float(np.sqrt(max(c * c - dbar * dbar, 0.0)))
    return {
        "n_ord": n_ord,
        "f_thz": round(f_thz, 5),
        "L_m": L,
        "n_grid": n_grid,
        "dbar_rad_per_m": round(dbar, 4),
        "g_ana": round(g_ana, 4),
        "g_meas_sinh": round(g_sinh, 4),
        "g_meas_sinh_sym": round(g_sinh_b, 4),
        "g_meas_cosh": round(g_cosh, 4),
        "energy_drift_pct": res["energy_drift_pct"],
        "ts": time.time(),
    }


def sweep(n_ord: int, L: float, n_grid: int, n_band: int, n_out: int) -> int:
    """Dense sweep for one ladder order; returns points appended."""
    f_root = rep.stmi_shift_thz_corrected(n_ord)
    # band half-width (THz): where dbar(f) = c (g -> 0); ddbar/df sampled
    # numerically over +-0.5 THz around the root.
    f_lo, f_hi = f_root - 0.5, f_root + 0.5
    s_lo = (
        rep.beta0(rep.PUMP_THZ + f_lo)
        + rep.beta0(rep.PUMP_THZ - f_lo)
        - 2 * rep.beta0(rep.PUMP_THZ)
    )
    s_hi = (
        rep.beta0(rep.PUMP_THZ + f_hi)
        + rep.beta0(rep.PUMP_THZ - f_hi)
        - 2 * rep.beta0(rep.PUMP_THZ)
    )
    ddbar = abs((s_hi - s_lo) / 2.0)
    half_w = (
        (2 / 3)
        * (2 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF)
        * rep.P0_W
        / (abs(ddbar) + 1e-6)
    )
    freqs = list(np.linspace(f_root - half_w, f_root + half_w, n_band))
    edge_lo, edge_hi = f_root - 3 * half_w, f_root + 3 * half_w
    freqs += list(np.linspace(edge_lo, f_root - half_w, n_out)[1:-1:2])
    freqs += list(np.linspace(f_root + half_w, edge_hi, n_out)[1:-1:2])
    freqs = sorted(set(round(float(f), 5) for f in freqs))
    done = 0
    with tqdm(freqs, desc=f"order {n_ord}", file=sys.stdout) as bar:
        for f in bar:
            rec = {"type": "start", "n_ord": n_ord, "f_thz": round(float(f), 5)}
            with open(OUT_JSONL, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            t0 = time.time()
            try:
                r = probe_point(n_ord, float(f), L, n_grid)
                r["elapsed_s"] = round(time.time() - t0, 1)
            except Exception as e:  # don't lose the sweep to one point
                r = {
                    "type": "error",
                    "n_ord": n_ord,
                    "f_thz": round(float(f), 5),
                    "error": repr(e),
                }
            r["type"] = "point"
            with open(OUT_JSONL, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
            done += 1
            if r.get("g_ana") is not None:
                bar.set_postfix(g=r["g_meas_sinh"], ga=r["g_ana"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--L", type=float, default=0.10)
    ap.add_argument("--n-grid", type=int, default=16384)
    ap.add_argument("--n-band", type=int, default=11)
    ap.add_argument("--n-out", type=int, default=16)
    args = ap.parse_args()

    rec = {
        "type": "meta",
        "orders": args.orders,
        "L_m": args.L,
        "n_grid": args.n_grid,
        "n_band": args.n_band,
        "n_out": args.n_out,
        "ts": time.time(),
    }
    with open(OUT_JSONL, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    total = 0
    for n in args.orders:
        total += sweep(n, args.L, args.n_grid, args.n_band, args.n_out)
    with open(OUT_SUM, "w") as fh:
        json.dump(
            {"total_points": total, "orders": args.orders, "done": time.time()},
            fh,
            indent=2,
        )
    tqdm.write(f"SWEEP DONE: {total} points -> {OUT_JSONL.name}")


if __name__ == "__main__":
    main()
