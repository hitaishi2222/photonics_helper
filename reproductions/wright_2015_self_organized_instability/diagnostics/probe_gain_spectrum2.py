"""Task 6.4b — measured-vs-analytic gain spectrum with a GROWTH test.

Session-1 sweep (probe_gain_spectrum.py) found the recovered exponent
~2.8-3.2/m across the whole +-0.03 THz sweep, where the analytic 2x2
mismatch band is only +-0.016 THz: a single-length asinh recovery cannot
distinguish true parametric growth |b2| ~ sinh(gL) from OFF-RESONANT
oscillatory coupling |b2| ~ sin(|dbar| L/2)·(c/|dbar|)·2 (both give
|b2|/a ~ 0.28 at detunings of a few thousandths of a THz).

Fix: propagate the same coherent probe and read |b2(z)| along the saved
trajectory. Growth -> monotone increase; oscillation -> bounded/oscillating.

Consumed by plot_gain_spectrum.py (gives the final measured-vs-analytic
figure with the discriminator overlay). JSONL output crash-safe.
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
OUT_JSONL = HERE / "gain_spectrum_ztraj.jsonl"

spec = importlib.util.spec_from_file_location("rep", str(HERE.parent / "reproduce.py"))
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def probe_ztraj(
    n_ord: int, f_thz: float, L: float, n_grid: int, nsaves: int = 21
) -> dict:
    """Coherent conjugate-tone probe with a |b2(z)| trajectory readout.

    Returns z and |b2|/a_sig, plus a growth classification:
      growth_frac = fraction of the last half of the trajectory where the
      monotone fit slope >= 0 AND |b2(end)|/|b2(start of second half)| > 1.5
    """
    rep.N_GRID = n_grid
    om = 2.0 * np.pi * f_thz * 1e12
    half_steps = int(round(L / rep.STEP_M_LADDER))
    dz = L / half_steps
    grid = rep.TemporalGrid(N=n_grid, Tmax=rep.Time(rep.WINDOW_PS * 1e-12, "s"))
    tt = np.asarray(grid.t.as_s if hasattr(grid.t, "as_s") else grid.t)
    tone = np.exp(1j * om * tt)
    a_sig = rep.PROBE_RMS * float(np.sqrt(rep.P0_W))

    def one_run(seed: tuple[float, float]):
        waves = []
        for ch in range(rep.N_MODES):
            ww = rep.Wave(
                grid=grid,
                envelope=rep.Envelope(
                    shape="gaussian",
                    peak_amplitude=float(np.sqrt(rep.P0_W)),
                    pulse_width=rep.Time(rep.WINDOW_PS * 1e-12 / 4.0, "s"),
                ),
                central_wavelength=rep.Wavelength(rep.LAMBDA0_NM, "nm"),
            )
            if ch == 0:
                field = np.full(n_grid, float(np.sqrt(rep.P0_W)), complex)
            elif ch == 1:
                field = a_sig * seed[0] * tone
            else:
                field = a_sig * seed[1] * np.conj(tone)
            waves.append(ww.with_field(np.asarray(field, complex)))
        fiber = rep.FiberProfile(
            n2=rep.N2,
            alpha=0.0,
            A_eff=rep.Area(rep.A_EFF, "m^2"),
            length=rep.Length(L, "m"),
        )
        eng = rep.MultimodeSplitStepEngine(
            waves,
            fiber,
            betas=[rep.BETAS_FLAT] * rep.N_MODES,
            betas_unit="s^k/m",
            phase_offsets=[0.0, float(-n_ord * rep.KAPPA), float(-n_ord * rep.KAPPA)],
            oam_l=[0, 0, 0],
            coef_model="lp_degenerate",
            include_fwm=True,
            fwm_pump_depletion=True,
            step_size=rep.Length(dz, "m"),
        )
        # nsaves along the propagation for the trajectory readout
        eng.propagate(half_steps, nsaves=nsaves)
        zs = np.linspace(0.0, L, len(eng.evolution))
        b2 = []
        for st in eng.evolution:
            f2 = (
                st[2]
                if not hasattr(st[2], "_pulse_train_field")
                else st[2]._pulse_train_field
            )
            f2 = np.asarray(f2, complex)
            a2 = np.abs(f2 @ np.conj(np.conj(tone))) / n_grid
            b2.append(a2 / a_sig)
        return zs, np.asarray(b2)

    zs, b2_a = one_run((1.0, 0.0))  # seed the a1 tone (sibling = b2)
    zs2, b1_b = one_run((0.0, 1.0))  # seed the a2 tone (sibling = b1)
    # growth classifier on the b2 branch (last half of the trajectory)
    half_i = len(zs) // 2
    v = b2_a[half_i:] + 1e-15
    slope_ok = float(np.corrcoef(zs[half_i:], b2_a[half_i:])[0, 1]) > 0.9
    ratio = float(b2_a[-1] / max(b2_a[half_i], 1e-15))
    return {
        "type": "point",
        "n_ord": n_ord,
        "f_thz": round(float(f_thz), 5),
        "L_m": L,
        "n_grid": n_grid,
        "z": [round(float(z), 5) for z in zs],
        "b2_traj": [round(float(v), 6) for v in b2_a],
        "b1_traj_sym": [round(float(v), 6) for v in b1_b],
        "growth_frac_test": {
            "slope_corr": slope_ok,
            "end_ratio": ratio,
            "classified_growth": bool(slope_ok and ratio > 1.5),
        },
        "ts": time.time(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--L", type=float, default=0.10)
    ap.add_argument("--n-grid", type=int, default=16384)
    ap.add_argument("--n-band", type=int, default=11)
    ap.add_argument("--n-out", type=int, default=16)
    ap.add_argument("--nsaves", type=int, default=21)
    args = ap.parse_args()

    with open(OUT_JSONL, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "meta",
                    "orders": args.orders,
                    "L_m": args.L,
                    "n_grid": args.n_grid,
                    "nsaves": args.nsaves,
                }
            )
            + "\n"
        )
    c = (2 / 3) * (2 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF) * rep.P0_W
    total = 0
    for n in args.orders:
        f_root = rep.stmi_shift_thz_corrected(n)
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
        ddbar = abs((s_hi - s_lo) / 2.0)  # rad/m per THz (numerical)
        half_w = c / (abs(ddbar) + 1e-6)  # analytic band half-width, THz
        freqs = list(np.linspace(f_root - half_w, f_root + half_w, args.n_band))
        freqs += list(
            np.linspace(f_root - 3 * half_w, f_root - half_w, args.n_out)[1:-1:2]
        )
        freqs += list(
            np.linspace(f_root + half_w, f_root + 3 * half_w, args.n_out)[1:-1:2]
        )
        freqs = sorted(set(round(float(f), 5) for f in freqs))
        with tqdm(freqs, desc=f"order {n}", file=sys.stdout) as bar:
            for f in bar:
                t0 = time.time()
                try:
                    r = probe_ztraj(n, float(f), args.L, args.n_grid, args.nsaves)
                    r["elapsed_s"] = round(time.time() - t0, 1)
                except Exception as e:
                    r = {
                        "type": "error",
                        "n_ord": n,
                        "f_thz": round(float(f), 5),
                        "error": repr(e),
                    }
                with open(OUT_JSONL, "a") as fh:
                    fh.write(json.dumps(r, default=float) + "\n")
                    fh.flush()
                total += 1
                g = r.get("growth_frac_test", {})
                bar.set_postfix(g=g.get("classified_growth"), r=g.get("end_ratio"))
    tqdm.write(f"ZTRAJ SWEEP DONE: {total} points -> {OUT_JSONL.name}")


if __name__ == "__main__":
    main()
