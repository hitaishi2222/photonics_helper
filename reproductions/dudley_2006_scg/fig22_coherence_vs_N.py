"""Fig. 22 — average coherence vs input soliton order N (Dudley 2006).

Paper reference
---------------
Dudley, Genty & Coen, RMP 78, 1135 (2006), Sec. VI.D.3.
Fig. 22: scatter of the average SC coherence against the input soliton
order N for pump wavelengths 790-900 nm, pulse durations 30-200 fs, and
peak powers 3-30 kW (10 cm PCF). Paper conclusion: N < 10 -> high
coherence; N > 30 -> low coherence.

N convention
------------
N = sqrt(L_D / L_NL) = sqrt(gamma P0 T0^2 / |beta_2|) with T0 the sech
parameter (FWHM = 1.763 T0), evaluated from the Table-I beta2 at 835 nm
(continuation of the Fig. 21 reproduction's dispersion caveat: the paper's
N uses the measured beta2(lambda); recorded, with beta2 fixed at the
835-nm value here").

Validation (structural, quick- and full-grid):
1. 0 <= g12 <= 1 per point.
2. The high-N ensemble's mean coherence is lower than the low-N ensemble's
   (the paper's coherence/N trend), evaluated on the same ladder points.
"""

from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np

from photonics_helper.noise import coherence_g12

from . import common

HERE = Path(__file__).resolve().parent

LENGTH_M = 10.0e-2
ZDW = common.PARAMS["zero_dispersion_wavelength_nm"]
WL0 = common.PARAMS["central_wavelength_nm"]


def _N(P0: float, dur_fs: float, wl_nm: float) -> float:
    """Soliton order from sech FWHM duration & power, beta2(Table I)."""
    T0 = dur_fs / 1.763 * 1e-15
    beta2 = abs(common.beta2_at_wavelength(wl_nm)) * 1e-24  # s^2/m
    L_D = T0**2 / beta2
    L_NL = 1.0 / (common.GAMMA * P0)
    return float(np.sqrt(L_D / L_NL))


def run_point(P0: float, dur_fs: float, wl_nm: float, n_real: int, n_steps: int):
    pulse = common.build_pulse(
        P0,
        T0_fs=dur_fs / 1.763,
        N_points=1024,
        Tmax_ps=float(common.PARAMS["figures"]["fig03_basic_scg"]["Tmax_ps"]),
        wl_nm=wl_nm,
    )
    fiber = common.build_fiber(pulse, length_m=LENGTH_M, raman=True)
    betas = np.asarray(common.BETAS, dtype=float)
    spectra = []
    for seed in range(n_real):
        solver = common.make_solver(
            pulse,
            fiber,
            betas,
            raman=True,
            shock=True,
            tau_shock=common.SHOCK_FS * 1e-15,
        )
        solver.propagate(
            num_steps=n_steps,
            nsaves=2,
            raman_noise=True,
            noise_seed=7000 + seed,
        )
        spectra.append(np.asarray(pulse.grid.fft(solver.evolution[-1].envelope_field)))
    g12 = coherence_g12(np.asarray(spectra, dtype=complex))
    return float(np.mean(g12))


def validate(fast: bool = True, make_plot: bool = True) -> dict:
    # ladder spanning low-N and high-N codimensions (paper: 790-900 nm,
    # 30-200 fs, 3-30 kW). Fast: 6 points; full: 12 points.
    if fast:
        points = [
            (4_000.0, 50.0, 835.0),  # N ~ 1
            (4_000.0, 100.0, 835.0),  # N ~ 2
            (10_000.0, 150.0, 835.0),  # N ~ 6 (the fig19/20b deck)
            (30_000.0, 200.0, 835.0),  # N ~ 20
            (30_000.0, 200.0, 880.0),
        ]
        n_real, n_steps = 4, 200
    else:
        points = [
            (4_000.0, 50.0, 800.0),
            (4_000.0, 50.0, 835.0),
            (4_000.0, 100.0, 800.0),
            (10_000.0, 150.0, 800.0),
            (10_000.0, 150.0, 835.0),
            (10_000.0, 150.0, 880.0),
            (30_000.0, 150.0, 835.0),
            (30_000.0, 200.0, 800.0),
            (30_000.0, 200.0, 835.0),
            (30_000.0, 200.0, 880.0),
        ]
        n_real, n_steps = 8, 400

    rows = []
    for P0, dur, wl in points:
        N = _N(P0, dur, wl)
        g = run_point(P0, dur, wl, n_real, n_steps)
        rows.append({"N": N, "g12_avg": g, "P0_W": P0, "dur_fs": dur, "wl_nm": wl})
    assert all(0.0 <= r["g12_avg"] <= 1.0 for r in rows)

    low = np.mean([r["g12_avg"] for r in rows if r["N"] < 10])
    high = np.mean([r["g12_avg"] for r in rows if r["N"] >= 15])
    results = {"rows": rows, "g12_lowN": float(low), "g12_highN": float(high)}
    # the paper's core claim, structurally: high-N ensembles lose coherence
    # relative to low-N ensembles (recorded with direction assert; the
    # absolute thresholds N<10 high / N>30 low require the measured beta2
    # per pump and the full ladder).
    assert results["g12_lowN"] >= results["g12_highN"], (
        f"low-N coherence {low:.3f} < high-N {high:.3f}"
    )

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        Ns = [r["N"] for r in rows]
        gs = [r["g12_avg"] for r in rows]
        ax.scatter(
            [r["N"] for r in rows],
            gs,
            c=[r["P0_W"] for r in rows],
            cmap="viridis",
            s=60,
            edgecolor="k",
        )
        ax.set_xlabel("input soliton order N")
        ax.set_ylabel("average |g12|")
        for r, g in zip(rows, gs):
            ax.annotate(
                f"{int(r['wl_nm'])}nm/{int(r['dur_fs'])}fs",
                (r["N"], g),
                fontsize=7,
                textcoords="offset points",
                xytext=(4, 3),
            )
        ax.axvspan(10, 30, color="0.9", zorder=0)
        ax.text(20, 0.05, "paper: transition band", ha="center", fontsize=8)
        ax.set_title("Dudley Fig. 22-style: coherence vs soliton order")
        fig.colorbar(
            plt.cm.ScalarMappable(cmap="viridis"), ax=ax, label="peak power (W)"
        )
        fig.tight_layout()
        fig.savefig(HERE / "fig22_coherence_vs_N.png", dpi=150)
        plt.close(fig)
        results["plot"] = "fig22_coherence_vs_N.png"
    return results


if __name__ == "__main__":
    import sys

    print(validate(fast="--fast" in sys.argv))
