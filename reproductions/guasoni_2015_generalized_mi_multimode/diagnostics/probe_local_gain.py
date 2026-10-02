"""ISSUES.md #8 — is the flat Eq.-(12) readout a SATURATION artifact?

The reproduction measures the Eq.-(12) amplification as a single end-to-end
ratio,

    A_hat(nu) = log( S_out(nu) / S_in(nu) ) / (2 L),

between `evolution[0]` and `evolution[-1]`, and finds it essentially flat
across |nu| <= 1.15 (band/edge 0.233/0.230) where the paper's Fig. 4/5 shows
banded structure (band ~0.64). The header already records that the log-ratio
reaches ~e^23, i.e. the sidebands have grown to the pump scale.

Hypothesis: once the growing bands are **depleted/saturated**, every frequency
inside the MI band has been pumped up to the same level, so an *end-to-end*
log-ratio can no longer show band structure even though the underlying gain
still is banded. The test is to measure the **local** gain over short segments
before saturation, where the exponential growth rate is still unsaturated, and
to check whether the banded structure appears there.

It also sweeps the noise seed level: the header records the readout *falling*
as the seed grows (1e-7 -> 0.233, 1e-5 -> 0.188, 1e-3 -> 0.141), which is what
seed-driven saturation looks like.

Run:  python diagnostics/probe_local_gain.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import reproduce as R  # noqa: E402

L_TOTAL = 5.0
NSAVES = 26  # -> 25 local segments of 0.2 m
AVG = 240


def local_gain_contrast(noise_w: float, seeds: int = 3) -> dict:
    """Per-segment local gain contrast (band max / edge mean) vs z."""
    R.NOISE_POWER_W = noise_w
    per_seed: list[list[np.ndarray]] = []
    z_seg: list[float] = []
    for seed in range(seeds):
        per_seed.append([])
        eng, grid = R.make_engine(L_TOTAL, seed)
        t0 = time.time()
        eng.propagate(1, nsaves=NSAVES, show_progress=False)
        specs = [
            np.abs(grid.fft(np.array([w._pulse_train_field for w in ev]))) ** 2
            for ev in eng.evolution
        ]
        z = np.asarray(eng.z_array, dtype=float)
        print(
            f"    seed {seed}: {len(specs)} snapshots, z 0..{z[-1]:.2f} m "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        dz = float(z[1] - z[0])
        ker = np.hanning(AVG) / np.hanning(AVG).sum()
        for k in range(len(specs) - 1):
            g = np.log(np.maximum(specs[k + 1], 1e-300) / np.maximum(specs[k], 1e-300))
            g = g / (2.0 * dz) * R.L_NL1  # paper's normalized unit
            g = np.array([np.convolve(g[m], ker, mode="same") for m in range(4)])
            per_seed[-1].append(g)
            if seed == 0:
                z_seg.append(float(0.5 * (z[k] + z[k + 1])))
    # average over seeds, keeping the segment axis
    mean = np.mean(np.stack(per_seed, axis=0), axis=0)  # (n_seg, 4, nfreq)

    nu = (grid.w / (2.0 * np.pi)) * R.T_NL1
    in_band = (np.abs(nu) > 0.05) & (np.abs(nu) < 0.65)
    edge = (np.abs(nu) > 0.7) & (np.abs(nu) < 1.15)

    rows = []
    for g, zc in zip(mean, z_seg, strict=True):
        b = float(g[1][in_band].max())
        e = float(g[1][edge].mean())
        rows.append(
            {
                "z_m": zc,
                "band_max_2x": b,
                "edge_mean_2x": e,
                "contrast": float(b / max(e, 1e-12)),
            }
        )
    return {"noise_power_W": noise_w, "segments": rows}


def main() -> dict:
    out = {"L_total_m": L_TOTAL, "n_segments": NSAVES - 1, "runs": {}}
    for noise_w in (1e-7, 1e-11):
        tag = f"{noise_w:.0e}"
        print(f"[{tag} W/sample]", flush=True)
        out["runs"][tag] = local_gain_contrast(noise_w)
        rows = out["runs"][tag]["segments"]
        for r in rows[:: max(1, len(rows) // 6)]:
            print(
                f"    z={r['z_m']:5.2f} m  band {r['band_max_2x']:8.3f}  "
                f"edge {r['edge_mean_2x']:8.3f}  contrast {r['contrast']:6.3f}",
                flush=True,
            )
        best = max(rows, key=lambda r: r["contrast"])
        print(
            f"  -> peak local contrast {best['contrast']:.3f} at z={best['z_m']:.2f} m "
            f"(end-to-end readout in the main deck is ~0.233 flat)",
            flush=True,
        )
    Path(HERE / "probe_local_gain.json").write_text(json.dumps(out, indent=2))
    print("wrote probe_local_gain.json", flush=True)
    return out


if __name__ == "__main__":
    main()
