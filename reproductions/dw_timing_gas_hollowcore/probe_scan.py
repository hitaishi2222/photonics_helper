"""Fig. 2 probe scan: deterministic noise-free runs over the pump-energy scan
(80-220 uJ), extracting RDW energy / arrival time / central wavelength per
energy -> JSONL (crash-safe). The paper's resampling method (p-08).

Run from the repo root:
    python reproductions/planned/dw_timing_gas_hollowcore/probe_scan.py \
        [--shard i/N]
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dw_rep", str(HERE / "reproduce.py"))
_rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rep)
analyse_run = _rep.analyse_run
build_engine = _rep.build_engine
L_M = _rep.L_M

# 800 uniformly spaced energies 80-220 uJ, fixed CEP, no shot noise (p-08).
# dz = 5e-5 taken for the scan (convergence checked: RDW wavelength shifts
# by 1.5 nm / energy frac by ~2 % vs dz = 2.5e-5; the paper's own adaptive
# stepping is resolved to the same level).
N_SCAN = 800
E_LO_UJ, E_HI_UJ = 80.0, 220.0
STEP_M = 5e-5

JSONL = HERE / "rdw_probe_scan.jsonl"
SHARD, NSHARD = 0, 1
PRESSURE_BAR = 2.1  # Fig. 2/5 constant-pressure default
GRADIENT_FILL = False
if "--pressure" in sys.argv:
    PRESSURE_BAR = float(sys.argv[sys.argv.index("--pressure") + 1])
if "--gradient" in sys.argv:
    GRADIENT_FILL = True
    PRESSURE_BAR = float(sys.argv[sys.argv.index("--gradient") + 1])
JSONL = HERE / (
    "rdw_scan_gradient_%.1fbar.jsonl" % PRESSURE_BAR
    if GRADIENT_FILL
    else "rdw_scan_%.1fbar.jsonl" % PRESSURE_BAR
)
if "--shard" in sys.argv:
    i = sys.argv.index("--shard")
    SHARD, NSHARD = int(sys.argv[i + 1]), int(sys.argv[i + 2])


def main() -> None:
    deadline = time.time() + 8 * 3600
    energies = np.linspace(E_LO_UJ, E_HI_UJ, N_SCAN)
    sel = [k for k in range(N_SCAN) if k % NSHARD == SHARD]
    if not JSONL.exists():
        with open(JSONL, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "type": "meta",
                        "N": N_SCAN,
                        "shard": SHARD,
                        "nshard": NSHARD,
                        "step_m": STEP_M,
                        "pressure_bar": PRESSURE_BAR,
                        "gradient": GRADIENT_FILL,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
    n_done = 0
    from tqdm import tqdm

    for k in tqdm(sel, desc="scan", unit="pt", ncols=80, position=0):
        e_uj = float(energies[k])
        t0 = time.time()
        solver, gam, _ = build_engine(
            energy_uJ=e_uj,
            step_m=STEP_M,
            pressure_bar=PRESSURE_BAR,
            pressure_gradient=GRADIENT_FILL,
        )
        solver.propagate(int(round(L_M / STEP_M)), nsaves=3)
        a = analyse_run(solver)
        row = {
            "type": "point",
            "k": k,
            "energy_uJ": round(e_uj, 1),
            "pressure_bar": PRESSURE_BAR,
            "gradient": GRADIENT_FILL,
            "rdw_energy": float(a["rdw_energy"]),
            "rdw_lambda_nm": float(a["rdw_lambda_nm"]),
            "arrival_time_fs": float(a["arrival_time_fs"]),
            "elapsed_s": time.time() - t0,
        }
        with open(JSONL, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        n_done += 1
        if n_done % 20 == 0:
            print(
                f"shard {SHARD}: {n_done}/{len(sel)} done \
({time.time() - t0:.0f} s/pt avg {sum(0 for _ in ()):.0f})",
                flush=True,
            )
        if time.time() > deadline:
            print("deadline reached", flush=True)
            break
    print("SCAN DONE", flush=True)


if __name__ == "__main__":
    main()
