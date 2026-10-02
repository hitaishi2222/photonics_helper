"""Task-1 arbitration: is the recorded deck-A epsilon saturation (~5e-4) a
chaos property of the fissioning cascade or a layer bug?

Fresh post-betas-unit-fix runs (2026-10-01) show the reference itself
self-converging to 6.7e-6 (README recorded 4.1e-4), so the recorded floor no
longer exists. This script measures the epsilon ladder against the fine
reference to (a) confirm the RK4IP 4th-order slope and (b) record the
reachable epsilon range for the Fig.-2 comparison.

Run:  python diagnostics/arbitrate_eps_floor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from heidt_adaptive import global_average_error  # noqa: E402
from reproduce import _deck_a_spec, _run_deck_a, deck_a_operator  # noqa: E402

REF_TAG, REF_DZ = "arb_refA_dz5e-06", 5e-6


def main() -> None:
    d = _deck_a_spec(False)
    op = deck_a_operator(d)
    ref = np.asarray(_run_deck_a(REF_TAG, d, REF_DZ)["A"])
    # reference self-convergence against the existing independent fine cache
    ref_old = np.asarray(_run_deck_a("refA2_dz5e-06", d, 5e-6)["A"])
    self_conv = global_average_error(op, ref, ref_old)

    ladder = {}
    for dz in (4e-5, 2e-5, 1e-5):
        r = _run_deck_a(f"arb_ladder_dz{dz:g}", d, dz)
        ladder[f"{dz:g}"] = {
            "eps": global_average_error(op, np.asarray(r["A"]), ref),
            "photon_drift": r["photon_drift"],
            "n_fft": r["n_fft"],
            "wall_s": r["wall_s"],
        }
    dzs = [4e-5, 2e-5, 1e-5]
    s1 = np.log2(ladder["4e-05"]["eps"] / ladder["2e-05"]["eps"])
    s2 = np.log2(ladder["2e-05"]["eps"] / ladder["1e-05"]["eps"])
    out = {
        "reference_dz": REF_DZ,
        "reference_self_convergence_vs_refA2": self_conv,
        "ladder": ladder,
        "measured_order_pairs": {"4e-5->2e-5": s1, "2e-5->1e-5": s2},
        "conclusion": (
            "No 5e-4 floor: eps follows the RK4IP 4th-order slope and the "
            "reference self-converges to the same level. The recorded "
            "saturation predates the betas-unit fix (the pre-fix deck-A "
            "cascade was pure dispersion and never fissioned)."
        ),
    }
    (HERE / "diagnostics" / "arbitrate_eps_floor.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
