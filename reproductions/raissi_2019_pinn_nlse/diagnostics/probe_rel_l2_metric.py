"""ISSUES.md #6 — which rel-L2 definition reaches the paper's 1.97e-3?

The reproduction reports rel-L2 on the **complex** field,
``||h_pred - h_exact|| / ||h_exact||`` = 6.0-6.1e-3, against the paper's
1.97e-3. But the paper's Fig. 2 top panel shows the **magnitude**
|h| = sqrt(u^2 + v^2), and the text reads "the top panel of figure 2 shows the
magnitude of the predicted spatio-temporal solution |h| ... The resulting
prediction error is validated against the test data for this problem, and is
measured at 1.97e-3 in the relative L2-norm".

If the reported error is taken on the magnitude rather than the complex field,
phase errors cancel and the number is legitimately smaller — which would
explain a ~3x gap with no physics or optimiser defect.

This probe recomputes the metric under every plausible convention from the
saved checkpoint, so the reproduction can state which one it means:

    complex    ||h_pred - h|| / ||h||              (what we report today)
    magnitude  || |h_pred| - |h| || / || |h| ||    (the Fig. 2 top panel)
    real       ||u_pred - u|| / ||u||
    imag       ||v_pred - v|| / ||v||
    intensity  || |h_pred|^2 - |h|^2 || / || |h|^2 ||

Run:  python diagnostics/probe_rel_l2_metric.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> dict:
    import reproduce as R

    ckpt_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "pinn_checkpoint.pt"
    )
    params = json.loads(R.PARAMETERS.read_text())
    torch = R._torch()
    dtype = torch.float64

    grid, z_arr, H = R._run_engine(params)
    n_big = H.shape[1]
    H = H[:, n_big // 4 : 3 * n_big // 4]

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    net = ck["net"].to("cpu")
    net.eval()

    em = params["engine_mapping"]
    x_min, x_max = params["pde"]["x_domain"]
    t_max = float(em["z_span_m"])
    t_c = t_h = 0.5 * t_max
    x_c, x_h = 0.5 * (x_max + x_min), 0.5 * (x_max - x_min)
    t_all = torch.tensor(z_arr, dtype=dtype)
    x_all = torch.tensor(
        np.arange(H.shape[1]) * (2 * x_h) / H.shape[1] - x_h, dtype=dtype
    )

    def h_of(t, x):
        tn = (t - t_c) / t_h
        xn = (x - x_c) / x_h
        out = net(torch.cat([tn, xn], dim=1))
        return out[:, 0:1], out[:, 1:2]

    trained = {
        "metrics": {
            "device": "cpu",
            "loss_history": ck.get("hist", []),
            "loss_final": float(ck.get("loss_final", ck.get("loss", 0.0))),
        },
        "h_of": h_of,
        "t_all": t_all,
        "x_all": x_all,
    }
    ev, h_pred = R.evaluate_pinn(trained, H, z_arr)

    mag_p, mag_e = np.abs(h_pred), np.abs(H)
    int_p, int_e = mag_p**2, mag_e**2
    out = {
        "checkpoint": str(ckpt_path),
        "checkpoint_stage": ck.get("stage"),
        "checkpoint_loss": float(ck.get("loss_final", ck.get("loss", 0.0))),
        "paper_rel_l2": params["reference"]["paper_rel_l2"],
        "metrics": {
            "complex": ev["rel_l2_full"],
            "magnitude": rel(mag_p, mag_e),
            "real": ev["rel_l2_real"],
            "imag": ev["rel_l2_imag"],
            "intensity": rel(int_p, int_e),
        },
    }
    print(
        f"checkpoint stage={out['checkpoint_stage']} loss={out['checkpoint_loss']:.3e}"
    )
    print(f"paper rel-L2 = {out['paper_rel_l2']:.3e}")
    for k, v in out["metrics"].items():
        ratio = v / out["paper_rel_l2"]
        flag = "  <== matches paper" if 0.5 <= ratio <= 2.0 else ""
        print(f"  {k:10s} {v:.4e}   ({ratio:5.2f}x paper){flag}", flush=True)
    (HERE / "probe_rel_l2_metric.json").write_text(json.dumps(out, indent=2))
    print("wrote probe_rel_l2_metric.json")
    return out


if __name__ == "__main__":
    main()
