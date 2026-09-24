"""Re-run the Raissi-2019 PINN validation from pinn_checkpoint.pt (no retrain).

Rebuilds the minimal `trained` dict (net + h_of closure + loss history),
re-runs the PINN evaluation on the engine grid, and regenerates the
figures via reproduce._make_plots. Used both as a CLI tool and by the
local-only test suite (`tests/test_reproductions.py`, marked slow):

    python reproductions/raissi_2019_pinn_nlse/replot_fig2.py [--no-plot]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def evaluate(make_plot: bool = True) -> dict:
    """Rebuild the trained net from the checkpoint and re-run the checks."""
    import numpy as np
    import torch

    import reproduce as R

    params = json.loads(R.PARAMETERS.read_text())
    torch = R._torch()
    device = "cpu"
    dtype = torch.float64

    # --- engine data (same pipeline as validate()) ---
    grid, z_arr, H = R._run_engine(params)
    n_big = H.shape[1]
    H = H[:, n_big // 4: 3 * n_big // 4]

    # --- rebuild the net from the checkpoint ---
    ck = torch.load(str(HERE / "pinn_checkpoint.pt"), map_location="cpu",
                    weights_only=False)
    net = ck["net"].to(device)
    net.eval()

    em = params["engine_mapping"]
    x_min, x_max = params["pde"]["x_domain"]
    t_max = float(em["z_span_m"])
    t_c, t_h = 0.5 * t_max, 0.5 * t_max
    x_c, x_h = 0.5 * (x_max + x_min), 0.5 * (x_max - x_min)
    t_all = torch.tensor(z_arr, dtype=dtype)
    x_all = torch.tensor(
        np.arange(H.shape[1]) * (2 * x_h * 1.0) / H.shape[1] - x_h,
        dtype=dtype)

    def h_of(t, x):
        tn = (t - t_c) / t_h
        xn = (x - x_c) / x_h
        out = net(torch.cat([tn, xn], dim=1))
        return out[:, 0:1], out[:, 1:2]

    trained = {"metrics": {"device": device,
                           "loss_history": ck.get("hist", []),
                           "loss_final": float(ck["loss_final"])},
               "h_of": h_of, "t_all": t_all, "x_all": x_all}

    # --- eval + plots (same as validate()) ---
    ev, h_pred = R.evaluate_pinn(trained, H, z_arr)
    trained["eval"] = ev
    print(f"rel_l2_full = {ev['rel_l2_full']:.4e}  (loss {ck['loss_final']:.3e})")

    results = {"derived": {
        "note": "units: t in the PINN is z in meters (L_s = 1 m)",
        "N2_breather_period_m": np.pi / 2.0,
    }}
    results["data_validation"] = R.validate_data(params)
    results["pinn"] = {
        "rel_l2_full": ev["rel_l2_full"],
        "rel_l2_real": ev["rel_l2_real"],
        "rel_l2_imag": ev["rel_l2_imag"],
        "cuts_rel_l2": {k: c["rel_l2"] for k, c in ev["cuts"].items()},
        "loss_final": float(ck["loss_final"]),
        "paper_rel_l2": params["reference"]["paper_rel_l2"],
        "accept_rel_l2": params["reference"]["accept_rel_l2"],
        "resumed_from": "pinn_checkpoint.pt (pre-eval-final)",
    }

    if make_plot:
        from scipy.stats import qmc

        p = params["pinn"]
        x0 = qmc.LatinHypercube(d=1, seed=p["seed"]).random(p["N0"])[:, 0]
        x0 = -5.0 + 10.0 * x0
        tb = qmc.LatinHypercube(d=1, seed=p["seed"] + 1).random(p["Nb"])[:, 0]
        tb = t_max * tb
        results["figures"] = R._make_plots(H, z_arr, h_pred, x0, tb,
                                           trained, HERE)
        print("regenerated:", results["figures"])

    assert ev["rel_l2_full"] < params["reference"]["accept_rel_l2"], \
        results["pinn"]
    return results


if __name__ == "__main__":
    evaluate(make_plot="--no-plot" not in sys.argv)
