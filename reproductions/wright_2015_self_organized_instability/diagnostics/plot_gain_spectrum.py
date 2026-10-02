"""Task 6.4 — render the measured-vs-analytic gain spectrum figure.

Consumes the JSONL produced by probe_gain_spectrum.py; makes one panel per
ladder order: measured growth exponent g_meas(f) (markers, from coherent
asinh/cosh projections) vs the analytic reduced-2x2 gain g_ana(f) (line),
with dbar(f) as a light reference. No spectral peak picking anywhere.

Run from the repo root AFTER the sweep completes:
    python reproductions/planned/wright_2015_self_organized_instability/\
diagnostics/plot_gain_spectrum.py
"""

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
JSONL = HERE / "gain_spectrum_points.jsonl"

pts = defaultdict(list)
for line in JSONL.read_text().splitlines():
    r = json.loads(line)
    if r.get("type") == "point" and "g_ana" in r and "error" not in r:
        pts[r["n_ord"]].append(r)

spec = importlib.util.spec_from_file_location("rep", str(HERE.parent / "reproduce.py"))
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

fig, axes = plt.subplots(1, len(pts), figsize=(6.0 * len(pts), 4.5), squeeze=False)
axes = axes[0]
for ax, n in zip(axes, sorted(pts)):
    rows = sorted(pts[n], key=lambda r: r["f_thz"])
    f = np.array([r["f_thz"] for r in rows])
    g_ana = np.array([r["g_ana"] for r in rows])
    g_m = np.array([r["g_meas_sinh"] for r in rows])
    g_mc = np.array([r["g_meas_cosh"] for r in rows])
    root = rep.stmi_shift_thz_corrected(n)
    fine = np.linspace(f.min(), f.max(), 400)
    c = (2 / 3) * (2 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF) * rep.P0_W
    sym = np.array(
        [
            rep.beta0(rep.PUMP_THZ + x)
            + rep.beta0(rep.PUMP_THZ - x)
            - 2 * rep.beta0(rep.PUMP_THZ)
            for x in fine
        ]
    )
    db = 0.5 * sym - n * rep.KAPPA - c
    ax.plot(
        fine,
        np.sqrt(np.clip(c * c - db * db, 0, None)),
        "r-",
        lw=1.4,
        label="analytic g(Ω) (2×2 model)",
    )
    ax.plot(fine, np.abs(db), "r:", lw=0.8, label="|Δβ̄(Ω)| (reference)")
    ax.plot(f, g_m, "o", ms=5, color="C0", label="measured (sinh arm)")
    ax.plot(f, g_mc, "s", ms=4, color="C1", mfc="none", label="measured (cosh arm)")
    ax.axvline(root, color="k", ls="--", lw=0.8, label="corrected root")
    ax.set_xlabel("detuning (THz)")
    ax.set_ylabel("parametric gain g (1/m)")
    ax.set_title(f"ladder order {n} (L = {rows[0]['L_m']} m)")
    ax.legend(fontsize=8)
fig.suptitle(
    "Wright 2015 STMI — engine measured gain vs analytic "
    "reduced-2×2 model (deterministic coherent probe)"
)
fig.tight_layout()
out = HERE.parent / "wright_2015_gain_spectrum.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
