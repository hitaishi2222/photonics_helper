"""Task 6.4b — render the L = 0.4 m |b2(z)| trajectory sweep (session 2).

Discriminating figure: measured |b2|/a at the END vs the MAX of the
trajectory per detuning, against the analytic band (|dbar| < c -> in-band
growth; outside -> oscillatory, b2_end ~ (2c/xi)|sin(xi L/2)| envelope).
Growth-classified points shaded; the analytic dbar(f) band edges marked.

Run from the repo root:
    python reproductions/planned/wright_2015_self_organized_instability/\
diagnostics/plot_gain_ztraj.py
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
JSONL = HERE / "gain_spectrum_ztraj.jsonl"

pts = defaultdict(list)
for line in JSONL.read_text().splitlines():
    r = json.loads(line)
    if r.get("type") == "point" and "b2_traj" in r:
        pts[r["n_ord"]].append(r)

spec = importlib.util.spec_from_file_location("rep", str(HERE.parent / "reproduce.py"))
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

c = (2 / 3) * (2 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF) * rep.P0_W

fig, axes = plt.subplots(1, len(pts), figsize=(6.4 * len(pts), 4.6), squeeze=False)
axes = axes[0]
for ax, n in zip(axes, sorted(pts)):
    rows = sorted(pts[n], key=lambda r: r["f_thz"])
    f = np.array([r["f_thz"] for r in rows])
    b2_end = np.array([r["b2_traj"][-1] for r in rows])
    b2_max = np.array([max(r["b2_traj"]) for r in rows])
    grow = np.array([r["growth_frac_test"]["classified_growth"] for r in rows])

    # analytic: dbar(f), band edges, oscillatory envelope at L
    fine = np.linspace(f.min(), f.max(), 600)
    sym = np.array(
        [
            rep.beta0(rep.PUMP_THZ + x)
            + rep.beta0(rep.PUMP_THZ - x)
            - 2 * rep.beta0(rep.PUMP_THZ)
            for x in fine
        ]
    )
    db = 0.5 * sym - n * rep.KAPPA - c
    L = rows[0]["L_m"]
    env = np.where(
        np.abs(db) < c,
        np.sinh(c * L),
        np.abs(2 * c / np.abs(db) * np.sin(np.abs(db) * L / 2)),
    )

    ax.plot(fine, env, "r-", lw=1.4, label="analytic |b2|/a envelope at L (2x2 model)")
    ax.axhline(np.sinh(c * L), color="r", ls=":", lw=0.8)
    ax.plot(f, b2_max, "o", ms=5, color="C0", label="measured max |b2(z)|")
    ax.plot(f, b2_end, "s", ms=5, color="C1", label="measured end |b2(L)|")
    ax.scatter(
        f[grow],
        b2_end[grow] + 0.06,
        marker="^",
        s=40,
        color="green",
        zorder=5,
        label="classified growth",
    )
    # analytic band edges: |dbar| = c
    sgn = np.sign(np.diff(db))
    for xi in np.where(np.diff(np.sign(np.abs(db) - c)) != 0)[0]:
        ax.axvline(fine[xi], color="k", ls="--", lw=0.8)
    ax.axhline(0.9 * np.sinh(np.min(np.abs(db)[np.abs(db) < c])) * 0, color="none")
    ax.set_xlabel("detuning (THz)")
    ax.set_ylabel("|b2|/a_sig")
    ax.set_title(f"ladder order {n} (L = {L} m, ztraj sweep)")
    ax.legend(fontsize=8)
fig.suptitle(
    "Wright 2015 STMI - L=0.4 m trajectory sweep: growth band vs "
    "analytic |dbar|<c band (corrected condition "
    "dbar = 0.5*sym - N*kappa - gamma*P0/3)"
)
fig.tight_layout()
out = HERE.parent / "wright_2015_gain_ztraj.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
