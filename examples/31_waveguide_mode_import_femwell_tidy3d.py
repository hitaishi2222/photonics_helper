#!/usr/bin/env python3
"""Import *real* waveguide mode data from femwell and Tidy3D.

``examples/generate_waveguide_mode_data.py`` solved a canonical silicon strip
(500 nm × 220 nm on SiO₂) with two independent eigenmode solvers and wrote
``n_eff(λ)`` tables in the ``WaveguideMode`` conventions. This example is the
*consumer* half: it loads those exports (CSV and NPZ), cross-checks the two
solvers, and turns the imported index into the library's dispersion objects.

It runs without femwell or Tidy3D installed — only the committed data files in
``examples/data/`` are needed.

Run with: python examples/31_waveguide_mode_import_femwell_tidy3d.py
"""

# The sys.path bootstrap below intentionally precedes the library imports.

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper import WaveguideMode, Wavelength
from photonics_helper.phase_matching import (
    PropagationConstantAdaptor,
    fwm_delta_beta_degenerate,
)

DATA = Path(__file__).resolve().parent / "data"
META = json.loads((DATA / "si_strip_meta.json").read_text())

# ── 1. Load the two solver exports (CSV and NPZ) ────────────────────────
modes = {
    "femwell": {
        "csv": WaveguideMode.from_csv(DATA / "si_strip_femwell.csv"),
        "npz": WaveguideMode.from_npz(DATA / "si_strip_femwell.npz"),
    },
    "tidy3d": {
        "csv": WaveguideMode.from_csv(DATA / "si_strip_tidy3d.csv"),
        "npz": WaveguideMode.from_npz(DATA / "si_strip_tidy3d.npz"),
    },
}

print("Imported FEM mode exports")
for solver, variants in modes.items():
    csv_mode, npz_mode = variants["csv"], variants["npz"]
    # CSV and NPZ must describe the same table.
    np.testing.assert_allclose(csv_mode.neff, npz_mode.neff)
    np.testing.assert_allclose(csv_mode.ng, npz_mode.ng)
    n1550 = csv_mode.neff_at(Wavelength(1550, "nm"))
    ng1550 = float(np.interp(1.55, csv_mode.wavelengths.as_um, csv_mode.ng))
    print(
        f"  {solver:8s} {csv_mode!r}\n"
        f"           n_eff(1550 nm) = {n1550:.6f},  ng(1550 nm) = {ng1550:.4f}"
    )

# ── 2. Cross-check the two independent solvers ──────────────────────────
wl_um = modes["femwell"]["csv"].wavelengths.as_um
neff_fw = modes["femwell"]["csv"].neff
neff_td = modes["tidy3d"]["csv"].neff
rel_dev = np.abs(neff_fw - neff_td) / neff_td
max_dev = float(rel_dev.max())
print(f"\nSolver agreement: max |Δn_eff|/n_eff = {max_dev:.2%} (over 1.5–1.6 µm)")
assert max_dev < 0.02, "the two solvers disagree by more than 2%"

# ── 3. n_eff -> PropagationConstant / Dispersion ────────────────────────
pc = modes["femwell"]["csv"].to_propagation_constant()
disp = modes["femwell"]["csv"].to_dispersion()
lam = Wavelength(1550, "nm")

beta2_pc = pc.beta2(lam)
beta2_D = disp.get_beta2(lam.as_nm)
D_1550 = disp.fn(lam) * 1e6
print("\nDerived dispersion (femwell import):")
print(f"  beta2(1550 nm) = {beta2_pc:.4e} s^2/m")
print(f"  beta2 via D(λ) = {beta2_D:.4e} s^2/m  (rel err {abs(beta2_pc - beta2_D) / abs(beta2_pc):.2e})")
print(f"  D(1550 nm)     = {D_1550:.2f} ps/(nm·km)")

# ── 4. The import plugs straight into phase_matching ────────────────────
# Degenerate FWM mismatch for a pump at 1550 nm and a nearby signal.
beta_fn = PropagationConstantAdaptor(pc)
omega_p = lam.to_omega().as_rad_s
omega_s = Wavelength(1560, "nm").to_omega().as_rad_s
delta_beta = fwm_delta_beta_degenerate(beta_fn, omega_p, omega_s)
print(f"\nphase_matching tie-in: Δβ(1550; 1560 nm) = {delta_beta:.3f} 1/m")

# ── 5. Plot ─────────────────────────────────────────────────────────────
wl_plot = np.linspace(wl_um.min(), wl_um.max(), 200)
neff_fw_plot = np.array([modes["femwell"]["csv"].neff_at(Wavelength(x, "um")) for x in wl_plot])
neff_td_plot = np.array([modes["tidy3d"]["csv"].neff_at(Wavelength(x, "um")) for x in wl_plot])
D_plot = np.array([disp.fn(Wavelength(x, "um")) for x in wl_plot]) * 1e6

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].plot(wl_plot, neff_fw_plot, "C0-", lw=2, label="femwell")
axes[0].plot(wl_plot, neff_td_plot, "C1--", lw=2, label="Tidy3D")
axes[0].plot(wl_um, neff_fw, "C0.", ms=3, alpha=0.6)
axes[0].plot(wl_um, neff_td, "C1.", ms=3, alpha=0.6)
axes[0].set_xlabel("Wavelength (µm)")
axes[0].set_ylabel(r"$n_\mathrm{eff}$")
axes[0].set_title("Imported effective index")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(wl_um, rel_dev * 100, "C3o-", lw=1.5, ms=4)
axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
axes[1].set_xlabel("Wavelength (µm)")
axes[1].set_ylabel(r"$|n_\mathrm{eff}^{femwell} - n_\mathrm{eff}^{tidy3d}|/n_\mathrm{eff}$ (%)")
axes[1].set_title(f"Solver deviation (max {max_dev:.2%})")
axes[1].grid(True, alpha=0.3)

axes[2].plot(wl_plot, D_plot, "C2-", lw=2)
axes[2].axhline(0.0, color="k", lw=0.8, ls=":")
axes[2].set_xlabel("Wavelength (µm)")
axes[2].set_ylabel("D (ps/(nm·km))")
axes[2].set_title("Dispersion from the imported index")
axes[2].grid(True, alpha=0.3)

geom = META["geometry"]
fig.suptitle(
    f"Si strip {geom['core_width_um'] * 1000:.0f} × {geom['core_height_um'] * 1000:.0f} nm "
    f"on SiO₂ — real femwell / Tidy3D exports imported with WaveguideMode",
    fontsize=11,
)
fig.tight_layout()
out = Path(__file__).resolve().parent / "images" / "31_waveguide_mode_import_femwell_tidy3d.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
