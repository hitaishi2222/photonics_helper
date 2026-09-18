#!/usr/bin/env python3
"""Waveguide mode import from an external FEM solver (``WaveguideMode``).

The library does **not** solve eigenmodes — Lumerical MODE, COMSOL Wave
Optics, etc. own that. What it does provide is a documented path to ingest
their exported ``n_eff(λ)`` table and turn it into the library's dispersion
objects.

This example:

- generates a synthetic FEM-style ``n_eff(λ)`` table (so the script is
  self-contained) and writes it in both supported formats:
  ``mode.csv`` (``wavelength_um, neff[, ng]``) and ``mode.npz``;
- loads it back with :meth:`WaveguideMode.from_csv` / ``from_npz``;
- builds a :class:`PropagationConstant` and checks ``beta2`` against the
  analytic second derivative ``d²β/dω²``;
- builds a :class:`Dispersion` ``D(λ)`` and plots ``n_eff(λ)`` and ``D(λ)``.

The resulting ``PropagationConstant`` / ``Dispersion`` are the same objects
used by ``phase_matching`` and ``gnlse``; ``chi2`` consumes phase mismatches
via ``delta_k_shg`` (see ``reproductions/shg_textbook/``).

Run with: python examples/29_waveguide_mode_import.py
"""

# The sys.path bootstrap below intentionally precedes the library imports.
# ruff: noqa: E402

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper import WaveguideMode, Wavelength

# ── 1. A synthetic "FEM export" ─────────────────────────────────────────
# Smooth, monotonic n_eff(λ) for a nitride-like waveguide. We keep an
# analytic form so beta2 has a closed-form cross-check.
N0, SLOPE, CURV, LAM0_UM = 2.42, -0.10, 0.05, 1.55


def neff_analytic(wl_um):
    """Ground-truth effective index used to fabricate the export."""
    d = np.asarray(wl_um, dtype=float) - LAM0_UM
    return N0 + SLOPE * d + CURV * d**2


def beta2_analytic(wl_nm, n_fine=40001):
    """d²β/dω² from a fine-grid central difference of the analytic β(ω)."""
    wl_um = np.linspace(1.40, 1.70, n_fine)
    omega = 2 * np.pi * 2.99792458e8 / (wl_um * 1e-6)
    beta = neff_analytic(wl_um) * omega / 2.99792458e8
    d2 = np.gradient(np.gradient(beta, omega), omega)
    return float(np.interp(wl_nm * 1e-3, wl_um, d2))


workdir = Path(tempfile.mkdtemp(prefix="waveguide_mode_"))
wl_um = np.linspace(1.40, 1.70, 61)
neff = neff_analytic(wl_um)

csv_path = workdir / "mode.csv"
with csv_path.open("w") as fh:
    fh.write("wavelength_um, neff, ng\n")
    for x, y in zip(wl_um, neff):
        fh.write(f"{x:.6f}, {y:.10f}, {N0:.10f}\n")

npz_path = workdir / "mode.npz"
np.savez(
    npz_path,
    wavelength_um=wl_um,
    neff=neff,
    ng=np.full_like(wl_um, N0),
    central_wavelength_nm=1550.0,
)

print(f"synthetic export written to {workdir}")
print(f"  {csv_path.name}: 61 rows, header 'wavelength_um, neff, ng'")
print(f"  {npz_path.name}: wavelength_um, neff, ng, central_wavelength_nm")

# ── 2. Load it back ─────────────────────────────────────────────────────
mode_csv = WaveguideMode.from_csv(csv_path)
mode_npz = WaveguideMode.from_npz(npz_path)
print(f"\nCSV import -> {mode_csv!r}")
print(f"NPZ import -> {mode_npz!r}")
np.testing.assert_allclose(mode_csv.neff, mode_npz.neff)

# ── 3. n_eff -> PropagationConstant -> beta2 ────────────────────────────
pc = mode_csv.to_propagation_constant()
lam_check = Wavelength(1550, "nm")

beta2_lib = pc.beta2(lam_check)
beta2_ref = beta2_analytic(lam_check.as_nm)
rel_err = abs(beta2_lib - beta2_ref) / abs(beta2_ref)
print("\nPropagationConstant from the imported table:")
print(f"  n_eff(1550 nm)      = {mode_csv.neff_at(lam_check):.6f}")
print(f"  beta2(1550 nm)      = {beta2_lib:.6e} s^2/m")
print(f"  analytic beta2      = {beta2_ref:.6e} s^2/m  (rel err {rel_err:.2e})")

# ── 4. n_eff -> Dispersion D(lambda) ────────────────────────────────────
disp = mode_csv.to_dispersion()
beta2_from_D = disp.get_beta2(lam_check.as_nm)
print("\nDispersion D(λ) from the same table:")
print(f"  D(1550 nm)          = {disp.fn(lam_check) * 1e6:.3f} ps/(nm·km)")
print(f"  beta2 via D         = {beta2_from_D:.6e} s^2/m")

# ── 5. Plot n_eff(λ) and D(λ) ───────────────────────────────────────────
wl_plot = np.linspace(1.40, 1.70, 301)
neff_plot = np.array([mode_csv.neff_at(Wavelength(x, "um")) for x in wl_plot])
D_plot = np.array([disp.fn(Wavelength(x, "um")) for x in wl_plot]) * 1e6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.plot(wl_plot, neff_plot, "C0-", lw=2, label="spline (imported)")
ax1.plot(wl_um, neff, "k.", ms=4, label="FEM samples")
ax1.set_xlabel("Wavelength (µm)")
ax1.set_ylabel(r"$n_\mathrm{eff}$")
ax1.set_title("Imported effective index")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(wl_plot, D_plot, "C3-", lw=2)
ax2.axhline(0.0, color="k", lw=0.8, ls=":")
ax2.set_xlabel("Wavelength (µm)")
ax2.set_ylabel("D (ps/(nm·km))")
ax2.set_title(r"Dispersion from $D = -\lambda/c \cdot d^2n_\mathrm{eff}/d\lambda^2$")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).resolve().parent / "images" / "29_waveguide_mode_import.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
