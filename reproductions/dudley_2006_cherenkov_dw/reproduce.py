"""Reproduction: Cherenkov dispersive-wave emission (Dudley et al., 2006).

Reference
---------
J. M. Dudley, G. Genty, S. Coen, "Supercontinuum generation in photonic crystal
fiber," Rev. Mod. Phys. 78, 1135 (2006) — see the dispersive-wave / Cherenkov
radiation discussion (Sec. IV).  Theory: N. Akhmediev and M. Karlsson,
"Cherenkov radiation emitted by solitons in optical fibers," Phys. Rev. A 51,
2602 (1995).

Physics
-------
A soliton sheds a dispersive wave at the frequency where the linear phase
mismatch vanishes,

    β(ω_DW) − β(ω₀) − β₁(ω₀)(ω_DW − ω₀) = 0.

For a PCF described by β₂ < 0 and β₃ > 0 the lowest-order root is

    Ω_DW = −3 β₂ / β₃        (blue side),

and the full β(ω) root is obtained with ``dispersive_wave_roots``.

Validation
----------
1. ``dispersive_wave_roots`` agrees with the analytic `−3β₂/β₃` value.
2. The GNLSE spectrum develops a blue dispersive-wave peak at that wavelength
   (within `dw_tolerance`).

Usage
-----
    python reproductions/dudley_2006_cherenkov_dw/reproduce.py
"""

from __future__ import annotations

import json
from math import factorial
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from photonics_helper.base import C_MS, AngularFrequencyArray, Length, Time, Wavelength
from photonics_helper.fiber import PropagationConstant
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.phase_matching import (
    PropagationConstantAdaptor,
    dispersive_wave_roots,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"


def validate(params: dict | None = None, make_plot: bool = True) -> dict:
    params = params or json.loads(PARAMETERS.read_text())
    wl0 = params["central_wavelength_nm"] * 1e-9
    beta2 = params["beta2_ps2_per_m"]
    beta3 = params["beta3_ps3_per_m"]
    betas = np.array([beta2, beta3])

    grid = TemporalGrid(
        N=params["grid_N"], Tmax=Time(params["grid_Tmax_ps"] * 1e-12, "s")
    )
    env = Envelope.from_fwhm(
        "sech",
        peak_amplitude=np.sqrt(params["peak_power_W"]),
        fwhm=Time(params["pulse_fwhm_fs"], "fs"),
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wl0, "m"))
    w0 = float(pulse.central_frequency)

    # Analytic Cherenkov condition (beta2, beta3 only).
    omega_dw = -3 * beta2 / beta3 * 1e12  # rad/ps → rad/s
    lambda_analytic = 2 * np.pi * C_MS / (w0 + omega_dw) * 1e9

    # Full-beta root finder.
    wl_arr = np.linspace(300, 2500, 5000) * 1e-9
    om = 2 * np.pi * C_MS / wl_arr
    om_ps = (om - w0) * 1e-12
    beta_vals = np.zeros_like(om)
    for k, bk in enumerate(betas, start=2):
        beta_vals += bk * om_ps**k / factorial(k)
    pc = PropagationConstant(
        values=beta_vals, x_values=AngularFrequencyArray(np.asarray(om), "rad/s")
    )
    roots = dispersive_wave_roots(
        PropagationConstantAdaptor(pc),
        w0,
        wl_range=(Wavelength(300, "nm"), Wavelength(2500, "nm")),
        n_brackets=1000,
    )
    lambda_root = (
        float(roots.wavelengths.as_nm[0])
        if roots.wavelengths.as_m.size
        else float("nan")
    )
    assert abs(lambda_root - lambda_analytic) / lambda_analytic < 0.02, (
        f"root finder {lambda_root:.1f} nm vs analytic {lambda_analytic:.1f} nm"
    )

    # GNLSE spectrum and the blue DW peak.
    fiber = FiberProfile.from_gamma(
        gamma=params["gamma_per_Wm"],
        n2=2.7e-20,
        omega0=w0,
        length=Length(params["length_m"], "m"),
    )
    solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=betas, include_raman=False)
    solver.propagate(num_steps=3000)
    W = np.abs(grid.fft(solver.evolution[-1].envelope_field)) ** 2
    lam = (2 * np.pi * C_MS / (w0 + grid.w)) * 1e9
    order = np.argsort(lam)
    Ls, Ws = lam[order], W[order]

    peaks, _ = find_peaks(Ws, height=Ws.max() * 0.01)
    blue = [(Ls[i], Ws[i] / Ws.max()) for i in peaks if Ls[i] < wl0 * 1e9 - 20]
    assert blue, "no blue-side dispersive-wave peak found"
    lambda_sim = max(blue, key=lambda x: x[1])[0]  # strongest blue peak

    rel_err = abs(lambda_sim - lambda_analytic) / lambda_analytic
    assert rel_err < params["dw_tolerance"], (
        f"simulated DW {lambda_sim:.1f} nm vs analytic {lambda_analytic:.1f} nm "
        f"({rel_err * 100:.1f}%)"
    )

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Ls, Ws / Ws.max(), color="C0")
        ax.axvline(
            lambda_analytic,
            color="k",
            ls=":",
            label=f"analytic DW {lambda_analytic:.0f} nm",
        )
        ax.axvline(
            lambda_sim, color="C3", ls="--", label=f"simulated DW {lambda_sim:.0f} nm"
        )
        ax.set_xlim(450, 950)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized spectrum")
        ax.set_title("Cherenkov dispersive wave (Dudley et al. 2006 PCF)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = HERE / "cherenkov_dw.png"
        fig.savefig(out_png, dpi=150)
        print(f"wrote {out_png}")

    print("Cherenkov DW reproduction passed:")
    print(f"  analytic -3β₂/β₃ = {lambda_analytic:.1f} nm")
    print(f"  dispersive_wave_roots = {lambda_root:.1f} nm")
    print(f"  GNLSE blue DW peak = {lambda_sim:.1f} nm (rel err {rel_err * 100:.2f}%)")
    return {
        "lambda_analytic_nm": lambda_analytic,
        "lambda_root_nm": lambda_root,
        "lambda_sim_nm": lambda_sim,
        "rel_err": rel_err,
    }


if __name__ == "__main__":
    validate()
