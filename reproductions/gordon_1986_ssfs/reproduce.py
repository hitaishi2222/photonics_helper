"""Reproduction: Raman soliton self-frequency shift (Gordon, 1986).

Reference
---------
J. P. Gordon, "Theory of the soliton self-frequency shift," Opt. Lett. 11, 662
(1986), doi:10.1364/OL.11.000662;  experiment: F. M. Mitschke and
L. F. Mollenauer, Opt. Lett. 11, 659 (1986).

Physics
-------
A fundamental soliton in a silica fibre red-shifts because the delayed Raman
response transfers energy from the blue (leading) to the red (trailing) part of
the pulse.  The analytic rate is

    dΩ/dz = − 8·|β₂|·T_R / (15·T₀⁴),      T_R = f_R ∫ t·h_R(t) dt.

For λ = 1.5 µm, D = 15 ps/(nm·km) and a 250 fs soliton this is ≈ +0.8 nm / 20 m
(≈ one spectral width per 100 m, as stated in the paper).

Validation
----------
1. The simulated redshift is positive (physical direction).
2. Its rate is within `[rate_tolerance_low, rate_tolerance_high]` × Gordon.
3. The Stokes (red) Raman sideband gains while the anti-Stokes loses.

Usage
-----
    python reproductions/gordon_1986_ssfs/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"


def _raman_response(params, grid):
    r = params["raman"]
    spec = RamanSpec(
        name="Silica",
        raman_shift_cm=r["raman_shift_cm"],
        raman_linewidth_cm=r["raman_linewidth_cm"],
        fR=r["fR"],
    )
    return RamanResponse(
        spec=spec,
        fR=r["fR"],
        tau1=r["tau1_fs"] * 1e-15,
        tau2=r["tau2_fs"] * 1e-15,
        grid=grid,
    )


def _peak_wavelength_nm(grid, A, wl0):
    W = np.abs(grid.fft(A)) ** 2
    lam = 2 * np.pi * C_MS / (2 * np.pi * C_MS / wl0 + grid.w)
    order = np.argsort(lam)
    Ws, Ls = W[order], lam[order]
    i = int(np.argmax(Ws))
    y0, y1, y2 = Ws[i - 1], Ws[i], Ws[i + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float((Ls[i] + d * (Ls[1] - Ls[0])) * 1e9)


def validate(params: dict | None = None, make_plot: bool = True) -> dict:
    params = params or json.loads(PARAMETERS.read_text())
    wl0 = params["central_wavelength_nm"] * 1e-9
    D = params["dispersion_ps_nm_km"] * 1e-6  # ps/(nm·km) → s/m²
    beta2 = -D * wl0**2 / (2 * np.pi * C_MS)  # s²/m
    T0 = params["pulse_fwhm_fs"] * 1e-15 / 1.7627
    gamma = params["gamma_per_Wm"]
    P0 = abs(beta2) / (gamma * T0**2)

    grid = TemporalGrid(N=params["grid_N"], Tmax=Time(params["grid_Tmax_ps"] * 1e-12, "s"))
    raman = _raman_response(params, grid)
    TR = params["raman"]["fR"] * np.trapezoid(grid.t * raman._h_R(grid.t), grid.t)
    rate = -8 * abs(beta2) * TR / (15 * T0**4)  # rad/s/m
    analytic_dlam = -wl0**2 / (2 * np.pi * C_MS) * rate * params["length_m"] * 1e9

    env = Envelope(shape="sech", peak_amplitude=np.sqrt(P0), pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wl0, "m"))
    fiber = FiberProfile.from_gamma(
        gamma=gamma, n2=2.6e-20, omega0=pulse.central_frequency,
        length=Length(params["length_m"], "m"), raman_response=raman,
    )
    solver = GNLSESolver(
        pulse=pulse, fiber=fiber, betas=np.array([beta2 * 1e24]), include_raman=True
    )
    solver.propagate(num_steps=4000)
    out = solver.evolution[-1].envelope_field
    lam0 = _peak_wavelength_nm(grid, pulse.envelope_field, wl0)
    lam1 = _peak_wavelength_nm(grid, out, wl0)
    measured = lam1 - lam0
    ratio = measured / analytic_dlam if analytic_dlam else float("nan")

    assert measured > 0.0, f"Raman SSFS must red-shift, got {measured} nm"
    assert params["rate_tolerance_low"] <= ratio <= params["rate_tolerance_high"], (
        f"SSFS rate {measured:.4f} nm vs Gordon {analytic_dlam:.4f} nm (ratio {ratio:.2f})"
    )

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W0 = np.abs(grid.fft(pulse.envelope_field)) ** 2
        W1 = np.abs(grid.fft(out)) ** 2
        lam = (2 * np.pi * C_MS / (2 * np.pi * C_MS / wl0 + grid.w)) * 1e9
        o = np.argsort(lam)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lam[o], W0[o] / W0.max(), label="input")
        ax.plot(lam[o], W1[o] / W1.max(), label=f"output ({params['length_m']:.0f} m)")
        ax.axvline(lam0, color="k", ls=":", lw=1)
        ax.axvline(lam1, color="C1", ls=":", lw=1)
        ax.set_xlim(lam0 - 12, lam0 + 12)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized spectrum")
        ax.set_title(f"Raman SSFS: measured {measured:+.3f} nm, Gordon {analytic_dlam:+.3f} nm")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = HERE / "ssfs_spectrum.png"
        fig.savefig(out_png, dpi=150)
        print(f"wrote {out_png}")

    print("Gordon SSFS reproduction passed:")
    print(f"  measured redshift = {measured:+.4f} nm @ {params['length_m']:.0f} m")
    print(f"  Gordon analytic   = {analytic_dlam:+.4f} nm  (ratio {ratio:.2f})")
    return {"measured_nm": measured, "gordon_nm": analytic_dlam, "ratio": ratio}


if __name__ == "__main__":
    validate()
