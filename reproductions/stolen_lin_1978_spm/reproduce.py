"""Reproduction: self-phase modulation in silica fibers (Stolen & Lin, 1978).

Reference
---------
R. H. Stolen and C. Lin, "Self-phase-modulation in silica optical fibers,"
Phys. Rev. A 17, 1448 (1978), doi:10.1103/PhysRevA.17.1448.

What is reproduced
------------------
The SPM-broadened spectrum of a Gaussian pulse after a purely nonlinear fibre
(no dispersion), which has the closed form

    A(L, t) = sqrt(P0) exp(-t^2/(2 T0^2)) exp[i phi_max exp(-t^2/T0^2)],

with phi_max = gamma P0 L.  The spectrum |A(L, w)|^2 develops the classic
interference fringes; the number of spectral peaks follows the textbook rule
N_peaks = floor(phi_max/pi) + 1 (Agrawal, *Nonlinear Fiber Optics*, Sec. 4.1).

Validation
----------
1. The GNLSE output field is compared with the closed-form Fourier integral
   (machine precision, since beta = 0 makes the split-step nonlinear operator
   exact).
2. The measured peak count follows the phi_max/pi rule.

Usage
-----
    python reproductions/stolen_lin_1978_spm/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"


def _analytic_spectrum(grid: TemporalGrid, P0: float, T0: float, phi_max: float) -> np.ndarray:
    t = grid.t
    A0 = np.sqrt(P0) * np.exp(-(t**2) / (2 * T0**2))
    A = A0 * np.exp(1j * phi_max * np.exp(-(t**2) / T0**2))
    W = np.abs(grid.fft(A)) ** 2
    return W / W.max()


def _simulated_field(grid, P0, T0, gamma, phi_max, lam):
    L = phi_max / (gamma * P0)
    env = Envelope(shape="gaussian", peak_amplitude=np.sqrt(P0), pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(lam, "m"))
    fiber = FiberProfile.from_gamma(
        gamma=gamma, n2=2.6e-20, omega0=2 * np.pi * C_MS / lam, length=Length(L, "m")
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=GNLSESolver.estimate_num_steps(pulse, fiber, np.array([0.0])))
    return solver.evolution[-1].envelope_field


def validate(params: dict | None = None, make_plot: bool = True) -> dict:
    """Run every phi_max case and return the validation metrics."""
    params = params or json.loads(PARAMETERS.read_text())
    lam = params["central_wavelength_nm"] * 1e-9
    T0 = params["pulse_width_T0_fs"] * 1e-15
    P0 = params["peak_power_W"]
    gamma = params["gamma_per_Wm"]
    grid = TemporalGrid(N=params["grid_N"], Tmax=Time(params["grid_Tmax_ps"] * 1e-12, "s"))

    results = []
    for phi_pi in params["phi_max_over_pi"]:
        phi_max = float(phi_pi) * np.pi
        W_num = np.abs(grid.fft(_simulated_field(grid, P0, T0, gamma, phi_max, lam))) ** 2
        W_num /= W_num.max()
        W_an = _analytic_spectrum(grid, P0, T0, phi_max)
        max_diff = float(np.max(np.abs(W_num - W_an)))
        peaks, _ = find_peaks(W_num, prominence=0.05)
        expected = int(np.floor(phi_pi)) + 1
        results.append(
            {
                "phi_max_over_pi": float(phi_pi),
                "n_peaks": int(len(peaks)),
                "n_peaks_expected": expected,
                "max_abs_spectrum_diff": max_diff,
            }
        )

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.2 * len(results)), sharex=True)
        omega = grid.w
        for ax, phi_pi, res in zip(np.atleast_1d(axes), params["phi_max_over_pi"], results):
            phi_max = float(phi_pi) * np.pi
            W_num = np.abs(grid.fft(_simulated_field(grid, P0, T0, gamma, phi_max, lam))) ** 2
            W_num /= W_num.max()
            W_an = _analytic_spectrum(grid, P0, T0, phi_max)
            ax.plot(omega * 1e-12, W_an, "k-", lw=1, label="analytic")
            ax.plot(omega * 1e-12, W_num, "r--", lw=1, label="GNLSE")
            ax.set_ylabel(r"$|A|^2$")
            ax.set_title(rf"$\phi_{{max}}={phi_pi}\pi$  (peaks {res['n_peaks']}, expected {res['n_peaks_expected']})")
        np.atleast_1d(axes)[-1].set_xlabel(r"$\Omega$ (rad/ps)")
        np.atleast_1d(axes)[0].legend()
        fig.tight_layout()
        out = HERE / "spm_spectra.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")

    max_diff = max(r["max_abs_spectrum_diff"] for r in results)
    peak_ok = all(r["n_peaks"] == r["n_peaks_expected"] for r in results)
    assert max_diff < params["spectrum_tolerance"], f"spectrum mismatch {max_diff}"
    assert peak_ok, f"peak-count rule failed: {results}"
    print("SPM validation passed:")
    for r in results:
        print(
            f"  phi={r['phi_max_over_pi']}pi: peaks={r['n_peaks']} "
            f"(expected {r['n_peaks_expected']}), max|Δ|={r['max_abs_spectrum_diff']:.2e}"
        )
    return {"cases": results, "max_abs_spectrum_diff": max_diff}


if __name__ == "__main__":
    validate()
