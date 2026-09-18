"""Fig. 19-style coherence of the Fig. 3 supercontinuum (DRAFT, step2).

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 19 and
Sec. VI.A.  The paper shows the modulus of the complex degree of first-order
coherence ``|g₁₂(λ)|`` across the supercontinuum: high in the SPM-dominated
center, degraded in the Raman-shifted soliton / dispersive-wave wings where
spontaneous scattering and MI amplify input noise.

Method
------
Ensemble of ``N_REAL`` realizations of the Fig. 3 config (Table I betas,
Raman on, shock on with the effective-area-corrected ``τ_shock = 0.56 fs``),
each with ``raman_noise=True`` and a distinct ``noise_seed``.  Output spectra
give ``g₁₂(λ)`` via :func:`photonics_helper.noise.coherence_g12`.

Validation
----------
1. ``0 ≤ g₁₂ ≤ 1`` everywhere.
2. Mean ``g₁₂`` in the pump band exceeds the mean in the far wings
   (coherence degrades away from the pump, as in Fig. 19).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from photonics_helper.noise import coherence_g12

from . import common

HERE = Path(__file__).resolve().parent

CFG = common.PARAMS["figures"]["fig03_basic_scg"]
P0 = float(CFG["peak_power_W"])
LENGTH_M = float(CFG["length_m"])

N_REAL = 20
N_REAL_FAST = 4
BETAS = common.BETAS  # full Table I


def run(n_real: int = N_REAL, fast: bool = False, show_progress: bool = False):
    """Run the ensemble; return (wavelength_nm, g12, spectra)."""
    n = N_REAL_FAST if fast else n_real
    n_points = 2048 if fast else int(CFG["N_points"])
    tmax_ps = float(CFG["Tmax_ps"])
    n_steps = 400 if fast else 1500
    spectra = []
    pulse = common.build_pulse(P0, N_points=n_points, Tmax_ps=tmax_ps)
    fiber = common.build_fiber(pulse, length_m=LENGTH_M, raman=True)
    omega0 = float(pulse.central_frequency)
    for seed in range(n):
        solver = common.make_solver(
            pulse,
            fiber,
            BETAS,
            raman=True,
            shock=True,
            tau_shock=common.SHOCK_FS * 1e-15,
        )
        solver.propagate(
            num_steps=n_steps,
            nsaves=5,
            show_progress=show_progress,
            raman_noise=True,
            noise_seed=1000 + seed,
        )
        spec = np.asarray(pulse.grid.fft(solver.evolution[-1].envelope_field))
        spectra.append(spec)
    spectra = np.asarray(spectra, dtype=complex)
    g12 = coherence_g12(spectra)
    w_abs = omega0 + pulse.grid.w
    wl_nm = 2 * np.pi * 3e8 / w_abs * 1e9
    order = np.argsort(wl_nm)
    return wl_nm[order], g12[order], spectra


def validate(fast: bool = True, make_plot: bool = True) -> dict:
    wl_nm, g12, _ = run(fast=fast)
    assert np.all((g12 >= 0.0) & (g12 <= 1.0)), "g12 out of [0, 1]"
    soliton = (wl_nm > 1000) & (wl_nm < 1200)
    blue_wing = (wl_nm > 650) & (wl_nm < 800)
    g_soliton = float(np.mean(g12[soliton]))
    g_blue = float(np.mean(g12[blue_wing]))
    assert g_soliton > g_blue, (
        f"soliton g12={g_soliton:.3f} <= blue-wing g12={g_blue:.3f}"
    )
    assert float(np.mean(g12)) < 1.0, "ensemble should partially decohere"
    out = {
        "g12_soliton": g_soliton,
        "g12_blue_wing": g_blue,
        "g12_mean": float(np.mean(g12)),
    }
    if make_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(wl_nm, g12, lw=1.0)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("|g12|")
        ax.set_title("Dudley Fig. 19-style SC coherence (draft)")
        ax.set_ylim(0, 1.05)
        fig.savefig(HERE / "fig19_coherence.png", dpi=150)
        plt.close(fig)
        out["plot"] = "fig19_coherence.png"
    return out


if __name__ == "__main__":
    print(validate(fast=True))
