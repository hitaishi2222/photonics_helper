"""Fig. 5 — periodic evolution of an ideal higher-order N = 3 soliton.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 5 and the
surrounding text (Sec. V.B.1).  The ideal NLSE case neglects all higher-order
nonlinear and dispersive effects: only β₂ is kept, fR = 0, τ_shock = 0 and all
noise sources are off.

Physics
-------
For the NLSE an injected higher-order soliton evolves periodically with the
soliton period ``z_sol = (π/2) L_D`` and ``L_D = T0²/|β₂|``.  With the
paper's parameters (T0 = 28.4 fs, β₂ = −11.83 ps²/km) we have
``L_D = 6.8 cm`` and ``z_sol = 10.6 cm``; Fig. 5 plots two soliton periods.

Validation
----------
1. The predicted scales match the paper: L_D ≈ 6.8 cm, z_sol ≈ 10.6 cm.
2. The spectrum at z = z_sol reproduce the input spectrum (normalised
   overlap > 0.99) — this is the periodicity of the higher-order soliton.
3. At z = z_sol/2 the temporal intensity is compressed by a factor > 3
   (the ideal N = 3 soliton breathes to a peak power ≈ N²P0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from .common import Evolution, soliton_scales

HERE = Path(__file__).resolve().parent

CFG = common.PARAMS["figures"]["fig05_ideal_soliton"]
P0 = float(CFG["peak_power_W"])


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised spectral overlap (Cauchy-Schwarz) of two power spectra."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.sqrt(a.sum() * b.sum())
    return float((np.sqrt(a * b).sum()) / denom) if denom > 0 else 0.0


def run(fast: bool = False, show_progress: bool = False) -> Evolution:
    scales = soliton_scales(P0=P0)
    length = float(CFG["length_soliton_periods"]) * scales.z_sol
    pulse = common.build_pulse(
        P0,
        T0_fs=common.T0_FS,
        N_points=int(CFG["N_points"]),
        Tmax_ps=float(CFG["Tmax_ps"]),
    )
    fiber = common.build_fiber(pulse, length_m=length, raman=False)
    n_steps = 1200 if fast else 3000
    evo = common.run_gnlse(
        pulse,
        fiber,
        np.array([common.BETA2_PS2_M]),  # β₂ only — ideal NLSE
        num_steps=n_steps,
        nsaves=201,
        raman=False,
        shock=False,
        show_progress=show_progress,
    )
    evo.metadata["scales"] = scales
    evo.metadata["z_sol"] = scales.z_sol
    evo.metadata["L_D"] = scales.L_D
    return evo


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    scales = soliton_scales(P0=P0)

    # 1. Paper length scales (Sec. V.B.1 and Fig. 5 caption).
    assert abs(scales.L_D - 0.068) / 0.068 < 0.05, f"L_D={scales.L_D*100:.2f} cm"
    assert abs(scales.z_sol - 0.106) / 0.106 < 0.05, f"z_sol={scales.z_sol*100:.2f} cm"
    assert abs(scales.N - 3.0) < 0.15, f"N={scales.N:.3f}"

    evo = run(fast=fast)
    spectra = evo.spectra
    z_sol = scales.z_sol

    # Snapshot index for z = z_sol (uniform save grid).
    z = evo.z
    i_full = int(np.argmin(np.abs(z - z_sol)))

    # 2. Periodicity: spectrum at z_sol equals the input spectrum.
    overlap = _overlap(spectra[0], spectra[i_full])
    assert overlap > 0.99, f"periodicity overlap at z_sol = {overlap:.5f}"

    # 3. Breathing / temporal compression within the first soliton period.
    #    The ideal N=3 bound state reaches roughly N² P0 at maximum
    #    compression (near z_sol/4 and 3 z_sol/4).
    within_first = z <= z_sol + 1e-12
    peak_ratio = float(
        (evo.intensity[within_first].max(axis=1) / evo.intensity[0].max()).max()
    )
    assert peak_ratio > 3.0, f"peak compression factor = {peak_ratio:.2f}"
    assert peak_ratio < 15.0, f"peak compression factor unexpectedly large: {peak_ratio:.2f}"

    result = {
        "L_D_cm": scales.L_D * 100,
        "L_NL_mm": scales.L_NL * 1e3,
        "N": scales.N,
        "z_sol_cm": scales.z_sol * 100,
        "periodicity_overlap": overlap,
        "peak_compression": peak_ratio,
    }

    if make_plot:
        _plot(evo, scales, result)
    return result


def _plot(evo: Evolution, scales, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Convert distance to soliton periods for the paper's Fig. 5 axis.
    z_sol = scales.z_sol
    evo.z = evo.z / z_sol

    common.plot_spectral_evolution(
        evo, axes[0], wl_min=500, wl_max=1200, dynamic_range_db=30.0
    )
    axes[0].set_ylabel("Distance (z / z_sol)")
    axes[0].set_title("(a) Spectral evolution")

    common.plot_temporal_evolution(
        evo, axes[1], t_min=-0.3, t_max=0.3, dynamic_range_db=30.0
    )
    axes[1].set_ylabel("Distance (z / z_sol)")
    axes[1].set_title("(b) Temporal evolution")

    fig.suptitle(
        f"Ideal N=3 soliton (NLSE only): z_sol={z_sol*100:.2f} cm, "
        f"periodicity overlap={result['periodicity_overlap']:.4f}",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig05_ideal_soliton_period.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate()
    print("Fig. 5 (ideal N=3 soliton period) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
