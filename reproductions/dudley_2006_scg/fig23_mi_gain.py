"""Fig. 23 — parametric (modulation-instability) gain versus pump wavelength.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 23 and
Sec. VII.A.2.  A CW pump of 500 W is assumed and the gain is plotted against
the frequency offset Ω/2π from the pump for a range of pump wavelengths.  A
dashed line marks the zero-dispersion wavelength.

Physics
-------
For the undepleted pump the parametric amplitude gain is::

    g(Ω) = sqrt[(γP0)² − (κ/2)²] ,

with the phase mismatch

    κ = 2γP0 + Σ_{m≥1} β_{2m}/(2m)! Ω^{2m} .

The maximum gain is ``g_max = γP0`` (i.e. a power gain ``2g_max = 2γP0``),
which the paper's Fig. 23 plots.  Pumping in the anomalous GVD regime
(λ > 780 nm) gives broad bands growing from the pump; pumping in the normal
GVD regime relies on higher-order dispersion and gives narrow bands displaced
far from the pump.

Caveat
------
The paper uses the full GVD curve of Fig. 2.  Here β(ω) is reconstructed from
the Table I Taylor coefficients at 835 nm; the reconstruction reproduces the
paper's ZDW (≈ 780 nm) and the gain structure, but the far-normal-dispersion
gain windows are an extrapolation and are treated qualitatively.

Validation
----------
1. The reconstructed ZDW is 780 ± 5 nm (paper Fig. 2).
2. At an anomalous pump (800 nm) the peak gain equals 2γP0 and the peak
   frequency matches the classical ``sqrt(2γP/|β2|)`` to within 15 %.
3. At a normal pump (750 nm) the classical β₂-only model has zero gain while
   the full-dispersion model has gain displaced further than the 800 nm peak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from photonics_helper.phase_matching import mi_gain_spectrum, mi_gain_spectrum_extended

from . import common

HERE = Path(__file__).resolve().parent
CFG = common.PARAMS["figures"]["fig23_mi_gain"]
P_CW = float(CFG["cw_power_W"])
GAMMA = common.GAMMA
PUMP_WLS = list(CFG["pump_wavelengths_nm"])
N_OMEGA = int(CFG["grid_points"])
OMEGA_MAX = 1.2e15  # rad/s, ±191 THz


def _beta_fn():
    return common.taylor_beta_fn()


def _pump_omega(wl_nm: float) -> float:
    from photonics_helper.base import C_MS

    return 2.0 * np.pi * C_MS / (wl_nm * 1e-9)


def gain_curve(wl_nm: float, omega_m: np.ndarray | None = None) -> dict:
    """Extended MI gain at one pump wavelength."""
    if omega_m is None:
        omega_m = np.linspace(-OMEGA_MAX, OMEGA_MAX, N_OMEGA)
    return mi_gain_spectrum_extended(
        _beta_fn(), _pump_omega(wl_nm), GAMMA, P_CW, omega_m=omega_m
    )


def zDW_nm() -> float:
    """Zero-dispersion wavelength of the reconstructed β₂(λ)."""
    grid = np.linspace(600.0, 1000.0, 20000)
    values = np.array([common.beta2_at_wavelength(x) for x in grid])
    for i in range(len(grid) - 1):
        if values[i] * values[i + 1] < 0:
            return float(grid[i] + grid[i + 1]) / 2.0
    return float("nan")


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    zdw = zDW_nm()
    assert abs(zdw - 780.0) < 5.0, f"reconstructed ZDW = {zdw:.1f} nm"

    # 2. Anomalous pump at 800 nm.
    ext_800 = gain_curve(800.0)
    gmax_800 = float(ext_800["gain"].max())
    expected_gmax = 2.0 * GAMMA * P_CW
    assert abs(gmax_800 - expected_gmax) / expected_gmax < 1e-3, (
        f"g_max={gmax_800:.4f} vs 2γP={expected_gmax:.4f} "
        "(grid-discretisation tolerance)"
    )
    beta2_800 = common.beta2_at_wavelength(800.0) * 1e-24  # ps²/m -> s²/m
    classical = mi_gain_spectrum(beta2_800, GAMMA, P_CW)
    assert isinstance(classical, dict), "mi_gain_spectrum should return a result dict"
    omega_peak_classical = float(classical["Omega_peak"])
    omega_peak_ext = abs(float(ext_800["Omega_peak"]))
    peak_rel_err = abs(omega_peak_ext - omega_peak_classical) / omega_peak_classical
    assert peak_rel_err < 0.15, (
        f"peak Ω_ext={omega_peak_ext / 2 / np.pi / 1e12:.2f} THz vs classical "
        f"{omega_peak_classical / 2 / np.pi / 1e12:.2f} THz"
    )

    # 3. Normal pump at 750 nm: classical zero, extended displaced further.
    ext_750 = gain_curve(750.0)
    gmax_750 = float(ext_750["gain"].max())
    beta2_750 = common.beta2_at_wavelength(750.0) * 1e-24
    classical_750 = mi_gain_spectrum(beta2_750, GAMMA, P_CW)
    assert classical_750 == 0.0, "classical β₂-only model should have no gain at 750 nm"
    assert gmax_750 > 0.0, "full-dispersion model should have gain at 750 nm"
    assert abs(float(ext_750["Omega_peak"])) > abs(float(ext_800["Omega_peak"])), (
        "normal-GVD gain should be displaced further than anomalous-GVD gain"
    )

    result = {
        "zdw_nm": zdw,
        "g_max_800_W_per_m": gmax_800,
        "g_max_theory_W_per_m": expected_gmax,
        "omega_peak_800_THz": abs(float(ext_800["Omega_peak"])) / 2 / np.pi / 1e12,
        "omega_peak_classical_800_THz": omega_peak_classical / 2 / np.pi / 1e12,
        "peak_rel_err": peak_rel_err,
        "omega_peak_750_THz": abs(float(ext_750["Omega_peak"])) / 2 / np.pi / 1e12,
    }

    if make_plot:
        _plot(result)
    return result


def _plot(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    omega = np.linspace(-OMEGA_MAX, OMEGA_MAX, N_OMEGA)
    omega_thz = omega / 2 / np.pi / 1e12

    pump_grid = np.linspace(660.0, 960.0, 121)
    gain_map = np.zeros((len(pump_grid), len(omega)))
    for i, wl in enumerate(pump_grid):
        gain_map[i] = gain_curve(wl, omega_m=omega)["gain"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = axes[0].pcolormesh(
        omega_thz,
        pump_grid,
        2.0 * gain_map,
        shading="auto",
        cmap="jet",
    )
    axes[0].axvline(0.0, color="w", lw=0.5, alpha=0.4)
    axes[0].axhline(result["zdw_nm"], color="w", ls="--", lw=1.0)
    axes[0].set_xlim(-120, 120)
    axes[0].set_xlabel("Frequency offset $\\Omega/2\\pi$ (THz)")
    axes[0].set_ylabel("Pump wavelength (nm)")
    axes[0].set_title("(a) Power gain 2g (1/m)")
    fig.colorbar(im, ax=axes[0])

    for wl, color in [(800.0, "C3"), (750.0, "C0")]:
        curve = gain_curve(wl, omega_m=omega)
        axes[1].plot(omega_thz, 2.0 * curve["gain"], color=color, label=f"{wl:.0f} nm")
    axes[1].set_xlim(-220, 220)
    axes[1].set_xlabel("Frequency offset $\\Omega/2\\pi$ (THz)")
    axes[1].set_ylabel("Power gain 2g (1/m)")
    axes[1].set_title("(b) Specific gain curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Fig. 23 — parametric gain (CW {P_CW:.0f} W); "
        f"reconstructed ZDW = {result['zdw_nm']:.0f} nm",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig23_mi_gain.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate()
    print("Fig. 23 (MI gain) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
