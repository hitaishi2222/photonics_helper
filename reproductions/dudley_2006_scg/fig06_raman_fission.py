"""Fig. 6 — Raman-induced soliton fission of an N = 3 soliton.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 6 and
Sec. V.B.1.  To isolate the fission process from dispersive perturbations the
model keeps only β₂ and the delayed Raman response: fR = 0.18, τ_shock = 0 and
β₃ = … = 0.  This is the paper's footnote 11.

Physics
-------
The injected N = 3 soliton first compresses and broadens, then fissions into
the constituent fundamental solitons whose parameters are given by the
Kodama-Hasegawa perturbation theory (see :func:`common.kodama_hasegawa`)::

    P_j = P0 (2N - 2j + 1)² / N² ,   T_j = T0 / (2N - 2j + 1).

Each ejected soliton subsequently shifts to longer wavelengths through the
Raman self-frequency shift, so the spectrum acquires a continuous red tail.

Validation
----------
1. The fission length estimate is L_fiss ~ L_D/N ≈ 2.2 cm.
2. The strongest ejected soliton's peak power and FWHM match the
   Kodama-Hasegawa j = 1 values to within 15 % and 20 % respectively.
3. The output spectrum is Raman red-shifted: mean wavelength grows from
   ≈ 835 nm to > 1000 nm over 0.5 m.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from .common import Evolution, kodama_hasegawa, soliton_scales

HERE = Path(__file__).resolve().parent

CFG = common.PARAMS["figures"]["fig06_raman_fission"]
P0 = float(CFG["peak_power_W"])
LENGTH_M = float(CFG["length_m"])
N_POINTS = int(CFG["N_points"])
TMAX_PS = float(CFG["Tmax_ps"])


_FAST_CACHE: Evolution | None = None


def run(fast: bool = False, show_progress: bool = False) -> Evolution:
    global _FAST_CACHE
    if fast and _FAST_CACHE is not None:
        return _FAST_CACHE
    pulse = common.build_pulse(P0, N_points=N_POINTS, Tmax_ps=TMAX_PS)
    fiber = common.build_fiber(pulse, length_m=LENGTH_M, raman=True)
    n_steps = 8000 if fast else 12000
    evo = common.run_gnlse(
        pulse,
        fiber,
        np.array([common.BETA2_PS2_M]),  # β₂ only, plus Raman
        num_steps=n_steps,
        nsaves=201,
        raman=True,
        shock=False,
        show_progress=show_progress,
    )
    if fast:
        _FAST_CACHE = evo
    return evo


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    scales = soliton_scales(P0=P0)
    kh = kodama_hasegawa(3.0, P0)
    kh1 = kh[0]

    evo = run(fast=fast)

    # 1. Fission length estimate (paper Sec. V.B.1).
    assert abs(scales.N - 3.0) < 0.15, f"N={scales.N:.3f}"
    assert abs(scales.L_fiss - 0.0223) / 0.0223 < 0.02, (
        f"L_fiss={scales.L_fiss * 100:.3f} cm"
    )

    # 2. Ejected fundamental solitons at the output.
    intensity = evo.intensity[-1]
    peaks = common.temporal_peaks(intensity, rel_height=0.02, min_distance=40)
    assert len(peaks) >= 2, f"expected >= 2 ejected solitons, found {len(peaks)}"

    strongest = int(peaks[np.argmax(intensity[peaks])])
    p_meas = float(intensity[strongest])
    fwhm_meas = common.measure_peak_fwhm(evo.t, intensity, strongest) * 1e15
    p_err = abs(p_meas - kh1["P_W"]) / kh1["P_W"]
    fwhm_err = abs(fwhm_meas - kh1["fwhm_fs"]) / kh1["fwhm_fs"]
    assert p_err < 0.15, (
        f"ejected P={p_meas:.0f} W vs KH {kh1['P_W']:.0f} W ({p_err * 100:.1f}%)"
    )
    assert fwhm_err < 0.20, (
        f"ejected FWHM={fwhm_meas:.2f} fs vs KH {kh1['fwhm_fs']:.2f} fs "
        f"({fwhm_err * 100:.1f}%)"
    )

    # 3. Raman self-frequency shift: mean wavelength grows with z.
    lambda_start = common.mean_spectral_wavelength_nm(evo, 0)
    lambda_end = common.mean_spectral_wavelength_nm(evo, -1)
    assert lambda_start < 850.0, f"input mean wavelength {lambda_start:.1f} nm"
    assert lambda_end > 1000.0, f"output mean wavelength {lambda_end:.1f} nm"

    result = {
        "L_fiss_cm": scales.L_fiss * 100,
        "n_ejected": int(len(peaks)),
        "ejected_power_W": p_meas,
        "ejected_power_kh_W": kh1["P_W"],
        "ejected_fwhm_fs": fwhm_meas,
        "ejected_fwhm_kh_fs": kh1["fwhm_fs"],
        "power_rel_err": p_err,
        "fwhm_rel_err": fwhm_err,
        "mean_wavelength_start_nm": lambda_start,
        "mean_wavelength_end_nm": lambda_end,
    }

    if make_plot:
        _plot(evo, result)
    return result


def _plot(evo: Evolution, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    common.plot_spectral_evolution(
        evo, axes[0], wl_min=450, wl_max=1150, dynamic_range_db=30.0
    )
    axes[0].set_title("(a) Spectral evolution")

    # Literature comoving time: the Raman soliton appears at positive delay.
    common.plot_temporal_evolution(
        evo, axes[1], t_bounds=(-1.0, 5.0), dynamic_range_db=30.0
    )
    axes[1].set_title("(b) Temporal evolution (Raman fission)")

    fig.suptitle(
        f"Raman-induced fission of an N=3 soliton  "
        f"(L_fiss≈{result['L_fiss_cm']:.2f} cm; ejected P={result['ejected_power_W']:.0f} W, "
        f"FWHM={result['ejected_fwhm_fs']:.1f} fs)",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig06_raman_fission.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate(fast=False)
    print("Fig. 6 (Raman-induced fission) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
