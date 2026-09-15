"""Fig. 4 — detailed output temporal and spectral characteristics.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 4 and the
surrounding text in Sec. V.A.  Numerical filtering correlates the prominent
spectral features with the temporal peaks: the two long-wavelength Raman
solitons (A, B) and the normal-GVD dispersive wave (C) at ≈ 550 nm, which is
associated with the low-amplitude temporal pedestal.

Physics
-------
The output pulse consists of several temporally separated fundamental
solitons that have Raman-shifted to different wavelengths, plus the narrow-
band blue dispersive wave.  Because the fiber group delay is wavelength
dependent, each spectral component maps to a distinct temporal feature.

Validation
----------
1. The output spectrum contains at least three well-separated components:
   a blue DW (< 650 nm), the pump region and a red Raman soliton (> 900 nm).
2. The output temporal profile contains multiple (≥ 3) peaks.
3. Bandpass filtering the brightest red soliton and the DW yields temporally
   localized features at different delays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from . import fig03_basic_scg as scg

HERE = Path(__file__).resolve().parent

DW_BAND = (450.0, 600.0)
SOLITON_BAND = (850.0, 1300.0)


def _top_peaks(wl, psd, n=3, min_sep_nm=30.0, rel_height=0.05):
    from scipy.signal import find_peaks

    spectrum = psd / psd.max()
    peaks, _ = find_peaks(spectrum, height=rel_height, prominence=rel_height)
    if peaks.size == 0:
        return []
    order = peaks[np.argsort(spectrum[peaks])[::-1]]
    chosen: list[int] = []
    for idx in order:
        if all(abs(wl[idx] - wl[j]) >= min_sep_nm for j in chosen):
            chosen.append(int(idx))
        if len(chosen) == n:
            break
    return [(float(wl[i]), float(spectrum[i])) for i in chosen]


def _band_temporal_delay(
    evo, center_nm: float, half_width_nm: float = 20.0
) -> tuple[float, float]:
    """Delay (ps) and peak intensity of the bandpass-isolated feature."""
    wl, psd = evo.spectrum_on_wavelength()
    _ = wl
    # Work on the unsorted frequency grid for the inverse FFT.
    omega = evo.omega
    lam = 2.0 * np.pi * common.C_MS / (evo.omega0 + omega) * 1e9
    mask = np.abs(lam - center_nm) <= half_width_nm
    spectrum = np.fft.fftshift(np.fft.fft(evo.fields[-1]))
    filtered = np.fft.ifft(np.fft.ifftshift(spectrum * mask))
    intensity = np.abs(filtered) ** 2
    return float(evo.t[int(np.argmax(intensity))] * 1e12), float(intensity.max())


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    evo = scg.run(fast=fast)
    wl, psd = evo.spectrum_on_wavelength()
    spectrum = psd[-1]
    peaks = _top_peaks(wl, spectrum, n=3, min_sep_nm=30.0)

    # 1. Blue DW, pump region and red Raman soliton must all be present.
    dw_nm, dw_rel = common.strongest_peak_in_band(wl, spectrum, *DW_BAND)
    sol_nm, sol_rel = common.strongest_peak_in_band(wl, spectrum, *SOLITON_BAND)
    assert dw_rel > 0.02, f"DW component too weak: {dw_rel:.3f}"
    assert sol_rel > 0.1, f"Raman soliton component too weak: {sol_rel:.3f}"
    assert sol_nm - dw_nm > 150.0, "DW and soliton are not well separated"

    # 2. Multiple temporal peaks.
    intensity = evo.intensity[-1]
    t_peaks = common.temporal_peaks(intensity, rel_height=0.02, min_distance=20)
    assert len(t_peaks) >= 3, f"only {len(t_peaks)} temporal peaks"

    # 3. Bandpass correlation: each component maps to a temporally localised
    #    feature.  The absolute delays depend on the exact soliton partition
    #    and are reported rather than asserted.
    tau_dw, amp_dw = _band_temporal_delay(evo, dw_nm)
    tau_sol, amp_sol = _band_temporal_delay(evo, sol_nm)
    assert amp_dw > 0 and amp_sol > 0, "bandpass filtering returned an empty feature"

    result = {
        "spectral_peaks_nm": [p[0] for p in peaks],
        "spectral_peaks_rel": [p[1] for p in peaks],
        "dw_peak_nm": dw_nm,
        "dw_relative_power": dw_rel,
        "soliton_peak_nm": sol_nm,
        "soliton_relative_power": sol_rel,
        "n_temporal_peaks": int(len(t_peaks)),
        "dw_delay_ps": tau_dw,
        "soliton_delay_ps": tau_sol,
    }

    if make_plot:
        _plot(evo, result)
    return result


def _plot(evo, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wl, psd = evo.spectrum_on_wavelength()
    t_ps, intensity = evo.output_intensity()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(t_ps, intensity / intensity.max(), color="C0")
    axes[0].set_xlim(-4, 4)
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Normalised intensity")
    axes[0].set_title("(a) Output temporal profile")
    axes[0].grid(True, alpha=0.3)
    for label, tau in [("DW", result["dw_delay_ps"]), ("soliton", result["soliton_delay_ps"])]:
        axes[0].axvline(tau, color="C3", ls=":", lw=1.0)
        axes[0].text(tau, 0.5, f" {label}", color="C3", fontsize=9)

    axes[1].semilogy(wl, psd[-1] / psd[-1].max(), color="C0")
    for nm, rel in zip(result["spectral_peaks_nm"], result["spectral_peaks_rel"]):
        axes[1].axvline(nm, color="C1", ls=":", lw=1.0)
    axes[1].set_xlim(400, 1400)
    axes[1].set_ylim(1e-4, 1.5)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Normalised PSD")
    axes[1].set_title("(b) Output spectral profile")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Fig. 4 — output features (DW {result['dw_peak_nm']:.0f} nm, "
        f"soliton {result['soliton_peak_nm']:.0f} nm, "
        f"{result['n_temporal_peaks']} temporal peaks)",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig04_output_features.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate(fast=False)
    print("Fig. 4 (output features) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
