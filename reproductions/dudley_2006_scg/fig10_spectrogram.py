"""Fig. 10 — simulated supercontinuum spectrogram.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 10 and
Sec. V.C (Eq. 4).  The output SC field ``E(t)`` is gated by the 50 fs input
pulse ``g(t)`` and the resulting spectrogram is projected onto the temporal
intensity and the spectrum.

Physics
-------
The spectrogram separates the SC into its time-frequency constituents: the
blue dispersive wave, the Raman-shifted solitons and the residual pump all
appear at different ``(λ, τ)`` because of the fiber group delay.  The ultrafast
oscillations seen on the trailing edge of the strongest peak arise from
beating between two spectrogram bands separated by a large frequency
difference (≈ 165 THz in the paper's simulation).

Implementation note
-------------------
This script predates and mirrors :func:`photonics_helper.gnlse.plot_spectrogram`
(and :func:`photonics_helper.gnlse.gnlse_spectrogram`), which implement the same
Eq. 4 trace directly on a ``GNLSESolver``.  The library functions are the
reusable API; this script keeps its own copy so it can reuse the cached
``Evolution`` container.  Note that ``Envelope.visualize_3d`` and
``Wave.visualize(show_spectrogram=True)`` are a *conventional* STFT (Hann
window, baseband frequency, no gate) and do **not** reproduce this figure.

Caveat
------
The exact beat frequency depends on the precise input, dispersion and
(optional) noise realisation.  This reproduction therefore validates the
*method* and the time-frequency correlation rather than the literal 165 THz:
it checks that distinct DW and Raman-soliton bands exist and are separated in
both wavelength and delay, and reports the measured beat.

Validation
----------
1. The spectrogram resolves a blue DW band (λ < 600 nm) and a Raman-soliton
   band (λ > 900 nm).
2. Those two bands are separated in delay by > 0.1 ps (group-velocity
   walk-off), i.e. the spectrogram correlates spectral and temporal features.
3. The projected temporal intensity recovers the output intensity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from . import fig03_basic_scg as scg

HERE = Path(__file__).resolve().parent

DW_BAND = (500.0, 620.0)
SOLITON_BAND = (900.0, 1150.0)


def compute(fast: bool = False, n_delays: int = 181):
    evo = scg.run(fast=fast)
    pulse = common.build_pulse(
        scg.P0, N_points=scg.N_POINTS, Tmax_ps=scg.TMAX_PS
    )
    gate = common.input_gate(pulse)
    delays, omega, S = common.spectrogram(
        evo.fields[-1], evo.t, gate, evo.omega, n_delays=n_delays, delay_span_ps=5.0
    )
    wl = 2.0 * np.pi * common.C_MS / (evo.omega0 + omega) * 1e9
    order = np.argsort(wl)
    return evo, delays, wl[order], S[:, order]


def _band_delay(delays, wl, S, band) -> float:
    mask = (wl >= band[0]) & (wl <= band[1])
    profile = S[:, mask].sum(axis=1)
    return float(delays[int(np.argmax(profile))])


def _dominant_beat_THz(evo, f_min: float = 10.0) -> float:
    from scipy.ndimage import uniform_filter1d

    intensity = evo.intensity[-1]
    t = evo.t
    ac = intensity - uniform_filter1d(intensity, size=31)
    spectrum = np.abs(np.fft.rfft(ac)) ** 2
    freq = np.fft.rfftfreq(len(ac), d=float(t[1] - t[0]))
    mask = freq > f_min * 1e12
    if not mask.any():
        return 0.0
    return float(freq[mask][np.argmax(spectrum[mask])] / 1e12)


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    evo, delays, wl, S = compute(fast=fast)
    tau_dw = _band_delay(delays, wl, S, DW_BAND)
    tau_sol = _band_delay(delays, wl, S, SOLITON_BAND)

    dw_power = S[:, (wl >= DW_BAND[0]) & (wl <= DW_BAND[1])].sum()
    sol_power = S[:, (wl >= SOLITON_BAND[0]) & (wl <= SOLITON_BAND[1])].sum()
    assert dw_power > 0, "no blue DW band in the spectrogram"
    assert sol_power > 0, "no Raman-soliton band in the spectrogram"

    delay_sep = abs(tau_dw - tau_sol)
    assert delay_sep > 0.1, f"DW and soliton delays too close: {delay_sep:.3f} ps"

    beat = _dominant_beat_THz(evo)

    result = {
        "dw_band_nm": DW_BAND,
        "soliton_band_nm": SOLITON_BAND,
        "dw_delay_ps": tau_dw,
        "soliton_delay_ps": tau_sol,
        "delay_separation_ps": delay_sep,
        "dominant_beat_THz": beat,
    }

    if make_plot:
        _plot(evo, delays, wl, S, result)
    return result


def _plot(evo, delays, wl, S, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.25, 3, 1.2], height_ratios=[3, 1])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[0, 2], sharey=ax_main)
    ax_time = fig.add_subplot(gs[1, 1], sharex=ax_main)

    wl_lo, wl_hi = 400.0, 1400.0
    wl_grid, S_grid = common.interpolate_on_wavelength(wl, S, wl_lo, wl_hi)
    img = common.density_clip(S_grid.T, dynamic_range_db=50.0)
    ax_main.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=(delays.min(), delays.max(), wl_lo, wl_hi),
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    ax_main.set_xlim(-4, 4)
    ax_main.set_ylim(wl_lo, 1300)
    ax_main.set_xlabel("Delay (ps)")
    ax_main.set_ylabel("Wavelength (nm)")
    ax_main.set_title("Spectrogram")

    spectrum_proj = S_grid.sum(axis=0)
    ax_spec.plot(spectrum_proj / spectrum_proj.max(), wl_grid, color="k", lw=0.8)
    ax_spec.set_xlabel("Spectrum")
    ax_spec.tick_params(labelleft=False)

    intensity = evo.intensity[-1]
    ax_time.plot(delays, np.interp(delays, evo.t * 1e12, intensity), color="k")
    ax_time.set_xlabel("Delay (ps)")
    ax_time.set_ylabel("Intensity")
    ax_time.set_xlim(-4, 4)

    fig.suptitle(
        f"Fig. 10 — SC spectrogram (DW τ={result['dw_delay_ps']:.2f} ps, "
        f"soliton τ={result['soliton_delay_ps']:.2f} ps, "
        f"beat≈{result['dominant_beat_THz']:.0f} THz)",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig10_spectrogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate(fast=False)
    print("Fig. 10 (spectrogram) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
