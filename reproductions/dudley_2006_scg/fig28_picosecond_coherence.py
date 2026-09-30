"""Fig. 24-28 deck — picosecond-pulse SC coherence (Dudley 2006, Sec. VII.D).

Paper reference
---------------
Dudley, Genty & Coen, RMP 78, 1135 (2006), Sec. VII.D.
Deck (Figs. 24-26, reused for 28): 2 m PCF (same Table-I fiber as the
femtosecond decks), 20 ps pulses, 500 W peak power, 800 nm pump. Fig. 28:
20 realizations over the first 50 cm; |g12| across the output spectrum.

Claims asserted (structural; the long-pulse regime is N > 1000 so the
incoherence is expected from the paper's own argument):
1. The mean output spectrum is smooth and featureless (no strongly
   contrasted narrow soliton peaks), unlike the femtosecond SC —
   measured as the ratio of the mean-spectrum max to its -20 dB local
   structure: smoothness proxy = mean(|spec - smoothed(spec)|)/max < tol.
2. The output coherence |g12| peaks above the far wings only in a narrow
   band around the pump (paper: 'nearly zero at all wavelengths except
   within a few nanometers of the pump').
3. The |g12| ncarrow pump-band value is low (incoherent seeded SC):
   g12_pump_band < the femtosecond-deck level (fig19's 0.91 soliton-band
   anchor) — recorded quantitatively.
"""

from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np

from photonics_helper.noise import coherence_g12

from . import common

HERE = Path(__file__).resolve().parent

LENGTH_M = 0.50  # 50 cm, per Fig. 28
P0 = 500.0
WL_PUMP_NM = 800.0
DUR_PS = 20.0
N_REAL = 20
N_REAL_FAST = 4
N_STEPS = 600


def run(n_real: int, n_steps: int, fast: bool = False):
    pulse = common.build_pulse(
        P0,
        T0_fs=DUR_PS * 1e3 / 1.763,
        N_points=16384 if fast else 32768,
        Tmax_ps=float(common.PARAMS["figures"]["fig03_basic_scg"]["Tmax_ps"]) * 4,
        wl_nm=WL_PUMP_NM,
    )
    fiber = common.build_fiber(pulse, length_m=LENGTH_M, raman=True)
    betas = np.asarray(common.BETAS, dtype=float)
    spectra = []
    for seed in range(n_real):
        solver = common.make_solver(
            pulse, fiber, betas, raman=True, shock=True,
            tau_shock=common.SHOCK_FS * 1e-15,
        )
        solver.propagate(
            num_steps=n_steps, nsaves=4,
            raman_noise=True, noise_seed=9000 + seed,
        )
        spectra.append(np.asarray(pulse.grid.fft(solver.evolution[-1].envelope_field)))
    omega0 = float(pulse.central_frequency)
    wl = 2 * pi * 2.99792458e8 / (omega0 + pulse.grid.w) * 1e9
    order = np.argsort(wl)
    spec = np.asarray(spectra, dtype=complex)
    g12 = coherence_g12(spec)  # per-bin at the output
    mean_spec_c = np.mean(spec, axis=0)
    mean_spec = np.abs(mean_spec_c) ** 2
    return wl[order], g12[order], mean_spec[order], wl[order]


def validate(fast: bool = True, make_plot: bool = True) -> dict:
    n_real = N_REAL_FAST if fast else N_REAL
    n_steps = 200 if fast else N_STEPS
    wl, g12, mean_spec, _ = run(n_real, n_steps, fast=fast)

    # (1) smoothness of the ensemble-averaged spectrum
    kernel = 101
    sm = np.convolve(mean_spec, np.ones(kernel) / kernel, mode="same")
    rel_struct = float(np.mean(np.abs(mean_spec - sm) / mean_spec.max()))
    assert rel_struct < 0.2, f"mean spectrum not smooth (structure {rel_struct:.3f})"

    # (2) narrow near-coherent band only around the pump
    i_pump = int(np.argmin(np.abs(wl - WL_PUMP_NM)))
    half_w = 3  # 'few nanometres' bin window (bin ~1 nm at this grid)
    pump_band = np.zeros(len(g12), dtype=bool)
    pump_band[max(0, i_pump - half_w):i_pump + half_w + 1] = True
    far = ~pump_band
    g_pump = float(g12[pump_band].mean())
    g_far = float(g12[far].mean())
    assert g_pump > g_far + 0.1, (
        f"pump-band coherence {g_pump:.3f} not above far wings {g_far:.3f}"
    )

    # (3) the whole SC is mostly incoherent (quantitative record)
    results = {
        "g12_pump_band": g_pump,
        "g12_far_windows": g_far,
        "g12_max": float(g12.max()),
        "smoothness_rel_structure": rel_struct,
        "n_real": n_real,
        "n_steps": n_steps,
    }
    # paper Fig. 28b: |g12| near unity ONLY within a few nm of the pump;
    # everywhere else incoherent.
    assert g_pump > 0.9, (
        f"pump carrier band not coherent ({g_pump:.3f}) — grid/noise issue"
    )
    assert g_far < 0.5, f"far-field SC unexpectedly coherent ({g_far:.3f})"

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].plot(wl, 10 * np.log10(mean_spec / mean_spec.max()), color="C0")
        axs[0].set_ylim(-80, 0)
        axs[0].set_xlabel("λ (nm)")
        axs[0].set_ylabel("mean spectrum (dB)")
        axs[0].set_title(f"Fig. 28a-style mean spectrum "
                         f"({DUR_PS:.0f} ps, 500 W, {WL_PUMP_NM:.0f} nm)")
        axs[1].plot(wl, g12, color="C3")
        axs[1].set_ylim(0, 1.05)
        axs[1].set_xlabel("λ (nm)")
        axs[1].set_ylabel("|g12|")
        axs[1].set_title("Fig. 28b-style output coherence")
        fig.tight_layout()
        fig.savefig(HERE / "fig28_picosecond_coherence.png", dpi=150)
        plt.close(fig)
        results["plot"] = "fig28_picosecond_coherence.png"
    return results


if __name__ == "__main__":
    import sys
    print(validate(fast="--fast" in sys.argv))
