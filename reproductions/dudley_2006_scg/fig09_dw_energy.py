"""Fig. 9 — Raman influence on dispersive-wave energy transfer.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 9 and
Sec. V.B.2.  For the j = 1 fundamental soliton of Fig. 8 the figure shows, as
a function of distance, (a) the mean soliton frequency/wavelength with and
without Raman scattering and (b) the relative energy radiated to the
dispersive wave.

Physics
-------
Without Raman the soliton frequency is fixed, so the DW resonance condition is
satisfied over the whole propagation and energy transfer continues.  With
Raman scattering the soliton continuously red-shifts, detuning it from the DW
resonance; efficient transfer therefore occurs only during the initial
propagation phase and the final DW energy is much lower.

Validation
----------
1. The Raman-on soliton red-shifts: mean soliton wavelength at the output is
   larger than the Raman-off value.
2. The DW energy fraction at the output is smaller with Raman on.
3. The DW energy is initially zero and grows with distance (both cases).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from . import fig08_dispersive_wave as dw

HERE = Path(__file__).resolve().parent
CFG = common.PARAMS["figures"]["fig09_dw_energy"]
DW_BAND = tuple(CFG["dw_band_nm"])
SOLITON_BAND = (850.0, 1400.0)


def _soliton_mean_wavelength(evo, soliton_band=SOLITON_BAND) -> np.ndarray:
    """Mean wavelength (nm) inside the soliton band, per z snapshot."""
    wl, psd = evo.spectrum_on_wavelength()
    mask = (wl >= soliton_band[0]) & (wl <= soliton_band[1])
    band = psd[:, mask]
    weights = band / np.maximum(band.sum(axis=1, keepdims=True), 1e-30)
    return np.asarray((weights * wl[mask]).sum(axis=1))


def _dw_fraction_series(evo, band=DW_BAND) -> np.ndarray:
    wl, psd = evo.spectrum_on_wavelength()
    mask = (wl >= band[0]) & (wl <= band[1])
    return np.asarray(
        psd[:, mask].sum(axis=1) / np.maximum(psd.sum(axis=1), 1e-30)
    )


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    evo_no = dw.run(raman=False, fast=fast)
    evo_yes = dw.run(raman=True, fast=fast)

    sol_no = _soliton_mean_wavelength(evo_no)
    sol_yes = _soliton_mean_wavelength(evo_yes)
    dw_no = _dw_fraction_series(evo_no)
    dw_yes = _dw_fraction_series(evo_yes)

    assert sol_yes[-1] > sol_no[-1], (
        f"Raman soliton should red-shift: {sol_no[-1]:.1f} -> {sol_yes[-1]:.1f} nm"
    )
    assert dw_yes[-1] < dw_no[-1], (
        f"Raman should reduce DW energy: {dw_no[-1]:.4f} -> {dw_yes[-1]:.4f}"
    )
    assert dw_no[-1] > dw_no[0], "DW energy should grow with distance"
    assert dw_yes[-1] > dw_yes[0], "DW energy should grow with distance (Raman on)"

    result = {
        "soliton_wl_no_raman_nm": float(sol_no[-1]),
        "soliton_wl_raman_nm": float(sol_yes[-1]),
        "dw_fraction_no_raman": float(dw_no[-1]),
        "dw_fraction_raman": float(dw_yes[-1]),
        "dw_growth_no_raman": float(dw_no[-1] - dw_no[0]),
        "dw_growth_raman": float(dw_yes[-1] - dw_yes[0]),
    }

    if make_plot:
        _plot(evo_no, evo_yes, sol_no, sol_yes, dw_no, dw_yes, result)
    return result


def _plot(evo_no, evo_yes, sol_no, sol_yes, dw_no, dw_yes, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    z_cm = evo_no.z * 100

    axes[0].plot(z_cm, sol_no, "k-", label="No Raman")
    axes[0].plot(z_cm, sol_yes, "--", color="C3", label="Raman")
    axes[0].set_xlabel("Distance (cm)")
    axes[0].set_ylabel("Mean soliton wavelength (nm)")
    axes[0].set_title("(a) Soliton wavelength vs z")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z_cm, dw_no, "k-", label="No Raman")
    axes[1].plot(z_cm, dw_yes, "--", color="C3", label="Raman")
    axes[1].set_xlabel("Distance (cm)")
    axes[1].set_ylabel("DW energy fraction")
    axes[1].set_title("(b) Energy radiated to the DW")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Fig. 9 — Raman vs no-Raman DW dynamics  "
        f"(output DW fraction {result['dw_fraction_no_raman']:.3f} → "
        f"{result['dw_fraction_raman']:.3f})",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig09_dw_energy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate(fast=False)
    print("Fig. 9 (DW energy fraction) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
