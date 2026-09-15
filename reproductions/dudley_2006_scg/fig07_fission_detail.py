"""Fig. 7 — soliton fission detail and the Kodama-Hasegawa ejected soliton.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 7 and
Sec. V.B.1.  The left panel shows the onset of Raman-induced fission of an
N = 3 soliton; the right panel compares the intensity profile of the first
(j = 1) ejected soliton with the Kodama-Hasegawa prediction.

Physics
-------
Kodama-Hasegawa perturbation theory gives the constituent fundamental solitons
of an ideal N-soliton::

    P_j = P0 (2N - 2j + 1)² / N² ,   T_j = T0 / (2N - 2j + 1)
    A_j = sqrt(P_j) sech(T / T_j).

For N = 3, P0 = 1.25 kW, T0 = 28.4 fs the j = 1 soliton has P₁ = 3.47 kW and
an intensity FWHM of 10.0 fs — exactly the values quoted in the paper.

Validation
----------
The strongest ejected soliton's temporal intensity profile matches the
normalised Kodama-Hasegawa sech field with overlap > 0.97, and its peak power
and FWHM agree to within 15 % and 20 %.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from . import fig06_raman_fission as fission
from .common import Evolution, kodama_hasegawa

HERE = Path(__file__).resolve().parent
P0 = float(common.PARAMS["figures"]["fig06_raman_fission"]["peak_power_W"])


def run(fast: bool = False, show_progress: bool = False) -> Evolution:
    return fission.run(fast=fast, show_progress=show_progress)


def _ejected_soliton(evo: Evolution) -> dict:
    intensity = evo.intensity[-1]
    peaks = common.temporal_peaks(intensity, rel_height=0.02, min_distance=40)
    idx = int(peaks[np.argmax(intensity[peaks])])
    return {
        "index": idx,
        "power_W": float(intensity[idx]),
        "fwhm_fs": common.measure_peak_fwhm(evo.t, intensity, idx) * 1e15,
        "time_ps": float(evo.t[idx] * 1e12),
    }


def _sech_overlap(evo: Evolution, idx: int, T_j_fs: float, n_widths: float = 8.0) -> float:
    """Normalised field overlap between the measured peak and sech(t/T_j)."""
    t = evo.t
    intensity = evo.intensity[-1]
    T_j = T_j_fs * 1e-15
    mask = np.abs(t - t[idx]) < n_widths * T_j
    measured = np.sqrt(intensity[mask])
    reference = 1.0 / np.cosh((t[mask] - t[idx]) / T_j)
    measured = measured / np.linalg.norm(measured)
    reference = reference / np.linalg.norm(reference)
    return float(measured @ reference)


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    kh = kodama_hasegawa(3.0, P0)[0]
    evo = run(fast=fast)
    sol = _ejected_soliton(evo)

    power_err = abs(sol["power_W"] - kh["P_W"]) / kh["P_W"]
    fwhm_err = abs(sol["fwhm_fs"] - kh["fwhm_fs"]) / kh["fwhm_fs"]
    overlap = _sech_overlap(evo, sol["index"], kh["T_fs"])

    assert power_err < 0.15, (
        f"ejected P = {sol['power_W']:.0f} W vs KH {kh['P_W']:.0f} W "
        f"({power_err*100:.1f}%)"
    )
    assert fwhm_err < 0.20, (
        f"ejected FWHM = {sol['fwhm_fs']:.2f} fs vs KH {kh['fwhm_fs']:.2f} fs "
        f"({fwhm_err*100:.1f}%)"
    )
    assert overlap > 0.97, f"sech shape overlap = {overlap:.4f}"

    result = {
        "ejected_power_W": sol["power_W"],
        "kh_power_W": kh["P_W"],
        "power_rel_err": power_err,
        "ejected_fwhm_fs": sol["fwhm_fs"],
        "kh_fwhm_fs": kh["fwhm_fs"],
        "fwhm_rel_err": fwhm_err,
        "sech_overlap": overlap,
    }

    if make_plot:
        _plot(evo, sol, kh, result)
    return result


def _plot(evo: Evolution, sol: dict, kh: dict, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: fission onset over the first 20 cm (temporal density).
    zoom = Evolution(
        z=evo.z.copy(),
        fields=evo.fields.copy(),
        t=evo.t,
        omega=evo.omega,
        omega0=evo.omega0,
    )
    common.plot_temporal_evolution(
        zoom, axes[0], t_min=-3, t_max=3, dynamic_range_db=30.0
    )
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(0, 20)
    axes[0].axhline(2.23, color="w", ls=":", lw=1.2)
    axes[0].text(-2.8, 2.9, r"$L_D/N \approx 2.2$ cm", color="w", fontsize=9)
    axes[0].set_title("(a) Fission onset")

    # Right: ejected soliton profile vs Kodama-Hasegawa sech.
    t = evo.t * 1e12
    intensity = evo.intensity[-1]
    idx = sol["index"]
    T_j = kh["T_fs"] / 1e3  # fs -> ps
    window = np.abs(t - t[idx]) < 6 * T_j
    axes[1].plot(
        t[window], intensity[window] / intensity[idx], "k-", lw=2, label="GNLSE (j = 1)"
    )
    axes[1].plot(
        t[window],
        1.0 / np.cosh((t[window] - t[idx]) / T_j) ** 2,
        "o",
        ms=3,
        mfc="none",
        color="C3",
        label="Kodama-Hasegawa",
    )
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Normalised intensity")
    axes[1].set_title(
        f"(b) Ejected soliton: P={sol['power_W']:.0f} W, FWHM={sol['fwhm_fs']:.1f} fs"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Fig. 7 — fission detail (sech overlap = {result['sech_overlap']:.4f})",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig07_fission_detail.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate(fast=False)
    print("Fig. 7 (fission detail) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
