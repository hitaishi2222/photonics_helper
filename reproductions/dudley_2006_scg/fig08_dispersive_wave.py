"""Fig. 8 — dispersive-wave (Cherenkov) generation from a fundamental soliton.

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 8 and
Sec. V.B.2.  The j = 1 ejected fundamental soliton of Figs. 6-7
(P0 = 3.48 kW, T0 = 5.67 fs, FWHM 10 fs at 835 nm, ZDW 780 nm) propagates in
the *full* PCF dispersion, (a) without and (b) with Raman scattering.

Physics
-------
Where the soliton's nonlinear phase matches a linear dispersive wave, energy
is resonantly transferred to the normal-GVD side of the ZDW.  The paper's
phase-matching condition is

    β(ω_s) − ω_s/v_g,s + (1 − fR) γ Ps = β(ω_DW) − ω_DW/v_g,s ,

i.e. ``Σ_{k≥2} β_k Ω^k/k! = (1−fR) γ Ps`` with ``Ω = ω_DW − ω0``.  With the
full Table I dispersion this predicts a blue DW at ≈ 663 nm.

Validation
----------
1. A blue-side DW peak appears in the output spectrum.
2. Its wavelength matches the full-dispersion nonlinear phase-matching root
   to within 6 % (``fast`` and ``full`` configurations both pass).
3. With Raman on, the mean output wavelength is red-shifted and the DW energy
   fraction is reduced relative to the Raman-free case (full run only).

Runtime
-------
``fast=True``  : ~5 s  (0.05 m, 4096 points) — used by the test suite.
``fast=False`` : ~3-4 min (0.5 m, 8192 points, both Raman cases).
"""

from __future__ import annotations

from pathlib import Path

from . import common
from .common import Evolution

HERE = Path(__file__).resolve().parent
CFG = common.PARAMS["figures"]["fig08_dispersive_wave"]
P0 = float(CFG["peak_power_W"])
T0_FS = float(CFG["T0_fs"])
TOL = float(CFG["dw_tolerance"])
DW_BAND = (500.0, 700.0)


def _cache_path(raman: bool, fast: bool) -> Path:
    tag = "raman" if raman else "noraman"
    mode = "fast" if fast else "full"
    return HERE / f"fig08_{tag}_{mode}.npz"


def run(
    raman: bool,
    fast: bool = True,
    show_progress: bool = False,
    *,
    use_cache: bool = True,
) -> Evolution:
    cache = _cache_path(raman, fast)
    if use_cache and cache.exists():
        return common.load_evolution(cache)
    cfg = CFG["fast"] if fast else CFG["full"]
    pulse = common.build_pulse(
        P0, T0_fs=T0_FS, N_points=int(cfg["N_points"]), Tmax_ps=float(cfg["Tmax_ps"])
    )
    fiber = common.build_fiber(pulse, length_m=float(cfg["length_m"]), raman=raman)
    evo = common.run_gnlse(
        pulse,
        fiber,
        common.BETAS,  # full Table I dispersion
        num_steps=int(cfg["num_steps"]),
        nsaves=51 if fast else 101,
        raman=raman,
        shock=False,
        show_progress=show_progress,
    )
    if use_cache:
        common.save_evolution(evo, cache)
        print(f"cached {cache}")
    return evo


def _dw_energy_fraction(evo: Evolution, band=(600.0, 720.0)) -> float:
    """Fraction of output spectral power inside the DW band."""
    wl, psd = evo.spectrum_on_wavelength()
    spectrum = psd[-1]
    mask = (wl >= band[0]) & (wl <= band[1])
    return float(spectrum[mask].sum() / spectrum.sum())


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    predicted = common.dispersive_wave_wavelength_nm(P0)
    evo_no_raman = run(raman=False, fast=fast)

    wl, psd = evo_no_raman.spectrum_on_wavelength()
    dw_nm, dw_rel = common.strongest_peak_in_band(wl, psd[-1], *DW_BAND)
    dw_err = abs(dw_nm - predicted) / predicted
    assert dw_rel > 0.05, f"DW peak too weak (rel={dw_rel:.3f})"
    assert dw_err < TOL, (
        f"DW {dw_nm:.1f} nm vs phase-matching {predicted:.1f} nm ({dw_err * 100:.2f}%)"
    )

    result = {
        "dw_predicted_nm": predicted,
        "dw_simulated_nm": dw_nm,
        "dw_relative_power": dw_rel,
        "dw_rel_err": dw_err,
    }

    if not fast:
        evo_raman = run(raman=True, fast=False)
        mean_no = common.mean_spectral_wavelength_nm(evo_no_raman, -1)
        mean_yes = common.mean_spectral_wavelength_nm(evo_raman, -1)
        frac_no = _dw_energy_fraction(evo_no_raman)
        frac_yes = _dw_energy_fraction(evo_raman)
        assert mean_yes > mean_no, (
            f"Raman should red-shift the output: {mean_no:.1f} -> {mean_yes:.1f} nm"
        )
        assert frac_yes < frac_no, (
            f"Raman should reduce DW energy: {frac_no:.3f} -> {frac_yes:.3f}"
        )
        result.update(
            {
                "mean_wl_no_raman_nm": mean_no,
                "mean_wl_raman_nm": mean_yes,
                "dw_fraction_no_raman": frac_no,
                "dw_fraction_raman": frac_yes,
            }
        )
        if make_plot:
            _plot(evo_no_raman, dw_nm, predicted)
    elif make_plot:
        _plot(evo_no_raman, dw_nm, predicted)
    return result


def _plot(evo: Evolution, dw_nm: float, predicted: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wl, psd = evo.spectrum_on_wavelength()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    common.plot_spectral_evolution(
        evo, axes[0], wl_min=450, wl_max=1100, dynamic_range_db=35.0
    )
    axes[0].set_title("(a) DW generation without Raman")
    axes[0].axvline(predicted, color="w", ls=":", lw=1.2)

    axes[1].plot(wl, psd[-1] / psd[-1].max(), color="C0")
    axes[1].axvline(
        predicted, color="k", ls=":", label=f"phase matching {predicted:.0f} nm"
    )
    axes[1].axvline(dw_nm, color="C3", ls="--", label=f"GNLSE DW {dw_nm:.0f} nm")
    axes[1].set_xlim(500, 900)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Normalised spectral power")
    axes[1].set_title("(b) Output spectrum")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Fig. 8 — Cherenkov dispersive wave from a fundamental soliton", fontsize=10
    )
    fig.tight_layout()
    out = HERE / "fig08_dispersive_wave.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    result = validate(fast=False)
    print("Fig. 8 (dispersive wave) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
