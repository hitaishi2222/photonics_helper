"""Fig. 3 — the reference femtosecond supercontinuum (all processes on).

Paper reference
---------------
Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 3 and
Sec. V.A.  15 cm of the highly nonlinear PCF, a 50 fs (FWHM) 10 kW sech pulse
at 835 nm, the full Table I dispersion, Raman scattering (fR = 0.18) and
self-steepening.

Physics
-------
The pulse has soliton order

    N = sqrt(L_D / L_NL) = sqrt(T0²/(|β₂| · 1/(γP0))) ≈ 8.5 ,

with L_D = 6.8 cm and z_sol = (π/2)L_D = 10.6 cm.  The initial compression is
followed by soliton fission (fission length ≈ L_D/N), Raman self-frequency
shifts of the ejected solitons, and emission of a blue dispersive wave in the
normal-GVD regime.  The output spectrum is continuous over more than an octave
at the −20 dB level (550-1100 nm in the paper).

Validation
----------
1. The soliton order and length scales match the paper (N ≈ 8.5, z_sol ≈ 10.6 cm).
2. The output is octave-spanning at −20 dB (span ratio > 1.8) and contains
   both a blue edge below 650 nm and a red edge above 1000 nm.
3. The spectral and temporal evolution density plots reproduce the morphology
   of Fig. 3 (initial broadening, fission, red-shifting solitons, blue DW).

Runtime
-------
``fast=True``  : ~10 s  (1500 steps) — used by the test suite.
``fast=False`` : ~40 s-2 min (6000 steps) for the paper-quality figure.

The full result is cached to ``fig03_cache.npz`` so that Figs. 4 and 10 can
reuse it without re-propagating.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from .common import Evolution, soliton_scales

HERE = Path(__file__).resolve().parent
CACHE = HERE / "fig03_cache.npz"

CFG = common.PARAMS["figures"]["fig03_basic_scg"]
P0 = float(CFG["peak_power_W"])
LENGTH_M = float(CFG["length_m"])
N_POINTS = int(CFG["N_points"])
TMAX_PS = float(CFG["Tmax_ps"])
NSAVES = int(CFG["nsaves"])
BETAS = common.BETAS  # full Table I


_FAST_CACHE: Evolution | None = None


def run(
    fast: bool = False,
    *,
    use_cache: bool = True,
    show_progress: bool = False,
) -> Evolution:
    global _FAST_CACHE
    if fast and _FAST_CACHE is not None:
        return _FAST_CACHE
    if use_cache and not fast and CACHE.exists():
        return common.load_evolution(CACHE)

    pulse = common.build_pulse(P0, N_points=N_POINTS, Tmax_ps=TMAX_PS)
    fiber = common.build_fiber(pulse, length_m=LENGTH_M, raman=True)
    n_steps = int(CFG["num_steps_quick"] if fast else CFG["num_steps_full"])
    evo = common.run_gnlse(
        pulse,
        fiber,
        BETAS,
        num_steps=n_steps,
        nsaves=41 if fast else NSAVES,
        raman=True,
        shock=True,
        show_progress=show_progress,
    )
    if fast:
        _FAST_CACHE = evo
    elif use_cache:
        common.save_evolution(evo, CACHE)
        print(f"cached {CACHE}")
    return evo


def _octave_span(wavelength_nm: np.ndarray, psd: np.ndarray, level_db: float = -20.0):
    spectrum = psd / psd.max()
    mask = spectrum >= 10.0 ** (level_db / 10.0)
    lo = float(wavelength_nm[mask].min())
    hi = float(wavelength_nm[mask].max())
    return lo, hi, hi / lo


def validate(fast: bool = False, make_plot: bool = True) -> dict:
    scales = soliton_scales(P0=P0)
    assert abs(scales.N - 8.5) / 8.5 < 0.05, f"N={scales.N:.2f}"
    assert abs(scales.z_sol - 0.106) / 0.106 < 0.05, (
        f"z_sol={scales.z_sol * 100:.2f} cm"
    )

    evo = run(fast=fast)
    wl, psd = evo.spectrum_on_wavelength()
    lo, hi, ratio = _octave_span(wl, psd[-1])
    assert lo < 650.0, f"blue edge only reaches {lo:.0f} nm"
    assert hi > 1000.0, f"red edge only reaches {hi:.0f} nm"
    assert ratio > 1.8, f"−20 dB span ratio = {ratio:.2f} (need > 1.8)"

    result = {
        "N": scales.N,
        "z_sol_cm": scales.z_sol * 100,
        "L_D_cm": scales.L_D * 100,
        "L_fiss_cm": scales.L_fiss * 100,
        "span_minus20dB_nm": (lo, hi),
        "span_ratio": ratio,
    }

    if make_plot:
        _plot(evo, result)
    return result


def _plot(
    evo: Evolution,
    result: dict,
    *,
    wl_bounds: tuple[float, float] | None = None,
    t_bounds: tuple[float, float] = (-1.0, 5.0),
    dynamic_range_db: float = 40.0,
    cmap: str = "jet",
    plotly: bool = False,
    annotate_features: bool = True,
    save: bool = True,
):
    """Assemble the four-panel Fig. 3 and return the figure.

    Parameters
    ----------
    evo : Evolution
        Cached GNLSE result.
    result : dict
        Output of :func:`validate` (used in the title / shading).
    wl_bounds : (float, float), optional
        Spectral window in nm.  Defaults to the carrier-wavelength-based
        :func:`common.default_wl_bounds` (≈ 0.5·λ0 to 1.6·λ0).
    t_bounds : (float, float)
        Time window in ps.  Defaults to ``(-1, 5)``; the temporal axis uses
        the literature comoving convention, so Raman solitons are at positive
        delay as in the paper's Fig. 3(b).
    dynamic_range_db, cmap : float, str
        Density-plot colour range and matplotlib colormap.
    plotly : bool
        If True, build an interactive 2x2 Plotly dashboard whose hover text
        labels each evolution cell as DW / SPM / Raman soliton.
    annotate_features : bool
        Mark the strongest cell of each feature class in the Plotly path.
    save : bool
        Write ``fig03_basic_scg.png`` (matplotlib path only).

    Returns
    -------
    matplotlib Figure or plotly Figure
        The figure, intentionally **not** closed, so the caller can adjust it.
    """
    if wl_bounds is None:
        wl_bounds = common.default_wl_bounds(common.carrier_wavelength_nm(evo))

    if plotly:
        return _plot_plotly(
            evo,
            result,
            wl_bounds=wl_bounds,
            t_bounds=t_bounds,
            dynamic_range_db=dynamic_range_db,
            annotate_features=annotate_features,
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.7])
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_evo_spec = fig.add_subplot(gs[1, 0])
    ax_evo_time = fig.add_subplot(gs[1, 1])

    # (a) output spectrum in dB vs wavelength
    wl, psd = evo.spectrum_on_wavelength()
    spec_db = 10.0 * np.log10(psd[-1] / psd[-1].max() + 1e-30)
    ax_spec.plot(wl, spec_db, color="C0", lw=1.2)
    ax_spec.axhline(-20.0, color="k", ls=":", lw=1.0)
    ax_spec.axvspan(
        result["span_minus20dB_nm"][0],
        result["span_minus20dB_nm"][1],
        color="C1",
        alpha=0.15,
    )
    ax_spec.set_xlim(*wl_bounds)
    ax_spec.set_ylim(-dynamic_range_db, 2.0)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Intensity (dB)")
    ax_spec.set_title("(a) Intensity (dB) vs wavelength")
    ax_spec.grid(True, which="both", alpha=0.3)

    # (b) output temporal intensity in dB vs time (literature comoving time)
    t_ps, inten = common.temporal_evolution_data(evo, time_reversal=True)
    inten_dB = 10.0 * np.log10(inten[-1] / inten[-1].max() + 1e-30)
    ax_time.plot(t_ps, inten_dB, color="C0", lw=1.2)
    ax_time.set_xlim(*t_bounds)
    ax_time.set_ylim(-dynamic_range_db, 2.0)
    ax_time.set_xlabel("Time (ps)")
    ax_time.set_ylabel("Intensity (dB)")
    ax_time.set_title("(b) Intensity (dB) vs time")
    ax_time.grid(True, which="both", alpha=0.3)

    # (c) spectral evolution
    common.plot_spectral_evolution(
        evo,
        ax_evo_spec,
        wl_bounds=wl_bounds,
        dynamic_range_db=dynamic_range_db,
        cmap=cmap,
    )
    ax_evo_spec.set_title("(c) Spectral evolution")

    # (d) temporal evolution
    common.plot_temporal_evolution(
        evo,
        ax_evo_time,
        t_bounds=t_bounds,
        dynamic_range_db=dynamic_range_db,
        cmap=cmap,
        time_reversal=True,
    )
    ax_evo_time.set_title("(d) Temporal evolution")

    fig.suptitle(
        f"Fig. 3 — basic SCG (N={result['N']:.2f}, z_sol={result['z_sol_cm']:.1f} cm, "
        f"−20 dB span {result['span_minus20dB_nm'][0]:.0f}-{result['span_minus20dB_nm'][1]:.0f} nm)",
        fontsize=10,
    )
    if save:
        out = HERE / "fig03_basic_scg.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    return fig


def _plot_plotly(
    evo: Evolution,
    result: dict,
    *,
    wl_bounds: tuple[float, float] | None = None,
    t_bounds: tuple[float, float] = (-1.0, 5.0),
    dynamic_range_db: float = 40.0,
    annotate_features: bool = True,
):
    """Interactive 2x2 Plotly version of Fig. 3 with feature-labelled hover.

    Thin wrapper over :func:`photonics_helper.gnlse.plot_scg_dashboard`; the
    reproduction prepares its own arrays because it stores an
    :class:`~reproductions.dudley_2006_scg.common.Evolution` rather than a
    live ``GNLSESolver``.
    """
    from photonics_helper.gnlse import plot_scg_dashboard

    if wl_bounds is None:
        wl_bounds = common.default_wl_bounds(common.carrier_wavelength_nm(evo))
    wl_lo, wl_hi = float(wl_bounds[0]), float(wl_bounds[1])

    grid, resampled = common.spectral_evolution_data(evo, (wl_lo, wl_hi))
    spec_db = common._db_clipped(resampled, dynamic_range_db)
    spec_labels = common.classify_spectral_features(
        grid, common.carrier_wavelength_nm(evo)
    )

    t_ps, intensity = common.temporal_evolution_data(evo, time_reversal=True)
    int_db = common._db_clipped(intensity, dynamic_range_db)
    t_labels = common.temporal_feature_labels(evo, time_reversal=True)

    wl, psd = evo.spectrum_on_wavelength()
    out_spec_db = 10.0 * np.log10(psd[-1] / psd[-1].max() + 1e-30)
    out_time_db = 10.0 * np.log10(intensity[-1] / intensity[-1].max() + 1e-30)

    return plot_scg_dashboard(
        z_axis=evo.z * 100.0,
        wl_nm=grid,
        spec_db=spec_db,
        t_ps=t_ps,
        int_db=int_db,
        out_wl_nm=wl,
        out_spec_db=out_spec_db,
        out_t_ps=t_ps,
        out_int_db=out_time_db,
        spec_labels=spec_labels,
        t_labels=t_labels,
        title=(
            f"Fig. 3 — basic SCG (N={result['N']:.2f}, "
            f"z_sol={result['z_sol_cm']:.1f} cm, "
            f"−20 dB span {result['span_minus20dB_nm'][0]:.0f}-"
            f"{result['span_minus20dB_nm'][1]:.0f} nm)"
        ),
        z_label="Distance (cm)",
        wl_bounds=(wl_lo, wl_hi),
        t_bounds=t_bounds,
        dynamic_range_db=dynamic_range_db,
        annotate_features=annotate_features,
    )


def write_html(
    evo: Evolution,
    result: dict,
    path: Path | None = None,
    **kwargs,
) -> Path:
    """Write the interactive Fig. 3 dashboard to an HTML file.

    With no *path*, the document is written next to this script as
    ``fig03_basic_scg.html``.  Extra keyword arguments are forwarded to
    :func:`_plot_plotly` (``wl_bounds``, ``t_bounds``, ``dynamic_range_db``,
    ``annotate_features``).
    """
    if path is None:
        path = HERE / "fig03_basic_scg.html"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = _plot_plotly(evo, result, **kwargs)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out}")
    return out


def main() -> None:
    result = validate(fast=False)
    print("Fig. 3 (basic SCG) reproduction passed:")
    for key, value in result.items():
        print(f"  {key} = {value}")
    write_html(run(fast=False), result)


if __name__ == "__main__":
    main()
