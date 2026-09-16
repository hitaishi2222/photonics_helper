"""Paper-reproduction regression tests.

Each reproduction lives under ``reproductions/<name>/`` with a
``parameters.json`` and a ``reproduce.py::validate`` that asserts its result
against an analytic/closed-form reference.
"""

import numpy as np
import pytest

from reproductions.macleod_quarter_wave_dbr.reproduce import validate as validate_dbr
from reproductions.stolen_lin_1978_spm.reproduce import validate as validate_spm
from reproductions.gordon_1986_ssfs.reproduce import validate as validate_gordon
from reproductions.dudley_2006_cherenkov_dw.reproduce import validate as validate_dw
from reproductions.kuznetsov_ma_2012_breather.reproduce import validate as validate_km
from reproductions.narhi_2016_mi_breathers.reproduce import validate as validate_narhi
from reproductions.tomlinson_1985_wave_breaking.reproduce import validate as validate_wb

from reproductions.dudley_2006_scg.fig03_basic_scg import validate as validate_fig03
from reproductions.dudley_2006_scg.fig04_output_features import validate as validate_fig04
from reproductions.dudley_2006_scg.fig05_ideal_soliton_period import (
    validate as validate_fig05,
)
from reproductions.dudley_2006_scg.fig06_raman_fission import validate as validate_fig06
from reproductions.dudley_2006_scg.fig07_fission_detail import validate as validate_fig07
from reproductions.dudley_2006_scg.fig08_dispersive_wave import validate as validate_fig08
from reproductions.dudley_2006_scg.fig10_spectrogram import validate as validate_fig10
from reproductions.dudley_2006_scg.fig23_mi_gain import validate as validate_fig23


def test_stolen_lin_1978_spm():
    """SPM spectrum matches the closed-form Fourier integral and the phi/pi peak rule."""
    result = validate_spm(make_plot=False)
    assert result["max_abs_spectrum_diff"] < 1e-6
    for case in result["cases"]:
        assert case["n_peaks"] == case["n_peaks_expected"]


def test_macleod_quarter_wave_dbr():
    """Quarter-wave DBR reflectance matches the exact characteristic-matrix closed form."""
    result = validate_dbr(make_plot=False)
    assert result["max_peak_error"] < 1e-6
    assert result["R_lambda0"] > 0.999
    assert result["width_rel_error"] < 0.15


def test_gordon_1986_ssfs():
    """Raman soliton self-frequency shift matches the Gordon analytic rate."""
    result = validate_gordon(make_plot=False)
    assert result["measured_nm"] > 0.0
    assert 0.5 <= result["ratio"] <= 2.0


def test_dudley_2006_cherenkov_dw():
    """Dispersive-wave root finder and GNLSE DW peak match the analytic -3β₂/β₃ value."""
    result = validate_dw(make_plot=False)
    assert abs(result["lambda_root_nm"] - result["lambda_analytic_nm"]) / result["lambda_analytic_nm"] < 0.02
    assert result["rel_err"] < 0.05


def test_kuznetsov_ma_2012_breather():
    """Exact Kuznetsov-Ma solution is reproduced by the GNLSE engine over one period."""
    result = validate_km(num_steps=4000, make_plot=False)
    assert abs(result["T0_ps"] - 4.894) < 0.02
    assert abs(result["period_km"] - 5.312) / 5.312 < 0.01
    assert result["peak_rel_err"] < 5e-3
    assert result["center_max_abs_err_W"] < 0.05
    assert result["intensity_rel_l2_max"] < 1e-2
    assert result["spectrum_rel_l2"] < 1e-2
    assert result["lossy_peak_W"] < result["peak_analytic_W"]


def test_narhi_2016_mi_breathers():
    """MI gain, exact Peregrine/Akhmediev breathers, and noise-seeded MI."""
    result = validate_narhi(fast=True, make_plot=False)
    assert abs(result["mi"]["omega_peak_GHz"] - 46.4) < 0.5
    assert 0.75 < result["mi_growth"]["ratio"] < 1.25
    assert result["peregrine"]["peak_rel_err"] < 0.01
    assert result["peregrine"]["profile_l2"] < 0.02
    assert result["akhmediev"]["peak_rel_err"] < 0.01
    assert result["akhmediev"]["profile_l2"] < 5e-3
    assert 30.0 <= result["spontaneous"]["sideband_peak_GHz"] <= 65.0
    assert result["spontaneous"]["max_peak_ratio"] > 4.0


def test_tomlinson_1985_wave_breaking():
    """Optical wave breaking: onset near z_WB and z ~ sqrt(L_D L_NL) ~ P0^-1/2 scaling."""
    result = validate_wb(fast=True, make_plot=False)
    assert 0.3 < result["z_onset_over_zWB"] < 1.5
    assert result["peak_steepness_over_gaussian"] > 1.5
    assert result["z_oscillation_m"] / result["sqrt_LD_LNL_m"] < 4.5
    assert -0.65 < result["scaling_slope"] < -0.35
    assert result["scaling_constant_spread"] < 1.35


# ---------------------------------------------------------------------------
# Dudley, Genty & Coen, Rev. Mod. Phys. 78, 1135 (2006) — figure reproductions
# ---------------------------------------------------------------------------


def test_dudley_fig05_ideal_soliton_period():
    """Ideal N=3 soliton is periodic with z_sol and breathes to >3x peak power."""
    result = validate_fig05(fast=True, make_plot=False)
    assert result["periodicity_overlap"] > 0.99
    assert result["peak_compression"] > 3.0
    assert abs(result["z_sol_cm"] - 10.7) < 0.5


def test_dudley_fig06_raman_fission():
    """Raman fission ejects a Kodama-Hasegawa j=1 soliton and red-shifts."""
    result = validate_fig06(fast=True, make_plot=False)
    assert result["n_ejected"] >= 2
    assert result["power_rel_err"] < 0.15
    assert result["fwhm_rel_err"] < 0.20
    assert result["mean_wavelength_end_nm"] > 1000.0


def test_dudley_fig07_fission_detail():
    """Ejected soliton matches the Kodama-Hasegawa sech profile."""
    result = validate_fig07(fast=True, make_plot=False)
    assert result["sech_overlap"] > 0.97
    assert result["power_rel_err"] < 0.15


def test_dudley_fig08_dispersive_wave():
    """Blue DW wavelength matches the full-beta phase-matching root."""
    result = validate_fig08(fast=True, make_plot=False)
    assert result["dw_relative_power"] > 0.05
    assert result["dw_rel_err"] < 0.06


def test_dudley_fig23_mi_gain():
    """MI gain: ZDW 780 nm, g_max = 2 gamma P, anomalous peak frequency."""
    result = validate_fig23(make_plot=False)
    assert abs(result["zdw_nm"] - 780.0) < 5.0
    assert abs(result["g_max_800_W_per_m"] - result["g_max_theory_W_per_m"]) < 1e-2
    assert result["peak_rel_err"] < 0.15
    assert result["omega_peak_750_THz"] > result["omega_peak_800_THz"]


def test_dudley_fig03_basic_scg():
    """Full SCG is octave-spanning at -20 dB with the paper's N and z_sol."""
    result = validate_fig03(fast=True, make_plot=False)
    assert abs(result["z_sol_cm"] - 10.7) < 0.5
    assert result["span_ratio"] > 1.8
    lo, hi = result["span_minus20dB_nm"]
    assert lo < 650.0 and hi > 1000.0


def test_dudley_fig04_output_features():
    """Output has a blue DW, a red Raman soliton and multiple temporal peaks."""
    result = validate_fig04(fast=True, make_plot=False)
    assert result["dw_peak_nm"] < 650.0
    assert result["soliton_peak_nm"] > 850.0
    assert result["n_temporal_peaks"] >= 3


def test_dudley_fig10_spectrogram():
    """Spectrogram resolves the DW and Raman-soliton bands at different delays."""
    result = validate_fig10(fast=True, make_plot=False)
    assert result["delay_separation_ps"] > 0.1
    assert result["dominant_beat_THz"] > 0.0


# ---------------------------------------------------------------------------
# Dudley Fig. 3 plotting helpers: time convention, wavelength bounds, backends
# ---------------------------------------------------------------------------


def test_dudley_default_wl_bounds_track_carrier():
    """Default spectral window is centred on the carrier, not hard-coded."""
    from reproductions.dudley_2006_scg import common

    lo, hi = common.default_wl_bounds(835.0)
    assert lo < 835.0 < hi
    assert lo == pytest.approx(417.5)
    assert hi == pytest.approx(1336.0)
    # A tighter fraction still brackets the carrier.
    lo2, hi2 = common.default_wl_bounds(835.0, (0.8, 1.2))
    assert lo2 < 835.0 < hi2
    with pytest.raises(ValueError):
        common.default_wl_bounds(835.0, (1.5, 0.5))


def test_dudley_temporal_reversal_lifts_soliton_right():
    """Raman solitons sit at positive delay in the literature convention."""
    from reproductions.dudley_2006_scg import common, fig03_basic_scg

    evo = fig03_basic_scg.run(fast=True)
    t_internal, i_internal = common.temporal_evolution_data(evo, time_reversal=False)
    t_literature, i_literature = common.temporal_evolution_data(evo, time_reversal=True)

    assert t_internal[np.argmax(i_internal[-1])] < 0.0
    assert t_literature[np.argmax(i_literature[-1])] > 0.0
    # Centre of mass mirrors about zero.
    assert np.sum(t_internal * i_internal) == pytest.approx(
        -np.sum(t_literature * i_literature)
    )


def test_dudley_evolution_plots_return_figures():
    """Both evolution helpers return the figure instead of closing it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reproductions.dudley_2006_scg import common, fig03_basic_scg

    evo = fig03_basic_scg.run(fast=True)
    fig_s = common.plot_spectral_evolution(evo, wl_bounds=(450.0, 1200.0))
    fig_t = common.plot_temporal_evolution(evo, t_bounds=(-1.0, 5.0))
    assert isinstance(fig_s, plt.Figure)
    assert isinstance(fig_t, plt.Figure)
    assert tuple(fig_s.axes[0].get_xlim()) == (450.0, 1200.0)
    assert tuple(fig_t.axes[0].get_xlim()) == (-1.0, 5.0)
    plt.close(fig_s)
    plt.close(fig_t)


def test_dudley_plotly_hover_labels_features():
    """plotly=True yields hover customdata naming DW / SPM / soliton."""
    go = pytest.importorskip("plotly.graph_objects")

    from reproductions.dudley_2006_scg import common, fig03_basic_scg

    evo = fig03_basic_scg.run(fast=True)
    fig = common.plot_temporal_evolution(evo, plotly=True, t_bounds=(-1.0, 5.0))
    assert isinstance(fig, go.Figure)
    labels = set(np.asarray(fig.data[0].customdata).ravel().tolist())
    assert "Raman soliton (red)" in labels
    assert "SPM / pump" in labels
    assert "Dispersive wave (blue)" in labels
    assert "Feature: %{customdata}" in fig.data[0].hovertemplate


def test_dudley_fig03_plot_returns_figure():
    """The assembled Fig. 3 returns a figure for both backends."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reproductions.dudley_2006_scg import fig03_basic_scg

    result = validate_fig03(fast=True, make_plot=False)
    evo = fig03_basic_scg.run(fast=True)
    mpl_fig = fig03_basic_scg._plot(evo, result, save=False)
    assert isinstance(mpl_fig, plt.Figure)
    plt.close(mpl_fig)

    go = pytest.importorskip("plotly.graph_objects")
    plotly_fig = fig03_basic_scg._plot(evo, result, plotly=True, save=False)
    assert isinstance(plotly_fig, go.Figure)
