"""Paper-reproduction regression tests.

Each reproduction lives under ``reproductions/<name>/`` with a
``parameters.json`` and a ``reproduce.py::validate`` that asserts its result
against an analytic/closed-form reference.
"""

import numpy as np
import pytest

from reproductions.dudley_2006_cherenkov_dw.reproduce import validate as validate_dw
from reproductions.dudley_2006_scg.fig03_basic_scg import validate as validate_fig03
from reproductions.dudley_2006_scg.fig04_output_features import (
    validate as validate_fig04,
)
from reproductions.dudley_2006_scg.fig05_ideal_soliton_period import (
    validate as validate_fig05,
)
from reproductions.dudley_2006_scg.fig06_raman_fission import validate as validate_fig06
from reproductions.dudley_2006_scg.fig07_fission_detail import (
    validate as validate_fig07,
)
from reproductions.dudley_2006_scg.fig08_dispersive_wave import (
    validate as validate_fig08,
)
from reproductions.dudley_2006_scg.fig10_spectrogram import validate as validate_fig10
from reproductions.dudley_2006_scg.fig23_mi_gain import validate as validate_fig23
from reproductions.gordon_1986_ssfs.reproduce import validate as validate_gordon
from reproductions.kuznetsov_ma_2012_breather.reproduce import validate as validate_km
from reproductions.macleod_quarter_wave_dbr.reproduce import validate as validate_dbr
from reproductions.narhi_2016_mi_breathers.reproduce import validate as validate_narhi
from reproductions.shg_textbook.reproduce import validate as validate_shg
from reproductions.stolen_lin_1978_spm.reproduce import validate as validate_spm
from reproductions.tomlinson_1985_wave_breaking.reproduce import validate as validate_wb


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
    assert (
        abs(result["lambda_root_nm"] - result["lambda_analytic_nm"])
        / result["lambda_analytic_nm"]
        < 0.02
    )
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


def test_shg_textbook():
    """Textbook SHG eta = tanh^2(kappa L) and first-order QPM recovery."""
    result = validate_shg(make_plot=False)
    assert result["max_rel_error"] < 0.01
    assert result["eta_off_qpm"] < 0.01
    assert result["qpm_rel_error"] < 0.01
    assert result["eta_on_qpm"] > 0.1


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
    """Raman solitons sit at POSITIVE delay in the engine's internal frame.

    Post-#0 (closed 2026-09-30) the internal time grid already matches the
    literature convention: the red-shifted soliton is slower (beta2 < 0 ->
    dbeta1/dlambda > 0) and drifts to later arrival, exactly as Dudley et
    al. (2006) Fig. 3(b) shows (soliton trail 0 -> ~4 ps over 15 cm).
    The historical time_reversal compensation is retired (default False).
    """
    from reproductions.dudley_2006_scg import common, fig03_basic_scg

    evo = fig03_basic_scg.run(fast=True)
    t_internal, i_internal = common.temporal_evolution_data(evo, time_reversal=False)

    assert t_internal[np.argmax(i_internal[-1])] > 0.0
    # The legacy flip (back-compat flag) still mirrors about zero.
    t_literature, i_literature = common.temporal_evolution_data(evo, time_reversal=True)
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


def test_gordon_ssfs():
    """Gordon SSFS: measured redshift vs the paper's law (direction + ratio)."""
    result = validate_gordon(make_plot=False)
    assert result["measured_nm"] > 0.0
    assert 0.5 <= result["ratio"] <= 2.0


from reproductions.wright_2015_self_organized_instability.reproduce import (
    validate as validate_wright,
)


@pytest.mark.slow
def test_wright_2015_self_organized_instability():
    """Wright-2015 STMI: check B' at the corrected root (heavy: ~15 min)."""
    result = validate_wright(fast=True, make_plot=False)
    assert "check_b_deterministic" in result


from reproductions.dw_timing_gas_hollowcore.reproduce import (
    validate as validate_dw_timing,
    validate_stats as validate_dw_timing_stats,
)


def test_dw_timing_gas_hollowcore_fast():
    """Brahms & Travers 2021 analytic tier (Boerzsoenyi He, Marcatili
    transmission, beta2/ZDW anchors, Eq. 12 fission length). Seconds."""
    r = validate_dw_timing(fast=True, make_plot=False)
    assert r["gamma_mode_factor"] > 2.0
    assert 0.85 < r["transmission_1m"] / 100 < 0.9


def test_dw_timing_gas_hollowcore_stats():
    """Fig. 5/6 jitter statistics over the completed scan decks. Reads
    stats_fig5.json (produced once by stats_fig56.py, ~8 min)."""
    r = validate_dw_timing_stats()
    assert r["all_decks"] == 10
    for k, v in r.items():
        if not isinstance(v, dict):
            continue
        assert v["tau_rho_energy"] > 0.5, (k, v)


from reproductions.renninger_wise_2013_grin_solitons.reproduce import (
    build_engine as build_engine_rw,
    mode_walkoffs as mode_walkoffs_rw,
    BETA2 as BETA2_RW,
    LAMBDA0 as LAMBDA0_RW,
    C_MS as C_MS_RW,
    OMEGA0 as OMEGA0_RW,
)


def test_renninger_wise_higher_mode_blue_shift():
    """#0 closure check (2026-09-30): under the current convention the
    higher-mode locked carriers are BLUE-shifted relative to the
    fundamental, as the paper's Fig. 2c kinematics require
    (dw = -db1/beta2 > 0). Light: nonlinear 52 m run, ~6 s."""
    import numpy as np

    from reproductions.renninger_wise_2013_grin_solitons.reproduce import run

    eng = run(True)
    fields = eng.fields_vs_z()
    grid = eng.grid
    lam = 2 * np.pi * C_MS_RW / (OMEGA0_RW + grid.w) * 1e9
    centres = []
    for p in range(3):
        s = np.abs(grid.fft(fields[p][-1])) ** 2
        centres.append(float(np.sum(lam * s) / np.sum(s)))
    rel = [centres[1] - centres[0], centres[2] - centres[0]]
    assert rel[0] < 0 and rel[1] < 0, rel
    # magnitude within a factor ~2 of the kinematic requirement
    req = (
        -(-mode_walkoffs_rw()[1:] / BETA2_RW)
        * LAMBDA0_RW**2
        / (2 * np.pi * C_MS_RW)
        * 1e9
    )
    assert abs(rel[0]) < 2.5 * abs(req[0]) and abs(rel[0]) > 0.3 * abs(req[0])
    assert abs(rel[1]) < 2.5 * abs(req[1]) and abs(rel[1]) > 0.3 * abs(req[1])


from reproductions.kibler_2010_peregrine.reproduce import (
    validate as validate_kibler,
)


def test_kibler_2010_peregrine():
    """Kibler 2010 Peregrine: AB evolution through the engine matches the
    analytic breather (growth-leg tracking + peak/FWHM/train-period
    anchors + Peregrine-limit profile). ~3 min."""
    r = validate_kibler(make_plot=False)
    assert r["growth_leg_max_rel_l2"] < 0.08
    assert (
        abs(r["deck_peak_ratio"] - r["deck_peak_theory"]) / r["deck_peak_theory"] < 0.15
    )
    assert r["peregrine_profile_l2"] < 0.1


from reproductions.huang_202x_pcgnlse_attractors.diagnostics.check_dark_layer import (
    check_ansatz as check_huang_dark_ansatz,
)


def test_huang_202x_dark_layer_structural():
    """Huang pcGNLSE dark-soliton structural identities (S47/S48, S104/S105,
    S102) hold to 1e-9 on the ansatz — the 2026-09-30 double-Γ fix in
    `dark_moments` regression. Seconds."""
    for row in check_huang_dark_ansatz():
        assert row["E_err"] < 1e-8
        assert row["M_err"] < 1e-9
        assert row["Omega_tilde_err"] < 1e-9


from reproductions.dudley_2006_scg.fig20_coherence_evolution import (
    validate as validate_fig20,
)


@pytest.mark.slow
def test_dudley_fig20_coherence_evolution():
    """Dudley 2006 Fig. 20: coherence evolution along the fibre for the
    100/150 fs decks (ensemble of 20; heavy: ~15–20 min)."""
    r = validate_fig20(fast=False, make_plot=False)
    assert r["near_unity_bw_100fs_nm"] > r["near_unity_bw_150fs_nm"]


from reproductions.dudley_2006_scg.fig21_coherence_vs_pump import (
    validate as validate_fig21,
)
from reproductions.dudley_2006_scg.fig22_coherence_vs_N import (
    validate as validate_fig22,
)
from reproductions.dudley_2006_scg.fig28_picosecond_coherence import (
    validate as validate_fig28,
)


def test_dudley_fig21_coherence_vs_pump_fast():
    """Fig. 21 fast smoke: [0,1] bound + -20 dB width peaks on the
    anomalous/near-ZDW side. Narrative asserts recorded (measured
    beta(lambda) needed)."""
    r = validate_fig21(fast=True, make_plot=False)
    assert r["width_peak_wl"] >= 790.0


def test_dudley_fig22_coherence_vs_N_fast():
    """Fig. 22 fast smoke: coherence drops with soliton order."""
    r = validate_fig22(fast=True, make_plot=False)
    assert r["g12_lowN"] >= r["g12_highN"]


def test_dudley_fig28_picosecond_coherence_fast():
    """Fig. 28 fast smoke: near-coherent only at the pump, smooth mean
    spectrum."""
    r = validate_fig28(fast=True, make_plot=False)
    assert r["g12_pump_band"] > 0.9
    assert r["g12_far_windows"] < 0.5
    assert 0.0 <= r["smoothness_rel_structure"] < 0.2


def test_dudley_common_analysis_channel_mirror_audit():
    """ISSUES.md #0 addendum — close the raw-np.fft audit candidates in
    ``dudley_2006_scg/common.py`` (spectrogram helper + the sorted-wavelength
    interpolation path). A coherent tone whose envelope is
    ``exp(+i*Omega_m*t)`` must land at ``grid.w = -Omega_m`` (red under the
    ``c/(omega0+grid.w)`` map) through every analysis helper, exactly as it
    does through ``TemporalGrid.fft``.
    """
    import numpy as np

    from photonics_helper.base import Time
    from photonics_helper.pulse import TemporalGrid

    from reproductions.dudley_2006_scg import common

    grid = TemporalGrid(N=8192, Tmax=Time(50e-12, "s"))
    t = np.asarray(grid.t, dtype=float)
    omega = np.asarray(grid.w)
    omega0 = 2 * np.pi * 3e8 / 835e-9
    Omega_m = 2 * np.pi * 3e12  # physical tone at w = -Omega_m
    field = np.exp(1j * Omega_m * t)

    # reference: engine kernel
    ref = np.abs(np.asarray(grid.fft(field))) ** 2
    imax_ref = int(np.argmax(ref))
    assert omega[imax_ref] < 0  # tone is red under the lambda map

    # (a) spectrogram helper: each gated slice uses the engine kernel
    gate = np.ones_like(field)
    _, _, S = common.spectrogram(field, t, gate, omega, n_delays=3, delay_span_ps=1.0)
    imax = int(np.argmax(S[0]))
    assert float(omega[imax]) == pytest.approx(float(omega[imax_ref]), abs=1e-6)

    # (b) Evolution.spectra / spectrum_on_wavelength / the uniform-wavelength
    # interpolation path: the sorted-PSD pipeline must keep the tone red
    # (a negative frequency offset maps to a LONGER wavelength).
    evo = common.Evolution(
        z=np.array([0.0]), fields=field[None, :], t=t, omega=omega, omega0=omega0
    )
    wl, psd = evo.spectrum_on_wavelength()
    assert float(wl[int(np.argmax(psd[0]))]) > 835.0  # red of the carrier
    grid_wl, resampled = common.interpolate_on_wavelength(wl, psd[:1], 700.0, 900.0)
    assert float(grid_wl[int(np.argmax(resampled[0]))]) > 835.0


from reproductions.krupa_2019_multimode.reproduce import (
    validate as validate_krupa,
)


@pytest.mark.slow
def test_krupa_2019_gpi_ladder():
    """Krupa 2019 GPI sideband ladder (APL Photonics 4, 110901). Heavy on
    first run (~35 min propagation, file-cached by case key in the folder);
    with warm caches the suite revalidation is ~seconds."""
    r = validate_krupa(make_plot=False)
    assert abs(r["xi_mm"] - 0.6157) < 0.01
    assert r["energy_drift"] < 0.10
    assert r["control_peaks_in_gpi_windows"] == 0
    rels = [v for v in r["rel_err"] if v is not None]
    assert len(rels) >= 5 and max(rels[:5]) < 0.02


def test_dw_timing_beta1_arrival_helper():
    """β₁-aware in-engine absolute-arrival readout (arrival_beta1 module,
    shipped as the ISSUES.md #0 addendum follow-up). Machine-precision vs
    the closed form (L-z_f)*(beta1(w_rdw)-beta1(w0)); zf monotone."""
    import numpy as np

    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "ab", "reproductions/dw_timing_gas_hollowcore/arrival_beta1.py"
    )
    ab = ilu.module_from_spec(spec)
    spec.loader.exec_module(ab)
    rep = ab._load_rep()

    w0 = ab.omega0_from_lambda(800.0)
    wr = ab.omega0_from_lambda(559.7)

    def b1(om):
        return rep.differentiate_beta(np.asarray(om), 2.1)[1]

    got = ab.absolute_arrival_time_fs(
        lambda zf: lambda om: b1(om),
        length_m=1.0,
        z_fission_m=0.4,
        omega0=w0,
        omega_rdw=wr,
    )
    _, b1r, _ = rep.differentiate_beta(np.array([wr]), 2.1)
    _, b1_0, _ = rep.differentiate_beta(np.array([w0]), 2.1)
    ana = 0.6 * (b1r[0] - b1_0[0]) * 1e15
    assert abs(got - ana) < 1e-9 * max(abs(ana), 1.0), (got, ana)
    ta = [
        ab.absolute_arrival_time_fs(
            lambda zf: lambda om: b1(om),
            length_m=1.0,
            z_fission_m=z,
            omega0=w0,
            omega_rdw=wr,
        )
        for z in (0.05, 0.6)
    ]
    # walk-off leg shrinks -> |tau| shrinks monotonically toward 0
    assert abs(ta[0]) > abs(ta[1]) and np.sign(ta[0]) == np.sign(ta[1])


from reproductions.heidt_2009_adaptive_step.reproduce import (
    validate as validate_heidt,
)


@pytest.mark.slow
def test_heidt_2009_adaptive_step_full():
    """Heidt 2009 adaptive step size (JLT 27, 3984) — full 10 cm / 400 km
    run. ~5 min with warm caches (all ladders re-run; only the two reference
    fields are file-cached). Asserts the paper's headline claims: Fig. 2
    efficiency (RK4IP-CQE 0.34x constant, 0.70x local), Fig. 3(b) collision
    step collapse + recovery, CQE-needs-conserved-quantity finding."""
    r = validate_heidt(fast=False)
    failing = [k for k, v in r["asserts"].items() if not v["ok"]]
    assert not failing, failing


def test_heidt_2009_integrator_layer():
    """Cheap guard for the same folder: RK4IP must integrate a linear flow
    exactly and measure orders 4 / 2 against the paper's eta = 5 / 3."""
    r = validate_heidt(fast=True)
    ig = r["integrators"]
    assert ig["rk4ip_linear_rel_error"] < 1e-10
    assert abs(ig["orders"]["rk4ip"] - 4) < 0.6
    assert abs(ig["orders"]["ssf"] - 2) < 0.6


from reproductions.poletti_2008_multimode.reproduce import (
    validate as validate_poletti,
)


def test_poletti_2008_overlap_selection_rules():
    """Poletti & Horak 2008 (JOSA B 25, 1645) Sec. 4.A — the Eq. (7) overlap
    coefficients of the Fig. 1 step-index fibre (V = 4.2726, ten modes).

    Asserts the paper's symmetry results, all of which *emerge* from the
    transverse quadrature rather than being hard-coded: the Eq. (18) spatial
    selection rules (1360 survivors per type, forbidden entries at round-off),
    the Eq. (19) polarisation rules (1360 -> 340, the discarded 1020 carrying
    |Q| up to 1.0), the eleven Eq. (16) permutation identities, the Sec. 3
    statement Q^(1) = Q^(2) for real-valued mode functions (exact in the real
    LP basis, false in the helical Eq.-17 basis), the Sec. 5 complexity saving,
    and the Eq. (14) polarisation closure on the engine."""
    r = validate_poletti(make_plot=False)

    m = r["selection_rules"]["measured"]
    assert m["type_1"]["matched_spatial"] and m["type_2"]["matched_spatial"]
    assert m["type_1"]["max_abs_violating_spatial"] < 1e-12
    assert m["type_1"]["n_dropped_by_pol_rule"] == 1020
    assert m["type_1"]["max_abs_dropped_by_pol_rule"] > 0.5
    assert r["selection_rules"]["masked_survivors"] == {"type_1": 340, "type_2": 340}
    assert r["selection_rules"]["max_abs_cross_polarisation_Q2"] == 0.0

    assert r["symmetries"]["eq16_worst"] < 1e-12
    assert r["symmetries"]["real_basis_equality"]["max_abs_diff"] < 1e-12
    assert r["symmetries"]["helical_max_abs_diff"] > 1e-3

    f = r["fig1_distribution"]
    assert f["n_total"]["all"] == 20000
    assert f["n_zero"]["all"] == 17280
    assert f["n_nonzero"]["after_rule_18_19"] == 680

    assert r["complexity_sec5"]["fit_exponent"] > 3.0
    assert min(r["complexity_sec5"]["reduction_factor"]) > 20.0

    e = r["engine"]
    assert e["run_a_10modes_xpm"]["leak_fraction"] == 0.0
    assert e["run_b_4modes_fwm"]["leak_fraction"] == 0.0
    assert e["run_a_10modes_xpm"]["energy_drift"] < 1e-6
    # the isotropic control must leak, or the closure check would be vacuous
    assert e["control_isotropic"]["leak_fraction"] > 1e-6
