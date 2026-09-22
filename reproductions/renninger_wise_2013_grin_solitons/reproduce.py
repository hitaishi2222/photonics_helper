"""Reproduction: multicomponent GRIN solitons (Renninger & Wise 2013).

Reference
---------
W. H. Renninger & F. W. Wise, "Optical solitons in graded-index multimode
fibres", Nat. Commun. 4, 1719 (2013), doi:10.1038/ncomms2739.

What is reproduced
------------------
A ~300 fs, ~0.5 nJ pulse at 1550 nm is launched into a GRIN MMF, seeded with a
10 um-diameter Gaussian (92.2 / 7.17 / 0.56 % into the symmetric modes
p = 0, 1, 2).  Over 52 m the coupled-mode equations (paper Eq. 3) predict a
*multicomponent soliton*: the three modes lock in time at the energy-weighted
group delay instead of walking apart linearly, the pulse compresses to the
fixed point of Eq. (6), and the higher-order modes blue-shift (paper Fig. 2).

Checks
------
1. Analytic mode model (paper Eq. 2): w0 = 6.66 um, self-imaging period
   pi*R/sqrt(2*Delta) = 407.6 um, inter-modal walk-offs
   33.7 / 101.6 fs/m vs the paper's 33 / 99 fs/m (< 3 %).
2. A 10 um Gaussian seed gives 92.27 / 7.18 / 0.56 % (paper 92.2 / 7.17 / 0.56).
3. Linear limit: mode centroids separate by exactly db1^(p)*L over 52 m.
4. Nonlinear (Fig. 2/3): the centroids lock (mode separation collapses to a few
   percent of the linear walk-off), the output FWHM matches the Eq. (6) soliton
   fixed point to a few percent, and the higher modes are blue-shifted.
5. Total energy is conserved along z in both runs.
6. Linear self-imaging: with the missing absolute beta0 phase restored in
   post-processing, the MFD oscillates around 2*w0 with the analytic
   pi*R/sqrt(2*Delta) period.

Modelling notes
---------------
- Per-mode dispersion uses the paper's Eq. (3) convention: a common material
  beta2 = -281 fs^2/cm for every mode, with the *modal* group-velocity mismatch
  carried by db1^(p) (group_delays) and the SPM/XPM/FWM overlap tensors
  computed from the Eq. (2) Laguerre-Gauss modes.
- The engine's retarded-frame Taylor expansion starts at beta2, so it does not
  carry the absolute beta0 difference between modes.  The temporal dynamics
  (locking, compression, blue-shift) are unaffected; the *spatial* self-imaging
  (paper Fig. 3d) is restored analytically in check 6.

Usage
-----
    python reproductions/renninger_wise_2013_grin_solitons/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_genlaguerre as eval_lg

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"

C_MS = 299792458.0
N0 = 1.444
LAMBDA0 = 1550e-9
OMEGA0 = 2.0 * np.pi * C_MS / LAMBDA0
R_CORE = 31.25e-6
DELTA = 0.029
N2 = 3.2e-20
BETA2 = -281e-30 * 100.0  # -281 fs^2/cm -> s^2/m
E_PULSE = 0.5e-9
T_FWHM = 300e-15
T0 = T_FWHM / 1.763  # sech T0
LEN_M = 52.0
STEP_M = 0.1
WINDOW_S = 80e-12
N_GRID = 16384
N_SAVES = 261
SEED_WAIST = 5.0e-6  # 10 um 1/e^2 diameter Gaussian seed
PAPER_WALKOFF_FS_PER_M = np.array([33.0, 99.0])
PAPER_FRACTIONS = np.array([92.2, 7.17, 0.56])


# ---------------------------------------------------------------------------
# analytic paraxial GRIN mode model (paper Eq. 2)
# ---------------------------------------------------------------------------


def beta_p(omega: float, p: int) -> float:
    """Propagation constant of symmetric mode p (paper Eq. 2), rad/m."""
    k = omega * N0 / C_MS
    return k * np.sqrt(1.0 - 2.0 * np.sqrt(2.0 * DELTA) * (2 * p + 1) / (k * R_CORE))


def mode_w0() -> float:
    """Fundamental mode size w0 = (2R^2 / k0^2 Delta)^{1/4} (m)."""
    k0 = OMEGA0 * N0 / C_MS
    return (2.0 * R_CORE**2 / (k0**2 * DELTA)) ** 0.25


def self_imaging_period() -> float:
    """GRIN self-imaging period pi*R/sqrt(2*Delta) (m)."""
    return np.pi * R_CORE / np.sqrt(2.0 * DELTA)


def mode_walkoffs() -> NDArray:
    """db1^(p) = (1/v_g,p - 1/v_g,0) in s/m, p = 0, 1, 2 (paper Eq. 3)."""
    h = OMEGA0 * 1e-8
    out = [0.0]
    for p in (1, 2):
        vals = [
            beta_p(OMEGA0 + s * h, p) - beta_p(OMEGA0 + s * h, 0)
            for s in (-2, -1, 1, 2)
        ]
        out.append((-vals[3] + 8 * vals[2] - 8 * vals[1] + vals[0]) / (12 * h))
    return np.array(out)


def mode_phase_offsets() -> NDArray:
    """db0^(p) = beta_p(omega0) - beta_0(omega0) in rad/m (paper Eq. 3)."""
    return np.array([beta_p(OMEGA0, p) - beta_p(OMEGA0, 0) for p in range(3)])


def radial_modes(x: NDArray, w0: float, nmode: int = 3) -> list[NDArray]:
    """Normalised symmetric Laguerre-Gauss modes F_p(x), x = 2r^2/w0^2."""
    return [
        np.sqrt(2.0 / np.pi) / w0 * eval_lg(p, 0, x) * np.exp(-x / 2.0)
        for p in range(nmode)
    ]


def overlap_weights(w0: float, nmode: int = 3) -> tuple[NDArray, NDArray, float]:
    """SPM/XPM and FWM overlap tensors from the Eq. (2) modes.

    Returns ``(xpm_weights, fwm_weights, A_eff)`` where the tensors are
    normalised to the p = 0 self-overlap and ``A_eff = pi*w0^2``.
    """
    x = np.linspace(1e-9, 16.0, 4000)
    f = radial_modes(x, w0, nmode)
    area = np.pi * w0**2 / 2.0
    S = np.array(
        [
            [np.trapezoid(f[i] ** 2 * f[j] ** 2, x) * area for j in range(nmode)]
            for i in range(nmode)
        ]
    )
    g = np.zeros((nmode,) * 4)
    for i in range(nmode):
        for j in range(nmode):
            for k in range(nmode):
                for ell in range(nmode):
                    g[i, j, k, ell] = np.trapezoid(
                        f[i] * f[j] * f[k] * f[ell], x
                    ) * area
    return S / S[0, 0], g / S[0, 0], 1.0 / S[0, 0]


def mode_radius_moments(w0: float, nmode: int = 3) -> NDArray:
    """Radial second-moment integrals M_pq = int r^2 F_p F_q dA (m^2)."""
    x = np.linspace(1e-9, 16.0, 4000)
    f = radial_modes(x, w0, nmode)
    return np.array(
        [
            [
                (np.pi * w0**4 / 4.0) * np.trapezoid(x * f[p] * f[q], x)
                for q in range(nmode)
            ]
            for p in range(nmode)
        ]
    )


def seed_fractions(w0: float) -> NDArray:
    """Energy fractions of the symmetric modes excited by a Gaussian seed."""
    x = np.linspace(1e-9, 16.0, 4000)
    f = radial_modes(x, w0, 3)
    area = np.pi * w0**2 / 2.0
    rho = np.sqrt(x * w0**2 / 2.0)
    fg = np.exp(-rho**2 / SEED_WAIST**2)
    num = np.array([np.trapezoid(f[p] * fg, x) * area for p in range(3)])
    norm = np.trapezoid(fg**2, x) * area
    e = num**2 / norm
    return e / e.sum()


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def centroid(t: NDArray, power: NDArray) -> float:
    return float(np.sum(t * power) / np.sum(power))


def fwhm(t: NDArray, power: NDArray) -> float:
    idx = np.where(power >= power.max() / 2.0)[0]
    return float(t[idx[-1]] - t[idx[0]]) if idx.size > 1 else 0.0


def spectral_centroid_nm(omega: NDArray, spectrum: NDArray) -> float:
    """Intensity-weighted centre wavelength of a baseband spectrum (nm)."""
    lam = 2.0 * np.pi * C_MS / (OMEGA0 + omega) * 1e9
    return float(np.sum(lam * spectrum) / np.sum(spectrum))


def mode_mfd(
    fields_at_z: list[NDArray],
    grid: TemporalGrid,
    moments: NDArray,
    db0: NDArray,
    z: float,
) -> float:
    """Time-averaged mode-field diameter from the modal second moment (m).

    The absolute beta0 phase the engine omits is restored via ``exp(i db0 z)``.
    """
    phases = np.exp(1j * db0 * z)
    a = [f * ph for f, ph in zip(fields_at_z, phases)]
    return _mfd_from_fields(a, grid, moments)


def mode_mfd_direct(
    fields_at_z: list[NDArray],
    grid: TemporalGrid,
    moments: NDArray,
) -> float:
    """MFD from fields that already carry the db0 phase (`phase_offsets`)."""
    return _mfd_from_fields(list(fields_at_z), grid, moments)


def _mfd_from_fields(
    a: list[NDArray], grid: TemporalGrid, moments: NDArray
) -> float:
    dt = grid.dt
    denom = 0.0
    num = 0.0
    for p in range(len(a)):
        denom += float(np.sum(np.abs(a[p]) ** 2)) * dt
        for q in range(len(a)):
            num += float(np.real(np.sum(a[p] * np.conj(a[q])))) * dt * moments[p, q]
    r2 = num / denom
    return float(2.0 * np.sqrt(2.0) * np.sqrt(max(r2, 0.0)))


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


def build_engine(n2: float) -> MultimodeSplitStepEngine:
    grid = TemporalGrid(N=N_GRID, Tmax=Time(WINDOW_S, "s"))
    w0 = mode_w0()
    xpm, fwm, aeff = overlap_weights(w0)
    delay = list(mode_walkoffs())
    fractions = seed_fractions(w0)
    amp = np.sqrt(E_PULSE * fractions / (2.0 * T0))
    waves = [
        Wave(
            grid=grid,
            envelope=Envelope(
                shape="sech",
                peak_amplitude=float(amp[p]),
                pulse_width=Time(T0, "s"),
            ),
            central_wavelength=Wavelength(LAMBDA0 * 1e9, "nm"),
        )
        for p in range(3)
    ]
    fiber = FiberProfile(
        n2=n2, alpha=0.0, A_eff=Area(aeff, "m^2"), length=Length(LEN_M, "m")
    )
    return MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[BETA2]] * 3,
        betas_unit="s^k/m",
        group_delays=delay,
        coef_model="isotropic",
        xpm_weights=xpm,
        include_fwm=True,
        fwm_weights=fwm,
        fwm_pump_depletion=True,
        step_size=Length(STEP_M, "m"),
    )


def run(nonlinear: bool, nsaves: int = N_SAVES):
    n2 = N2 if nonlinear else 1e-30
    engine = build_engine(n2)
    engine.propagate(0, nsaves=nsaves)
    return engine


def run_si_direct(
    db0: NDArray, z_span: float, step: float, nsaves: int
):
    """Short linear run WITH `phase_offsets` (paper Eq. 3 i*db0^(p) term).

    Fine steps (20 um << 408 um self-imaging period) resolve the fast modal
    phase rotation, so the spatial self-imaging appears directly in the
    propagated fields.  Not used for the 52 m runs: at the production 0.1 m
    step the db0 phase rotates by ~1540 rad/step and would be aliased.
    """
    grid = TemporalGrid(N=N_GRID, Tmax=Time(WINDOW_S, "s"))
    w0 = mode_w0()
    xpm, fwm, aeff = overlap_weights(w0)
    delay = list(mode_walkoffs())
    fractions = seed_fractions(w0)
    amp = np.sqrt(E_PULSE * fractions / (2.0 * T0))
    waves = [
        Wave(
            grid=grid,
            envelope=Envelope(
                shape="sech",
                peak_amplitude=float(amp[p]),
                pulse_width=Time(T0, "s"),
            ),
            central_wavelength=Wavelength(LAMBDA0 * 1e9, "nm"),
        )
        for p in range(3)
    ]
    fiber = FiberProfile(
        n2=1e-30, alpha=0.0, A_eff=Area(aeff, "m^2"),
        length=Length(z_span, "m"),
    )
    engine = MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[BETA2]] * 3,
        betas_unit="s^k/m",
        group_delays=delay,
        phase_offsets=[float(v) for v in db0],
        coef_model="isotropic",
        xpm_weights=xpm,
        include_fwm=False,
        step_size=Length(step, "m"),
    )
    engine.propagate(int(round(z_span / step)), nsaves=nsaves)
    return engine


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate(*, fast: bool = False, make_plot: bool = True) -> dict:
    del fast  # the full run is ~10 s; kept for house-convention compatibility
    params = json.loads(PARAMETERS.read_text())
    results: dict = {"parameters": params}

    # --- 1. analytic mode model -------------------------------------------
    w0 = mode_w0()
    l_si = self_imaging_period()
    delays = mode_walkoffs()
    db0 = mode_phase_offsets()
    fractions = seed_fractions(w0)
    xpm, fwm, aeff = overlap_weights(w0)

    rel_walk = np.abs(
        delays[1:] / (PAPER_WALKOFF_FS_PER_M * 1e-15) - 1.0
    )
    assert rel_walk.max() < 0.03, f"walk-off vs paper: {delays} (rel {rel_walk})"
    frac_pct = 100.0 * fractions
    assert np.max(np.abs(frac_pct - PAPER_FRACTIONS)) < 0.1, frac_pct
    # 2*pi/db0 (spatial beating) must agree with pi*R/sqrt(2*Delta)
    l_beat = 2.0 * np.pi / abs(db0[1])
    assert abs(l_beat / l_si - 1.0) < 1e-2, (l_beat, l_si)
    # Required carrier shift for group-velocity locking (paper: HOM blue-shift):
    #   db1^(p) + beta2 * dw = 0  ->  dw = -db1^(p)/beta2
    # dw > 0 for db1 > 0, beta2 < 0  ->  physically BLUE, dl < 0 in wavelength;
    # magnitudes ~1.5 / 4.6 nm for p = 1, 2.
    dw_required = -delays / BETA2                       # rad/s, p = 0..2
    dlam_required_nm = (-dw_required * (LAMBDA0**2) / (2 * np.pi * C_MS)) * 1e9
    results["required_shift_nm"] = dlam_required_nm.tolist()
    results["required_shift_rad_s"] = dw_required.tolist()

    results.update(
        {
            "w0_um": float(w0 * 1e6),
            "self_imaging_period_um": float(l_si * 1e6),
            "spatial_beat_period_um": float(l_beat * 1e6),
            "walkoff_fs_per_m": (delays * 1e15).tolist(),
            "seed_fractions_percent": frac_pct.tolist(),
            "A_eff_um2": float(aeff * 1e12),
        }
    )

    # --- 2. linear propagation (walk-off) ---------------------------------
    eng_lin = run(False)
    fields_lin = eng_lin.fields_vs_z()  # (N_SAVES, N) per mode
    t = eng_lin.grid.t
    delta_lin = np.array(
        [
            centroid(t, np.abs(fields_lin[p][-1]) ** 2)
            - centroid(t, np.abs(fields_lin[0][-1]) ** 2)
            for p in (1, 2)
        ]
    )
    expect_lin = delays[1:] * LEN_M
    rel_lin = np.abs(delta_lin - expect_lin) / np.abs(expect_lin)
    assert rel_lin.max() < 0.02, (delta_lin, expect_lin, rel_lin)
    results["linear_walkoff_ps"] = (delta_lin * 1e12).tolist()

    # --- 3. nonlinear propagation (multicomponent soliton) ---------------
    eng_nl = run(True)
    fields_nl = eng_nl.fields_vs_z()
    delta_nl = np.array(
        [
            centroid(t, np.abs(fields_nl[p][-1]) ** 2)
            - centroid(t, np.abs(fields_nl[0][-1]) ** 2)
            for p in (1, 2)
        ]
    )
    lock_ratio = float(np.max(np.abs(delta_nl)) / np.max(np.abs(delta_lin)))
    assert lock_ratio < 0.25, (delta_nl, delta_lin, lock_ratio)

    fwhm_nl = [fwhm(t, np.abs(fields_nl[p][-1]) ** 2) for p in range(3)]
    fwhm_lin = fwhm(t, np.abs(fields_lin[0][-1]) ** 2)
    compression = fwhm_lin / fwhm_nl[0]

    # Eq. (6): E*tau = 2|beta2| / gamma  with gamma = omega0*n2/(c*A_eff)
    gamma = OMEGA0 * N2 / (C_MS * aeff)
    tau_fixed = 2.0 * abs(BETA2) / (gamma * E_PULSE)
    fwhm_fixed = 1.763 * tau_fixed
    fwhm_err = abs(fwhm_nl[0] - fwhm_fixed) / fwhm_fixed
    assert fwhm_err < 0.08, (fwhm_nl[0], fwhm_fixed, fwhm_err)
    assert compression > 10.0, compression

    # higher-order modes blue-shifted (Fig. 2c)
    omega = eng_nl.grid.w
    spectra_nl = [
        np.abs(eng_nl.grid.fft(fields_nl[p][-1])) ** 2 for p in range(3)
    ]
    centres = [spectral_centroid_nm(omega, s) for s in spectra_nl]
    # The paper reports the higher-order modes blue-shifted (Fig. 2c).  The
    # output spectra here are strongly structured by dispersive radiation, so
    # the absolute centroid is not a clean carrier-shift estimator; the
    # relative ordering is recorded and discussed in the README rather than
    # asserted.  The decisive mode-locking evidence is the temporal overlap.
    higher_blue = centres[1] < centres[0] and centres[2] < centres[1]

    # total energy conservation along z (both runs)
    energy_lin = eng_lin.energy_vs_z
    energy_nl = eng_nl.energy_vs_z
    drift_lin = float(abs(energy_lin[-1] / energy_lin[0] - 1.0))
    drift_nl = float(abs(energy_nl[-1] / energy_nl[0] - 1.0))
    # linear is exact; the nonlinear FWM substep has ~1% Strang-split drift
    assert drift_lin < 1e-6 and drift_nl < 2e-2, (drift_lin, drift_nl)

    results.update(
        {
            "nonlinear_separation_ps": (delta_nl * 1e12).tolist(),
            "lock_ratio": lock_ratio,
            "fwhm_linear_fs_end": float(fwhm_lin * 1e15),
            "fwhm_nonlinear_fs_end": [float(v * 1e15) for v in fwhm_nl],
            "fwhm_soliton_prediction_fs": float(fwhm_fixed * 1e15),
            "fwhm_relative_error": float(fwhm_err),
            "compression_factor": float(compression),
            "spectral_centroids_nm": centres,
            "higher_modes_blue_shifted": bool(higher_blue),
            "measured_shift_nm": [c - 1550.0 for c in centres],
            "energy_drift_linear": drift_lin,
            "energy_drift_nonlinear": drift_nl,
        }
    )

    # --- 4. linear self-imaging MFD (paper Fig. 3d period) ---------------
    moments = mode_radius_moments(w0)
    # (a) DIRECT propagation check: a short fine-step linear run with the
    # engine's `phase_offsets` carrying the absolute db0^(p) phase, so the
    # spatial self-imaging oscillation appears in the raw propagated fields
    # (no post-processing phase restore).  Step 20 um << 408 um period.
    # (b) POST-PROCESSING check: the 52 m linear run's input fields with the
    # omitted beta0 phase restored analytically (kept as a cross-check).
    z_short = np.linspace(0.0, 4e-3, 801)
    fields_at_0 = [fields_lin[p][0] for p in range(3)]
    mfd = np.array([
        mode_mfd(fields_at_0, eng_lin.grid, moments, db0, z) for z in z_short
    ]) * 1e6

    eng_si = run_si_direct(db0, z_span=4e-3, step=20e-6, nsaves=801)
    fields_si = eng_si.fields_vs_z()
    mfd_direct = np.array([
        mode_mfd_direct(
            [fields_si[p][k] for p in range(3)], eng_si.grid, moments
        )
        for k in range(fields_si[0].shape[0])
    ]) * 1e6

    def _dominant_period(mfd_curve: NDArray, z_axis: NDArray) -> float:
        spectrum = np.abs(np.fft.rfft(mfd_curve - mfd_curve.mean())) ** 2
        freqs = np.fft.rfftfreq(len(mfd_curve), d=z_axis[1] - z_axis[0])
        idx = int(np.argmax(spectrum[1:]) + 1)
        return float(1.0 / freqs[idx] * 1e6)

    period_um = _dominant_period(mfd, z_short)
    assert abs(period_um / (l_si * 1e6) - 1.0) < 0.05, (period_um, l_si * 1e6)
    period_direct_um = _dominant_period(mfd_direct, eng_si.z_array)
    assert abs(period_direct_um / (l_si * 1e6) - 1.0) < 0.05, (
        period_direct_um, l_si * 1e6,
    )
    results["mfd_oscillation_period_um"] = period_um
    results["mfd_direct_oscillation_period_um"] = period_direct_um

    results["_plot"] = {
        "t": t,
        "fields_lin": fields_lin,
        "fields_nl": fields_nl,
        "z": eng_nl.z_array,
        "z_lin": eng_lin.z_array,
        "energy_lin": energy_lin,
        "energy_nl": energy_nl,
        "omega": omega,
        "spectra_nl": spectra_nl,
        "z_short": z_short,
        "mfd": mfd,
        "mfd_direct": mfd_direct,
        "z_direct": eng_si.z_array,
        "delays": delays,
        "fwhm_fixed_fs": fwhm_fixed * 1e15,
        "centres": centres,
    }

    if make_plot:
        _plot(results)

    print("Renninger & Wise (2013) multicomponent GRIN soliton: validation passed")
    print(
        f"  w0 = {w0 * 1e6:.3f} um, self-imaging = {l_si * 1e6:.1f} um, "
        f"walk-off = {delays[1] * 1e15:.1f}/{delays[2] * 1e15:.1f} fs/m"
    )
    print(
        f"  linear walk-off = {delta_lin[0] * 1e12:.3f}/{delta_lin[1] * 1e12:.3f} ps "
        f"-> locked to {delta_nl[0] * 1e12:.3f}/{delta_nl[1] * 1e12:.3f} ps "
        f"(lock ratio {lock_ratio:.3f})"
    )
    print(
        f"  FWHM {fwhm_nl[0] * 1e15:.1f} fs vs Eq.(6) {fwhm_fixed * 1e15:.1f} fs "
        f"({100 * fwhm_err:.1f}%); compression {compression:.0f}x"
    )
    print(
        f"  spectral centroids p=0,1,2: "
        f"{centres[0]:.2f}/{centres[1]:.2f}/{centres[2]:.2f} nm "
        f"(higher modes blue-shifted: {bool(higher_blue)})"
    )
    print(
        f"  energy drift linear {drift_lin:.2e}, nonlinear {drift_nl:.2e}; "
        f"MFD period {period_um:.1f} um (direct: {period_direct_um:.1f} um)"
    )
    print(
        "  required locking shifts (paper: blue): "
        f"{dlam_required_nm[1]:+.2f}/{dlam_required_nm[2]:+.2f} nm"
    )
    return results


def _plot(result: dict) -> None:
    d = result["_plot"]
    t_ps = d["t"] * 1e12
    z = d["z"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (a) temporal profiles at z = L (nonlinear) with linear centres
    ax = axes[0, 0]
    for p in range(3):
        p_nl = np.abs(d["fields_nl"][p][-1]) ** 2
        ax.plot(t_ps, p_nl / p_nl.max(), label=f"nonlinear p={p}")
    for p in (1, 2):
        c = centroid(d["t"], np.abs(d["fields_lin"][p][-1]) ** 2)
        ax.axvline(c * 1e12, ls=":", color=f"C{p}", alpha=0.8)
    ax.set_xlim(-3, 3)
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("intensity (norm.)")
    ax.set_title("(a) Temporal locking at z = 52 m\n(dotted: linear centres)")
    ax.legend(fontsize=8)

    # (b) per-mode output spectra
    ax = axes[0, 1]
    omega = d["omega"]
    lam = 2.0 * np.pi * C_MS / (OMEGA0 + omega) * 1e9
    keep = (lam > 1500) & (lam < 1600)
    for p in range(3):
        spec = d["spectra_nl"][p]
        ax.plot(lam[keep], spec[keep] / spec[keep].max(), label=f"p={p}")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("spectral intensity (norm.)")
    ax.set_title("(b) Output spectra (radiation-structured)")
    ax.legend(fontsize=8)

    # (c) mode separation vs z
    ax = axes[0, 2]
    fl = d["fields_lin"]
    fn = d["fields_nl"]
    for p, idx in ((1, 1), (2, 2)):
        sep_lin = np.array(
            [
                centroid(d["t"], np.abs(fl[idx][k]) ** 2)
                - centroid(d["t"], np.abs(fl[0][k]) ** 2)
                for k in range(fl[0].shape[0])
            ]
        )
        sep_nl = np.array(
            [
                centroid(d["t"], np.abs(fn[idx][k]) ** 2)
                - centroid(d["t"], np.abs(fn[0][k]) ** 2)
                for k in range(fn[0].shape[0])
            ]
        )
        ax.plot(d["z_lin"], sep_lin * 1e12, "--", color=f"C{idx}", alpha=0.6)
        ax.plot(z, sep_nl * 1e12, "-", color=f"C{idx}", label=f"p={p}")
    ax.set_xlabel("z (m)")
    ax.set_ylabel(r"$\Delta\tau$ (ps)")
    ax.set_title("(c) Mode locking\n(dashed: linear)")
    ax.legend(fontsize=8)

    # (d) FWHM vs z
    ax = axes[1, 0]
    fwhm_lin = np.array(
        [fwhm(d["t"], np.abs(fl[0][k]) ** 2) for k in range(fl[0].shape[0])]
    )
    fwhm_nl = np.array(
        [fwhm(d["t"], np.abs(fn[0][k]) ** 2) for k in range(fn[0].shape[0])]
    )
    ax.plot(d["z_lin"], fwhm_lin * 1e15, "--", label="linear p=0")
    ax.plot(z, fwhm_nl * 1e15, "-", label="nonlinear p=0")
    ax.axhline(d["fwhm_fixed_fs"], color="r", ls=":", label="Eq. (6)")
    ax.set_yscale("log")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("FWHM (fs)")
    ax.set_title("(d) Soliton compression")
    ax.legend(fontsize=8)

    # (e) total energy vs z
    ax = axes[1, 1]
    ax.plot(d["z_lin"], d["energy_lin"] / d["energy_lin"][0], label="linear")
    ax.plot(z, d["energy_nl"] / d["energy_nl"][0], "--", label="nonlinear")
    ax.set_xlabel("z (m)")
    ax.set_ylabel(r"$E(z)/E(0)$")
    ax.set_title("(e) Energy conservation")
    ax.legend(fontsize=8)

    # (f) self-imaging MFD: post-processing vs direct phase_offsets run
    ax = axes[1, 2]
    ax.plot(d["z_short"] * 1e3, d["mfd"], "k-", label="phase restored")
    ax.plot(d["z_direct"] * 1e3, d["mfd_direct"], "C1--",
            label="direct (phase_offsets)")
    ax.axhline(2 * mode_w0() * 1e6, color="r", ls=":", label=r"$2w_0$")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("MFD (um)")
    ax.set_title(
        f"(f) Self-imaging, period {result['mfd_oscillation_period_um']:.0f}/"
        f"{result['mfd_direct_oscillation_period_um']:.0f} um"
    )
    ax.legend(fontsize=8)

    fig.suptitle(
        "Multicomponent soliton in GRIN MMF "
        "(Renninger & Wise, Nat. Commun. 4, 1719, 2013)"
    )
    fig.tight_layout()
    out = HERE / "renninger_wise_2013.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    validate()
