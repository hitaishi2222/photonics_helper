"""Reproduction: spontaneous MI breathers and rogue waves (Narhi et al., 2016).

Reference
---------
M. Narhi, B. Wetzel, C. Billet, S. Toenger, T. Sylvestre, J.-M. Merolla,
R. Morandotti, F. Dias, G. Genty, J. M. Dudley, "Real-time measurements of
spontaneous breathers and rogue wave events in optical fibre modulation
instability", Nature Communications 7, 13675 (2016),
doi:10.1038/ncomms13675.

What is reproduced
------------------
The paper studies spontaneous modulation instability (MI) of a noisy CW field in
SMF-28 and interprets the emerging localized structures as nonlinear Schrodinger
breathers. The exact analytic structures it compares against are reproduced here:

* **MI gain.** For ``i A_z = (beta2/2) A_TT - gamma |A|^2 A`` with ``beta2 < 0``
  the maximum-gain modulation frequency is ``Omega = sqrt(2 gamma P0 / |beta2|)``
  and ``g_max = 2 gamma P0`` (paper: 46.4 GHz for P0 = 0.7 W).
* **Peregrine soliton (PS).** Paper Fig. 4b uses the exact profile
  ``I_PS(t) = (1 - 4/(1 + 4 gamma P0 t^2 / |beta2|))^2`` (normalised to the
  background), whose peak-to-background ratio is exactly 9.
* **Akhmediev breather (AB).** The family (paper's Ref. 31) grows and decays once
  and is periodic in time; its exact form is propagated here.
* **Spontaneous MI.** A noise-seeded CW field is propagated with the GNLSE
  engine and the ensemble-averaged spectrum is shown to develop sidebands in the
  MI band, with breather peaks close to and above the ratio-9 rogue-wave limit.

The Peregrine and Akhmediev exact solutions are the ``a -> 1/2`` and ``a < 1/2``
members of the same soliton-on-finite-background family used in the companion
``kuznetsov_ma_2012_breather`` reproduction (see that README for Eq. 2).

Usage
-----
    python reproductions/narhi_2016_mi_breathers/reproduce.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver, SplitStepEngine
from photonics_helper.phase_matching import mi_gain_spectrum
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

# The exact breather solutions and the stochastic seed now live in the library;
# this reproduction validates the solver against them.
from photonics_helper.breathers import (
    akhmediev_breather as akhmediev,
    peregrine_soliton as peregrine,
)
from photonics_helper.noise import add_noise

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"


@dataclass(frozen=True)
class MIParameters:
    """Physical parameters of the SMF-28 MI experiment."""

    lam_m: float
    beta2_si: float
    gamma: float
    alpha_per_m: float
    P0: float
    LNL_m: float
    T0_s: float

    @property
    def T0_ps(self) -> float:
        return self.T0_s * 1e12

    @property
    def omega_peak(self) -> float:
        """Maximum-gain modulation frequency sqrt(2 gamma P0 / |beta2|)."""
        return float(np.sqrt(2 * self.gamma * self.P0 / abs(self.beta2_si)))

    @property
    def omega_cutoff(self) -> float:
        return 2.0 * self.omega_peak

    @property
    def g_max(self) -> float:
        return 2.0 * self.gamma * self.P0


def derive(params: dict) -> MIParameters:
    beta2 = float(params["beta2_s2_per_m"])
    gamma = float(params["gamma_per_Wm"])
    P0 = float(params["background_power_W"])
    lam = float(params["central_wavelength_nm"]) * 1e-9
    alpha = float(params["loss_dB_per_km"]) / (10.0 * np.log10(np.e)) / 1e3
    LNL = 1.0 / (gamma * P0)
    T0 = float(np.sqrt(abs(beta2) * LNL))
    return MIParameters(
        lam_m=lam,
        beta2_si=beta2,
        gamma=gamma,
        alpha_per_m=alpha,
        P0=P0,
        LNL_m=LNL,
        T0_s=T0,
    )


# ---------------------------------------------------------------------------
# Exact breather solutions (photonics_helper.breathers) are imported above.
# ---------------------------------------------------------------------------


def _soliton_pulse(
    mp: MIParameters, grid: TemporalGrid, initial: NDArray, name: str
) -> Wave:
    """Wrap an arbitrary initial field in a ``Wave`` (custom-shape envelope)."""
    field = np.asarray(initial, dtype=complex)

    def func(t: NDArray, _T0: float, _A0: float) -> NDArray:
        del t, _T0, _A0
        return field

    env = Envelope(
        shape="custom",
        peak_amplitude=float(np.sqrt(mp.P0)),
        pulse_width=Time(mp.T0_s, "s"),
        func=func,
    )
    del name
    return Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(mp.lam_m, "m"),
    )


def _propagate_soliton(
    mp: MIParameters,
    grid: TemporalGrid,
    initial: NDArray,
    length_m: float,
    num_steps: int,
    nsaves: int,
) -> tuple[NDArray, list[Wave]]:
    """Propagate a pure-NLSE (beta2-only) breather and return ``(z, evolution)``."""
    pulse = _soliton_pulse(mp, grid, initial, "breather")
    fiber = FiberProfile.from_gamma(
        gamma=mp.gamma,
        n2=2.6e-20,
        omega0=2 * np.pi * C_MS / mp.lam_m,
        length=Length(length_m, "m"),
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([mp.beta2_si * 1e24, 0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=num_steps, nsaves=nsaves)
    return solver.z_array, solver.evolution


# ---------------------------------------------------------------------------
# Part A / B — MI gain
# ---------------------------------------------------------------------------


def check_mi_gain(mp: MIParameters) -> dict:
    info = mi_gain_spectrum(mp.beta2_si, mp.gamma, mp.P0)
    assert isinstance(info, dict)
    return {
        "omega_peak_GHz": float(info["Omega_peak"]) / (2 * np.pi) / 1e9,
        "omega_cutoff_GHz": float(info["Omega_cutoff"]) / (2 * np.pi) / 1e9,
        "g_max_per_m": float(info["g_max"]),
        "g_max_theory": mp.g_max,
    }


def deterministic_mi_growth(mp: MIParameters, params: dict, fast: bool) -> dict:
    """Seed two weak sidebands at the MI peak and measure the exponential growth."""
    Tmax = 40 * (2 * np.pi / mp.omega_peak)
    N = 4096 if fast else 8192
    grid = TemporalGrid(N=N, Tmax=Time(Tmax, "s"))
    dw = 2 * np.pi / Tmax
    length = 2000.0
    dz = 20.0
    fiber = FiberProfile.from_gamma(
        gamma=mp.gamma,
        n2=2.6e-20,
        omega0=2 * np.pi * C_MS / mp.lam_m,
        length=Length(length, "m"),
    )
    omega = mp.omega_peak
    seed = np.sqrt(mp.P0) * (1 + 1e-4 * np.cos(omega * grid.t))
    pulse = _soliton_pulse(mp, grid, seed, "mi-seed")
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([mp.beta2_si * 1e24, 0.0]),
        step_size=Length(dz, "m"),
    )
    engine.propagate(int(length / dz), nsaves=5)
    z = engine.z_array
    i0 = N // 2
    k = int(round(omega / dw))
    powers = np.array(
        [
            (
                np.abs(grid.fft(w.envelope_field))[i0 + k] ** 2
                + np.abs(grid.fft(w.envelope_field))[i0 - k] ** 2
            )
            / 2
            for w in engine.evolution
        ]
    )
    g_meas = float(np.polyfit(z, np.log(powers), 1)[0])
    g_th = float(mi_gain_spectrum(mp.beta2_si, mp.gamma, mp.P0, omega))
    return {"g_measured": g_meas, "g_theory": g_th, "ratio": g_meas / g_th}


# ---------------------------------------------------------------------------
# Part C — Peregrine soliton
# ---------------------------------------------------------------------------


def peregrine_solution(mp: MIParameters, params: dict, fast: bool) -> dict:
    cfg = params["peregrine"]
    xi0, xi1 = float(cfg["xi_start"]), float(cfg["xi_end"])
    N = 2048 if fast else int(cfg["grid_N"])
    grid = TemporalGrid(N=N, Tmax=Time(float(cfg["grid_Tmax_ps"]) * 1e-12, "s"))
    initial = np.sqrt(mp.P0) * peregrine(xi0, grid.t / mp.T0_s)
    steps = 4000 if fast else int(cfg["num_steps"])
    z, evolution = _propagate_soliton(
        mp, grid, initial, (xi1 - xi0) * mp.LNL_m, steps, nsaves=61
    )
    xi = xi0 + z / mp.LNL_m
    mid = grid.N // 2
    center = np.array([abs(w.envelope_field[mid]) ** 2 for w in evolution])
    center_an = np.array([mp.P0 * abs(peregrine(x, 0.0)) ** 2 for x in xi])

    i_peak = int(np.argmax(center))
    A = evolution[i_peak].envelope_field
    Aa = np.sqrt(mp.P0) * peregrine(xi[i_peak], grid.t / mp.T0_s)
    profile_l2 = float(
        np.linalg.norm(np.abs(A) ** 2 - np.abs(Aa) ** 2)
        / np.linalg.norm(np.abs(Aa) ** 2)
    )
    ratio = float(center[i_peak] / mp.P0)
    return {
        "peak_ratio": ratio,
        "peak_ratio_theory": 9.0,
        "peak_rel_err": abs(ratio - 9.0) / 9.0,
        "profile_l2": profile_l2,
        "center_max_abs_err_W": float(np.max(np.abs(center - center_an))),
        "xi_peak": float(xi[i_peak]),
        "_plot": (xi, center, center_an, grid.t, A, Aa, xi[i_peak]),
    }


# ---------------------------------------------------------------------------
# Part D — Akhmediev breather
# ---------------------------------------------------------------------------


def akhmediev_solution(mp: MIParameters, params: dict, fast: bool) -> dict:
    cfg = params["akhmediev"]
    a = float(cfg["a"])
    xi0, xi1 = float(cfg["xi_start"]), float(cfg["xi_end"])
    periods = int(cfg["modulation_periods"])
    nu = 2 * np.sqrt(1 - 2 * a)
    Tmax = periods * (2 * np.pi / nu) * mp.T0_s
    N = 1024 if fast else int(cfg["grid_N"])
    grid = TemporalGrid(N=N, Tmax=Time(Tmax, "s"))
    initial = np.sqrt(mp.P0) * akhmediev(xi0, grid.t / mp.T0_s, a)
    steps = 2000 if fast else int(cfg["num_steps"])
    z, evolution = _propagate_soliton(
        mp, grid, initial, (xi1 - xi0) * mp.LNL_m, steps, nsaves=61
    )
    xi = xi0 + z / mp.LNL_m
    mid = grid.N // 2
    center = np.array([abs(w.envelope_field[mid]) ** 2 for w in evolution])
    center_an = np.array([mp.P0 * abs(akhmediev(x, 0.0, a)) ** 2 for x in xi])
    i_peak = int(np.argmax(center))
    A = evolution[i_peak].envelope_field
    Aa = np.sqrt(mp.P0) * akhmediev(xi[i_peak], grid.t / mp.T0_s, a)
    profile_l2 = float(
        np.linalg.norm(np.abs(A) ** 2 - np.abs(Aa) ** 2)
        / np.linalg.norm(np.abs(Aa) ** 2)
    )
    return {
        "a": a,
        "peak_ratio": float(center[i_peak] / mp.P0),
        "peak_ratio_theory": float(abs(akhmediev(0.0, 0.0, a)) ** 2),
        "peak_rel_err": abs(center[i_peak] / mp.P0 - abs(akhmediev(0.0, 0.0, a)) ** 2)
        / abs(akhmediev(0.0, 0.0, a)) ** 2,
        "profile_l2": profile_l2,
        "center_max_abs_err_W": float(np.max(np.abs(center - center_an))),
        "_plot": (xi, center, center_an, grid.t, A, Aa, xi[i_peak]),
    }


# ---------------------------------------------------------------------------
# Part E — spontaneous (noise-seeded) MI
# ---------------------------------------------------------------------------


def spontaneous_mi(mp: MIParameters, params: dict, fast: bool) -> dict:
    cfg = params["spontaneous_mi"]
    periods = int(cfg["modulation_periods"])
    Tmax = periods * (2 * np.pi / mp.omega_peak)
    N = int(cfg["grid_N"])
    grid = TemporalGrid(N=N, Tmax=Time(Tmax, "s"))
    length = float(cfg["length_m"])
    dz = float(cfg["step_size_m"])
    n_real = int(cfg["fast_realizations"]) if fast else int(cfg["num_realizations"])
    eps = float(params["noise_amplitude_contrast"])

    fiber = FiberProfile.from_gamma(
        gamma=mp.gamma,
        n2=2.6e-20,
        omega0=2 * np.pi * C_MS / mp.lam_m,
        alpha=mp.alpha_per_m,
        length=Length(length, "m"),
    )

    spectrum_avg = np.zeros(N, dtype=float)
    peak_ratios: list[float] = []
    cw_field = np.full(N, np.sqrt(mp.P0), dtype=complex)
    for seed in range(n_real):
        base = _soliton_pulse(mp, grid, cw_field, "cw")
        pulse = add_noise(base, eps, seed=seed)
        engine = SplitStepEngine(
            pulse=pulse,
            fiber=fiber,
            betas=np.array([mp.beta2_si * 1e24, 0.0]),
            step_size=Length(dz, "m"),
        )
        engine.propagate(int(length / dz), nsaves=2)
        A = engine.evolution[-1].envelope_field
        spectrum_avg += np.abs(grid.fft(A)) ** 2
        intensity = np.abs(A) ** 2
        peak_ratios.append(float(intensity.max() / intensity.mean()))
    spectrum_avg /= n_real

    freqs = grid.w / (2 * np.pi) / 1e9
    # Restrict to the MI band around Omega_peak (the paper's 46.4 GHz sideband);
    # the low-frequency pedestal from pump self-phase modulation is not the
    # MI sideband and would otherwise win the argmax for a small ensemble.
    band = (np.abs(freqs) > 20.0) & (np.abs(freqs) < 70.0)
    sideband_peak_GHz = float(np.abs(freqs[band])[np.argmax(spectrum_avg[band])])

    ratios = np.array(peak_ratios)
    return {
        "sideband_peak_GHz": sideband_peak_GHz,
        "omega_peak_theory_GHz": mp.omega_peak / (2 * np.pi) / 1e9,
        "g_max_per_m": mp.g_max,
        "max_peak_ratio": float(ratios.max()),
        "mean_peak_ratio": float(ratios.mean()),
        "n_realizations": n_real,
        "input_intensity_contrast_pct": float(eps * np.sqrt(2) * 100),
        "_plot": (freqs, spectrum_avg, ratios),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    params: dict | None = None,
    *,
    fast: bool = False,
    make_plot: bool = True,
) -> dict:
    params = params or json.loads(PARAMETERS.read_text())
    mp = derive(params)
    tol = params["tolerances"]

    gain = check_mi_gain(mp)
    growth = deterministic_mi_growth(mp, params, fast)
    ps = peregrine_solution(mp, params, fast)
    ab = akhmediev_solution(mp, params, fast)
    smi = spontaneous_mi(mp, params, fast)

    result = {
        "T0_ps": mp.T0_ps,
        "LNL_m": mp.LNL_m,
        "mi": gain,
        "mi_growth": growth,
        "peregrine": {k: v for k, v in ps.items() if k != "_plot"},
        "akhmediev": {k: v for k, v in ab.items() if k != "_plot"},
        "spontaneous": {k: v for k, v in smi.items() if k != "_plot"},
    }

    # --- assertions ---
    assert (
        abs(gain["omega_peak_GHz"] - params["derived_reference"]["omega_peak_GHz"])
        < tol["omega_peak_GHz"]
    ), gain
    assert abs(gain["omega_peak_GHz"] - float(params["mi_sideband_GHz"])) < 0.5, gain
    assert abs(gain["g_max_per_m"] - mp.g_max) < 1e-9
    assert tol["gain_ratio_low"] < growth["ratio"] < tol["gain_ratio_high"], growth
    assert ps["peak_rel_err"] < tol["peregrine_peak_rel"], ps
    assert ps["profile_l2"] < tol["peregrine_profile_l2"], ps
    assert ab["peak_rel_err"] < tol["akhmediev_peak_rel"], ab
    assert ab["profile_l2"] < tol["akhmediev_profile_l2"], ab
    assert (
        tol["spontaneous_sideband_GHz_low"]
        <= smi["sideband_peak_GHz"]
        <= tol["spontaneous_sideband_GHz_high"]
    ), smi
    assert smi["max_peak_ratio"] > tol["spontaneous_max_ratio_min"], smi

    if make_plot:
        _plot(mp, params, ps, ab, smi)

    print("Narhi et al. (2016) MI / breather validation passed:")
    print(
        f"  MI: Omega_peak = {gain['omega_peak_GHz']:.2f} GHz (paper 46.4), "
        f"g_max = {gain['g_max_per_m']:.3e} /m, cutoff = {gain['omega_cutoff_GHz']:.1f} GHz"
    )
    print(
        f"  deterministic growth: measured {growth['g_measured']:.3e} vs theory "
        f"{growth['g_theory']:.3e} /m (ratio {growth['ratio']:.2f})"
    )
    print(
        f"  Peregrine: peak/background {ps['peak_ratio']:.4f} (theory 9), "
        f"profile L2 {ps['profile_l2']:.2e}"
    )
    print(
        f"  Akhmediev (a={ab['a']}): peak/background {ab['peak_ratio']:.4f} "
        f"(theory {ab['peak_ratio_theory']:.4f}), profile L2 {ab['profile_l2']:.2e}"
    )
    print(
        f"  spontaneous MI ({smi['n_realizations']} realisations): sideband {smi['sideband_peak_GHz']:.1f} GHz, "
        f"max peak/mean {smi['max_peak_ratio']:.2f}, mean {smi['mean_peak_ratio']:.2f}"
    )
    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot(mp: MIParameters, params: dict, ps: dict, ab: dict, smi: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (a) MI gain curve.
    omega = np.linspace(0, 1.3 * mp.omega_cutoff, 500)
    g = np.array(
        [float(mi_gain_spectrum(mp.beta2_si, mp.gamma, mp.P0, w)) for w in omega]
    )
    axes[0, 0].plot(omega / (2 * np.pi) / 1e9, g, "k-")
    axes[0, 0].axvline(
        mp.omega_peak / (2 * np.pi) / 1e9, color="r", ls="--", label=r"$\Omega_{peak}$"
    )
    axes[0, 0].set_xlabel("frequency (GHz)")
    axes[0, 0].set_ylabel("MI gain $g$ (1/m)")
    axes[0, 0].set_title("(a) MI gain (paper: 46.4 GHz)")
    axes[0, 0].legend()

    # (b) Peregrine centre-power evolution.
    xi, center, center_an, t, A, Aa, xi_peak = ps["_plot"]
    axes[0, 1].plot(xi, center_an, "k-", lw=2, label="analytic PS")
    axes[0, 1].plot(xi, center, "r--", label="GNLSE")
    axes[0, 1].axhline(9 * mp.P0, color="b", ls=":", label="9 P$_0$")
    axes[0, 1].set_xlabel(r"$\xi$")
    axes[0, 1].set_ylabel("centre power (W)")
    axes[0, 1].set_title("(b) Peregrine soliton")
    axes[0, 1].legend(fontsize=8)

    # (c) Peregrine profile at peak.
    axes[0, 2].plot(
        t * 1e12,
        mp.P0 * np.abs(peregrine(xi_peak, t / mp.T0_s)) ** 2,
        "k-",
        label="analytic",
    )
    axes[0, 2].plot(t * 1e12, np.abs(A) ** 2, "r--", label="GNLSE")
    axes[0, 2].plot(
        t * 1e12,
        mp.P0 * (1 - 4 / (1 + 4 * mp.gamma * mp.P0 * t**2 / abs(mp.beta2_si))) ** 2,
        "b:",
        label="paper Fig. 4b",
    )
    axes[0, 2].set_xlim(-20, 20)
    axes[0, 2].set_xlabel("time (ps)")
    axes[0, 2].set_ylabel("$|A|^2$ (W)")
    axes[0, 2].set_title("(c) PS profile at $\\xi=0$")
    axes[0, 2].legend(fontsize=8)

    # (d) Akhmediev centre-power evolution.
    xi, center, center_an, t, A, Aa, xi_peak = ab["_plot"]
    axes[1, 0].plot(xi, center_an, "k-", lw=2, label="analytic AB")
    axes[1, 0].plot(xi, center, "r--", label="GNLSE")
    axes[1, 0].set_xlabel(r"$\xi$")
    axes[1, 0].set_ylabel("centre power (W)")
    axes[1, 0].set_title(f"(d) Akhmediev breather (a={ab['a']})")
    axes[1, 0].legend(fontsize=8)

    # (e) AB profile at peak.
    axes[1, 1].plot(
        t * 1e12,
        mp.P0 * np.abs(akhmediev(xi_peak, t / mp.T0_s, ab["a"])) ** 2,
        "k-",
        label="analytic",
    )
    axes[1, 1].plot(t * 1e12, np.abs(A) ** 2, "r--", label="GNLSE")
    axes[1, 1].set_xlim(-25, 25)
    axes[1, 1].set_xlabel("time (ps)")
    axes[1, 1].set_ylabel("$|A|^2$ (W)")
    axes[1, 1].set_title("(e) AB profile at peak")
    axes[1, 1].legend(fontsize=8)

    # (f) Spontaneous MI average spectrum.
    freqs, spectrum, ratios = smi["_plot"]
    mask = np.abs(freqs) < 150
    axes[1, 2].plot(
        freqs[mask], 10 * np.log10(spectrum[mask] / spectrum[mask].max()), "k-"
    )
    axes[1, 2].axvline(
        smi["sideband_peak_GHz"],
        color="r",
        ls="--",
        label=f"peak {smi['sideband_peak_GHz']:.0f} GHz",
    )
    axes[1, 2].axvline(-smi["sideband_peak_GHz"], color="r", ls="--")
    axes[1, 2].set_xlabel("frequency (GHz)")
    axes[1, 2].set_ylabel("normalised spectral intensity (dB)")
    axes[1, 2].set_ylim(-40, 0)
    axes[1, 2].set_title("(f) Spontaneous MI (ensemble avg)")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(
        "Narhi et al., Nat. Commun. 7, 13675 (2016) — SMF-28 MI breathers "
        f"($P_0$ = {mp.P0} W, $\\beta_2$ = {mp.beta2_si * 1e27:.1f}×10⁻²⁷ s²/m)",
        fontsize=11,
    )
    fig.tight_layout()
    out = HERE / "mi_breathers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_contact_sheet() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pages = sorted((HERE / "paper_pages").glob("page*.png"))
    if not pages:
        return
    n = len(pages)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, pages):
        ax.imshow(mpimg.imread(p))
        ax.set_title(p.stem, fontsize=8)
        ax.axis("off")
    for ax in axes[len(pages) :]:
        ax.axis("off")
    fig.suptitle(
        "Narhi et al., Nat. Commun. 7, 13675 (2016) — spontaneous MI breathers",
        fontsize=11,
    )
    fig.tight_layout()
    out = HERE / "paper_pages" / "contact_sheet.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    validate()
    make_contact_sheet()
