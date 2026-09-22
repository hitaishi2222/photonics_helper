"""Reproduction: Manakov-PMD equation (Marcuse, Menyuk & Wai, JLT 1997).

Reference
---------
D. Marcuse, C. R. Menyuk, P. K. A. Wai, *Application of the Manakov-PMD
equation to studies of signal propagation in optical fibers with randomly
varying birefringence*, J. Lightwave Technol. **15**, 1735 (1997),
doi:10.1109/50.622902.

What is reproduced (three quantitative checks, paper Sec. III/IV)
-----------------------------------------------------------------
1. **Eq. (30) soliton duration law with the explicit 8/9** — in soliton
   units the Manakov soliton carries total peak power

       tau_m = [ 9 / ( 8 (|U|^2 + |V|^2)_peak ) ]^(1/2)

   so compared with the scalar NLSE soliton (|beta2| = gamma P0 T0^2) the
   equal-split two-axis soliton has (9/8) |beta2| = gamma P_peak T0^2.
   Two runs: a scalar control (single axis, coupling='incoherent') and the
   equal-split Manakov soliton (coupling='manakov'); both must hold their
   shape over 30 z0.

2. **Fig. 5 (core result)** — the same Manakov soliton propagated in a
   fiber with random birefringence (RandomBirefringenceEngine: rapid SU(2)
   frame rotations of the Eq. (34)/(35) class). The pulse keeps the
   Manakov shape while the peak |U|^2 + |V|^2 fluctuates by only ~1 %.

3. **Fig. 4 / NRZ system (reduced runtime)** — the Sec. IV-A lossless
   dispersion-map system: lambda = 1.55 µm, A_eff = 52 µm^2,
   n2 = 2.6e-20 m^2/W, period 100 km = 80 km normal (D = −1 ps/(nm·km)) +
   20 km compensation (D = +4 ps/(nm·km)), cumulative dispersion zero,
   no higher-order dispersion. Input: 16-bit NRZ word, 0.2 ns bit slot
   (5 Gb/s), shaped by a 4th-order 8.75 GHz Bessel filter, peak 20 mW;
   detection: square law + 5 GHz electrical Bessel filter. The paper runs
   500 km = 5 periods at 1.67 m resolution; for runtime this reproduction
   runs **2 periods (200 km)** with 100 m frame-rotation steps
   (birefringent run) vs the deterministic manakov-coupling run, and
   compares the detected and filtered currents (the paper's claim: the
   CNLS and Manakov solutions are "exactly the same"; nonlinear PMD gives
   no observable effect in this regime).

Usage
-----
    python reproductions/planned/marcuse_menyuk_wai_1997_manakov_pmd/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.core.grids import TemporalGrid
from photonics_helper.gnlse import FiberProfile
from photonics_helper.pulse import Envelope, Wave
from photonics_helper.vector_gnlse import (
    MANAKOV_FACTOR,
    RandomBirefringenceEngine,
    VectorSplitStepEngine,
)

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"

C_MS = 2.99792458e8

NRZ_BITS = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]
BIT_PERIOD_S = 0.2e-9
NRZ_PEAK_W = 0.020
NRZ_T0_S = -1.0e-9  # centre of bit 0

SOLITON_T0_PS = 10.0
SOLITON_BETA2_PS2_PER_KM = -21.0
SOLITON_Z0_COUNT = 30.0


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------


def load_params() -> dict:
    with open(PARAMETERS) as fh:
        return json.load(fh)


class PhysicalSetup:
    """Dimensional quantities (paper Sec. IV-A: λ, A_eff, n2; lossless)."""

    def __init__(self, wavelength_nm: float, n2: float, a_eff_um2: float):
        self.wavelength_nm = wavelength_nm
        self.n2 = n2
        self.A_eff_um2 = a_eff_um2

    @property
    def omega0(self) -> float:
        return 2.0 * np.pi * C_MS / (self.wavelength_nm * 1e-9)

    @property
    def gamma_per_W_km(self) -> float:
        """γ = n₂ ω₀ / (c A_eff), in W⁻¹ km⁻¹."""
        gamma_m = self.n2 * self.omega0 / (C_MS * self.A_eff_um2 * 1e-12)
        return gamma_m * 1e3

    def D_to_beta2_ps2_per_km(self, D_ps_nm_km: float) -> float:
        """β₂ = −D λ² / (2π c). D in ps/(nm·km), β₂ in ps²/km."""
        wl = self.wavelength_nm * 1e-9
        D_si = D_ps_nm_km * 1e-6  # ps/(nm km) → s/m²
        return -D_si * wl**2 / (2.0 * np.pi * C_MS) * 1e27

    def fiber(self, length_m: float, alpha_dB_km: float = 0.0) -> FiberProfile:
        alpha_np = alpha_dB_km * np.log(10.0) / 10.0 / 1e3  # dB/km → 1/m
        return FiberProfile(
            n2=self.n2,
            alpha=alpha_np,
            A_eff=Area(self.A_eff_um2 * 1e-12, "m^2"),
            length=Length(length_m, "m"),
        )


def _relative_l2(a: NDArray, b: NDArray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _shape_l2(power: NDArray, ref: NDArray) -> float:
    """Relative L2 of two peak-normalized temporal power profiles."""
    a = power / power.max()
    r = ref / ref.max()
    return float(np.linalg.norm(a - r) / np.linalg.norm(r))


def _sech_envelope(peak_W: float, T0_ps: float) -> Envelope:
    return Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(peak_W),
        pulse_width=Time(T0_ps * 1e-12, "s"),
    )


def _bessel_sos(order: int, cutoff_Hz: float, fs_Hz: float):
    return signal.bessel(order, cutoff_Hz, fs=fs_Hz, output="sos")


def _soliton_peak_powers(phys: PhysicalSetup) -> tuple[float, float]:
    """Scalar-soliton peak and the Manakov equal-split total peak (W)."""
    gamma_km = phys.gamma_per_W_km
    p_scalar = abs(SOLITON_BETA2_PS2_PER_KM) / (gamma_km * SOLITON_T0_PS**2)
    return p_scalar, (9.0 / 8.0) * p_scalar


def _soliton_z0_km() -> float:
    beta2_ps2_m = SOLITON_BETA2_PS2_PER_KM * 1e-3
    return SOLITON_T0_PS**2 * 1e-24 / abs(beta2_ps2_m * 1e-24) * 1e-3


# ---------------------------------------------------------------------------
# Check 1 — Eq. (30): scalar control vs equal-split Manakov soliton
# ---------------------------------------------------------------------------


def check_eq30(phys: PhysicalSetup, make_plot: bool = True) -> dict:
    wl = Wavelength(phys.wavelength_nm, "nm")
    grid = TemporalGrid(N=2048, Tmax=Time(160e-12, "s"))
    beta2_ps2_m = SOLITON_BETA2_PS2_PER_KM * 1e-3  # ps²/m
    length_m = _soliton_z0_km() * SOLITON_Z0_COUNT * 1e3
    step_m = 250.0
    n_steps = int(round(length_m / step_m))

    p_scalar, p_total = _soliton_peak_powers(phys)

    # (a) scalar control: one axis only, scalar-soliton power
    env = _sech_envelope(p_scalar, SOLITON_T0_PS)
    wx = Wave(grid=grid, envelope=env, central_wavelength=wl)
    wy = Wave(grid=grid, envelope=env, central_wavelength=wl).with_field(
        np.zeros(grid.N, dtype=complex)
    )
    ref_x = np.abs(env.field(grid.t)) ** 2
    eng = VectorSplitStepEngine(
        wx, wy, phys.fiber(length_m), [beta2_ps2_m],
        coupling="incoherent", step_size=Length(step_m, "m"),
    )
    eng.propagate(n_steps)
    out_x = np.abs(eng.evolution_x[-1].envelope_field) ** 2
    scalar_shape = _shape_l2(out_x, ref_x)
    scalar_peak_rel = float(out_x.max() / ref_x.max() - 1.0)

    # (b) equal-split Manakov soliton: per-axis peak p_total/2
    env = _sech_envelope(p_total / 2.0, SOLITON_T0_PS)
    wx = Wave(grid=grid, envelope=env, central_wavelength=wl)
    wy = Wave(grid=grid, envelope=env, central_wavelength=wl)
    ref_tot = 2.0 * np.abs(env.field(grid.t)) ** 2
    eng = VectorSplitStepEngine(
        wx, wy, phys.fiber(length_m), [beta2_ps2_m],
        coupling="manakov", step_size=Length(step_m, "m"),
    )
    eng.propagate(n_steps)
    out_tot = (
        np.abs(eng.evolution_x[-1].envelope_field) ** 2
        + np.abs(eng.evolution_y[-1].envelope_field) ** 2
    )
    manakov_shape = _shape_l2(out_tot, ref_tot)
    manakov_peak_rel = float(out_tot.max() / ref_tot.max() - 1.0)

    if make_plot:
        _plot_eq30(grid, eng, ref_tot, out_tot)

    return {
        "p_scalar_W": float(p_scalar),
        "p_manakov_total_W": float(p_total),
        "peak_ratio_by_construction": float(p_total / p_scalar),
        "z0_km": float(_soliton_z0_km()),
        "propagation_z0": SOLITON_Z0_COUNT,
        "scalar_shape_l2": scalar_shape,
        "scalar_peak_rel": scalar_peak_rel,
        "manakov_shape_l2": manakov_shape,
        "manakov_peak_rel": manakov_peak_rel,
        "manakov_factor": MANAKOV_FACTOR,
    }


def _plot_eq30(grid: TemporalGrid, eng, ref_tot: NDArray, out_tot: NDArray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_ps = grid.t * 1e12
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(t_ps, ref_tot / ref_tot.max(), "k--", lw=1.2, label="initial |U|²+|V|²")
    ax.plot(
        t_ps, out_tot / out_tot.max(), lw=1.4,
        label=f"after {SOLITON_Z0_COUNT:.0f} z₀,  {MANAKOV_FACTOR:.3g}γ(|U|²+|V|²)",
    )
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("normalized peak power")
    ax.set_title(
        "Marcuse/Menyuk/Wai Eq. (30): τ_m = [9/(8|U|²+|V|²)_peak]^½\n"
        "equal-split Manakov soliton holds its shape over 30 z₀"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "fig_eq30_manakov_soliton.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 2 — Fig. 5: random birefringence, ~1 % peak fluctuation
# ---------------------------------------------------------------------------


def check_fig5(phys: PhysicalSetup, seed: int = 20250101,
               make_plot: bool = True) -> dict:
    wl = Wavelength(phys.wavelength_nm, "nm")
    grid = TemporalGrid(N=2048, Tmax=Time(160e-12, "s"))
    beta2_ps2_m = SOLITON_BETA2_PS2_PER_KM * 1e-3
    length_m = _soliton_z0_km() * SOLITON_Z0_COUNT * 1e3
    step_m = 20.0  # fluctuation converged for step <= 20 m (README)

    _, p_total = _soliton_peak_powers(phys)
    env = _sech_envelope(p_total / 2.0, SOLITON_T0_PS)
    wx = Wave(grid=grid, envelope=env, central_wavelength=wl)
    wy = Wave(grid=grid, envelope=env, central_wavelength=wl)
    ref_tot = 2.0 * np.abs(env.field(grid.t)) ** 2

    rng_master = np.random.default_rng(seed)
    fluctuations: list[float] = []
    shape_l2s: list[float] = []
    energies: list[float] = []
    for k in range(4):  # four statistically independent fibres (Fig. 5(b) ensemble)
        eng = RandomBirefringenceEngine(
            wx, wy, phys.fiber(length_m), [beta2_ps2_m],
            step_size=Length(step_m, "m"),
        )
        eng.seed = int(rng_master.integers(0, 2**31 - 1))
        eng.propagate(int(round(length_m / step_m)))

        peaks = np.asarray(
            [
                np.max(
                    np.abs(sx.envelope_field) ** 2
                    + np.abs(sy.envelope_field) ** 2
                )
                for sx, sy in zip(eng.evolution_x, eng.evolution_y)
            ]
        )
        fluctuations.append(float((peaks.max() - peaks.min()) / peaks.mean()))
        out_tot = (
            np.abs(eng.evolution_x[-1].envelope_field) ** 2
            + np.abs(eng.evolution_y[-1].envelope_field) ** 2
        )
        shape_l2s.append(_shape_l2(out_tot, ref_tot))
        energies.append(abs(eng._energy_vs_z[-1] / eng._energy_vs_z[0] - 1.0))

        if make_plot and k == 0:
            _plot_fig5(eng, grid, ref_tot)

    return {
        "z0_km": float(_soliton_z0_km()),
        "length_km": float(_soliton_z0_km() * SOLITON_Z0_COUNT),
        "frame_step_m": step_m,
        "peak_fluctuation_pct_mean": float(np.mean(fluctuations) * 100.0),
        "peak_fluctuation_pct_worst": float(np.max(fluctuations) * 100.0),
        "shape_l2_mean": float(np.mean(shape_l2s)),
        "shape_l2_worst": float(np.max(shape_l2s)),
        "energy_drift_max": float(np.max(energies)),
    }


def _plot_fig5(eng, grid: TemporalGrid, ref_tot: NDArray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(eng._z_positions)
    z0 = z[-1] / SOLITON_Z0_COUNT
    peaks = np.asarray(
        [
            np.max(
                np.abs(sx.envelope_field) ** 2
                + np.abs(sy.envelope_field) ** 2
            )
            for sx, sy in zip(eng.evolution_x, eng.evolution_y)
        ]
    )

    t_ps = grid.t * 1e12
    out_tot = (
        np.abs(eng.evolution_x[-1].envelope_field) ** 2
        + np.abs(eng.evolution_y[-1].envelope_field) ** 2
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(z / z0, peaks / ref_tot.max(), lw=0.8)
    ax.set_xlabel("z / z₀")
    ax.set_ylabel("peak |U|²+|V|² / P_peak")
    ax.set_title("Fig. 5(b) analogue: soliton peak under random birefringence")
    ax = axes[1]
    ax.plot(t_ps, ref_tot / ref_tot.max(), "k--", lw=1.2, label="Manakov soliton")
    ax.plot(t_ps, out_tot / out_tot.max(), lw=1, alpha=0.85,
            label="after 30 z₀, random SU(2) frames")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("normalized power")
    ax.legend()
    ax.set_title("Shape invariance at the 8/9 nonlinearity")
    fig.tight_layout()
    fig.savefig(HERE / "fig5_soliton_random_birefringence.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 3 — Fig. 4: NRZ over the dispersion map (2 of the paper's 5 periods)
# ---------------------------------------------------------------------------


def _nrz_field(grid: TemporalGrid) -> tuple[NDArray, NDArray]:
    """16-bit NRZ word through the 8.75 GHz Bessel prefilter, 20 mW peak.

    The paper's Fig. 3(a) places the first bit around t ≈ 0 with the word
    extending to ≈ 3.2 ns on a 3.3 ns window; here the bit centres begin at
    the left window edge for padding symmetry.
    """
    t = grid.t
    raw = np.zeros_like(t)
    for k, b in enumerate(NRZ_BITS):
        c = NRZ_T0_S + k * BIT_PERIOD_S
        raw += b * (np.abs(t - c) <= BIT_PERIOD_S / 2.0)
    sos = _bessel_sos(4, 8.75e9, fs_Hz=1.0 / grid.dt)
    shaped = signal.sosfiltfilt(sos, raw)  # zero-phase; cosmetic rounding
    shaped *= np.sqrt(NRZ_PEAK_W) / shaped.max()
    field = shaped.astype(np.complex128)
    return field / np.sqrt(2.0), field / np.sqrt(2.0)


def _nrz_run(grid: TemporalGrid, phys: PhysicalSetup, rotations: bool,
             periods: int = 2, seed: int = 4242) -> NDArray:
    """Propagate the NRZ word through the paper's map; returns total power."""
    wl = Wavelength(phys.wavelength_nm, "nm")
    fx, fy = _nrz_field(grid)

    def _cont_wave(f: NDArray) -> Wave:
        env = Envelope(
            shape="custom",
            peak_amplitude=1.0,
            pulse_width=Time(1e-12, "s"),
            func=lambda t, T0, A0: np.ones_like(t),
        )
        return Wave(grid=grid, envelope=env, central_wavelength=wl).with_field(f)

    w_x = _cont_wave(fx)
    w_y = _cont_wave(fy)

    b2_normal = phys.D_to_beta2_ps2_per_km(-1.0)  # ps²/km
    b2_anom = phys.D_to_beta2_ps2_per_km(4.0)

    sections = [(80.0e3, b2_normal), (20.0e3, b2_anom)] * periods
    master = np.random.default_rng(seed)
    for l_sec, beta2_ps2_km in sections:
        step_m = 100.0 if rotations else 500.0
        if rotations:
            eng = RandomBirefringenceEngine(
                w_x, w_y, phys.fiber(l_sec), [beta2_ps2_km * 1e-3],
                step_size=Length(step_m, "m"),
            )
            eng.seed = int(master.integers(0, 2**31 - 1))
        else:
            eng = VectorSplitStepEngine(
                w_x, w_y, phys.fiber(l_sec), [beta2_ps2_km * 1e-3],
                coupling="manakov", step_size=Length(step_m, "m"),
            )
        eng.propagate(int(round(l_sec / step_m)))
        w_x = _cont_wave(eng.A_x.copy())
        w_y = _cont_wave(eng.A_y.copy())

    return _current_power(w_x, w_y, grid)


def _current_power(w_x: Wave, w_y: Wave, grid: TemporalGrid) -> NDArray:
    ax = w_x._pulse_train_field
    ay = w_y._pulse_train_field
    assert ax is not None and ay is not None
    return np.abs(ax) ** 2 + np.abs(ay) ** 2


def _electrical_filter(grid: TemporalGrid, power: NDArray) -> NDArray:
    """5 GHz electrical Bessel low-pass after square-law detection."""
    sos = _bessel_sos(4, 5.0e9, fs_Hz=1.0 / grid.dt)
    return signal.sosfiltfilt(sos, power)


def check_nrz(phys: PhysicalSetup, make_plot: bool = True) -> dict:
    grid = TemporalGrid(N=4096, Tmax=Time(8e-9, "s"))
    p_biref = _nrz_run(grid, phys, rotations=True)
    p_avg = _nrz_run(grid, phys, rotations=False)

    i_biref = _electrical_filter(grid, p_biref)
    i_avg = _electrical_filter(grid, p_avg)
    l2 = _relative_l2(i_biref, i_avg)

    # detected-pulse peak positions (three strongest pulses) tolerance check
    thr = 0.35 * i_biref.max()
    pk_b, _ = signal.find_peaks(i_biref, height=thr)
    pk_a, _ = signal.find_peaks(i_avg, height=thr)
    top_b = pk_b[np.argsort(i_biref[pk_b])[::-1][:3]]
    top_a = pk_a[np.argsort(i_avg[pk_a])[::-1][:3]]
    shift_ps = float(np.max(np.abs(grid.t[top_b] - grid.t[top_a]))) * 1e12

    if make_plot:
        _plot_nrz(grid, p_biref, p_avg, i_biref, i_avg)

    return {
        "periods_run": 2,
        "total_length_km": 200.0,
        "grid_N": grid.N,
        "current_l2_rel_biref_vs_manakov": l2,
        "pulse_peak_shift_ps_max": shift_ps,
        "peak_power_biref_mW": float(p_biref.max() * 1e3),
        "peak_power_avg_mW": float(p_avg.max() * 1e3),
    }


def _plot_nrz(grid: TemporalGrid, p_biref: NDArray, p_avg: NDArray,
              i_biref: NDArray, i_avg: NDArray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_ns = grid.t * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(t_ns, p_biref * 1e3, lw=1, label="random birefringence (SU(2) frames)")
    ax.plot(t_ns, p_avg * 1e3, lw=1, ls="--", alpha=0.85, label="Poincaré-averaged Manakov")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("optical power (mW)")
    ax.set_title("Fig. 4 analogue: NRZ output after {0:.0f} km of the dispersion map".format(200.0))
    ax.legend()
    ax = axes[1]
    ax.plot(t_ns, i_biref, lw=1, label="detected + 5 GHz filter (birefringent)")
    ax.plot(t_ns, i_avg, lw=1, ls="--", alpha=0.85, label="detected + 5 GHz filter (Manakov)")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("filtered detected current (W)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "fig4_nrz_dispersion_map.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# validate() / CLI
# ---------------------------------------------------------------------------


def validate(make_plot: bool = True) -> dict:
    phys = PhysicalSetup(wavelength_nm=1550.0, n2=2.6e-20, a_eff_um2=52.0)
    with open(PARAMETERS) as fh:
        tol = json.load(fh)["tolerances"]

    eq30 = check_eq30(phys, make_plot=make_plot)
    fig5 = check_fig5(phys, make_plot=make_plot)
    nrz = check_nrz(phys, make_plot=make_plot)

    # --- asserts against the analytic targets / published claims ----------
    assert eq30["peak_ratio_by_construction"] == 9.0 / 8.0
    assert abs(eq30["scalar_peak_rel"]) < tol["manakov_soliton_peak_rel"], eq30
    assert eq30["scalar_shape_l2"] < tol["manakov_soliton_shape_l2"], eq30
    assert abs(eq30["manakov_peak_rel"]) < tol["manakov_soliton_peak_rel"], eq30
    assert eq30["manakov_shape_l2"] < tol["manakov_soliton_shape_l2"], eq30

    assert fig5["peak_fluctuation_pct_mean"] < tol["random_peak_fluctuation_pct"], fig5
    assert fig5["shape_l2_worst"] < tol["random_shape_l2"], fig5
    assert fig5["energy_drift_max"] < tol["energy_drift"], fig5

    assert nrz["current_l2_rel_biref_vs_manakov"] < tol["random_vs_manakov_current_l2"], nrz
    assert nrz["pulse_peak_shift_ps_max"] < tol["pulse_peak_time_shift_ps"], nrz

    return {"eq30": eq30, "fig5": fig5, "nrz": nrz}


def main() -> int:
    import sys

    make_plot = "--no-plot" not in sys.argv[1:]
    print(json.dumps(validate(make_plot=make_plot), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
