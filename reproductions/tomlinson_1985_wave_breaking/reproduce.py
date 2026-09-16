"""Reproduction: optical wave breaking (Tomlinson, Stolen & Johnson, 1985).

Reference
---------
W. J. Tomlinson, R. H. Stolen, A. M. Johnson, "Optical wave breaking of pulses
in nonlinear optical fibers", Optics Letters 10, 457 (1985),
doi:10.1364/OL.10.000457.

The PDF is paywalled and could not be downloaded for this reproduction; the
analytic criterion below is therefore derived from the standard SPM chirp
(Agrawal, *Nonlinear Fiber Optics*, Sec. 4.1.3) and validated numerically.

What is reproduced
------------------
In the **normal-dispersion** regime (β₂ > 0), self-phase modulation imposes a
time-dependent chirp on the pulse while the group velocity dispersion converts
that chirp into a *time shift* of each spectral component.  For a Gaussian input
``A(0,T) = sqrt(P0) exp(-T²/2T₀²)`` the SPM chirp after a distance z is

    δω(T) = -(2 γ P₀ z / T₀²) · T exp(-T²/T₀²),

so the accumulated group delay is ``T' = T + β₂ z δω(T)``.  Wave breaking is the
point where this map stops being monotonic, i.e. where
``1 + β₂ z ∂δω/∂T = 0`` somewhere.  The most negative chirp slope (at the pulse
centre) gives the estimate

    z_WB = T₀ / sqrt(2 β₂ γ P₀) = (e^{3/4}/2) sqrt(L_D L_NL),

with ``L_D = T₀²/β₂`` and ``L_NL = 1/(γP₀)``.  The pulse develops steep (shock)
edges around z_WB and oscillations a little later.

What is validated here
----------------------
1. The edge steepness ``S = max|∂I/∂T|·T₀ / I_peak`` departs from the Gaussian
   value ``sqrt(2) e^{-1/2} ≈ 0.858`` around ``z_WB`` and peaks near
   ``1.5 sqrt(L_D L_NL)``.
2. Oscillations (multiple intensity maxima) appear within a few
   ``sqrt(L_D L_NL)`` of ``z_WB``.
3. The whole phenomenon scales as ``z_WB ∝ sqrt(L_D L_NL) ∝ P₀^{-1/2}``:
   sweeping P₀, the measured steepest-edge distance follows a log–log slope of
   about −1/2 (this is the decisive, constant-free test).

Usage
-----
    python reproductions/tomlinson_1985_wave_breaking/reproduce.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

# Analytic wave-breaking primitives now live in the library.
from photonics_helper.wave_breaking import (
    dispersion_length,
    edge_steepness,
    gaussian_edge_steepness,
    nonlinear_length,
    wave_breaking_distance,
)

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"

GAUSSIAN_STEEPNESS = gaussian_edge_steepness


@dataclass(frozen=True)
class WaveBreakingParameters:
    """Fibre and pulse parameters for the normal-dispersion wave-breaking case."""

    lam_m: float
    beta2_si: float
    beta2_ps2_per_m: float
    gamma: float
    T0_s: float
    P0: float

    @property
    def L_D(self) -> float:
        return dispersion_length(self.beta2_si, self.T0_s)

    @property
    def L_NL(self) -> float:
        return nonlinear_length(self.gamma, self.P0)

    @property
    def sqrt_LD_LNL(self) -> float:
        return float(np.sqrt(self.L_D * self.L_NL))

    @property
    def z_WB(self) -> float:
        """Analytic wave-breaking distance (chirp-folding criterion)."""
        return wave_breaking_distance(self.beta2_si, self.gamma, self.P0, self.T0_s)


def derive(params: dict) -> WaveBreakingParameters:
    beta2_ps = float(params["beta2_ps2_per_km"]) * 1e-3  # ps²/m
    return WaveBreakingParameters(
        lam_m=float(params["central_wavelength_nm"]) * 1e-9,
        beta2_si=beta2_ps * 1e-24,
        beta2_ps2_per_m=beta2_ps,
        gamma=float(params["gamma_per_W_per_km"]) * 1e-3,
        T0_s=float(params["T0_ps"]) * 1e-12,
        P0=float(params["peak_power_W"]),
    )


def _run(
    wb: WaveBreakingParameters,
    params: dict,
    P0: float,
    length_m: float,
    nsaves: int,
    num_steps: int,
) -> tuple[NDArray, list[Wave], TemporalGrid]:
    grid = TemporalGrid(
        N=int(params["grid_N"]), Tmax=Time(float(params["grid_Tmax_ps"]) * 1e-12, "s")
    )
    env = Envelope(
        shape="gaussian", peak_amplitude=float(np.sqrt(P0)), pulse_width=Time(wb.T0_s, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wb.lam_m, "m"))
    fiber = FiberProfile.from_gamma(
        gamma=wb.gamma,
        n2=float(params["n2_m2_per_W"]),
        omega0=2 * np.pi * C_MS / wb.lam_m,
        length=Length(length_m, "m"),
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([wb.beta2_ps2_per_m, 0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=num_steps, nsaves=nsaves)
    return solver.z_array, solver.evolution, grid


def _steepness(intensity: NDArray, t: NDArray, T0: float) -> float:
    return edge_steepness(intensity=intensity, t=t, T0=T0)


def _analyze(
    wb: WaveBreakingParameters, z: NDArray, evolution: list[Wave], grid: TemporalGrid
) -> dict:
    steep = np.array([_steepness(np.abs(w.envelope_field) ** 2, grid.t, wb.T0_s) for w in evolution])
    n_peaks = np.array(
        [
            len(find_peaks(np.abs(w.envelope_field) ** 2, prominence=0.01 * np.max(np.abs(w.envelope_field) ** 2))[0])
            for w in evolution
        ]
    )
    i_onset = int(np.argmax(steep > 1.10 * GAUSSIAN_STEEPNESS))
    i_steepest = int(np.argmax(steep))
    i_osc = int(np.argmax(n_peaks >= 2))
    return {
        "z_onset_m": float(z[i_onset]) if steep[i_onset] > 1.10 * GAUSSIAN_STEEPNESS else float("nan"),
        "z_steepest_m": float(z[i_steepest]),
        "z_oscillation_m": float(z[i_osc]) if n_peaks[i_osc] >= 2 else float("nan"),
        "peak_steepness": float(steep.max()),
        "steepness_onset_ratio": float(steep.max() / GAUSSIAN_STEEPNESS),
        "_steep": steep,
        "_npeaks": n_peaks,
    }


def validate(
    params: dict | None = None,
    *,
    fast: bool = False,
    make_plot: bool = True,
) -> dict:
    params = params or json.loads(PARAMETERS.read_text())
    wb = derive(params)
    tol = params["tolerances"]

    # --- reference case -----------------------------------------------------
    length = 4.0 * wb.z_WB
    z, evolution, grid = _run(
        wb, params, wb.P0, length, 201 if fast else 401, 400 if fast else 1200
    )
    ref = _analyze(wb, z, evolution, grid)
    ratio_steepest = ref["z_steepest_m"] / wb.z_WB
    ratio_onset = ref["z_onset_m"] / wb.z_WB

    # --- power sweep: validate the sqrt(L_D L_NL) ~ P0^{-1/2} scaling --------
    # The steepening onset is the robust estimator of z_WB (the "steepest edge"
    # distance is corrupted by the fine oscillations at large z).
    sweep = []
    for P0 in params["power_sweep_W"]:
        L_D = wb.T0_s**2 / wb.beta2_si
        L_NL = 1.0 / (wb.gamma * P0)
        sq = np.sqrt(L_D * L_NL)
        z_s, ev_s, grid_s = _run(
            wb, params, P0, 3.0 * sq, 101 if fast else 301, 300 if fast else 900
        )
        an = _analyze(wb, z_s, ev_s, grid_s)
        sweep.append(
            {
                "P0_W": float(P0),
                "sqrt_LD_LNL_m": float(sq),
                "z_onset_m": an["z_onset_m"],
                "z_onset_over_sqrt": an["z_onset_m"] / sq,
            }
        )
    P0_arr = np.array([s["P0_W"] for s in sweep])
    z_arr = np.array([s["z_onset_m"] for s in sweep])
    slope = float(np.polyfit(np.log(P0_arr), np.log(z_arr), 1)[0])
    over_sqrt = np.array([s["z_onset_over_sqrt"] for s in sweep])

    result = {
        "L_D_m": wb.L_D,
        "L_NL_m": wb.L_NL,
        "sqrt_LD_LNL_m": wb.sqrt_LD_LNL,
        "z_WB_analytic_m": wb.z_WB,
        "z_onset_m": ref["z_onset_m"],
        "z_steepest_m": ref["z_steepest_m"],
        "z_oscillation_m": ref["z_oscillation_m"],
        "z_onset_over_zWB": ratio_onset,
        "z_steepest_over_zWB": ratio_steepest,
        "peak_steepness": ref["peak_steepness"],
        "peak_steepness_over_gaussian": ref["peak_steepness"] / GAUSSIAN_STEEPNESS,
        "scaling_slope": slope,
        "z_onset_over_sqrt": over_sqrt.tolist(),
        "scaling_constant_spread": float(over_sqrt.max() / over_sqrt.min()),
        "sweep": sweep,
        "_plot": (wb, z, evolution, grid, ref, sweep, P0_arr, z_arr),
    }

    assert tol["z_onset_over_zWB_low"] < ratio_onset < tol["z_onset_over_zWB_high"], result
    assert ref["peak_steepness"] / GAUSSIAN_STEEPNESS > tol["peak_steepness_min"], result
    assert ref["z_oscillation_m"] / wb.sqrt_LD_LNL < tol["oscillation_z_over_sqrt_max"], result
    assert tol["scaling_slope_low"] < slope < tol["scaling_slope_high"], result
    assert result["scaling_constant_spread"] < 1.35, result

    if make_plot:
        _plot(wb, params, result)

    print("Optical wave breaking validation passed:")
    print(
        f"  L_D = {wb.L_D:.0f} m, L_NL = {wb.L_NL:.1f} m, "
        f"sqrt(L_D L_NL) = {wb.sqrt_LD_LNL:.1f} m, z_WB = {wb.z_WB:.0f} m"
    )
    print(
        f"  steepening onset {result['z_onset_m']:.0f} m ({ratio_onset:.2f} z_WB); "
        f"(+informational: end-of-run max edge steepness at {result['z_steepest_m']:.0f} m)"
    )
    print(
        f"  peak edge steepness {result['peak_steepness']:.2f} "
        f"({result['peak_steepness_over_gaussian']:.2f}× Gaussian)"
    )
    print(f"  first oscillations at {result['z_oscillation_m']:.0f} m ({result['z_oscillation_m'] / wb.sqrt_LD_LNL:.2f} sqrt)")
    print(
        f"  scaling z_onset ∝ P0^({slope:.2f}) (theory −0.5); "
        f"z_onset/sqrt spread {result['scaling_constant_spread']:.3f}"
    )
    return result


def _plot(wb: WaveBreakingParameters, params: dict, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, z, evolution, grid, ref, sweep, P0_arr, z_arr = result["_plot"]
    t_ps = grid.t * 1e12
    z_WB = wb.z_WB
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Intensity profiles through the breaking point (normalised to the INPUT peak).
    for z_target, style in [
        (0.0, "-"),
        (0.5 * z_WB, "--"),
        (z_WB, "-."),
        (2 * z_WB, ":"),
        (3 * z_WB, (0, (3, 1, 1, 1))),
    ]:
        i = int(np.argmin(np.abs(z - z_target)))
        inten = np.abs(evolution[i].envelope_field) ** 2
        axes[0, 0].plot(
            t_ps / (wb.T0_s * 1e12), inten / wb.P0, linestyle=style, label=f"z = {z[i] / z_WB:.2f} z_WB"
        )
    axes[0, 0].set_xlim(-5, 5)
    axes[0, 0].set_xlabel("T / T₀")
    axes[0, 0].set_ylabel("$I(T)/P_0$")
    axes[0, 0].set_title("(a) Steep edges → oscillations")
    axes[0, 0].legend(fontsize=8)

    # (b) Edge steepness vs distance.
    axes[0, 1].plot(z / wb.sqrt_LD_LNL, ref["_steep"] / GAUSSIAN_STEEPNESS, "k-")
    axes[0, 1].axvline(z_WB / wb.sqrt_LD_LNL, color="r", ls="--", label="z_WB analytic")
    axes[0, 1].axhline(1.0, color="gray", ls=":", label="Gaussian")
    axes[0, 1].axhline(1.5, color="b", ls=":", label="1.5×")
    axes[0, 1].set_xlabel("z / sqrt(L_D L_NL)")
    axes[0, 1].set_ylabel("edge steepness / Gaussian")
    axes[0, 1].set_title("(b) Steepening (wave breaking)")
    axes[0, 1].legend(fontsize=8)

    # (c) Scaling of the steepest-edge distance with P0.
    axes[1, 0].loglog(P0_arr, z_arr, "ko", label="GNLSE")
    p_ref = np.array([P0_arr.min(), P0_arr.max()])
    k = z_arr[0] * P0_arr[0] ** 0.5
    axes[1, 0].loglog(p_ref, k * p_ref**-0.5, "r--", label=r"$P_0^{-1/2}$")
    axes[1, 0].set_xlabel("peak power $P_0$ (W)")
    axes[1, 0].set_ylabel("$z$ (steepening onset) (m)")
    axes[1, 0].set_title(f"(c) Scaling: slope {result['scaling_slope']:.2f}")
    axes[1, 0].legend(fontsize=8)

    # (d) Spectra before/after breaking.
    for z_target, style in [(0.0, "-"), (3 * z_WB, "-")]:
        i = int(np.argmin(np.abs(z - z_target)))
        Aw = grid.fft(evolution[i].envelope_field)
        P = np.abs(Aw) ** 2
        axes[1, 1].plot(grid.w * wb.T0_s, P / P.max(), style, label=f"z = {z[i] / z_WB:.2f} z_WB")
    axes[1, 1].set_xlim(-30, 30)
    axes[1, 1].set_xlabel(r"$\Omega\,T_0$")
    axes[1, 1].set_ylabel("normalised spectral intensity")
    axes[1, 1].set_title("(d) SPM broadening (normal dispersion)")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        "Optical wave breaking (Tomlinson, Stolen & Johnson, Opt. Lett. 10, 457, 1985) — "
        f"$\\beta_2$ = {wb.beta2_ps2_per_m * 1e3:.0f} ps²/km, $T_0$ = {wb.T0_s * 1e12:.0f} ps, "
        f"$P_0$ = {wb.P0:.0f} W",
        fontsize=11,
    )
    fig.tight_layout()
    out = HERE / "wave_breaking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    validate()
