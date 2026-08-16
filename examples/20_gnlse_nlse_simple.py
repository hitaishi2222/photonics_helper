"""
Example: NLSE_simple (photonics_helper)
========================================

Replicates the laserfun ``NLSE_simple`` demo using only photonics_helper.
Layout matches the standard laserfun four-panel figure:

  top-left    — initial / final spectrum (frequency, dB)
  top-right   — initial / final temporal profile (dB)
  bottom-left — spectral evolution contour
  bottom-right — temporal evolution contour
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.gnlse import (
    FiberProfile,
    GNLSESolver,
    plot_spectral_evolution,
    plot_temporal_evolution,
)
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.base import Wavelength, Time, Length

# ── Parameters (match laserfun NLSE_simple) ───────────────────────────────────

CENTRAL_WL_NM = 1550.0
FIBER_LENGTH_M = 0.010
FWHM_PS = 0.05
PULSE_ENERGY_J = 50e-12
GAMMA_W_M = 1.0
N2 = 2.6e-20
BETAS = np.array([-0.12, 0.0, 5e-6])
TIME_WINDOW_PS = 7.0
N = 2**11

# Plot limits (similar to laserfun results.plot defaults)
F_MIN_THZ = 100.0
F_MAX_THZ = 300.0
T_MIN_PS = -0.9
T_MAX_PS = 1.1
DYNAMIC_RANGE_DB = 40.0


def make_pulse() -> Wave:
    grid = TemporalGrid(N=N, Tmax=Time(TIME_WINDOW_PS * 1e-12, "s"))
    env = Envelope.from_fwhm("sech", peak_amplitude=1.0, fwhm=Time(FWHM_PS, "ps"))
    T0 = env.pulse_width.as_s
    env.peak_amplitude = np.sqrt(PULSE_ENERGY_J / (2.0 * T0))
    return Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(CENTRAL_WL_NM, "nm"),
    )


def spectrum_db_thz(A: np.ndarray, grid: TemporalGrid, omega0: float) -> tuple[np.ndarray, np.ndarray]:
    """Absolute-frequency spectrum (THz) in dB relative to peak."""
    spec = np.abs(grid.fft(A)) ** 2
    f_thz = (grid.w + omega0) / (2 * np.pi) / 1e12
    idx = np.argsort(f_thz)
    f_thz, spec = f_thz[idx], spec[idx]
    mask = (f_thz >= F_MIN_THZ) & (f_thz <= F_MAX_THZ)
    sp_db = 10 * np.log10(spec[mask] / spec.max() + 1e-30)
    return f_thz[mask], sp_db


def temporal_db(A: np.ndarray, t_ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Temporal intensity in dB relative to peak."""
    I = np.abs(A) ** 2
    mask = (t_ps >= T_MIN_PS) & (t_ps <= T_MAX_PS)
    I_db = 10 * np.log10(I[mask] / I.max() + 1e-30)
    return t_ps[mask], I_db


def main() -> None:
    pulse = make_pulse()
    omega0 = pulse.central_frequency
    fiber = FiberProfile.from_gamma(
        gamma=GAMMA_W_M,
        n2=N2,
        omega0=omega0,
        length=Length(FIBER_LENGTH_M, "m"),
    )

    num_steps = GNLSESolver.estimate_num_steps(
        pulse, fiber, BETAS, include_self_steepening=True
    )
    print(f"Propagating with {num_steps} split-steps …")

    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=BETAS,
        include_raman=False,
        include_self_steepening=True,
        include_tpa=False,
    )
    solver.propagate(num_steps=num_steps)

    t_ps = pulse.grid.t * 1e12
    A_in = pulse.envelope_field
    A_out = solver.evolution[-1].envelope_field

    f_in, sp_in = spectrum_db_thz(A_in, pulse.grid, omega0)
    f_out, sp_out = spectrum_db_thz(A_out, pulse.grid, omega0)
    t_in, I_in = temporal_db(A_in, t_ps)
    t_out, I_out = temporal_db(A_out, t_ps)

    # ── laserfun-style 2×2 figure ───────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # Top-left: spectrum
    ax = axes[0, 0]
    ax.plot(f_in, sp_in, "b-", lw=1.5, label="Initial")
    ax.plot(f_out, sp_out, "r-", lw=1.5, label="Final")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Intensity (dB, norm.)")
    ax.set_xlim(F_MIN_THZ, F_MAX_THZ)
    ax.set_ylim(-80, 0)
    ax.grid(True, alpha=0.3)

    # Top-right: temporal
    ax = axes[0, 1]
    ax.plot(t_in, I_in, "b-", lw=1.5, label="Initial")
    ax.plot(t_out, I_out, "r-", lw=1.5, label="Final")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Intensity (dB, norm.)")
    ax.set_xlim(T_MIN_PS, T_MAX_PS)
    ax.set_ylim(-80, 0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: spectral evolution
    plot_spectral_evolution(
        solver,
        ax=axes[1, 0],
        f_min_THz=F_MIN_THZ,
        f_max_THz=F_MAX_THZ,
        n_points=400,
        dynamic_range_db=DYNAMIC_RANGE_DB,
        cmap="viridis",
        z_scale="mm",
        use_imshow=True,
    )
    axes[1, 0].set_title("")

    # Bottom-right: temporal evolution
    plot_temporal_evolution(
        solver,
        ax=axes[1, 1],
        t_min=T_MIN_PS,
        t_max=T_MAX_PS,
        dynamic_range_db=DYNAMIC_RANGE_DB,
        cmap="viridis",
        z_scale="mm",
        use_imshow=True,
    )
    axes[1, 1].set_title("")

    root = Path(__file__).resolve().parents[1]
    out = root / "examples/images/20_gnlse_nlse_simple.png"
    comparison_out = root / "comparison_output/NLSE_simple.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(comparison_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {comparison_out}")


if __name__ == "__main__":
    main()
