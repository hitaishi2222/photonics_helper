"""
Example: NLSE_dudley (photonics_helper)
========================================

Replicates the laserfun ``NLSE_dudley`` supercontinuum demo (Dudley et al.,
RMP 78, 1135, 2006, Fig. 3) using only photonics_helper.

835 nm, 15 cm PCF, high-order dispersion, Kerr nonlinearity, Dudley Raman
response, and self-steepening. Produces spectral and temporal evolution
contour plots side by side.

Self-steepening (shock term) is enabled to match laserfun's ``NLSE`` default
(``shock=True``). When comparing against laserfun, both libraries must use the
same self-steepening setting; a mismatch (one on, one off) produces 10s of dB
of spectral difference and dominates any solver-level comparison.

Step count is chosen automatically from adaptive dispersion / nonlinear length
limits (no fixed floor). Only ``nsaves`` field snapshots are kept for plotting
(laserfun Dudley uses 200), while the integrator may take tens of thousands of
internal steps for accuracy. Propagation may take several minutes; plotting RAM
is ~``nsaves × N × 16`` bytes for stored fields.

Usage::

    python examples/21_gnlse_nlse_dudley.py
    python examples/21_gnlse_nlse_dudley.py --safety-factor 4   # even finer steps
    python examples/21_gnlse_nlse_dudley.py --no-progress       # plain text only
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.gnlse import (
    FiberProfile,
    GNLSESolver,
    SplitStepEngine,
    plot_spectral_evolution,
    plot_temporal_evolution,
)
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.raman import RamanResponse, RamanSpec
from photonics_helper.base import Wavelength, Time, Length

# ── Dudley / laserfun NLSE_dudley parameters ────────────────────────────────

CENTRAL_WL_NM = 835.0
FIBER_LENGTH_M = 0.15
PEAK_POWER_W = 10_000.0
GAMMA_W_M = 0.11
N2 = 2.7e-20
FWHM_PS = 0.0284 * 1.76
TIME_WINDOW_PS = 12.5
N = 2**13

# β₂ … β₁₀ in ps^n / m
BETAS = np.array([
    -11.830e-3,
    8.1038e-5,
    -9.5205e-8,
    2.0737e-10,
    -5.3943e-13,
    1.3486e-15,
    -2.5495e-18,
    3.0524e-21,
    -1.7140e-24,
])

# Dudley Raman: fR = 0.18, τ₁ = 12.2 fs, τ₂ = 32 fs
RAMAN_FR = 0.18
RAMAN_TAU1_S = 0.0122e-12
RAMAN_TAU2_S = 0.032e-12

# Output snapshots along z (laserfun NLSE_dudley uses nsaves=200)
NSAVES = 200


def make_pulse() -> Wave:
    grid = TemporalGrid(N=N, Tmax=Time(TIME_WINDOW_PS * 1e-12, "s"))
    env = Envelope.from_fwhm(
        "sech",
        peak_amplitude=np.sqrt(PEAK_POWER_W),
        fwhm=Time(FWHM_PS, "ps"),
    )
    return Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(CENTRAL_WL_NM, "nm"),
    )


def make_fiber(pulse: Wave) -> FiberProfile:
    omega0 = pulse.central_frequency
    raman_spec = RamanSpec(
        name="Silica",
        raman_shift_cm=440,
        raman_linewidth_cm=45,
        fR=RAMAN_FR,
    )
    raman = RamanResponse(
        spec=raman_spec,
        fR=RAMAN_FR,
        tau1=RAMAN_TAU1_S,
        tau2=RAMAN_TAU2_S,
        grid=pulse.grid,
    )
    return FiberProfile.from_gamma(
        gamma=GAMMA_W_M,
        n2=N2,
        omega0=omega0,
        length=Length(FIBER_LENGTH_M, "m"),
        raman_response=raman,
    )


def compute_high_accuracy_steps(
    pulse: Wave,
    fiber: FiberProfile,
    betas: np.ndarray,
    *,
    safety_factor: float = 2.0,
) -> tuple[int, float]:
    """Return (num_steps, adaptive_dz_limit_m) for accurate supercontinuum.

    Combines :meth:`GNLSESolver.estimate_num_steps` with the engine's local
    adaptive limit so ``dz`` never exceeds twice the nonlinear/dispersion bound
    (the threshold used by the solver's coarse-step warning).
    """
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=True,
        include_self_steepening=True,
    )
    dz_adaptive = engine._adaptive_step_size(engine.A)
    length = fiber.length.as_m

    from_estimate = GNLSESolver.estimate_num_steps(
        pulse,
        fiber,
        betas,
        include_self_steepening=True,
        include_raman=True,
        safety_factor=safety_factor,
    )
    # Satisfy dz_base ≤ 2·dz_adaptive (solver warning threshold).
    from_adaptive = int(np.ceil(length / max(dz_adaptive, 1e-30)))
    from_warning_limit = int(np.ceil(length / max(2.0 * dz_adaptive, 1e-30)))

    num_steps = max(from_estimate, from_adaptive, from_warning_limit)
    return num_steps, dz_adaptive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dudley et al. Fig. 3 supercontinuum (photonics_helper NLSE_dudley replica).",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=2.0,
        metavar="F",
        help="Step-density multiplier passed to estimate_num_steps (default: 2.0; "
        "larger → more steps, higher accuracy, longer runtime).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        metavar="N",
        help="Override automatic step count (default: compute from adaptive limits).",
    )
    parser.add_argument(
        "--nsaves",
        type=int,
        default=NSAVES,
        metavar="N",
        help=f"Number of z snapshots to store for plots (default: {NSAVES}, "
        "matching laserfun NLSE_dudley).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/images/21_gnlse_nlse_dudley.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    if show_progress:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            print(
                "tqdm is not installed — progress bars disabled.\n"
                "Install with: pip install tqdm   or   pip install -e \".[examples]\"",
                file=sys.stderr,
            )
            show_progress = False

    pulse = make_pulse()
    fiber = make_fiber(pulse)

    if args.num_steps is not None:
        num_steps = args.num_steps
        dz_adaptive = float("nan")
    else:
        num_steps, dz_adaptive = compute_high_accuracy_steps(
            pulse, fiber, BETAS, safety_factor=args.safety_factor
        )

    length = fiber.length.as_m
    dz_base = length / num_steps
    mem_gb = args.nsaves * N * 16 / 1e9  # complex128 fields for saved snapshots

    print("NLSE_dudley (photonics_helper)")
    print(f"  Grid points N        : {N:,}")
    print(f"  Fiber length         : {length * 100:.1f} cm")
    print(f"  Integration steps    : {num_steps:,} (target; may be higher adaptively)")
    print(f"  Saved snapshots      : {args.nsaves:,} (for contour plots)")
    if not np.isnan(dz_adaptive):
        print(f"  Adaptive dz limit    : {dz_adaptive:.3e} m")
    print(f"  Mean step dz         : {dz_base:.3e} m")
    print(f"  Est. snapshot RAM    : ~{mem_gb * 1000:.0f} MB")
    print(f"  Self-steepening      : ON (matches laserfun shock=True)")
    print()

    # include_self_steepening=True: matches laserfun's default shock=True so
    # that cross-library comparisons (Dudley et al. Fig. 3) are apples-to-apples.
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=BETAS,
        include_raman=True,
        include_self_steepening=True,
        include_tpa=False,
    )

    t0 = time.perf_counter()
    solver.propagate(
        num_steps=num_steps,
        nsaves=args.nsaves,
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nPropagation finished in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"Stored {len(solver.evolution)} field snapshots for plotting")

    # ── Dudley-style dual contour figure ────────────────────────────────────

    print("Rendering contour plots …")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "NLSE_dudley — supercontinuum (photonics_helper)",
        fontweight="bold",
    )

    plot_spectral_evolution(
        solver,
        ax=axes[0],
        wl_min=400,
        wl_max=1350,
        n_points=400,
        dynamic_range_db=40,
        cmap="jet",
        z_scale="m",
    )
    axes[0].set_title("Spectral evolution")

    plot_temporal_evolution(
        solver,
        ax=axes[1],
        t_min=-0.5,
        t_max=5.0,
        dynamic_range_db=40,
        cmap="jet",
        z_scale="m",
    )
    axes[1].set_title("Temporal evolution")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
