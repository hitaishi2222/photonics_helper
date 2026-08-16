"""
Example: GNLSE Soliton Evolution
=================================

Demonstrates fundamental soliton propagation — the balance between anomalous
dispersion and Kerr nonlinearity that produces a shape-preserving pulse.

A fundamental soliton satisfies:

    N² = γ · P₀ · L_D / L_NL = 1

where N is the soliton order, P₀ is peak power, L_D = T₀²/|β₂| is the
dispersion length, and L_NL = 1/(γ·P₀) is the nonlinear length.

For N=1 the pulse propagates unchanged. For N>1 the pulse undergoes
periodic compression and broadening (soliton breathing).

This example shows:
  - N=1: perfect soliton (shape-preserving)
  - N=3: soliton fission — higher-order pulse splits into N fundamental solitons
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.base import Wavelength, Time, Length, Area


def make_soliton_pulse(wavelength_nm=1064, T0_fs=200, peak_power_W=1000):
    """Create a sech-shaped soliton pulse.

    A₀ = sqrt(P₀) gives peak power P₀ in the GNLSE convention (|A|² = power).
    """
    grid = TemporalGrid(N=2**11, Tmax=Time(5 * T0_fs * 1e-15, "s"))
    A0 = np.sqrt(peak_power_W)
    env = Envelope(shape="sech", peak_amplitude=A0, pulse_width=Time(T0_fs, "fs"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(wavelength_nm, "nm"),
    )
    return pulse


def make_soliton_fiber(beta2_ps2_per_km=-2.0, length_m=0.1):
    """Silica-like fiber with anomalous dispersion (negative β₂).

    β₂ in ps²/m (matching get_betas() convention).
    """
    beta2_ps2_m = beta2_ps2_per_km * 1e-3
    n2 = 2.6e-20  # m²/W
    omega0 = 2 * np.pi * 299792458.0 / (1064e-9)
    gamma = n2 * omega0 / (299792458.0 * 80e-12)  # ~1.45 1/(W·km)

    A_eff = n2 * omega0 / (gamma * 299792458.0)

    return (
        FiberProfile(
            n2=n2,
            alpha=0.0,
            A_eff=Area(A_eff, "m^2"),
            length=length_m,
        ),
        beta2_ps2_m,
    )


def compute_soliton_order(beta2_si, gamma, T0, P0):
    """Compute soliton order N = sqrt(γ·P₀·T₀² / |β₂|)."""
    L_D = T0**2 / abs(beta2_si)
    L_NL = 1.0 / (gamma * P0)
    return np.sqrt(L_D / L_NL)


def main():
    # ── Parameters ──────────────────────────────────────────────────────────
    central_wl = Wavelength(1064, "nm")
    T0 = Time(200, "fs")
    grid = TemporalGrid(N=2**11, Tmax=Time(10 * T0.as_s, "s"))

    # Fiber: anomalous dispersion, silica-like gamma
    n2 = 2.6e-20
    A_eff = Area(80, "um^2")
    omega0 = 2 * np.pi * 299792458.0 / central_wl.as_m
    gamma = n2 * omega0 / (299792458.0 * A_eff.as_m2)
    beta2_ps2_m = -2.0 * 1e-3  # -2 ps²/km = -0.002 ps²/m (engine units)

    print(f"γ = {gamma*1000:.3f} 1/(W·km)")
    print(f"β₂ = {beta2_ps2_m*1e3:.1f} ps²/km ({beta2_ps2_m} ps²/m)")

    # ── Case 1: Fundamental soliton (N=1) ──────────────────────────────────

    beta2_si = beta2_ps2_m * 1e-24  # ps²/m → s²/m for P0 calculation
    P0_N1 = abs(beta2_si) / (gamma * T0.as_s**2)  # N=1 peak power (SI)
    print(f"\n— N=1 soliton: P₀ = {P0_N1:.1f} W —")

    L_D = Length(T0.as_s**2 / abs(beta2_si), "m")

    pulse_N1 = make_soliton_pulse(
        wavelength_nm=central_wl.as_nm, T0_fs=T0.as_fs, peak_power_W=P0_N1
    )
    fiber = FiberProfile(n2=n2, alpha=0.0, A_eff=A_eff, length=L_D)
    betas = np.array([beta2_ps2_m])

    solver_N1 = GNLSESolver(
        pulse=pulse_N1,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_N1.propagate(num_steps=100)

    # ── Case 2: Higher-order soliton (N=3) ──────────────────────────────────

    P0_N3 = P0_N1 * 3**2  # N=3 requires N² times the power
    print(f"— N=3 soliton: P₀ = {P0_N3:.0f} W —")

    pulse_N3 = make_soliton_pulse(
        wavelength_nm=central_wl.as_nm, T0_fs=T0.as_fs, peak_power_W=P0_N3
    )

    solver_N3 = GNLSESolver(
        pulse=pulse_N3,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_N3.propagate(num_steps=100)

    # ── Plot 1: N=1 soliton — shape preservation ───────────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        "Fundamental Soliton (N=1) — Shape Preservation", fontsize=14, fontweight="bold"
    )

    t_ps = pulse_N1.grid.t * 1e12

    # Temporal profile at selected distances
    z_steps_N1 = np.linspace(0, fiber.length.as_m, len(solver_N1.evolution))
    display_indices = [0, 10, 30, 50, 70, 99]
    colors = plt.cm.viridis(np.linspace(0, 1, len(display_indices)))

    for idx, di in enumerate(display_indices):
        wave = solver_N1.evolution[di]
        axes[0].plot(
            t_ps,
            np.abs(wave.envelope_field),
            color=colors[idx],
            linewidth=1.5,
            label=f"z={z_steps_N1[di]*1e3:.2f} mm",
        )
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Field amplitude")
    axes[0].set_title("Temporal profile at selected distances")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim([-2, 2])  # Zoom to ±2 ps

    # Peak power evolution
    peak_powers = np.array([w.peak_power() for w in solver_N1.evolution])
    z_mm = z_steps_N1 * 1e3
    axes[1].plot(z_mm, peak_powers, "b-", linewidth=2.0, label="N=1 peak power")
    axes[1].axhline(
        y=P0_N1,
        color="r",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"Input P₀ = {P0_N1:.1f} W",
    )
    axes[1].set_xlabel("Propagation distance (mm)")
    axes[1].set_ylabel("Peak power (W)")
    axes[1].set_title("Peak power is constant (N=1 soliton)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("examples/images/13_gnlse_soliton_N1.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/images/13_gnlse_soliton_N1.png")
    plt.close()

    # ── Plot 2: N=3 soliton — soliton fission ──────────────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        "Higher-Order Soliton (N=3) — Breathing and Fission",
        fontsize=14,
        fontweight="bold",
    )

    t_ps_N3 = pulse_N3.grid.t * 1e12
    z_steps_N3 = np.linspace(0, fiber.length.as_m, len(solver_N3.evolution))

    # Temporal profile at selected distances
    display_indices = [0, 10, 25, 50, 75, 99]
    for idx, di in enumerate(display_indices):
        wave = solver_N3.evolution[di]
        axes[0].plot(
            t_ps_N3,
            np.abs(wave.envelope_field),
            color=colors[idx],
            linewidth=1.5,
            label=f"z={z_steps_N3[di]*1e3:.2f} mm",
        )
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Field amplitude")
    axes[0].set_title("Temporal profile — pulse breathes and broadens")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim([-3, 3])  # Zoom to ±3 ps

    # Peak power evolution
    peak_powers_N3 = np.array([w.peak_power() for w in solver_N3.evolution])
    z_mm_N3 = z_steps_N3 * 1e3
    axes[1].plot(z_mm_N3, peak_powers_N3, "r-", linewidth=2.0, label="N=3 peak power")
    axes[1].axhline(
        y=P0_N1,
        color="b",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"N=1 power = {P0_N1:.1f} W",
    )
    axes[1].axhline(
        y=P0_N3,
        color="g",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"N=3 input = {P0_N3:.0f} W",
    )
    axes[1].set_xlabel("Propagation distance (mm)")
    axes[1].set_ylabel("Peak power (W)")
    axes[1].set_title("Peak power oscillates — soliton fission regime")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("examples/images/13_gnlse_soliton_N3.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/images/13_gnlse_soliton_N3.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    print(f"\n— Soliton parameters —")
    print(f"Dispersion length L_D = T₀²/|β₂| = {L_D.as_m*1e3:.1f} mm")
    print(f"Nonlinear length L_NL = 1/(γ·P₀) = {1/(gamma*P0_N1)*1e3:.1f} mm")
    N1 = compute_soliton_order(beta2_si, gamma, T0.as_s, P0_N1)
    N3 = compute_soliton_order(beta2_si, gamma, T0.as_s, P0_N3)
    print(f"Soliton order N=1: confirmed ({N1:.2f})")
    print(f"Soliton order N=3: confirmed ({N3:.2f})")


if __name__ == "__main__":
    main()
