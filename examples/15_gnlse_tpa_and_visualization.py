"""
Example: GNLSE Two-Photon Absorption and Visualization
=======================================================

Demonstrates two-photon absorption (TPA) and the visualization tools
provided by the GNLSE module:

- **TPA** removes energy from the pulse as carriers are generated. The
  attenuation depends on the TPA cross-section σ and carrier lifetime τ_c.
  Important in materials with small bandgaps (e.g., silicon at 1550 nm).

- **Visualization**: the module provides waterfall plots, spectrum-vs-distance
  plots, and intensity-metrics plots for analyzing propagation results.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.gnlse import (
    FiberProfile,
    GNLSESolver,
    plot_waterfall,
    plot_spectrum_vs_distance,
    plot_intensity_metrics,
)
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.base import Wavelength, Time, Length, Area


def make_pulse(wavelength_nm=1550, T0_fs=200, peak_power_W=1e6):
    """Create a Gaussian pulse with high peak power for TPA."""
    grid = TemporalGrid(N=2**10, Tmax=Time(5 * T0_fs * 1e-15, "s"))
    A0 = np.sqrt(peak_power_W)
    env = Envelope(shape="gaussian", peak_amplitude=A0, pulse_width=Time(T0_fs, "fs"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(wavelength_nm, "nm"),
    )
    return pulse


def main():
    # ── Fiber parameters demonstrating TPA ─────────────────────────────────
    # Using parameters that show visible TPA effects
    central_wl = Wavelength(1550, "nm")
    T0 = Time(200, "fs")
    fiber_length = Length(2, "mm")
    A_eff = Area(80, "um^2")

    omega0 = 2 * np.pi * 299792458.0 / central_wl.as_m
    n2 = 2.6e-20  # m²/W — silica nonlinear index
    # Use a moderate TPA cross-section — enhanced for visualization but
    # still shows gradual decay over 2 mm propagation (silicon ~4.5e-11 m²/W)
    sigma_tpa = 1e-8  # m²/W — enhanced for visualization
    tau_c = Time(1, "ns")  # carrier lifetime (nanoseconds)
    beta2 = -2.0 * 1e-3  # -2 ps²/km = -0.002 ps²/m

    gamma = n2 * omega0 / (299792458.0 * A_eff.as_m2)

    print("Fiber with TPA:")
    print(f"  γ = {gamma*1e3:.3f} 1/(W·km)")
    print("  β₂ = -2.0 ps²/km")
    print(f"  σ_TPA = {sigma_tpa:.1e} m²/W (enhanced for visualization)")
    print(f"  τ_c = {tau_c.as_s*1e9:.0f} ns")

    # ── Case 1: No TPA ──────────────────────────────────────────────────────

    pulse = make_pulse(wavelength_nm=central_wl.as_nm, T0_fs=T0.as_fs, peak_power_W=1e6)
    fiber_no_tpa = FiberProfile(
        n2=n2,
        alpha=0.0,
        A_eff=A_eff,
        length=fiber_length,
        sigma_tpa=0.0,
    )

    solver_no_tpa = GNLSESolver(
        pulse=pulse,
        fiber=fiber_no_tpa,
        betas=np.array([beta2]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_no_tpa.propagate(num_steps=50)

    # ── Case 2: With TPA ────────────────────────────────────────────────────

    fiber_tpa = FiberProfile(
        n2=n2,
        alpha=0.0,
        A_eff=A_eff,
        length=fiber_length,
        sigma_tpa=sigma_tpa,
        carrier_lifetime=tau_c,
    )

    solver_tpa = GNLSESolver(
        pulse=pulse,
        fiber=fiber_tpa,
        betas=np.array([beta2]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=True,
    )
    solver_tpa.propagate(num_steps=50)

    # ── Plot 1: Waterfall plot (envelope evolution) ─────────────────────────

    fig_no_tpa = plot_waterfall(solver_no_tpa, dB=True)
    fig_no_tpa.savefig(
        "examples/images/15_gnlse_waterfall_no_tpa.png", dpi=150, bbox_inches="tight"
    )
    print("Saved: examples/images/15_gnlse_waterfall_no_tpa.png")
    plt.close(fig_no_tpa)

    fig_tpa = plot_waterfall(solver_tpa, dB=True)
    fig_tpa.savefig(
        "examples/images/15_gnlse_waterfall_with_tpa.png", dpi=150, bbox_inches="tight"
    )
    print("Saved: examples/images/15_gnlse_waterfall_with_tpa.png")
    plt.close(fig_tpa)

    # ── Plot 2: Spectrum vs distance ────────────────────────────────────────

    fig_spec = plot_spectrum_vs_distance(solver_tpa, dB=True)
    fig_spec.savefig(
        "examples/images/15_gnlse_spectrum_vs_distance.png", dpi=150, bbox_inches="tight"
    )
    print("Saved: examples/images/15_gnlse_spectrum_vs_distance.png")
    plt.close(fig_spec)

    # ── Plot 3: Intensity metrics ───────────────────────────────────────────

    fig_metrics = plot_intensity_metrics(solver_tpa)
    fig_metrics.savefig(
        "examples/images/15_gnlse_intensity_metrics.png", dpi=150, bbox_inches="tight"
    )
    print("Saved: examples/images/15_gnlse_intensity_metrics.png")
    plt.close(fig_metrics)

    # ── Plot 4: TPA effect comparison — energy decay ───────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        "Two-Photon Absorption in Silicon Waveguide", fontsize=14, fontweight="bold"
    )

    z_mm_no_tpa = solver_no_tpa.z_array * 1e3
    z_mm_tpa = solver_tpa.z_array * 1e3

    # Panel 1: Peak power comparison
    pp_no_tpa = np.array([w.peak_power() for w in solver_no_tpa.evolution])
    pp_tpa = np.array([w.peak_power() for w in solver_tpa.evolution])

    axes[0].plot(z_mm_no_tpa, pp_no_tpa, "b-", linewidth=2.0, label="No TPA")
    axes[0].plot(z_mm_tpa, pp_tpa, "r-", linewidth=2.0, label="With TPA")
    axes[0].set_xlabel("Propagation distance (mm)")
    axes[0].set_ylabel("Peak power (W)")
    axes[0].set_title("TPA attenuates peak power")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # Panel 2: Energy vs distance
    def energy(wave):
        return np.sum(np.abs(wave.envelope_field) ** 2) * wave.grid.dt

    E_no_tpa = np.array([energy(w) for w in solver_no_tpa.evolution])
    E_tpa = np.array([energy(w) for w in solver_tpa.evolution])

    axes[1].plot(
        z_mm_no_tpa, E_no_tpa / E_no_tpa[0] * 100, "b-", linewidth=1.5, label="No TPA"
    )
    axes[1].plot(z_mm_tpa, E_tpa / E_tpa[0] * 100, "r-", linewidth=1.5, label="With TPA")
    axes[1].set_xlabel("Propagation distance (mm)")
    axes[1].set_ylabel("Energy (% of input)")
    axes[1].set_title("TPA causes exponential-like energy decay")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("examples/images/15_gnlse_tpa_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/images/15_gnlse_tpa_comparison.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    E0 = energy(solver_no_tpa.pulse)
    E_final_no_tpa = E_no_tpa[-1]
    E_final_tpa = E_tpa[-1]

    print("\n— Energy summary —")
    print(f"Input energy:        {E0:.4e} W·s")
    print(f"Final (no TPA):      {E_final_no_tpa:.4e}  ({E_final_no_tpa/E0*100:.2f}%)")
    print(f"Final (with TPA):    {E_final_tpa:.4e}  ({E_final_tpa/E0*100:.2f}%)")
    print(
        f"Energy lost to TPA:  {E0 - E_final_tpa:.4e}  ({(1-E_final_tpa/E0)*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
