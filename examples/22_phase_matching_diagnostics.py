#!/usr/bin/env python3
"""Phase-matching diagnostics example.

Demonstrates:
- FWM scan (Δβ and efficiency vs signal wavelength)
- MI gain spectrum
- Dispersive wave root finder
- Simulation readiness assessment
- Spectrum overlay with PM predictions

Run with: python examples/22_phase_matching_diagnostics.py
"""

import sys
from pathlib import Path

# Prefer the repository package over any older site-packages install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.base import C_MS, PI, Wavelength, WavelengthArray, Area, Length, Time
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.gnlse import GNLSESolver, FiberProfile
from photonics_helper.fiber import Dispersion
from photonics_helper.phase_matching import (
    DispersionAdaptor,
    scan_fwm_detuning,
    mi_gain_spectrum,
    mi_sideband_frequencies,
    dispersive_wave_roots,
    assess_simulation_readiness,
    plot_fwm_efficiency,
    plot_readiness_report,
    plot_spectrum_with_pm_overlay,
)


def make_gaussian_pulse(central_wl_nm: float, T0_ps: float, power_W: float,
                        N: int = 2048) -> Wave:
    """Create a Gaussian pulse for GNLSE simulations."""
    t0 = T0_ps * 1e-12  # seconds
    dt = 4 * t0 / N
    t = np.arange(-N // 2, N // 2) * dt  # plain numpy array in seconds
    envelope_field = np.exp(-t**2 / (2 * t0**2)) * np.sqrt(power_W)

    Tmax = Time(4 * t0, "s")
    grid = TemporalGrid(N=N, Tmax=Tmax)

    central_wl = Wavelength(central_wl_nm, "nm")
    pulse = Wave(
        grid=grid,
        envelope=Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(T0_ps, "ps")),
        central_wavelength=central_wl,
    )
    pulse._pulse_train_field = envelope_field
    return pulse


def main():
    # ========================================================================
    # 1. Setup: fiber and dispersion
    # ========================================================================
    central_wl_nm = 1550.0
    omega0 = 2 * PI * C_MS / (central_wl_nm * 1e-9)

    # Fiber with anomalous dispersion (SCG regime)
    beta2_ps2_per_m = -100.0  # ps²/m at 1550 nm
    gamma = 10.0              # 1/(W·m)

    # Create a Dispersion object from known D(λ) values
    wl_nm = np.linspace(1400, 1700, 61)
    wl_m = wl_nm * 1e-9
    # D(λ) from β₂: D = -β₂·(2πc)/λ²
    D_vals = -beta2_ps2_per_m * 1e-24 * (2 * PI * C_MS) / wl_m**2 * 1e6  # ps/(nm·km)
    wl_arr = WavelengthArray(wl_nm, "nm")
    dispersion = Dispersion(
        wavelengths=wl_arr,
        values=D_vals,
        unit="ps/nm.km",
        central_wavelength=Wavelength(central_wl_nm, "nm"),
    )

    # Get Taylor betas in ps^k/m
    betas = dispersion.get_betas(polyOrder=3, wavelength=Wavelength(central_wl_nm, "nm"))

    print("=" * 60)
    print("Phase-Matching Diagnostics Example")
    print("=" * 60)

    # ========================================================================
    # 2. FWM Scan
    # ========================================================================
    print("\n--- FWM Scan ---")
    adaptor = DispersionAdaptor(dispersion, omega0)

    # Pump and signal
    P_pump = 500.0  # W
    L = 0.05  # 5 cm

    # Scan signal wavelengths around pump
    signal_wl_nm = np.linspace(1450, 1650, 201)
    omega_signal = 2 * PI * C_MS / (signal_wl_nm * 1e-9)

    fwm_result = scan_fwm_detuning(
        adaptor, omega0, omega_signal, P_pump, gamma, L=L
    )

    print(f"Pump wavelength: {central_wl_nm} nm")
    print(f"Signal range: {signal_wl_nm[0]:.1f} – {signal_wl_nm[-1]:.1f} nm")
    print(f"Δβ = 0 near signal wavelength: "
          f"{signal_wl_nm[np.argmin(np.abs(fwm_result.delta_beta))]:.1f} nm")

    # Plot FWM
    fig_fwm = plot_fwm_efficiency(fwm_result)
    fig_fwm.savefig("22_fwm_efficiency.png", dpi=150, bbox_inches="tight")
    print("Saved: 22_fwm_efficiency.png")

    # ========================================================================
    # 3. MI Gain Spectrum
    # ========================================================================
    print("\n--- MI Gain Spectrum ---")
    beta2_si = betas[0] * 1e-24  # ps²/m → s²/m

    omega_m = np.linspace(-5e13, 5e13, 501)
    gain = mi_gain_spectrum(beta2_si, gamma, P_pump, omega_m)

    # Find peak
    peak_idx = np.argmax(np.abs(gain))
    print(f"Peak MI gain: {np.max(gain):.2f} 1/m at Ω = {omega_m[peak_idx]/1e12:.2f} THz")

    # Sideband frequencies
    sidebands = mi_sideband_frequencies(beta2_si, gamma, P_pump)
    sb_wl_nm = 2 * PI * C_MS / (omega0 + sidebands) * 1e9
    print(f"MI sidebands: {sb_wl_nm[0]:.1f} nm, {sb_wl_nm[1]:.1f} nm")

    # Plot MI gain
    fig_mi = plt.figure(figsize=(10, 6))
    plt.plot(omega_m / 1e12, gain * 1e3, "b-", linewidth=1)
    plt.xlabel("Modulation Frequency (THz)")
    plt.ylabel(r"g(Ω) (1/mm)")
    plt.title("Modulation Instability Gain Spectrum")
    plt.axvline(x=0, color="k", linestyle=":", alpha=0.3)
    plt.grid(True, alpha=0.3)
    fig_mi.tight_layout()
    fig_mi.savefig("22_mi_gain.png", dpi=150, bbox_inches="tight")
    print("Saved: 22_mi_gain.png")

    # ========================================================================
    # 4. Dispersive Wave Root Finder
    # ========================================================================
    print("\n--- Dispersive Wave Root Finder ---")
    dw_result = dispersive_wave_roots(
        adaptor, omega0,
        wl_range=(Wavelength(1000, "nm"), Wavelength(2500, "nm")),
        n_brackets=200,
    )

    if len(dw_result.wavelengths) > 0:
        print(f"Found {len(dw_result.wavelengths)} DW root(s):")
        for i, wl in enumerate(dw_result.wavelengths):
            delta_wl = abs(wl.as_nm - central_wl_nm)
            print(f"  DW{i+1}: {wl.as_nm:.1f} nm (Δλ = {delta_wl:.1f} nm from pump)")
    else:
        print("No DW roots found in search range (DispersionAdaptor may not capture higher-order dispersion).")

    # β₂/β₃ analytic estimate for comparison
    if len(betas) > 1 and abs(betas[1]) > 1e-15:
        beta3_si = betas[1] * 1e-27
        delta_omega_dw = -2 * beta2_si / beta3_si
        omega_dw = omega0 + delta_omega_dw
        if omega_dw > 0:
            dw_wl_analytic = 2 * PI * C_MS / omega_dw * 1e9
            print(f"β₂/β₃ analytic estimate: {dw_wl_analytic:.1f} nm")
        else:
            print("β₂/β₃ analytic estimate: DW frequency would be non-positive")
    else:
        print("β₂/β₃ analytic estimate: β₃ not available or too small")

    # ========================================================================
    # 5. Simulation Readiness Assessment
    # ========================================================================
    print("\n--- Simulation Readiness Assessment ---")
    T0_ps = 1.0
    P_peak = 2000.0
    pulse = make_gaussian_pulse(central_wl_nm, T0_ps, P_peak)

    fiber = FiberProfile(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80, "um^2"),
        length=Length(0.05, "m"),
        confinement_factor=1.0,
    )

    report = assess_simulation_readiness(pulse, fiber, dispersion, betas=betas)

    print(f"Soliton order N: {report.soliton_order:.2f}")
    print(f"Dispersion length L_D: {report.dispersion_length:.4f} m")
    print(f"Nonlinear length L_NL: {report.nonlinear_length:.6f} m")
    print(f"Fission length L_fiss: {report.fission_length:.4f} m")
    print(f"Dispersion covers grid: {report.dispersion_covers_grid}")
    print(f"Predicted processes: {', '.join(report.predicted_processes) if report.predicted_processes else 'None'}")
    if report.dw_predictions:
        print(f"Predicted DW wavelengths: {[f'{w:.1f}' for w in report.dw_predictions]} nm")

    # Plot readiness
    fig_readiness = plot_readiness_report(report)
    fig_readiness.savefig("22_readiness_report.png", dpi=150, bbox_inches="tight")
    print("Saved: 22_readiness_report.png")

    # ========================================================================
    # 6. Full GNLSE Simulation with PM Diagnostics
    # ========================================================================
    print("\n--- GNLSE Simulation with PM Diagnostics ---")

    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
        check_phase_matching=True,
    )

    # Access preflight report
    preflight = solver.preflight_report
    if preflight:
        print(f"Preflight N: {preflight.soliton_order:.2f}")
        print(f"Preflight warnings: {len(preflight.warnings)}")
        for w in preflight.warnings:
            print(f"  ⚠ {w}")

    # Propagate
    num_steps = 200
    print(f"Propagating with {num_steps} steps...")
    solver.propagate(num_steps=num_steps)

    # Plot spectrum with PM overlay
    fig_spectrum = plot_spectrum_with_pm_overlay(solver, report)
    fig_spectrum.savefig("22_spectrum_pm_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: 22_spectrum_pm_overlay.png")

    # ========================================================================
    # 7. Energy Conservation Check
    # ========================================================================
    print("\n--- Energy Conservation ---")
    initial_energy = np.trapezoid(np.abs(pulse.envelope_field)**2, pulse.grid.t)
    final_energy = np.trapezoid(np.abs(solver.evolution[-1].envelope_field)**2, pulse.grid.t)
    if abs(initial_energy) > 1e-40:
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy) * 100
    else:
        energy_drift = 0.0
    print(f"Initial energy: {initial_energy:.3e} J")
    print(f"Final energy:   {final_energy:.3e} J")
    print(f"Energy drift:   {energy_drift:.2f}%")
    if energy_drift < 5:
        print("✓ Energy conservation OK (< 5% drift)")
    else:
        print("⚠ Large energy drift — consider increasing num_steps")

    # ========================================================================
    # 8. Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("All diagnostics complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  22_fwm_efficiency.png        — FWM Δβ and efficiency curves")
    print("  22_mi_gain.png               — MI gain spectrum")
    print("  22_readiness_report.png      — Dispersion coverage panel")
    print("  22_spectrum_pm_overlay.png   — Final spectrum with PM predictions")


if __name__ == "__main__":
    main()
