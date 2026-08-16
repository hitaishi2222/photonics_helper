#!/usr/bin/env python3
"""Soliton analysis example: soliton fission in chalcogenide waveguide.

Demonstrates:
- Creating a SolitonAnalyzer from GNLSE results
- Computing soliton order, fission length, dispersive wave wavelength
- Visualizing soliton trajectories and Raman shift
- Using confinement_factor for waveguide simulations
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.soliton import SolitonAnalyzer
from photonics_helper.raman import RamanResponse, RamanSpec
from photonics_helper.base import Wavelength, Time, Area, Length, C_MS, PI


def main():
    # ── Pulse parameters ──────────────────────────────────────────────
    T0 = 100e-15  # 100 fs pulse width
    P_peak = 5.0  # 5 W peak power
    central_wl = 1550e-9  # 1550 nm pump

    grid = TemporalGrid(N=2**14, Tmax=Time(50e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0, "s"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(central_wl * 1e9, "nm"),
    )

    # ── Waveguide parameters (GeAsSe) ─────────────────────────────────
    # Confinement factor Gamma < 1.0 for waveguide (vs 1.0 for fiber)
    fiber = FiberProfile(
        n2=6.0e-18,  # GeAsSe n2
        alpha=0.0,
        A_eff=Area(0.2e-12, "m^2"),  # 0.2 um^2 effective area
        length=Length(5e-3, "m"),  # 5 mm waveguide
        confinement_factor=0.8,  # Waveguide confinement
    )

    # ── Dispersion (anomalous, near zero-dispersion wavelength) ───────
    # beta2 = -2 ps^2/m, beta3 = -0.05 ps^3/m
    betas = np.array([-2.0, -0.05])

    # ── Raman response for GeAsSe ─────────────────────────────────────
    geasse_spec = RamanSpec.from_database("GeAsSe")
    raman_grid = TemporalGrid(N=2**14, Tmax=Time(50e-12, "s"))
    raman_response = RamanResponse(
        spec=geasse_spec, grid=raman_grid,
        response_type="time_domain",
    )
    fiber.raman_response = raman_response

    # ── Run GNLSE simulation ──────────────────────────────────────────
    print("Running GNLSE simulation...")
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=True,
        include_self_steepening=True,
    )
    solver.propagate(num_steps=100)
    print(f"  Propagated {len(solver.evolution)} steps")

    # ── Create SolitonAnalyzer ────────────────────────────────────────
    analyzer = SolitonAnalyzer(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        z_array=solver.z_array,
        spectra_vs_z=solver.spectra_vs_z,
    )

    # ── Compute soliton diagnostics ───────────────────────────────────
    N = analyzer.soliton_order()
    L_D = analyzer.dispersion_length()
    L_NL = analyzer.nonlinear_length()
    L_fiss = analyzer.fission_length()
    gamma = analyzer.gamma

    print(f"\nSoliton diagnostics:")
    print(f"  Soliton order N = {N:.2f}")
    print(f"  Dispersion length L_D = {L_D*1e3:.2f} mm")
    print(f"  Nonlinear length L_NL = {L_NL*1e3:.2f} mm")
    print(f"  Fission length L_fiss = {L_fiss*1e3:.2f} mm")
    print(f"  Nonlinear coeff gamma = {gamma:.4f} 1/(W*m)")
    print(f"  Confinement factor Gamma = {fiber.confinement_factor}")

    try:
        lambda_dw = analyzer.dispersive_wave_wavelength()
        print(f"  Dispersive wave wavelength = {lambda_dw*1e9:.2f} nm")
    except ValueError:
        print(f"  Dispersive wave wavelength: not available (beta3 needed)")

    # ── Count solitons in output spectrum ─────────────────────────────
    n_solitons = analyzer.count_solitons()
    print(f"  Solitons detected in output: {n_solitons}")

    # ── Plot results ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Spectrum evolution
    ax = axes[0, 0]
    omega, spectra = solver.spectra_vs_z
    omega_abs = omega + solver.omega0
    wavelength_nm = 2 * PI * C_MS / omega_abs * 1e9
    sort_idx = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[sort_idx]
    spectra = spectra[:, sort_idx]

    spectra_dB = 10 * np.log10(spectra + 1e-30)
    max_dB = spectra_dB.max()
    spectra_dB = spectra_dB - max_dB + 50
    z_mm = solver.z_array * 1e3
    ax.pcolormesh(wavelength_nm, z_mm, spectra_dB, shading="auto", cmap="hot")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Spectrum Evolution")

    # Soliton trajectories
    ax = axes[0, 1]
    traj = analyzer.soliton_trajectories()
    if traj:
        z_vals = np.array([t[0] for t in traj]) * 1e3
        lam_vals = np.array([t[1] for t in traj])
        ax.scatter(z_vals, lam_vals, s=3, alpha=0.5)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Peak wavelength (nm)")
    ax.set_title("Soliton Trajectories")

    # Raman shift
    ax = axes[1, 0]
    rate = analyzer.raman_shift_rate()
    ax.text(0.5, 0.5, f"RSFS Rate: {rate:.4f} nm/mm",
            ha="center", va="center", transform=ax.transAxes, fontsize=14)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Peak wavelength (nm)")
    ax.set_title("Raman Self-Frequency Shift")
    ax.axis("off")

    # Final spectrum with DW marker
    ax = axes[1, 1]
    final_spec = spectra[-1]
    max_val = np.max(final_spec)
    if max_val > 0:
        ax.plot(wavelength_nm, final_spec / max_val, "b-", linewidth=1)
    ax.axvline(x=1550, color="k", linestyle=":", alpha=0.5, label="Pump")
    try:
        dw_nm = analyzer.dispersive_wave_wavelength() * 1e9
        ax.axvline(x=dw_nm, color="r", linestyle="--", alpha=0.5, label="DW")
    except ValueError:
        pass
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized spectrum")
    ax.set_title("Final Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/soliton_analysis.png", dpi=100)
    print(f"\nPlot saved to /tmp/soliton_analysis.png")


if __name__ == "__main__":
    main()
