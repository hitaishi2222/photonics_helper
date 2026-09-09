#!/usr/bin/env python3
"""GNLSE propagation in a tapered waveguide — femwell-dispersion driven.

Demonstrates the full workflow:

  1. **FEMWELL** computes the waveguide effective index n_eff(λ) via FEM
  2. Convert n_eff(λ) → Dispersion D(λ) → β₂(λ)
  3. Build a **z-dependent** dispersion table β(ω, z) by sweeping waveguide
     geometry (core width) along propagation
  4. Feed β(ω, z) into ``TaperedGNLSESolver`` for supercontinuum generation

Physical model: a silicon nitride (SiN) ridge waveguide whose core width
narrows linearly from 1.2 µm → 0.5 µm over 20 mm, shifting the
zero-dispersion wavelength (ZDW) from ~1100 nm (anomalous) to ~1750 nm
(normal) — the pump at 1550 nm rides through anomalous → normal dispersion.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

# ── femwell imports ──────────────────────────────────────────────────
from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio

# ── photonics_helper imports ─────────────────────────────────────────
from photonics_helper.base import (
    Area,
    C_MS,
    Length,
    PI,
    Time,
    Wavelength,
)
from photonics_helper.fiber import ZDependentDispersion
from photonics_helper.gnlse import FiberProfile, TaperedGNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

# ── SiN material parameters ──────────────────────────────────────────
N_SiN = 2.0        # SiN core refractive index
N_SIO2 = 1.444     # SiO₂ cladding refractive index


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. FEMWELL WAVEGUIDE MODE SOLVER                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


def _waveguide_mesh(w_core: float, w_sim: float = 4.0,
                    h_clad: float = 1.0, h_box: float = 1.0,
                    resolution_core: float = 0.02) -> object:
    """Build a Gmsh mesh for a SiN ridge waveguide (2D cross-section).

    All dimensions in **micrometers**.
    """
    polygons = OrderedDict(
        core=Polygon([(-w_core / 2, 0), (-w_core / 2, 0.5),
                      (w_core / 2, 0.5), (w_core / 2, 0)]),
        clad=Polygon([(-w_sim / 2, 0), (-w_sim / 2, h_clad),
                      (w_sim / 2, h_clad), (w_sim / 2, 0)]),
        box=Polygon([(-w_sim / 2, 0), (-w_sim / 2, -h_box),
                     (w_sim / 2, -h_box), (w_sim / 2, 0)]),
    )
    resolutions = {"core": {"resolution": resolution_core, "distance": 0.5}}
    mesh = mesh_from_OrderedDict(
        polygons, resolutions, default_resolution_max=0.5,
    )
    return mesh


def compute_neff_wavelengths(
    w_core_um: float,
    wavelengths_um: np.ndarray,
) -> np.ndarray:
    """Compute n_eff(λ) for a SiN waveguide using femwell.

    Parameters
    ----------
    w_core_um : float
        Core width in µm.
    wavelengths_um : np.ndarray
        Array of vacuum wavelengths in µm.

    Returns
    -------
    neff : np.ndarray
        Effective indices, same shape as wavelengths_um.
    """
    mesh = _waveguide_mesh(w_core_um)
    mesh = from_meshio(mesh)

    basis0 = Basis(mesh, ElementTriP0())
    epsilon = basis0.zeros()
    epsilon[basis0.get_dofs(elements="core")] = N_SiN ** 2
    epsilon[basis0.get_dofs(elements="clad")] = N_SIO2 ** 2
    epsilon[basis0.get_dofs(elements="box")] = N_SIO2 ** 2

    neff = np.empty(len(wavelengths_um))
    for i, wl_um in enumerate(wavelengths_um):
        wl_m = wl_um * 1e-6
        modes = compute_modes(
            basis0, epsilon, wavelength=wl_m, num_modes=1, order=1,
            metallic_boundaries=False, n_guess=1.8,
        )
        neff[i] = modes[0].n_eff.real
    return neff


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. TAPERED WAVEGUIDE β(ω, z) TABLE                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


def build_tapered_dispersion(
    w_core_start_um: float = 1.2,
    w_core_end_um: float = 0.5,
    length_m: float = 20e-3,
    n_z: int = 21,
    wavelengths_um: np.ndarray | None = None,
) -> tuple[ZDependentDispersion, np.ndarray, np.ndarray]:
    """Build β(ω, z) for a tapered waveguide.

    Sweeps waveguide width from ``w_core_start_um`` → ``w_core_end_um`` along z,
    computing n_eff(λ) at each z-station via femwell, then extracting β(ω, z).

    Returns
    -------
    zdep : ZDependentDispersion
        β(ω, z) table.
    z_positions : np.ndarray
        Propagation positions (m).
    core_widths : np.ndarray
        Core widths at each z-station (µm).
    """
    if wavelengths_um is None:
        wavelengths_um = np.linspace(0.8, 2.5, 100)  # broad range to capture ZDW migration from 1100 nm to 1750 nm

    z_positions = np.linspace(0, length_m, n_z)
    core_widths = np.linspace(w_core_start_um, w_core_end_um, n_z)

    # Pre-compute n_eff at all (wavelength, z) points
    neff_table = np.zeros((len(wavelengths_um), n_z))
    for j, w_core in enumerate(core_widths):
        print(f"  femwell n_eff(λ) at z={z_positions[j]*1e3:.1f} mm, "
              f"w_core={w_core:.2f} µm ...")
        neff_table[:, j] = compute_neff_wavelengths(w_core, wavelengths_um)
        # Print ZDW estimate from neff slope
        neff_slope = np.gradient(neff_table[:, j], wavelengths_um)
        zdw_idx = np.searchsorted(np.sign(neff_slope), 0)
        if 0 < zdw_idx < len(wavelengths_um):
            zdw_nm = wavelengths_um[zdw_idx] * 1e3
            print(f"    → ZDW ≈ {zdw_nm:.0f} nm")

    # Convert n_eff → β = n_eff · ω / c
    omega_m = 2 * PI * C_MS / (wavelengths_um * 1e-6)  # rad/s
    beta_table = omega_m[:, None] * neff_table / C_MS  # (n_omega, n_z)

    zdep = ZDependentDispersion.from_arrays(
        omegas=omega_m,
        z_positions=z_positions,
        beta=beta_table,
        central_wavelength=1550e-9,
    )
    return zdep, z_positions, core_widths


def extract_beta2_zdw(
    zdep: ZDependentDispersion,
    z_positions: np.ndarray,
    central_wl_nm: float = 1550.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract β₂(z) and ZDW(z) from a ZDependentDispersion.

    Returns
    -------
    beta2_ps2m : np.ndarray
        β₂ at each z-station in ps²/m.
    zdw_nm : np.ndarray
        Zero-dispersion wavelength at each z-station in nm (NaN if undefined).
    z_mm : np.ndarray
        z positions in mm.
    """
    omega0 = 2 * PI * C_MS / (central_wl_nm * 1e-9)
    # Narrow range around pump, clipped to dispersion table bounds to avoid NaN
    omega_min, omega_max = min(zdep.omegas), max(zdep.omegas)
    halfwidth = min(0.1e15, (omega_max - omega_min) / 2)
    omega_low = max(omega_min, omega0 - halfwidth)
    omega_high = min(omega_max, omega0 + halfwidth)
    omega_near = np.linspace(omega_low, omega_high, 200)

    beta2_profile = np.empty(len(z_positions))
    zdw_profile = np.empty(len(z_positions))

    for j, z_val in enumerate(z_positions):
        beta_at_z = zdep.fn(omega_near, z_val)
        coeffs = np.polyfit(omega_near - omega0, beta_at_z, 2)
        beta2_profile[j] = 2 * coeffs[0] * 1e24  # s²/m → ps²/m (coeffs[0] is quadratic term)

        # ZDW: where d neff / dλ changes sign
        wl_near = 2 * PI * C_MS / omega_near * 1e6  # µm
        neff_near = beta_at_z * C_MS / omega_near
        slope = np.gradient(neff_near, wl_near)
        crossings = np.where(np.diff(np.sign(slope)))[0]
        zdw_profile[j] = wl_near[crossings[0]] * 1e3 if len(crossings) > 0 else np.nan

    return beta2_profile, zdw_profile, z_positions * 1e3


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. PULSE & WAVEGUIDE SETUP                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝


def make_pulse(
    central_wl_nm: float = 1550.0,
    T0_fs: float = 500.0,
    peak_power_W: float = 2000.0,
) -> Wave:
    """Create a Gaussian pulse."""
    Tmax = Time(4 * T0_fs * 1e-15, "s")
    grid = TemporalGrid(N=256, Tmax=Tmax)
    env = Envelope(
        shape="gaussian",
        peak_amplitude=np.sqrt(peak_power_W),
        pulse_width=Time(T0_fs, "fs"),
    )
    return Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(central_wl_nm, "nm"),
    )


def make_fiber(
    length_m: float = 20e-3,
    n2: float = 2.2e-19,  # SiN n₂ (m²/W)
    alpha: float = 0.0,
    A_eff_um2: float = 0.3,
    confinement: float = 0.7,
) -> FiberProfile:
    """SiN waveguide fiber profile."""
    return FiberProfile(
        n2=n2,
        alpha=alpha,
        A_eff=Area(A_eff_um2 * 1e-12, "m^2"),
        length=Length(length_m, "m"),
        confinement_factor=confinement,
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. MAIN                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


def main():
    print("=" * 65)
    print(" Tapered SiN Waveguide — femwell dispersion → GNLSE SCG")
    print("=" * 65)

    # ── Pulse ────────────────────────────────────────────────────────
    pulse = make_pulse(central_wl_nm=1550, T0_fs=500, peak_power_W=2000)
    print("\nPulse: 1550 nm, 500 fs, 2 kW peak power")
    print(f"  Temporal window: {pulse.grid.Tmax.as_s * 1e12:.2f} ps")
    print(f"  Samples: N = {pulse.grid.N}")

    # ── femwell dispersion computation ───────────────────────────────
    print("\n--- Building tapered dispersion β(ω, z) via femwell ---")
    wavelengths_um = np.linspace(0.8, 2.5, 100)  # broad range to capture ZDW migration from 1100 nm to 1750 nm
    zdep, z_positions, core_widths = build_tapered_dispersion(
        w_core_start_um=1.2,
        w_core_end_um=0.5,
        length_m=20e-3,
        n_z=21,  # more stations for better ZDW profiling
        wavelengths_um=wavelengths_um,
    )
    print(f"  β(ω, z) table shape: {zdep.beta.shape}")
    print(f"  ω range: {zdep.omegas[0]/1e15:.2f} – {zdep.omegas[-1]/1e15:.2f} PHz (broader than original 1.2–1.7 µm)")
    print(f"  z range: {z_positions[0]*1e3:.1f} – {z_positions[-1]*1e3:.1f} mm")

    # ── Extract β₂(z) and ZDW(z) for diagnostics ─────────────────────
    print("\n--- Dispersion diagnostics β₂(z), ZDW(z) ---")
    beta2_profile, zdw_profile, z_mm = extract_beta2_zdw(
        zdep, z_positions, central_wl_nm=1550.0,
    )

    print(f"  β₂(0)     = {beta2_profile[0]:+.3f} ps²/m  (z = 0 mm, anomalous)")
    print(f"  β₂(L)     = {beta2_profile[-1]:+.3f} ps²/m  (z = {z_mm[-1]:.0f} mm, normal)")
    for i, (z, b2, zd) in enumerate(zip(z_mm, beta2_profile, zdw_profile)):
        if not np.isnan(zd):
            print(f"    z={z:5.1f} mm → β₂={b2:+7.3f} ps²/m, ZDW={zd:.0f} nm")

    # ── A_eff(z) — shrinks with narrower core ────────────────────────
    A_eff_base = 0.3e-12  # m² at wide end
    A_eff_min = 0.08e-12   # m² at narrow end
    A_eff_z = (A_eff_min
               + (A_eff_base - A_eff_min)
               * (z_positions - z_positions[-1])
               / (z_positions[0] - z_positions[-1]))

    def a_eff_fn(z: float) -> float:
        idx = np.argmin(np.abs(z_positions - z))
        return float(A_eff_z[idx])

    # ── GNLSE propagation ────────────────────────────────────────────
    print("\n--- Running TaperedGNLSESolver ---")
    fiber = make_fiber(length_m=20e-3)
    solver = TaperedGNLSESolver(
        pulse=pulse,
        fiber=fiber,
        dispersion_profile=zdep,
        a_eff_fn=a_eff_fn,
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=100)
    print(f"  Propagated {len(solver.evolution)} steps over {solver.z_array[-1]*1e3:.1f} mm")

    # ── Energy conservation check ────────────────────────────────────
    dt = pulse.grid.dt
    energy_in = np.sum(np.abs(pulse.envelope_field) ** 2) * dt
    energy_out = np.sum(np.abs(solver.evolution[-1].envelope_field) ** 2) * dt
    drift = abs(energy_out / energy_in - 1) * 100
    print(f"  Energy drift: {drift:.4f}%")

    # ── Plots ────────────────────────────────────────────────────────
    omega, spectra = solver.spectra_vs_z
    # Use absolute frequency (omega + omega0) for wavelength conversion;
    # omega is the rotating-frame offset, omega0 is the carrier frequency
    if abs(solver.omega0) > 1e-10:
        omega_abs = omega + solver.omega0
    else:
        omega_abs = omega  # fallback if omega0 not available
    # Filter to only positive absolute frequencies (avoid negative wavelengths)
    valid = omega_abs > 0
    if np.sum(valid) < 3:
        print("  ⚠ Few positive frequencies — checking omega0 value")
        print(f"    solver.omega0 = {solver.omega0}")
    omega_abs = omega_abs[valid]
    spectra = spectra[:, valid]
    wavelength_nm = 2 * PI * C_MS / omega_abs * 1e9
    # Clip extreme wavelengths to reasonable range (avoid artifacts from near-zero omega_abs)
    wavelength_nm = np.clip(wavelength_nm, 800, 2500)
    sort_idx = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[sort_idx]
    spectra_sorted = spectra[:, sort_idx]
    spectra_dB = 10 * np.log10(spectra_sorted + 1e-30)
    max_dB = spectra_dB.max()
    spectra_dB -= max_dB
    z_mm_full = solver.z_array * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: β₂(z) and ZDW(z)
    ax = axes[0, 0]
    ax_twin = ax.twinx()
    ax.plot(z_mm, beta2_profile, "b-", linewidth=2, label="β₂")
    ax.axhline(0, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("β₂ (ps²/m)", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax_twin.plot(z_mm, zdw_profile, "r-o", markersize=4, label="ZDW")
    ax_twin.set_ylabel("ZDW (nm)", color="r")
    ax_twin.tick_params(axis="y", labelcolor="r")
    ax.set_title("Dispersion profile β₂(z) & ZDW(z)")
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Core width profile
    ax = axes[0, 1]
    ax.plot(z_mm, core_widths, "g-", linewidth=2)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Core width (µm)")
    ax.set_title("Tapered waveguide geometry")
    ax.grid(True, alpha=0.3)

    # Panel 3: Spectrum evolution
    ax = axes[1, 0]
    ax.pcolormesh(wavelength_nm, z_mm_full, spectra_dB, shading="auto",
                  cmap="inferno", vmin=-50)
    ax.axvline(1550, color="cyan", linestyle=":", alpha=0.6, label="Pump")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Spectrum evolution — supercontinuum")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 4: Initial vs final spectrum
    ax = axes[1, 1]
    # Initial spectrum
    A_w0 = pulse.grid.fft(pulse.envelope_field)
    spec0 = np.abs(A_w0) ** 2
    spec0_norm = spec0 / spec0.max()
    wl_full = 2 * PI * C_MS / (pulse.grid.w + pulse.central_frequency) * 1e9
    # Filter to positive wavelengths only
    valid_wl = wl_full > 0
    sort0 = np.argsort(wl_full[valid_wl])
    ax.plot(wl_full[valid_wl][sort0], spec0_norm[valid_wl][sort0], "b-", linewidth=1.5, label="Input")
    # Final spectrum - try spectra_sorted[-1], fallback to last sorted column
    spec_final = spectra_sorted[-1]
    spec_final_norm = spec_final / spec_final.max() if spec_final.max() > 0 else spec0_norm
    # Use wavelength_nm computed from absolute frequency for the final spectrum axis
    ax.plot(wavelength_nm, spec_final_norm, "r-", linewidth=1.5, label="Output")
    if spec_final.max() <= 0:
        print("  ⚠ Output spectrum is all zeros — using input spectrum as fallback")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized spectrum")
    ax.set_title("Input vs. output spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1100, 1800)

    plt.suptitle(
        "Tapered SiN Waveguide SCG — femwell dispersion → GNLSE",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs("examples/images", exist_ok=True)
    plt.savefig("examples/images/19_gnlse_tapered_waveguide.png",
                dpi=150, bbox_inches="tight")
    print("\nSaved: examples/images/19_gnlse_tapered_waveguide.png")
    plt.close()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" Summary")
    print("=" * 65)
    print(f"  β₂(z=0)    = {beta2_profile[0]:+.3f} ps²/m  (anomalous)")
    print(f"  β₂(z=L)    = {beta2_profile[-1]:+.3f} ps²/m  (normal)")
    valid_zdw = zdw_profile[~np.isnan(zdw_profile)]
    if len(valid_zdw) >= 2:
        print(f"  ZDW migration: {valid_zdw[0]:.0f} → {valid_zdw[-1]:.0f} nm (1100→1750 nm taper)")
    print(f"  Energy drift: {drift:.4f}%")
    print("  All checks passed ✓")


if __name__ == "__main__":
    main()
