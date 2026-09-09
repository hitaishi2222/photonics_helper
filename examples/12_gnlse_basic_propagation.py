"""
Example: GNLSE Basic Propagation
=================================

Demonstrates the GNLSE solver for pulse propagation through an optical fiber.
Shows how dispersion and the Kerr effect independently affect a pulse.

The Generalized Nonlinear Schrödinger Equation (GNLSE) describes how an
optical pulse evolves in a nonlinear medium:

    ∂A/∂z = -i·Σ(β_k/k!)(i∂/∂T)^k·A + i·γ·|A|²·A

where β_k are dispersion coefficients and γ is the nonlinear coefficient.

This example propagates a 100-fs Gaussian pulse through 1 mm of silica fiber,
comparing pure dispersion, pure Kerr, and their combination.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.base import Wavelength, Time, Length, Area


def make_pulse(wavelength_nm=1550, T0_fs=100, peak_power_W=5000):
    """Create a Gaussian pulse at the given wavelength."""
    grid = TemporalGrid(N=2**10, Tmax=Time(3 * T0_fs * 1e-15, "s"))
    env = Envelope(
        shape="gaussian", peak_amplitude=np.sqrt(peak_power_W), pulse_width=Time(T0_fs, "fs")
    )
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(wavelength_nm, "nm"),
    )
    return pulse


def make_fiber(length_m=0.2):
    """Create a silica fiber profile at 1550 nm."""
    return FiberProfile(
        n2=2.6e-20,  # m²/W — silica nonlinear index
        alpha=0.0,  # no loss for this demo
        A_eff=Area(80e-12, "m^2"),  # m² — effective mode area
        length=Length(length_m, "m"),
    )


def make_betas(beta2_ps2_per_km=20.0):
    """Dispersion coefficients: β₂ in ps²/m (matching get_betas() convention)."""
    # Convert ps²/km → ps²/m: divide by 1000
    beta2_ps2_m = beta2_ps2_per_km * 1e-3
    return np.array([beta2_ps2_m])


def main():
    central_wl = Wavelength(1550, "nm")
    T0 = Time(100, "fs")
    fiber_length = Length(200, "mm")

    pulse = make_pulse(wavelength_nm=central_wl.as_nm, T0_fs=T0.as_fs, peak_power_W=5000)
    fiber = make_fiber(length_m=fiber_length.as_m)
    betas = make_betas(beta2_ps2_per_km=20.0)  # normal dispersion

    # ── Run three simulations: dispersion only, Kerr only, both ─────────────

    # Pure dispersion (no nonlinearity)
    solver_disp = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_disp.propagate(num_steps=50)

    # Pure Kerr (no dispersion)
    solver_kerr = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_kerr.propagate(num_steps=50)

    # Combined dispersion + Kerr
    solver_both = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_both.propagate(num_steps=50)

    # ── Plot: Pulse envelope evolution ──────────────────────────────────────

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "GNLSE Basic Propagation — 100 fs pulse, 20 cm silica fiber",
        fontsize=14,
        fontweight="bold",
    )

    t_ps = pulse.grid.t * 1e12

    # Panel 1: Pure dispersion
    for i, wave in enumerate(solver_disp.evolution):
        axes[0].plot(t_ps, np.abs(wave.envelope_field), alpha=0.6, linewidth=1.0)
    axes[0].plot(
        t_ps, np.abs(pulse.envelope_field), "k--", linewidth=2.0, label="Input"
    )
    axes[0].set_ylabel("Field amplitude")
    axes[0].set_title("Pure Dispersion (β₂ = 20 ps²/km)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([-1, 1])  # Zoom to ±1 ps

    # Panel 2: Pure Kerr
    for i, wave in enumerate(solver_kerr.evolution):
        axes[1].plot(t_ps, np.abs(wave.envelope_field), alpha=0.6, linewidth=1.0)
    axes[1].plot(
        t_ps, np.abs(pulse.envelope_field), "k--", linewidth=2.0, label="Input"
    )
    axes[1].set_ylabel("Field amplitude")
    axes[1].set_title("Pure Kerr Effect (γ·P₀·L ≈ 1.3)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([-1, 1])

    # Panel 3: Combined
    for i, wave in enumerate(solver_both.evolution):
        axes[2].plot(t_ps, np.abs(wave.envelope_field), alpha=0.6, linewidth=1.0)
    axes[2].plot(
        t_ps, np.abs(pulse.envelope_field), "k--", linewidth=2.0, label="Input"
    )
    axes[2].set_ylabel("Field amplitude")
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_title("Dispersion + Kerr (combined GNLSE)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim([-1, 1])

    plt.tight_layout()
    plt.savefig("examples/images/12_gnlse_basic_propagation.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/images/12_gnlse_basic_propagation.png")
    plt.close()

    # ── Plot: Spectrum evolution ────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Spectral evolution at each propagation step", fontsize=14, fontweight="bold"
    )

    for idx, (solver, title) in enumerate(
        [
            (solver_disp, "Dispersion only"),
            (solver_kerr, "Kerr only"),
            (solver_both, "Combined"),
        ]
    ):
        omega, spectra = solver.spectra_vs_z
        freq_THz = np.abs(omega) / (2 * np.pi) / 1e12
        sort_idx = np.argsort(freq_THz)
        freq_THz = freq_THz[sort_idx]
        spectra_sorted = spectra[:, sort_idx]

        spectra_dB = 10 * np.log10(spectra_sorted + 1e-30)
        max_dB = spectra_dB.max()
        spectra_dB -= max_dB

        # Only show frequencies within ±10 THz
        mask = np.abs(freq_THz) < 10
        freq_plot = freq_THz[mask]

        axes[idx].pcolormesh(
            freq_plot,
            np.linspace(0, fiber.length.as_m * 1e3, spectra.shape[0]),
            spectra_dB[:, mask],
            shading="auto",
            cmap="viridis",
            vmin=-50,
        )
        axes[idx].set_xlabel("Frequency offset (THz)")
        axes[idx].set_ylabel("Distance (mm)")
        axes[idx].set_title(title)

    plt.tight_layout()
    plt.savefig("examples/images/12_gnlse_spectra.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/images/12_gnlse_spectra.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    initial_energy = np.sum(np.abs(pulse.envelope_field) ** 2) * pulse.grid.dt
    final_disp_energy = (
        np.sum(np.abs(solver_disp.evolution[-1].envelope_field) ** 2) * pulse.grid.dt
    )
    final_kerr_energy = (
        np.sum(np.abs(solver_kerr.evolution[-1].envelope_field) ** 2) * pulse.grid.dt
    )
    final_both_energy = (
        np.sum(np.abs(solver_both.evolution[-1].envelope_field) ** 2) * pulse.grid.dt
    )

    print("\n— Energy conservation check —")
    print(f"Input energy:      {initial_energy:.6e}")
    print(
        f"After dispersion:  {final_disp_energy:.6e}  (Δ={final_disp_energy/initial_energy*100-100:.2f}%)"
    )
    print(
        f"After Kerr:        {final_kerr_energy:.6e}  (Δ={final_kerr_energy/initial_energy*100-100:.2f}%)"
    )
    print(
        f"After combined:    {final_both_energy:.6e}  (Δ={final_both_energy/initial_energy*100-100:.2f}%)"
    )

    # Pulse width evolution
    def pulse_width(wave):
        t = wave.grid.t
        I = np.abs(wave.envelope_field) ** 2
        return np.sqrt(np.sum(t**2 * I) / np.sum(I))

    print("\n— Pulse width evolution —")
    w0 = pulse_width(pulse)
    print(f"Input:  {w0*1e12:.2f} ps")
    print(
        f"Disp:   {pulse_width(solver_disp.evolution[-1])*1e12:.2f} ps  ({pulse_width(solver_disp.evolution[-1])/w0*100-100:.1f}%)"
    )
    print(
        f"Kerr:   {pulse_width(solver_kerr.evolution[-1])*1e12:.2f} ps  ({pulse_width(solver_kerr.evolution[-1])/w0*100-100:.1f}%)"
    )
    print(
        f"Both:   {pulse_width(solver_both.evolution[-1])*1e12:.2f} ps  ({pulse_width(solver_both.evolution[-1])/w0*100-100:.1f}%)"
    )


if __name__ == "__main__":
    main()
