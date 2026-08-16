"""
Example: GNLSE Raman Scattering and Self-Steepening
====================================================

Demonstrates two key physical effects that become important at high peak
powers or for ultrashort pulses:

1. **Raman scattering** — delayed nonlinear response of the medium causes
   energy transfer from higher to lower frequencies (redshift). In silica,
   the Raman shift is ~440 cm⁻¹ (~13.2 THz). This is the mechanism behind
   Raman solitons and supercontinuum generation.

2. **Self-steepening** — the intensity-dependent group velocity causes the
   pulse trailing edge to steepen (optical shock formation). Proportional
   to 1/ω₀, so it matters more at shorter wavelengths.

This example compares propagation with and without these effects.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.raman import RamanResponse, RamanSpec
from photonics_helper.base import Wavelength, Time, Length, Area, Power


def make_pulse(wavelength_nm=1064, T0_fs=50):
    """Create a 50-fs Gaussian pulse."""
    grid = TemporalGrid(N=2**10, Tmax=Time(8 * T0_fs * 1e-15, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(T0_fs, "fs"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(wavelength_nm, "nm"),
    )
    return pulse


def make_high_power_pulse(wavelength_nm=1064, T0_fs=50, peak_power_W=10000):
    """Create a high-power pulse for nonlinear effects."""
    grid = TemporalGrid(N=2**13, Tmax=Time(4 * T0_fs * 1e-15, "s"))
    A0 = np.sqrt(peak_power_W)
    env = Envelope(shape="gaussian", peak_amplitude=A0, pulse_width=Time(T0_fs, "fs"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(wavelength_nm, "nm"),
    )
    return pulse


def main():
    # ── Parameters ──────────────────────────────────────────────────────────
    central_wl = Wavelength(1064, "nm")
    T0 = Time(50, "fs")
    omega0 = 2 * np.pi * 299792458.0 / central_wl.as_m

    # Fiber: anomalous dispersion for soliton dynamics
    n2 = 2.6e-20
    A_eff = Area(80, "um^2")
    gamma = n2 * omega0 / (299792458.0 * A_eff.as_m2)
    beta2_ps2_m = -2.0 * 1e-3  # -2 ps²/km = -0.002 ps²/m

    beta2_si = beta2_ps2_m * 1e-24  # ps²/m → s²/m for P0/L_D calculation
    P0 = abs(beta2_si) / (gamma * T0.as_s**2)
    L_D = Length(T0.as_s**2 / abs(beta2_si), "m")

    print(f"γ = {gamma*1e3:.3f} 1/(W·km)")
    print(f"β₂ = -2.0 ps²/km = {beta2_ps2_m} ps²/m")
    print(f"Peak power P₀ = {P0:.1f} W (N=1 soliton)")
    print(f"L_D = {L_D.as_m:.2f} m, fiber length = {3*L_D.as_m:.2f} m")

    # Create a silica Raman response
    silica_spec = RamanSpec(
        name="Silica",
        raman_shift_cm=440.0,
        raman_linewidth_cm=45.0,
        fR=0.18,
    )
    raman_grid = TemporalGrid(N=2**13, Tmax=Time(500, "fs"))
    raman_response = RamanResponse(spec=silica_spec, grid=raman_grid,
                                   tau1=Time(12.2, "fs").as_s, tau2=Time(32, "fs").as_s)

    fiber = FiberProfile(
        n2=n2,
        alpha=0.0,
        A_eff=A_eff,
        length=Length(3 * L_D.as_m, "m"),
        raman_response=raman_response,
    )

    # ── Run simulations ────────────────────────────────────────────────────

    # Case A: Dispersion + Kerr only (baseline)
    pulse = make_high_power_pulse(
        wavelength_nm=central_wl.as_nm, T0_fs=T0.as_fs, peak_power_W=P0
    )
    betas_arr = np.array([beta2_ps2_m])
    n_steps_base = GNLSESolver.estimate_num_steps(
        pulse, fiber, betas_arr, include_self_steepening=False
    )
    n_steps_steep = GNLSESolver.estimate_num_steps(
        pulse, fiber, betas_arr, include_self_steepening=True
    )

    solver_A = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2_ps2_m]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_A.propagate(num_steps=n_steps_base)

    # Case B: + Raman scattering
    solver_B = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2_ps2_m]),
        include_raman=True,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver_B.propagate(num_steps=n_steps_base)

    # Case C: + Self-steepening
    solver_C = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2_ps2_m]),
        include_raman=False,
        include_self_steepening=True,
        include_tpa=False,
    )
    solver_C.propagate(num_steps=n_steps_steep)

    # Case D: + Both Raman + self-steepening
    solver_D = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2_ps2_m]),
        include_raman=True,
        include_self_steepening=True,
        include_tpa=False,
    )
    solver_D.propagate(num_steps=max(n_steps_steep, n_steps_base))

    # ── Plot 1: Spectrum comparison ────────────────────────────────────────
    # Use zero-padded FFT for sub-bin spectral resolution (5 THz is too coarse
    # for a 50-fs pulse whose spectral shifts are ~0.1 THz over 3 L_D).

    N_pad = 2**17
    dt = pulse.grid.dt
    freq_fine = np.fft.fftshift(np.fft.fftfreq(N_pad, d=dt)) * 2 * np.pi
    freq_offset_THz_fine = freq_fine / (2 * np.pi) / 1e12

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Effect of Raman and Self-Steepening on Spectrum",
        fontsize=14,
        fontweight="bold",
    )

    cases = [
        (solver_A, axes[0, 0], "Dispersion + Kerr only"),
        (solver_B, axes[0, 1], "+ Raman scattering"),
        (solver_C, axes[1, 0], "+ Self-steepening"),
        (solver_D, axes[1, 1], "+ Raman + Self-steepening"),
    ]

    z_display_indices = [0, len(solver_A.evolution) // 4, len(solver_A.evolution) // 2,
                         3 * len(solver_A.evolution) // 4, len(solver_A.evolution) - 1]
    z_display_indices = sorted(set(min(i, len(solver_A.evolution) - 1) for i in z_display_indices))
    cmap = plt.cm.viridis
    z_colors = [cmap(i / max(len(z_display_indices) - 1, 1)) for i in range(len(z_display_indices))]

    for solver, ax, title in cases:
        z_array = solver.z_array
        mask = np.abs(freq_offset_THz_fine) < 10
        for k, zi in enumerate(z_display_indices):
            A = solver.evolution[zi].envelope_field
            A_pad = np.zeros(N_pad, dtype=complex)
            A_pad[:len(A)] = A
            spec = np.abs(np.fft.fftshift(np.fft.fft(A_pad))) ** 2
            spec_dB = 10 * np.log10(spec[mask] + 1e-30)
            spec_dB -= spec_dB.max()
            ax.plot(freq_offset_THz_fine[mask], spec_dB, color=z_colors[k], linewidth=1.5,
                    label=f"z={z_array[zi]*1e3:.0f} mm")

        ax.set_xlabel("Frequency offset (THz)")
        ax.set_ylabel("Normalized spectrum (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=1.0)
        ax.set_xlim([-10, 10])
        ax.set_ylim([-40, 2])
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        "examples/images/14_gnlse_raman_steepening_spectra.png", dpi=150, bbox_inches="tight"
    )
    print("Saved: examples/images/14_gnlse_raman_steepening_spectra.png")
    plt.close()

    # ── Plot 2: Temporal profile with self-steepening ─────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        "Temporal Effects: Self-Steepening and Raman", fontsize=14, fontweight="bold"
    )

    t_ps = pulse.grid.t * 1e12

    # Panel 1: Self-steepening effect on temporal profile
    for i, wave in enumerate(solver_C.evolution):
        axes[0].plot(t_ps, np.abs(wave.envelope_field), alpha=0.7, linewidth=1.2)
    axes[0].plot(
        t_ps, np.abs(pulse.envelope_field), "k--", linewidth=2.0, label="Input"
    )
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Field amplitude")
    axes[0].set_title("Self-steepening: trailing edge steepens (optical shock)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([-1, 1])  # Zoom to ±1 ps

    # Panel 2: Raman-induced redshift (zero-padded FFT for sub-bin resolution)
    N_pad = 2**17
    freq_offset_THz_fine = np.fft.fftshift(np.fft.fftfreq(N_pad, d=pulse.grid.dt)) / 1e12
    center_idx = N_pad // 2

    z_A = solver_A.z_array
    z_D = solver_D.z_array

    def compute_centroids(evolution):
        centroids = []
        for wave in evolution:
            A = wave.envelope_field
            A_pad = np.zeros(N_pad, dtype=complex)
            A_pad[:len(A)] = A
            spec = np.abs(np.fft.fftshift(np.fft.fft(A_pad))) ** 2
            total = np.sum(spec)
            centroid = np.sum(freq_offset_THz_fine * spec) / total if total > 0 else 0
            centroids.append(centroid)
        return np.array(centroids)

    centroids_A = compute_centroids(solver_A.evolution)
    centroids_D = compute_centroids(solver_D.evolution)

    axes[1].plot(
        z_A * 1e3, centroids_A, "b-", linewidth=2.0, label="Kerr only"
    )
    axes[1].plot(
        z_D * 1e3, centroids_D, "r-", linewidth=2.0, label="Kerr + Raman",
    )
    axes[1].set_xlabel("Propagation distance (mm)")
    axes[1].set_ylabel("Spectral centroid shift (THz)")
    axes[1].set_title("Raman scattering causes soliton self-frequency shift (redshift)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(
        "examples/images/14_gnlse_raman_steepening_temporal.png", dpi=150, bbox_inches="tight"
    )
    print("Saved: examples/images/14_gnlse_raman_steepening_temporal.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    print("\n— Effect comparison —")

    def spec_fwhm(evolution, label=""):
        N_pad = 2**17
        A = evolution[-1].envelope_field
        A_pad = np.zeros(N_pad, dtype=complex)
        A_pad[:len(A)] = A
        spec = np.abs(np.fft.fftshift(np.fft.fft(A_pad))) ** 2
        freq = np.fft.fftshift(np.fft.fftfreq(N_pad, d=pulse.grid.dt)) / 1e12
        peak_idx = np.argmax(spec)
        half_max = spec[peak_idx] / 2
        above = np.where(spec >= half_max)[0]
        if len(above) < 2:
            return 0.0
        left, right = above[0], above[-1]
        if left > 0:
            fl, fc = freq[left - 1], freq[left]
            sl, sc = spec[left - 1], spec[left]
            left_interp = fc - (fc - fl) * (half_max - sl) / (sc - sl) if sc != sl else fc
        else:
            left_interp = freq[left]
        if right < len(freq) - 1:
            fc, fr = freq[right], freq[right + 1]
            sc, sr = spec[right], spec[right + 1]
            right_interp = fc + (fr - fc) * (half_max - sc) / (sr - sc) if sr != sc else fc
        else:
            right_interp = freq[right]
        return right_interp - left_interp

    for name, s in [
        ("Kerr only", solver_A),
        ("+ Raman", solver_B),
        ("+ Steepening", solver_C),
        ("+ Both", solver_D),
    ]:
        fwhm = spec_fwhm(s.evolution)
        print(f"  {name:15s}: spectral FWHM = {fwhm:.3f} THz")


if __name__ == "__main__":
    main()
