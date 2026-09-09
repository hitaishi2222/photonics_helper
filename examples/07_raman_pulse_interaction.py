"""
Example: Pulse-Raman Interaction (Layer 4)
============================================

Demonstrates how an optical pulse interacts with a Raman-active medium.
Computes the nonlinear polarization P_NL(t) = n₂ · (R(t) ⊗ I(t)) via
FFT-based convolution of the pulse intensity with the Raman response.

When an intense pulse propagates through a Raman-active material:
1. The pulse intensity I(t) = |E(t)|² drives the nonlinear polarization
2. The instantaneous (Kerr) component responds immediately with the pulse
3. The delayed (lattice) component builds up during the pulse and decays after
4. This delayed polarization causes stimulated Raman scattering (SRS)

The convolution naturally produces a polarization that:
- Tracks the pulse intensity (instantaneous part)
- Peaks slightly after the pulse maximum (delayed part)
- Oscillates and decays after the pulse exits (Raman scattering)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.raman import RamanSpec, RamanResponse, RamanPulseInteraction
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.base import Wavelength, Time


def main():
    # ── Create a Gaussian pulse ─────────────────────────────────────────────

    grid = TemporalGrid(N=2**14, Tmax=Time(20e-12, "s"))
    envelope = Envelope(
        shape="gaussian",
        peak_amplitude=1.0,
        pulse_width=Time(100, "fs"),  # 100 fs FWHM ≈ 118 fs
    )
    wave = Wave(
        grid=grid,
        envelope=envelope,
        central_wavelength=Wavelength(800, "nm"),
    )
    fwhm_fs = envelope.fwhm.as_fs if hasattr(envelope.fwhm, 'as_fs') else envelope.fwhm * 1e15
    print(f"Pulse: {envelope.shape}, T₀={envelope.pulse_width.as_fs:.1f} fs, "
              f"FWHM={fwhm_fs:.1f} fs, peak power={wave.peak_power():.3f} W")

    # ── Create Raman responses for different materials ──────────────────────

    materials = ["Silica", "CdS", "Diamond", "As2Se3"]
    interactions = {}

    for name in materials:
        spec = RamanSpec.from_database(name)
        response = RamanResponse(spec=spec, grid=grid)
        interaction = RamanPulseInteraction(
            pulse=wave, response=response, spec=spec
        )
        interactions[name] = interaction
        P_NL = interaction.nonlinear_polarization
        print(f"{name}: max|P_NL|={np.max(np.abs(P_NL)):.3e}, "
              f"τ1={response.tau1*1e15:.2f} fs, τ2={response.tau2*1e15:.2f} fs")

    # ── Plot 1: 4-panel interaction for Silica ──────────────────────────────

    silica = interactions["Silica"]
    t_ps = silica.grid.t * 1e12
    I_t = silica.pulse.envelope_intensity
    P_NL = silica.nonlinear_polarization

    fig = silica.plot_interaction(backend="matplotlib", figsize=(12, 12))
    plt.savefig("examples/images/07_raman_pulse_interaction_silica.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/07_raman_pulse_interaction_silica.png")
    plt.close()

    # ── Plot 2: Compare P_NL for different materials ────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Nonlinear Polarization P_NL(t) — Material Comparison",
                 fontsize=14, fontweight="bold")

    colors = {"Silica": "#00d4ff", "CdS": "#a78bfa",
              "Diamond": "#34d399", "As2Se3": "#fbbf24"}

    for idx, name in enumerate(materials):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        interaction = interactions[name]
        t_ps = interaction.grid.t * 1e12
        P_NL = interaction.nonlinear_polarization
        I_t = interaction.pulse.envelope_intensity

        # Normalize both for comparison
        I_norm = I_t / np.max(I_t) if np.max(I_t) > 0 else I_t
        PNL_norm = P_NL / np.max(np.abs(P_NL)) if np.max(np.abs(P_NL)) > 0 else P_NL

        ax.plot(t_ps, I_norm, color="#555555", linewidth=1.0, alpha=0.5,
                label="Input pulse")
        ax.plot(t_ps, PNL_norm, color=colors[name], linewidth=1.5, label="P_NL(t)")
        ax.set_xlabel("Time (ps)", fontsize=10)
        ax.set_ylabel("Normalized amplitude", fontsize=10)
        ax.set_title(f"{name}  (n₂={interaction.spec.n2:.1e} m²/W)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("examples/images/07_raman_pulse_interaction_comparison.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/07_raman_pulse_interaction_comparison.png")
    plt.close()

    # ── Plot 3: Delayed polarization lag ────────────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Delayed Polarization Lag — Silica",
                 fontsize=14, fontweight="bold")

    t_ps = silica.grid.t * 1e12
    I_t = silica.pulse.envelope_intensity
    P_NL = silica.nonlinear_polarization
    R_delayed = silica.response.delayed_response(silica.grid.t)

    # Normalize for visualization
    I_norm = I_t / np.max(I_t)
    PNL_norm = P_NL / np.max(np.abs(P_NL))
    R_norm = R_delayed / np.max(np.abs(R_delayed)) if np.max(np.abs(R_delayed)) > 0 else R_delayed

    ax.plot(t_ps, I_norm, color="#00d4ff", linewidth=1.5, label="|E(t)|² (pulse)")
    ax.plot(t_ps, R_norm * 0.3, color="#a78bfa", linewidth=1.0, alpha=0.7,
            label="Delayed response h_R(t)")
    ax.plot(t_ps, PNL_norm, color="#34d399", linewidth=2.0, label="P_NL(t)")

    # Mark peak positions
    I_peak_t = t_ps[np.argmax(I_t)]
    PNL_peak_t = t_ps[np.argmax(P_NL)]
    ax.axvline(I_peak_t, color="#00d4ff", linestyle="--", alpha=0.5,
               label=f"Pulse peak: {I_peak_t:.3f} ps")
    ax.axvline(PNL_peak_t, color="#34d399", linestyle="-.", alpha=0.5,
               label=f"P_NL peak: {PNL_peak_t:.3f} ps")

    lag_ps = (PNL_peak_t - I_peak_t) * 1e12  # convert to fs
    ax.text(0.02, 0.95, f"Delay: {lag_ps:.1f} fs", transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Time (ps)", fontsize=11)
    ax.set_ylabel("Normalized amplitude", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig("examples/images/07_raman_pulse_interaction_lag.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/07_raman_pulse_interaction_lag.png")
    plt.close()

    # ── Plot 4: Effect of different pulse shapes ────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Pulse Shape Effect on P_NL — Silica",
                 fontsize=14, fontweight="bold")

    shapes = ["gaussian", "sech", "lorentzian"]
    shape_labels = ["Gaussian", "Sech²", "Lorentzian"]

    for idx, (shape, label) in enumerate(zip(shapes, shape_labels)):
        ax = axes[idx]

        env = Envelope(shape=shape, peak_amplitude=1.0, pulse_width=Time(100, "fs"))
        w = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(800, "nm"))
        resp = RamanResponse(spec=silica.spec, grid=grid)
        interact = RamanPulseInteraction(pulse=w, response=resp, spec=silica.spec)

        t_ps = interact.grid.t * 1e12
        I_t = interact.pulse.envelope_intensity
        P_NL = interact.nonlinear_polarization

        I_norm = I_t / np.max(I_t) if np.max(I_t) > 0 else I_t
        PNL_norm = P_NL / np.max(np.abs(P_NL)) if np.max(np.abs(P_NL)) > 0 else P_NL

        ax.plot(t_ps, I_norm, color="#00d4ff", linewidth=1.0, alpha=0.7,
                label="|E(t)|²")
        ax.plot(t_ps, PNL_norm, color="#34d399", linewidth=1.5, label="P_NL(t)")
        ax.set_xlabel("Time (ps)", fontsize=10)
        ax.set_ylabel("Normalized", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("examples/images/07_raman_pulse_interaction_shapes.png",
                dpi=150, bbox_inches="tight")
    print("Saved: examples/07_raman_pulse_interaction_shapes.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    print("\n— Layer 4 Summary —")
    print("The nonlinear polarization P_NL(t) = n₂ · (R(t) ⊗ I(t)) captures")
    print("how a pulse induces a delayed response in a Raman-active medium.")
    print("Key observations:")
    print("  • P_NL tracks the pulse intensity (instantaneous Kerr)")
    print("  • P_NL peaks slightly after the pulse (delayed Raman)")
    print("  • Materials with larger n₂ produce stronger polarization")
    print("  • The delay depends on τ1 (oscillation period) of the material")


if __name__ == "__main__":
    main()
