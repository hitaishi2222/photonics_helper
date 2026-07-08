"""
Example: Raman Frequency Response (Layer 3)
=============================================

Demonstrates the frequency-domain Raman response H(Ω) = F{h_R(t)} for
different materials. Shows how the time-domain lattice oscillation transforms
into the measured Raman gain spectrum.

The frequency-domain response reveals:
- **Im(H(Ω))** — The Raman gain spectrum (what's measured in experiments)
- **Re(H(Ω))** — The dispersive part (Kerr index change)
- **|H(Ω)|** — The magnitude spectrum
- **∠H(Ω)** — The phase (π/2 shift at resonance for causal response)

Key observations:
- Peak of Im(H) = Raman shift frequency
- Width of Im(H) peak = Raman linewidth
- Q factor = f_resonance / FWHM
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.raman import RamanSpec, RamanResponse, RamanFrequencyResponse
from photonics_helper.pulse import TemporalGrid


def main():
    # ── Create frequency responses for different materials ──────────────────

    materials = ["Silica", "CdS", "Diamond", "As2Se3"]
    freq_responses = {}

    print("Material Frequency Response Summary")
    print("=" * 70)
    for name in materials:
        spec = RamanSpec.from_database(name)
        if spec.fR is None or spec.fR == 0:
            print(f"{name:10s}: fR=0 — no delayed response (pure Kerr)")
            continue
        grid = TemporalGrid(N=2**14, Tmax=10e-12)
        resp = RamanResponse(spec=spec, grid=grid)
        fr = RamanFrequencyResponse(response=resp, grid=grid)
        freq_responses[name] = fr

        print(f"{name:10s}: f_res={fr.resonance_frequency_THz:8.2f} THz, "
              f"FWHM={fr.resonance_FWHM_THz:7.3f} THz, "
              f"Q={fr.quality_factor:7.1f}")
    print()

    # ── Plot 1: Im(H) comparison — Raman gain spectra ───────────────────────

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Raman Gain Spectra Im(H(Ω)) — Material Comparison",
                 fontsize=14, fontweight="bold")

    colors = {"Silica": "#00d4ff", "CdS": "#a78bfa", "As2Se3": "#fbbf24"}

    for name in freq_responses:
        fr = freq_responses[name]
        w_THz = fr.grid.w / (2 * np.pi * 1e12)
        # Plot only positive frequencies
        pos = w_THz > 0
        ax.plot(w_THz[pos], fr.H_imag[pos], color=colors[name], linewidth=1.5, label=name)
        ax.axvline(fr.resonance_frequency_THz, color=colors[name], linestyle="--",
                   alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Frequency (THz)", fontsize=12)
    ax.set_ylabel("Im(H(Ω))", fontsize=12)
    ax.set_title("Raman Gain Spectrum (Imaginary part of frequency response)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("examples/images/06_raman_frequency_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/06_raman_frequency_comparison.png")
    plt.close()

    # ── Plot 2: Silica full frequency response (4 panels) ───────────────────

    silica = freq_responses["Silica"]
    w_THz = silica.grid.w / (2 * np.pi * 1e12)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Silica Raman Frequency Response  "
        f"(f_res={silica.resonance_frequency_THz:.2f} THz, "
        f"Q={silica.quality_factor:.1f})",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Real part
    ax1 = axes[0, 0]
    ax1.plot(w_THz, silica.H_real, linewidth=1.5, color="#00d4ff")
    ax1.set_ylabel("Re(H(Ω))", fontsize=10)
    ax1.set_title("Real Part — Dispersive Response", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(silica.resonance_frequency_THz, color="r", linestyle="--", alpha=0.5,
                label=f"Resonance: {silica.resonance_frequency_THz:.2f} THz")
    ax1.legend(fontsize=8)

    # Panel 2: Imaginary part (gain spectrum)
    ax2 = axes[0, 1]
    ax2.plot(w_THz, silica.H_imag, linewidth=1.5, color="#a78bfa")
    ax2.set_ylabel("Im(H(Ω))", fontsize=10)
    ax2.set_title("Imaginary Part — Raman Gain Spectrum", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axvline(silica.resonance_frequency_THz, color="r", linestyle="--", alpha=0.5)
    # Shade FWHM region
    fwhm = silica.resonance_FWHM_THz
    f_res = silica.resonance_frequency_THz
    ax2.axvspan(f_res - fwhm/2, f_res + fwhm/2, alpha=0.15, color="orange",
                label=f"FWHM: {fwhm:.3f} THz")
    ax2.legend(fontsize=8)

    # Panel 3: Magnitude
    ax3 = axes[1, 0]
    ax3.plot(w_THz, silica.H_magnitude, linewidth=1.5, color="#34d399")
    ax3.set_ylabel("|H(Ω)|", fontsize=10)
    ax3.set_title("Magnitude Spectrum", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color="k", linewidth=0.5)
    ax3.axvline(silica.resonance_frequency_THz, color="r", linestyle="--", alpha=0.5)

    # Panel 4: Phase
    ax4 = axes[1, 1]
    ax4.plot(w_THz, silica.H_phase, linewidth=1.5, color="#fbbf24")
    ax4.set_xlabel("Frequency (THz)", fontsize=10)
    ax4.set_ylabel("∠H(Ω) (rad)", fontsize=10)
    ax4.set_title("Phase Spectrum", fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color="k", linewidth=0.5)
    ax4.axhline(np.pi/2, color="r", linestyle=":", alpha=0.5, label="π/2 (causal)")
    ax4.axvline(silica.resonance_frequency_THz, color="r", linestyle="--", alpha=0.5)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("examples/images/06_raman_frequency_silica.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/06_raman_frequency_silica.png")
    plt.close()

    # ── Plot 3: Effect of linewidth on resonance sharpness ──────────────────

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Linewidth Effect on Raman Resonance", fontsize=14, fontweight="bold")

    linewidths_cm = [5, 45, 100]  # cm⁻¹
    labels = ["Narrow (5 cm⁻¹)", "Silica (45 cm⁻¹)", "Wide (100 cm⁻¹)"]

    for idx, (lw, label) in enumerate(zip(linewidths_cm, labels)):
        ax = axes[idx]
        spec = RamanSpec(
            name=f"Linewidth={lw}",
            raman_shift_cm=440,
            raman_linewidth_cm=lw,
            fR=0.18,
        )
        grid = TemporalGrid(N=2**14, Tmax=10e-12)
        resp = RamanResponse(spec=spec, grid=grid)
        fr = RamanFrequencyResponse(response=resp, grid=grid)

        w_THz = fr.grid.w / (2 * np.pi * 1e12)
        pos = w_THz > 0
        ax.plot(w_THz[pos], fr.H_imag[pos], linewidth=1.5, color="#a78bfa")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Im(H(Ω))")
        ax.set_title(f"{label}\nQ = {fr.quality_factor:.1f}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(fr.resonance_frequency_THz, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("examples/images/06_raman_linewidth_effect.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/06_raman_linewidth_effect.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    print("\n— Key Observations —")
    print("• Peak of Im(H(Ω)) = Raman gain spectrum (what's measured)")
    print("• Width of peak = Raman linewidth")
    print("• Q = f_resonance / FWHM (higher Q = sharper resonance)")
    print("• Phase jumps by π/2 at resonance (causal response)")


if __name__ == "__main__":
    main()
