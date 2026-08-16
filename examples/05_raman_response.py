"""
Example: Raman Response Function (Layer 2)
============================================

Demonstrates the time-domain Raman response function h_R(t) for different
materials. Shows the delayed lattice oscillation, instantaneous Kerr response,
and their combination.

The Raman response models how a material's lattice responds to an intense
optical field. The response has two components:

1. **Instantaneous (Electronic Kerr)** — δ(t)-like spike, represents the
   nearly-instantaneous electronic nonlinear response.
2. **Delayed (Lattice Oscillation)** — damped sinusoid h_R(t), represents
   the slower nuclear/lattice response that lags behind the field.

For Silica (the most common fiber material):
  - Raman shift: 440 cm⁻¹ (~13.2 THz)
  - Linewidth: 45 cm⁻¹
  - τ1 (1/ν_R) ≈ 75.8 fs
  - τ2 (damping time) ≈ 236 fs
  - fR (Raman fraction) = 0.18
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonics_helper.raman import RamanSpec, RamanResponse
from photonics_helper.pulse import TemporalGrid
from photonics_helper.base import Time


def main():
    # ── Create Raman responses for different materials ──────────────────────

    materials = ["Silica", "CdS", "Diamond", "As2Se3"]
    responses = {}

    for name in materials:
        spec = RamanSpec.from_database(name)
        # Grid wide enough to capture damped oscillation (tau2 = 1/(π·linewidth))
        tau2 = 1.0 / (np.pi * spec.linewidth_Hz)
        grid = TemporalGrid(N=2**14, Tmax=Time(max(10e-12, 20 * tau2), "s"))
        resp = RamanResponse(spec=spec, grid=grid)
        responses[name] = resp
        print(f"{name}: τ1={resp.tau1*1e15:.2f} fs, τ2={resp.tau2*1e15:.2f} fs, fR={resp.fR}")

    # ── Plot 1: Compare delayed response for all materials ──────────────────

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Raman Delayed Response h_R(t) — Material Comparison",
                 fontsize=14, fontweight="bold")

    colors = {"Silica": "#00d4ff", "CdS": "#a78bfa", "Diamond": "#34d399", "As2Se3": "#fbbf24"}

    for idx, name in enumerate(materials):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        resp = responses[name]
        t_ps = resp.grid.t * 1e12

        # Plot only the delayed response (not the delta spike)
        delayed = resp.delayed_response()
        # Zoom to first few picoseconds where oscillation is visible
        mask = (t_ps >= 0) & (t_ps <= 5)
        ax.plot(t_ps[mask], delayed[mask], color=colors[name], linewidth=1.5, label=name)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("h_R(t) (arb.)")
        ax.set_title(f"{name}  (shift={resp.spec.raman_shift_cm} cm⁻¹, fwhm={resp.spec.raman_linewidth_cm} cm⁻¹)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("examples/images/05_raman_response_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/05_raman_response_comparison.png")
    plt.close()

    # ── Plot 2: Silica response components in detail ────────────────────────

    silica = responses["Silica"]
    t_ps = silica.grid.t * 1e12

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Silica Raman Response  "
        f"(fR={silica.fR:.2f}, τ₁={silica.tau1*1e15:.2f} fs, τ₂={silica.tau2*1e15:.2f} fs)",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Instantaneous (Kerr) response
    inst = silica.instantaneous_response()
    axes[0].plot(t_ps, inst, color="#00d4ff", linewidth=1.5)
    axes[0].set_ylabel("Amplitude (arb.)", fontsize=10)
    axes[0].set_title("Instantaneous Response (1-fR)·δ(t) — Electronic Kerr", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", linewidth=0.5)

    # Panel 2: Delayed (Raman) response
    delayed = silica.delayed_response()
    axes[1].plot(t_ps, delayed, color="#a78bfa", linewidth=1.5)
    axes[1].set_ylabel("Amplitude (arb.)", fontsize=10)
    axes[1].set_title("Delayed Response fR·h_R(t) — Lattice Oscillation", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="k", linewidth=0.5)
    # Show the oscillation envelope
    envelope = np.abs(delayed)
    axes[1].fill_between(t_ps, 0, envelope, alpha=0.15, color="#a78bfa")

    # Panel 3: Combined response
    combined = silica.combined_response()
    axes[2].plot(t_ps, combined, color="#34d399", linewidth=1.5, label="R(t)")
    axes[2].set_xlabel("Time (ps)", fontsize=10)
    axes[2].set_ylabel("Amplitude (arb.)", fontsize=10)
    axes[2].set_title("Combined Response R(t) = (1-fR)δ(t) + fR·h_R(t)", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("examples/images/05_raman_response_silica.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/05_raman_response_silica.png")
    plt.close()

    # ── Plot 3: Effect of varying fR ────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Effect of Raman Fraction fR on Response Function",
                 fontsize=14, fontweight="bold")

    fr_values = [0.0, 0.05, 0.18, 1.0]
    fr_labels = ["Pure Kerr", "Weak Raman", "Silica (default)", "Pure Raman"]

    for idx, (fr, label) in enumerate(zip(fr_values, fr_labels)):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # Create response with custom fR
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        resp = RamanResponse(spec=silica.spec, fR=fr, grid=grid)
        t_ps = resp.grid.t * 1e12

        inst = resp.instantaneous_response()
        delayed = resp.delayed_response()
        combined = resp.combined_response()

        ax.plot(t_ps, inst, color="#00d4ff", linewidth=1.0, alpha=0.7, label="(1-fR)δ(t)")
        ax.plot(t_ps, delayed, color="#a78bfa", linewidth=1.0, alpha=0.7, label="fR·h_R(t)")
        ax.plot(t_ps, combined, color="#34d399", linewidth=1.5, label="R(t)")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Amplitude (arb.)")
        ax.set_title(f"fR = {fr:.2f} — {label}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig("examples/images/05_raman_response_fr_effect.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/05_raman_response_fr_effect.png")
    plt.close()

    # ── Summary ─────────────────────────────────────────────────────────────

    print("\n— Material Parameters —")
    for name in materials:
        spec = RamanSpec.from_database(name)
        print(spec.summary())
        print()


if __name__ == "__main__":
    main()
