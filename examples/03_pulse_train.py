"""
Example: Pulse Train Visualization
====================================

Demonstrates mode-locked laser pulse trains with repetition rate.
Shows how pulses repeat at the cavity round-trip frequency.
"""

import numpy as np
from photonics_helper.pulse import Envelope, Wave, TemporalGrid
from photonics_helper.base import Wavelength, Frequency, Time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a 50 fs Gaussian pulse
pulse = Envelope(
    shape="gaussian",
    peak_amplitude=1.0,
    pulse_width=Time(50, "fs"),
)

# Mode-locked laser parameters
repetition_rate = Frequency(1, "GHz")  # 1 GHz (1 ns spacing)
n_pulses = 11  # 11 pulses centered at t=0 (−5 to +5 ns)
central_wavelength = Wavelength(1550, "nm")

# Create pulse train
# 11 pulses at 1 GHz spacing (1 ns apart), centered at t=0.
grid = TemporalGrid.for_pulse_train(
    repetition_rate=repetition_rate,
    n_pulses=n_pulses,
    pulse_width=pulse.pulse_width,
    N=2**20,
)
wave = Wave.from_pulse_train(
    envelope=pulse,
    central_wavelength=central_wavelength,
    grid=grid,
    repetition_rate=repetition_rate,
    n_pulses=n_pulses,
)

# Visualize the pulse train — shows all pulses with carrier oscillations
fig = wave.visualize(
    t_unit="fs",
    w_unit="THz",
    t_scale=1e15,  # Convert seconds to femtoseconds
    w_scale=1 / (2 * np.pi * 1e12),  # Convert rad/s to THz
    show_electric_field=True,
    show_phase=True,
    show_spectrogram=True,
    figsize=(14, 12),
)

fig.savefig("examples/images/pulse_train_matplotlib.png", dpi=150, bbox_inches="tight")
print("Saved pulse_train_matplotlib.png")
plt.close(fig)

# Also show a zoomed-in view of a single pulse
fig_zoom = plt.figure(figsize=(14, 8))
gs_zoom = plt.GridSpec(2, 2, figure=fig_zoom, hspace=0.35, wspace=0.3)

# Zoom into pulse at t=0
intensity_zoom = np.abs(wave.envelope_field[np.abs(wave.grid.t) <= 2e-12]) ** 2
t_zoom = wave.grid.t[np.abs(wave.grid.t) <= 2e-12]

ax_zoom_t = fig_zoom.add_subplot(gs_zoom[0, :])
ax_zoom_t.plot(t_zoom * 1e15, intensity_zoom, color="#00d4ff", linewidth=1.5)
ax_zoom_t.set_xlabel("Time (fs)")
ax_zoom_t.set_ylabel("Intensity")
ax_zoom_t.set_title("Zoomed-in: Single Pulse at t=0")
ax_zoom_t.grid(True, alpha=0.3)

# Spectral profile
spec = np.abs(wave.spectrum) ** 2
w_thz = wave.grid.w / (2 * np.pi * 1e12)
ax_zoom_w = fig_zoom.add_subplot(gs_zoom[1, 0])
spec_shifted = np.fft.fftshift(spec)
w_shifted = np.fft.fftshift(w_thz)
ax_zoom_w.plot(w_shifted, spec_shifted, color="#a78bfa", linewidth=1.5)
ax_zoom_w.set_xlabel("Frequency (THz)")
ax_zoom_w.set_ylabel("Spectral Intensity")
ax_zoom_w.set_title("Spectral Profile (single pulse)")
ax_zoom_w.grid(True, alpha=0.3)

# Pulse train overview — show all 11 pulses
ax_overview = fig_zoom.add_subplot(gs_zoom[1, 1])
t_ns = wave.grid.t * 1e9
intensity_full = np.abs(wave.envelope_field) ** 2
intensity_norm = intensity_full / np.max(intensity_full)
ax_overview.plot(t_ns, intensity_norm, color="#00d4ff", linewidth=0.5)
ax_overview.set_xlim(-6, 6)
ax_overview.set_xlabel("Time (ns)")
ax_overview.set_ylabel("Normalized Intensity")
ax_overview.set_title("Pulse Train Overview (11 pulses, −5 to +5 ns)")
ax_overview.grid(True, alpha=0.3)

fig_zoom.savefig("examples/images/pulse_train_zoom.png", dpi=150, bbox_inches="tight")
print("Saved pulse_train_zoom.png")
plt.close(fig_zoom)

# Also show a single pulse for comparison
single_fig = pulse.visualize_2d(backend="matplotlib", figsize=(14, 10))
single_fig.savefig("examples/images/single_pulse_comparison.png", dpi=150, bbox_inches="tight")
print("Saved single_pulse_comparison.png")
plt.close(single_fig)

print("\n✓ Pulse train examples saved to examples/")
