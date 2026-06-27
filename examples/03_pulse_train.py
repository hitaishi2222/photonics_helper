"""
Example: Pulse Train Visualization
====================================

Demonstrates mode-locked laser pulse trains with repetition rate.
Shows how pulses repeat at the cavity round-trip frequency.
"""

import numpy as np
from photonics_helper.pulse import Envelope, Wave
from photonics_helper.base import Wavelength, Frequency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a 50 fs Gaussian pulse
pulse = Envelope(
    shape="gaussian",
    peak_amplitude=1.0,
    pulse_width=50e-15,
)

# Mode-locked laser parameters
repetition_rate = Frequency(1, "GHz")  # 1 GHz (1 ns spacing)
n_pulses = 10  # Show 10 pulses
central_wavelength = Wavelength(1550, "nm")

# Create pulse train
grid = Envelope._make_grid(pulse, N=2**14)
wave = Wave.from_pulse_train(
    envelope=pulse,
    central_wavelength=central_wavelength,
    grid=grid,
    repetition_rate=repetition_rate,
    n_pulses=n_pulses,
)

# Visualize the pulse train
fig = wave.visualize(
    t_unit="ns",
    w_unit="THz",
    t_scale=1e9,  # Convert seconds to nanoseconds
    w_scale=1e12,  # Convert rad/s to THz
    show_electric_field=True,
    show_phase=True,
    show_spectrogram=True,
    figsize=(14, 12),
)

fig.savefig("examples/images/pulse_train_matplotlib.png", dpi=150, bbox_inches="tight")
print("Saved pulse_train_matplotlib.png")
plt.close(fig)

# Also show a single pulse for comparison
single_fig = pulse.visualize_2d(backend="matplotlib", figsize=(14, 10))
single_fig.savefig("examples/images/single_pulse_comparison.png", dpi=150, bbox_inches="tight")
print("Saved single_pulse_comparison.png")
plt.close(single_fig)

print("\n✓ Pulse train examples saved to examples/")
