"""
Example: Light vs Dark Theme Comparison
=========================================

Demonstrates the theme parameter for visualize_2d() and visualize_3d().
"""

import numpy as np
from photonics_helper.base import Time
from photonics_helper.pulse import Envelope
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a chirped pulse (good for showing phase in both themes) — 50 fs
pulse = Envelope(
    shape="gaussian",
    peak_amplitude=1.0,
    pulse_width=Time(50, "fs"),
    chirp=2.0,
)

# Plotly: light vs dark
fig_light = pulse.visualize_2d(backend="plotly", theme="light")
fig_light.write_html("examples/images/pulse_light_plotly.html")
print("Saved pulse_light_plotly.html")

fig_dark = pulse.visualize_2d(backend="plotly", theme="dark")
fig_dark.write_html("examples/images/pulse_dark_plotly.html")
print("Saved pulse_dark_plotly.html")

# Matplotlib: light vs dark
fig_light = pulse.visualize_2d(backend="matplotlib", theme="light", figsize=(14, 10))
fig_light.savefig("examples/images/pulse_light_matplotlib.png", dpi=150, bbox_inches="tight")
print("Saved pulse_light_matplotlib.png")
plt.close(fig_light)

fig_dark = pulse.visualize_2d(backend="matplotlib", theme="dark", figsize=(14, 10))
fig_dark.savefig("examples/images/pulse_dark_matplotlib.png", dpi=150, bbox_inches="tight")
print("Saved pulse_dark_matplotlib.png")
plt.close(fig_dark)

# 3D: light vs dark
fig_3d_light = pulse.visualize_3d(theme="light")
fig_3d_light.write_html("examples/images/pulse_3d_light_plotly.html")
print("Saved pulse_3d_light_plotly.html")

fig_3d_dark = pulse.visualize_3d(theme="dark")
fig_3d_dark.write_html("examples/images/pulse_3d_dark_plotly.html")
print("Saved pulse_3d_dark_plotly.html")

print("\n✓ Theme comparison examples saved to examples/")
