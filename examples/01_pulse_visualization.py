"""
Example: Pulse Envelope Visualization
======================================

Demonstrates the visualization capabilities of the Envelope class.
Shows different pulse shapes, chirp effects, and 3D views.
"""

from photonics_helper.base import Time
from photonics_helper.pulse import Envelope
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG export
import matplotlib.pyplot as plt

# Create example pulses — 50 fs pulse, transform-limited Gaussian
pulses = {
    "gaussian": Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(50, "fs")),
    "sech": Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(50, "fs")),
    "chirped_gaussian": Envelope(
        shape="gaussian", peak_amplitude=1.0, pulse_width=Time(50, "fs"), chirp=2.0
    ),
    "airy": Envelope(shape="airy", peak_amplitude=1.0, pulse_width=Time(50, "fs")),
    "cosine": Envelope(shape="cosine", peak_amplitude=1.0, pulse_width=Time(50, "fs")),
    "triangular": Envelope(shape="triangular", peak_amplitude=1.0, pulse_width=Time(50, "fs")),
    "parabolic": Envelope(shape="parabolic", peak_amplitude=1.0, pulse_width=Time(50, "fs")),
}

# Generate 2D visualizations (Plotly HTML - interactive)
print("Generating Plotly HTML (interactive)...")
for name, pulse in pulses.items():
    fig = pulse.visualize_2d(backend="plotly")
    fig.write_html(f"examples/images/{name}_2d_plotly.html")
    print(f"  Saved {name}_2d_plotly.html")

# Generate 2D visualizations (Matplotlib PNG - static)
print("\nGenerating Matplotlib PNG (static)...")
for name, pulse in pulses.items():
    fig = pulse.visualize_2d(backend="matplotlib", figsize=(20, 14))
    fig.savefig(f"examples/images/{name}_2d_matplotlib.png", dpi=300, bbox_inches="tight")
    print(f"  Saved {name}_2d_matplotlib.png")
    plt.close(fig)

# Generate 3D visualizations (Plotly HTML)
print("\nGenerating 3D Plotly HTML...")
for name, pulse in pulses.items():
    fig = pulse.visualize_3d()
    fig.write_html(f"examples/images/{name}_3d_plotly.html")
    print(f"  Saved {name}_3d_plotly.html")

# Additional: chirped pulse comparison
print("\nGenerating chirped pulse example...")
fig = pulses["chirped_gaussian"].visualize_2d(backend="matplotlib", figsize=(20, 14))
fig.savefig("examples/images/chirped_gaussian_2d_matplotlib.png", dpi=300, bbox_inches="tight")
print("  Saved chirped_gaussian_2d_matplotlib.png")
plt.close(fig)

print("\n✓ All examples saved to examples/")
print("\nNote: For PNG export of Plotly figures, install Chrome:")
print("  $ plotly_get_chrome")
