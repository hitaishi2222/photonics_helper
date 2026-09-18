"""
Example: Unified Dashboard (Raman Explorer + GNLSE Viewer)
============================================================

Launches the combined Dash dashboard:

- **Raman Explorer** tab — the multi-layer Raman browser (material selector,
  fR/τ sliders, pump wavelength, n/k database).
- **GNLSE Viewer** tab — propagate a sech pulse with the GNLSE and render the
  interactive four-panel spectral/temporal summary.

Run with::

    python examples/32_unified_dashboard.py

This launches the Dash server at http://localhost:8050.
"""

from photonics_helper.dashboard import app


def main():
    print("Launching unified Photonics Helper dashboard...")
    print("Tabs: Raman Explorer | GNLSE Viewer")
    print("Open http://localhost:8050 in your browser.")
    print("Press Ctrl+C to stop.\n")
    app().run(debug=False, port=8050)


if __name__ == "__main__":
    main()
