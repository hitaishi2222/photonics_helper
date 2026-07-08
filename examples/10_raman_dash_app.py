"""
Example: Dash Interactive App (Layer 7)
==========================================

Launches the interactive Raman scattering explorer built with Dash.

The app ties all 6 layers together in a single web interface:
- Sidebar: material selector, fR/τ1/τ2 sliders, pump wavelength slider,
  and an "add to compare" button.
- Main area: layer selector tabs (1–6) with live visualizations
  and a summary data panel below.

Run with::

    python examples/10_raman_dash_app.py

This launches the Dash server at http://localhost:8050.
"""

from photonics_helper.raman import app


def main():
    print("Launching Raman Explorer Dash app...")
    print("Open http://localhost:8050 in your browser.")
    print("Press Ctrl+C to stop.\n")

    # Run the Dash app
    # In production, disable debug mode:
    #   app().run(debug=False, port=8050)
    app().run(debug=True, port=8050)


if __name__ == "__main__":
    main()
