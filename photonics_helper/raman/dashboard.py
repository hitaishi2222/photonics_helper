"""Dash dashboard app for browsing Raman material data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..base import (
    Wavelength,
)
from ..pulse import TemporalGrid

if TYPE_CHECKING:
    import dash


from .db import RamanDatabase
from .explorer import MaterialComparison, PumpWavelengthExplorer
from .reference import RAMAN_MATERIALS
from .response import RamanFrequencyResponse, RamanPulseInteraction, RamanResponse
from .spec import RamanSpec

# ─── Dash App (Layer 7) ──────────────────────────────────────────────────────


def _nk_browser_style(layer: str) -> dict:
    """Show the n/k browser only on the 7th tab; hide it elsewhere."""
    return {"display": "block" if layer == "layer-7-nk" else "none"}




def _import_dash():
    """Import Dash lazily and return the pieces the app needs."""
    try:
        import dash
        from dash import Input, Output, State, dcc, html
    except ImportError as exc:
        raise ImportError(
            "Dash is required for the interactive app. "
            "Install it with: pip install dash>=2.18.0"
        ) from exc
    return dash, dcc, html, Input, Output, State


def build_layout(*, dcc, html):
    """Interactive Dash app tying all 6 Raman layers together.

    Layout::

        ┌─────────────────────────────────────────────────┐
        │  Photonics Helper — Raman Explorer              │
        ├──────────┬──────────────────────────────────────┤
        │ Sidebar  │  Main content area                   │
        │          │                                      │
        │ Material │  [Layer selector tabs]               │
        │ selector │  ┌──────────────────────────────┐    │
        │          │  │                              │    │
        │ fR slider│  │   Active visualization       │    │
        │          │  │                              │    │
        │ τ1 slider│  │                              │    │
        │          │  └──────────────────────────────┘    │
        │ τ2 slider│                                      │
        │          │  [Data table / summary]              │
        │ Pump λ   │                                      │
        │ slider   │                                      │
        │          │                                      │
        │ Compare  │                                      │
        │ [+ Add]  │                                      │
        └──────────┴──────────────────────────────────────┘

    Returns
    -------
    dash.Dash — configured Dash application object.

    Notes
    -----
    Requires `dash` to be installed. Run with::

        if __name__ == "__main__":
            app().run(debug=True)
    """
    # ── Build layout ──────────────────────────────────────────────────────────

    sidebar = html.Div(
        [
            html.H4("Material", style={"marginBottom": "5px"}),
            dcc.Dropdown(
                id="material-selector",
                options=[
                    {"label": name, "value": name}
                    for name in sorted(RAMAN_MATERIALS.keys())
                ],
                value="Silica",
                clearable=False,
            ),
            html.Hr(),
            html.H4("Response Parameters", style={"marginBottom": "5px"}),
            html.Label("fR:"),
            dcc.Slider(
                id="fr-slider",
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.18,
                marks={0.0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"},
            ),
            html.Label("τ1 (fs):"),
            dcc.Slider(
                id="tau1-slider",
                min=0.1,
                max=50.0,
                step=0.1,
                marks={1: "1", 5: "5", 10: "10", 20: "20", 50: "50"},
            ),
            html.Label("τ2 (fs):"),
            dcc.Slider(
                id="tau2-slider",
                min=0.5,
                max=100.0,
                step=0.5,
                marks={1: "1", 10: "10", 25: "25", 50: "50", 100: "100"},
            ),
            html.Div(id="tau1-display", style={"fontSize": "11px", "color": "#666"}),
            html.Div(id="tau2-display", style={"fontSize": "11px", "color": "#666"}),
            html.Hr(),
            html.H4("Pump Wavelength", style={"marginBottom": "5px"}),
            dcc.Slider(
                id="pump-wl-slider",
                min=400,
                max=2500,
                step=10,
                marks={
                    500: "500nm",
                    800: "800nm",
                    1000: "1μm",
                    1550: "1550nm",
                    2000: "2μm",
                },
                value=800,
            ),
            html.Div(id="pump-wl-display", style={"fontSize": "11px", "color": "#666"}),
            html.Hr(),
            html.H4("Compare", style={"marginBottom": "5px"}),
            html.Button("+ Add to Compare", id="add-to-compare-btn", n_clicks=0),
            html.Div(id="compare-list", style={"marginTop": "8px", "fontSize": "12px"}),
        ],
        style={
            "width": "280px",
            "minWidth": "280px",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #dee2e6",
            "overflowY": "auto",
            "height": "100vh",
        },
    )

    main_content = html.Div(
        [
            html.H2(
                "Photonics Helper — Raman Explorer",
                style={"textAlign": "center", "marginBottom": "5px"},
            ),
            html.P(
                "Interactive Raman scattering explorer — connect equations, intuition, and visualization",
                style={"textAlign": "center", "color": "#666", "marginBottom": "15px"},
            ),
            # Layer selector tabs
            dcc.Tabs(
                id="layer-tabs",
                value="layer-2-response",
                children=[
                    dcc.Tab(label="1 — Material", value="layer-1-material"),
                    dcc.Tab(label="2 — Response", value="layer-2-response"),
                    dcc.Tab(label="3 — Frequency", value="layer-3-frequency"),
                    dcc.Tab(label="4 — Pulse", value="layer-4-pulse"),
                    dcc.Tab(label="5 — Pump λ", value="layer-5-pump"),
                    dcc.Tab(label="6 — Compare", value="layer-6-compare"),
                    dcc.Tab(label="7 — n/k Database", value="layer-7-nk"),
                ],
            ),
            html.Br(),
            # Output area
            html.Div(id="output-container", style={"margin": "10px 0"}),
            # Summary / data table
            html.Div(
                id="summary-container",
                style={
                    "padding": "10px",
                    "backgroundColor": "#fff",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "4px",
                    "fontFamily": "monospace",
                    "whiteSpace": "pre-wrap",
                },
            ),
            # n/k dataset browser (only visible on the 7th tab)
            html.Div(
                id="nk-browser-container",
                style={"display": "none"},
                children=[
                    html.H3(
                        "Tabulated n/k Dataset Browser",
                        style={"marginBottom": "4px"},
                    ),
                    html.P(
                        "Every stored (λ, n, k) dataset with source/author provenance. "
                        "Filter below, then select a dataset to plot its n(λ) and k(λ).",
                        style={
                            "color": "#666",
                            "marginBottom": "12px",
                            "fontSize": "13px",
                        },
                    ),
                    html.Label("Filter:", style={"fontWeight": "bold"}),
                    dcc.Input(
                        id="nk-filter",
                        type="text",
                        placeholder="type to filter material or source…",
                        style={
                            "width": "100%",
                            "padding": "6px",
                            "boxSizing": "border-box",
                        },
                    ),
                    html.Br(),
                    html.Label(
                        "Dataset:",
                        style={"fontWeight": "bold", "marginTop": "8px"},
                    ),
                    dcc.Dropdown(
                        id="nk-dataset-dropdown",
                        options=[],
                        placeholder="select a dataset to plot…",
                        clearable=False,
                    ),
                    html.Div(
                        id="nk-dataset-info",
                        style={
                            "marginTop": "8px",
                            "fontSize": "12px",
                            "color": "#444",
                            "maxHeight": "120px",
                            "overflowY": "auto",
                        },
                    ),
                    html.Br(),
                    html.Div(id="nk-plot-output"),
                    html.Div(id="nk-table-output"),
                ],
            ),
        ],
        style={"flex": "1", "padding": "10px", "overflowY": "auto"},
    )

    return html.Div(
        [
            html.Div(
                [sidebar, main_content],
                style={
                    "display": "flex",
                    "minHeight": "100vh",
                },
            ),
        ]
    )



def register_callbacks(dash_app, *, dcc, html, Input, Output, State):
    """Register every Raman Explorer callback on ``dash_app``."""
    # ── Callbacks ─────────────────────────────────────────────────────────────

    @dash_app.callback(
        [Output("tau1-display", "children"), Output("tau2-display", "children")],
        Input("material-selector", "value"),
        Input("fr-slider", "value"),
        Input("tau1-slider", "value"),
        Input("tau2-slider", "value"),
    )
    def _update_tau_displays(selected_material, fr_val, tau1_val, tau2_val):
        """Show τ1/τ2 values in sidebar."""
        spec = RamanSpec.from_database(selected_material)
        # Auto-derived tau1/tau2 from the material
        if spec.raman_shift_Hz > 0:
            auto_tau1_fs = (1.0 / spec.raman_shift_Hz) * 1e15
        else:
            auto_tau1_fs = 0.0
        if spec.linewidth_Hz > 0:
            auto_tau2_fs = (1.0 / (np.pi * spec.linewidth_Hz)) * 1e15
        else:
            auto_tau2_fs = 0.0
        return (
            f"Auto τ1 = {auto_tau1_fs:.2f} fs (slider: {tau1_val:.1f} fs)",
            f"Auto τ2 = {auto_tau2_fs:.2f} fs (slider: {tau2_val:.1f} fs)",
        )

    @dash_app.callback(
        Output("compare-list", "children"),
        Input("add-to-compare-btn", "n_clicks"),
        State("material-selector", "value"),
        prevent_initial_call=True,
    )
    def _add_to_compare(n_clicks, material_name):
        """Add current material to the comparison list (display only)."""
        if n_clicks == 0:
            return html.P("No materials added yet.")
        return html.P(f"Added: {material_name}")

    @dash_app.callback(
        Output("output-container", "children"),
        Output("summary-container", "children"),
        Output("nk-browser-container", "style"),
        Input("layer-tabs", "value"),
        Input("material-selector", "value"),
        Input("fr-slider", "value"),
        Input("tau1-slider", "value"),
        Input("tau2-slider", "value"),
        Input("pump-wl-slider", "value"),
    )
    def _update_output(layer, material_name, fr_val, tau1_val, tau2_val, pump_wl_nm):
        """Main callback: render the active layer visualization."""
        import matplotlib

        matplotlib.use("Agg")
        import base64
        from io import BytesIO

        import matplotlib.pyplot as plt

        spec = RamanSpec.from_database(material_name)
        pump_wl = Wavelength(pump_wl_nm, "nm")

        # Build RamanResponse with user overrides
        resp = RamanResponse(
            spec=spec, fR=fr_val, tau1=tau1_val * 1e-15, tau2=tau2_val * 1e-15
        )

        summary_lines = [spec.summary()]
        img_html = ""

        if layer == "layer-1-material":
            # Layer 1: Material spectrum + phonons
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            # Spectrum
            shift = np.linspace(-200, 800, 1000)
            center = spec.raman_shift_cm or 0
            width = (spec.raman_linewidth_cm or 1) / 2
            intensity = (
                (width / np.pi) / ((shift - center) ** 2 + width**2)
                if width > 0
                else np.zeros_like(shift)
            )
            intensity /= np.max(intensity) if np.max(intensity) > 0 else 1
            ax1.plot(shift, intensity, linewidth=2, color="#00d4ff")
            ax1.axvline(x=center, color="r", linestyle="--", alpha=0.5)
            ax1.set_xlabel("Raman shift (cm⁻¹)")
            ax1.set_ylabel("Intensity (arb.)")
            ax1.set_title(f"Raman Spectrum: {spec.name}")
            ax1.grid(True, alpha=0.3)
            # Phonons
            if spec.lo_phonon_cm:
                ax2.barh(["LO"], [spec.lo_phonon_cm], color="blue", alpha=0.7)
            if spec.to_phonon_cm:
                ax2.barh(["TO"], [spec.to_phonon_cm], color="red", alpha=0.7)
            ax2.set_xlabel("Wavenumber (cm⁻¹)")
            ax2.set_title("Phonon Modes")
            plt.tight_layout()
            summary_lines.append(f"Q factor: {spec.quality_factor:.1f}")
            summary_lines.append(
                f"Stokes @ {pump_wl_nm}nm: {spec.stokes_wavelength(pump_wl).as_nm:.1f} nm"
            )
            summary_lines.append(
                f"Anti-Stokes @ {pump_wl_nm}nm: {spec.anti_stokes_wavelength(pump_wl).as_nm:.1f} nm"
            )

        elif layer == "layer-2-response":
            # Layer 2: Time-domain response
            fig = resp.plot_components(backend="matplotlib", figsize=(12, 8))
            summary_lines.append(f"fR = {fr_val:.2f}")
            summary_lines.append(f"τ1 = {tau1_val:.2f} fs, τ2 = {tau2_val:.2f} fs")
            summary_lines.append(
                f"h_R(0) = {resp.delayed_response(np.array([0.0]))[0]:.2e}"
            )

        elif layer == "layer-3-frequency":
            # Layer 3: Frequency response
            freq_resp = RamanFrequencyResponse(response=resp)
            fig = freq_resp.plot_all(backend="matplotlib", figsize=(12, 9))
            summary_lines.append(
                f"Resonance: {freq_resp.resonance_frequency_THz:.2f} THz"
            )
            summary_lines.append(f"FWHM: {freq_resp.resonance_FWHM_THz:.2f} THz")
            summary_lines.append(f"Q = {freq_resp.quality_factor:.1f}")

        elif layer == "layer-4-pulse":
            # Layer 4: Pulse interaction
            from photonics_helper.base import Time
            from photonics_helper.pulse import Envelope, Wave

            grid = TemporalGrid(N=2**14, Tmax=Time(20e-12, "s"))
            envelope = Envelope(
                shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs")
            )
            wave = Wave(grid=grid, envelope=envelope, central_wavelength=pump_wl)
            interaction = RamanPulseInteraction(pulse=wave, response=resp, spec=spec)
            fig = interaction.plot_interaction(backend="matplotlib", figsize=(12, 10))
            summary_lines.append(
                f"P_NL max: {np.max(np.abs(interaction.nonlinear_polarization)):.2e}"
            )
            summary_lines.append(f"n₂ = {spec.n2 or 'N/A'} m²/W")

        elif layer == "layer-5-pump":
            # Layer 5: Pump wavelength
            explorer = PumpWavelengthExplorer(spec=spec)
            fig = explorer.plot_both(pump_wl, backend="matplotlib", figsize=(14, 5))
            stokes = explorer.pump_to_stokes(pump_wl)
            anti = explorer.pump_to_anti_stokes(pump_wl)
            summary_lines.append(f"Pump: {pump_wl_nm} nm")
            summary_lines.append(
                f"Stokes: {stokes.as_nm:.1f} nm (Δλ = {stokes.as_nm - pump_wl_nm:.1f} nm)"
            )
            summary_lines.append(
                f"Anti-Stokes: {anti.as_nm:.1f} nm (Δλ = {pump_wl_nm - anti.as_nm:.1f} nm)"
            )

        elif layer == "layer-6-compare":
            # Layer 6: Material comparison (current + Silica as default comparison)
            comp = MaterialComparison(
                materials=[spec, RamanSpec.from_database("Silica")]
            )
            grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
            fig = comp.plot_all(backend="matplotlib", grid=grid, figsize=(12, 11))
            summary_lines.append(comp.comparison_table())

        elif layer == "layer-7-nk":
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "Use the n/k Database browser below.\n"
                "Filter datasets, then select one to plot n(λ) and k(λ).",
                ha="center",
                va="center",
                transform=ax.transAxes,
                textAlign="center",
            )
            ax.axis("off")
            summary_lines.append(
                "n/k dataset browser — see the table and dropdown below."
            )

        else:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "Select a layer",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            summary_lines.append("Select a layer tab to view visualization.")

        # Convert figure to base64 image
        buf = BytesIO()
        if hasattr(fig, "savefig"):
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")  # type: ignore[union-attr]
            plt.close(fig)  # type: ignore[arg-type]
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        img_html = html.Img(
            src=f"data:image/png;base64,{img_base64}", style={"width": "100%"}
        )

        return img_html, "\n".join(summary_lines), _nk_browser_style(layer)

    @dash_app.callback(
        Output("nk-dataset-dropdown", "options"),
        Output("nk-dataset-dropdown", "value"),
        Input("nk-filter", "value"),
        prevent_initial_call=True,
    )
    def _update_nk_datasets(filter_text):
        """Populate the dataset dropdown, filtered by free-text search."""
        summaries = RamanDatabase().list_nk_dataset_summaries()
        q = (filter_text or "").strip().lower()
        if q:
            summaries = [
                s
                for s in summaries
                if q in str(s["material"]).lower() or q in str(s["source"]).lower()
            ]
        options = [
            {
                "label": f"{s['material']}  •  {s['source']}  "
                f"[{s['wl_min_um']:.2f}–{s['wl_max_um']:.2f} μm, {s['n_points']} pts]",
                "value": s["source"],
            }
            for s in summaries
        ]
        # Sort case-insensitively by material then source for a stable listing.
        order = {s["source"]: i for i, s in enumerate(summaries)}
        options.sort(key=lambda o: order.get(o["value"], 0))
        default = options[0]["value"] if options else None
        return options, default

    @dash_app.callback(
        Output("nk-plot-output", "children"),
        Output("nk-dataset-info", "children"),
        Input("nk-dataset-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _update_nk_plot(source):
        """Load a selected dataset via from_material_database and plot it."""
        if not source:
            return html.P("Select a dataset to plot."), ""
        db = RamanDatabase()
        summaries = {s["source"]: s for s in db.list_nk_dataset_summaries()}
        info = summaries.get(source, {})
        info_lines = [
            f"Material:        {info.get('material', '?')}",
            f"Source:          {info.get('source', source)}",
            f"Valid range:     [{info.get('wl_min_um', 0):.3f}, {info.get('wl_max_um', 0):.3f}] μm",
            f"Points:          {info.get('n_points', 0)}",
        ]
        citation = db.list_nk_citations(info.get("material", "")).get(source, "")
        if citation:
            info_lines.append(f"Citation:        {citation}")
        try:
            from photonics_helper.materials import RefractiveIndex

            ri = RefractiveIndex.from_material_database(source)
        except ValueError as exc:
            return html.P(str(exc)), "\n".join(info_lines)

        import matplotlib

        matplotlib.use("Agg")
        import base64
        from io import BytesIO

        import matplotlib.pyplot as plt

        wl = ri.wl.as_um
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(wl, ri.n)
        ax1.set_xlabel("Wavelength [μm]")
        ax1.set_ylabel("n(λ)")
        ax1.title = f"{info.get('material', '?')} — {source}"
        ax1.grid(True, alpha=0.3)
        ax2.plot(wl, ri.k)
        ax2.set_xlabel("Wavelength [μm]")
        ax2.set_ylabel("k(λ)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        return (
            html.Img(
                src=f"data:image/png;base64,{img_base64}", style={"width": "100%"}
            ),
            "\n".join(info_lines),
        )



def app() -> dash.Dash:  # type: ignore[valid-type]
    """Build the standalone Raman Explorer Dash app.

    Returns
    -------
    dash.Dash — configured Dash application object.

    Notes
    -----
    Requires ``dash`` to be installed. Run with::

        if __name__ == "__main__":
            app().run(debug=True)
    """
    dash, dcc, html, Input, Output, State = _import_dash()
    dash_app = dash.Dash(__name__, title="Photonics Helper — Raman Explorer")
    dash_app.layout = build_layout(dcc=dcc, html=html)
    register_callbacks(
        dash_app, dcc=dcc, html=html, Input=Input, Output=Output, State=State
    )
    return dash_app  # type: ignore[no-any-return]
