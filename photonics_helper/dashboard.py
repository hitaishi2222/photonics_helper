"""Unified interactive dashboard: Raman explorer + GNLSE result viewer.

One Dash app, two top-level tabs:

* **Raman Explorer** — the full multi-layer Raman browser from
  :mod:`photonics_helper.raman.dashboard` (material selector, fR/τ sliders, pump
  wavelength, n/k database browser).
* **GNLSE Viewer** — a small form that propagates a pulse with
  :class:`~photonics_helper.gnlse.GNLSESolver` and renders the interactive
  four-panel spectral/temporal summary
  (:func:`~photonics_helper.gnlse.plot_spectral_temporal_summary`).

The Raman layout and callbacks are reused unchanged, so the standalone
``photonics_helper.raman.app()`` keeps working.

Requires the ``webapp`` extra (``pip install 'photonics-helper[webapp]'``) and
``plotting`` (Plotly) for the GNLSE tab. Run with::

    from photonics_helper.dashboard import app

    app().run(debug=True, port=8050)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dash

from .raman.dashboard import (
    build_layout as build_raman_layout,
)
from .raman.dashboard import (
    register_callbacks as register_raman_callbacks,
)

__all__ = ["app", "build_layout", "register_gnlse_callbacks"]


def _import_dash():
    """Import Dash lazily and return the pieces the app needs."""
    try:
        import dash
        from dash import Input, Output, State, dcc, html
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "Dash is required for the unified dashboard. "
            "Install it with: pip install 'photonics-helper[webapp]'"
        ) from exc
    return dash, dcc, html, Input, Output, State


def _gnlse_viewer(dcc, html):
    """Controls + output container for the GNLSE viewer tab."""
    field_style = {"display": "flex", "flexDirection": "column", "gap": "2px"}
    controls = html.Div(
        [
            html.H3("GNLSE Result Viewer", style={"marginBottom": "4px"}),
            html.P(
                "Propagate a pulse and inspect the spectral/temporal evolution. "
                "Defaults are a 50 fs sech pulse in a 10 cm highly-nonlinear fiber.",
                style={"color": "#666", "marginBottom": "12px", "fontSize": "13px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Wavelength (nm)"),
                            dcc.Input(id="g-wl", type="number", value=1550, step=10),
                        ],
                        style=field_style,
                    ),
                    html.Div(
                        [
                            html.Label("T₀ (fs)"),
                            dcc.Input(id="g-t0", type="number", value=50, step=5),
                        ],
                        style=field_style,
                    ),
                    html.Div(
                        [
                            html.Label("Peak power (W)"),
                            dcc.Input(id="g-p0", type="number", value=1000, step=100),
                        ],
                        style=field_style,
                    ),
                    html.Div(
                        [
                            html.Label("Length (m)"),
                            dcc.Input(id="g-length", type="number", value=0.1, step=0.01),
                        ],
                        style=field_style,
                    ),
                    html.Div(
                        [
                            html.Label("β₂ (ps²/km)"),
                            dcc.Input(id="g-beta2", type="number", value=-20, step=1),
                        ],
                        style=field_style,
                    ),
                    html.Div(
                        [
                            html.Label("γ (1/W/m)"),
                            dcc.Input(id="g-gamma", type="number", value=0.01, step=0.005),
                        ],
                        style=field_style,
                    ),
                    html.Div(
                        [
                            html.Label("Raman"),
                            dcc.Checklist(
                                id="g-raman",
                                options=[{"label": " include", "value": "on"}],
                                value=["on"],
                            ),
                        ],
                        style=field_style,
                    ),
                    html.Div(
                        [
                            html.Label("Steps"),
                            dcc.Input(id="g-steps", type="number", value=200, step=50),
                        ],
                        style=field_style,
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(120px, 1fr))",
                    "gap": "12px",
                    "marginBottom": "12px",
                },
            ),
            html.Button("Run GNLSE", id="g-run", n_clicks=0),
        ]
    )
    output = dcc.Loading(html.Div(id="g-output"), type="default")
    return html.Div([controls, html.Hr(), output])


def build_layout(*, dcc, html):
    """Top-level tabbed layout (Raman Explorer + GNLSE Viewer)."""
    raman_layout = build_raman_layout(dcc=dcc, html=html)
    return html.Div(
        [
            html.H2(
                "Photonics Helper — Dashboard",
                style={"textAlign": "center", "margin": "10px 0 4px"},
            ),
            dcc.Tabs(
                id="app-tabs",
                value="tab-raman",
                children=[
                    dcc.Tab(
                        label="Raman Explorer",
                        value="tab-raman",
                        children=[raman_layout],
                    ),
                    dcc.Tab(
                        label="GNLSE Viewer",
                        value="tab-gnlse",
                        children=[_gnlse_viewer(dcc, html)],
                    ),
                ],
            ),
        ]
    )


def _run_gnlse(
    wl_nm: float,
    t0_fs: float,
    p0_w: float,
    length_m: float,
    beta2_ps2_km: float,
    gamma: float,
    raman_on: bool,
    steps: int,
):
    """Propagate one pulse and return the Plotly summary figure."""
    import numpy as np

    from .base import Length, Time, Wavelength
    from .gnlse import (
        FiberProfile,
        GNLSESolver,
        plot_spectral_temporal_summary,
    )
    from .pulse import Envelope, TemporalGrid, Wave
    from .raman import RamanResponse, RamanSpec

    t0_s = float(t0_fs) * 1e-15
    grid = TemporalGrid(N=2**11, Tmax=Time(max(20 * t0_s, 4e-12), "s"))
    envelope = Envelope(
        shape="sech", peak_amplitude=np.sqrt(float(p0_w)), pulse_width=Time(t0_s, "s")
    )
    pulse = Wave(
        grid=grid,
        envelope=envelope,
        central_wavelength=Wavelength(float(wl_nm), "nm"),
    )
    raman_response = None
    if raman_on:
        spec = RamanSpec(
            name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18
        )
        raman_response = RamanResponse(spec=spec, grid=grid)
    fiber = FiberProfile.from_gamma(
        gamma=float(gamma),
        n2=2.6e-20,
        omega0=pulse.central_frequency,
        length=Length(float(length_m), "m"),
        raman_response=raman_response,
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([float(beta2_ps2_km) * 1e-3]),
        include_raman=bool(raman_on),
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(int(steps), nsaves=20)
    return plot_spectral_temporal_summary(solver, plotly=True)


def register_gnlse_callbacks(dash_app, *, dcc, html, Input, Output, State):
    """Register the GNLSE viewer callback on ``dash_app``."""

    @dash_app.callback(
        Output("g-output", "children"),
        Input("g-run", "n_clicks"),
        State("g-wl", "value"),
        State("g-t0", "value"),
        State("g-p0", "value"),
        State("g-length", "value"),
        State("g-beta2", "value"),
        State("g-gamma", "value"),
        State("g-raman", "value"),
        State("g-steps", "value"),
        prevent_initial_call=True,
    )
    def _update_gnlse(
        n_clicks, wl_nm, t0_fs, p0_w, length_m, beta2, gamma, raman, steps
    ):
        if not n_clicks:
            return html.P("Press “Run GNLSE”.")
        try:
            fig = _run_gnlse(
                wl_nm=wl_nm or 1550,
                t0_fs=t0_fs or 50,
                p0_w=p0_w or 1000,
                length_m=length_m or 0.1,
                beta2_ps2_km=beta2 if beta2 is not None else -20.0,
                gamma=gamma or 0.01,
                raman_on=bool(raman),
                steps=int(steps or 200),
            )
        except Exception as exc:  # surface solver/plotly errors in the UI
            return html.Div(
                f"GNLSE run failed: {exc}",
                style={"color": "#b00020", "padding": "8px"},
            )
        return dcc.Graph(figure=fig)


def app() -> dash.Dash:  # type: ignore[valid-type]
    """Build the unified dashboard Dash application.

    Returns
    -------
    dash.Dash — configured application with the Raman Explorer and GNLSE
    Viewer tabs, all callbacks registered.
    """
    dash, dcc, html, Input, Output, State = _import_dash()
    dash_app = dash.Dash(__name__, title="Photonics Helper — Dashboard")
    dash_app.layout = build_layout(dcc=dcc, html=html)
    register_raman_callbacks(
        dash_app, dcc=dcc, html=html, Input=Input, Output=Output, State=State
    )
    register_gnlse_callbacks(
        dash_app, dcc=dcc, html=html, Input=Input, Output=Output, State=State
    )
    return dash_app  # type: ignore[no-any-return]
