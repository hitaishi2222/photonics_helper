"""Tests for the unified dashboard (:mod:`photonics_helper.dashboard`).

The Dash app is built (not served) and its layout/callbacks are inspected, so
these run headless and need only the ``webapp`` extra.
"""

import pytest

dash = pytest.importorskip("dash")

from photonics_helper.dashboard import (
    _run_gnlse,
    app,
    build_layout,
)


def test_app_builds_with_two_tabs():
    dash_app = app()
    assert isinstance(dash_app, dash.Dash)
    # Title + Tabs
    assert len(dash_app.layout.children) == 2
    tabs = dash_app.layout.children[1]
    labels = [tab.label for tab in tabs.children]
    assert labels == ["Raman Explorer", "GNLSE Viewer"]


def test_build_layout_contains_raman_and_gnlse():
    from dash import dcc, html

    layout = build_layout(dcc=dcc, html=html)
    tabs = layout.children[1]
    raman_tab, gnlse_tab = tabs.children
    # Raman controls reused from raman.dashboard
    raman_ids = _collect_ids(raman_tab)
    assert "material-selector" in raman_ids
    assert "layer-tabs" in raman_ids
    # GNLSE viewer controls
    gnlse_ids = _collect_ids(gnlse_tab)
    assert {"g-run", "g-wl", "g-output"} <= gnlse_ids


def test_callbacks_registered():
    dash_app = app()
    # Raman (main output) + GNLSE (g-output) callbacks, at least.
    assert len(dash_app.callback_map) >= 5
    assert any("g-output" in key for key in dash_app.callback_map)


def test_run_gnlse_returns_plotly_figure():
    pytest.importorskip("plotly")
    fig = _run_gnlse(
        wl_nm=1550,
        t0_fs=50,
        p0_w=1000,
        length_m=0.03,
        beta2_ps2_km=-20,
        gamma=0.01,
        raman_on=True,
        steps=20,
    )
    assert hasattr(fig, "data") and len(fig.data) > 0


def test_run_gnlse_without_raman():
    pytest.importorskip("plotly")
    fig = _run_gnlse(
        wl_nm=1064,
        t0_fs=80,
        p0_w=500,
        length_m=0.02,
        beta2_ps2_km=10,
        gamma=0.005,
        raman_on=False,
        steps=20,
    )
    assert hasattr(fig, "data") and len(fig.data) > 0


def test_dashboard_app_exported():
    import photonics_helper

    assert "dashboard_app" in photonics_helper.__all__
    assert callable(photonics_helper.dashboard_app)


# ── helpers ─────────────────────────────────────────────────────────────


def _collect_ids(component) -> set:
    """Recursively collect every ``id`` in a Dash component tree."""
    found: set = set()
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            found |= _collect_ids(child)
    return found


class _CaptureApp:
    """Minimal Dash stand-in that exposes the registered callback."""

    def __init__(self):
        self.fn = None

    def callback(self, *args, **kwargs):
        def register(fn):
            self.fn = fn
            return fn

        return register


def test_gnlse_viewer_honours_explicit_zero(monkeypatch):
    """gamma=0, p0=0, length=0 must reach the solver as 0 (audit finding 8)."""
    from dash import dcc, html

    from photonics_helper import dashboard

    captured = {}

    def fake_run_gnlse(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dashboard, "_run_gnlse", fake_run_gnlse)
    stub = _CaptureApp()
    dashboard.register_gnlse_callbacks(
        stub,
        dcc=dcc,
        html=html,
        Input=lambda *a, **k: None,
        Output=lambda *a, **k: None,
        State=lambda *a, **k: None,
    )
    stub.fn(1, 1550, 50, 0, 0, -20.0, 0, False, 200)
    assert captured["p0_w"] == 0
    assert captured["gamma"] == 0
    assert captured["length_m"] == 0


def test_gnlse_viewer_bounds_an_oversized_run(monkeypatch):
    from dash import dcc, html

    from photonics_helper import dashboard

    monkeypatch.setattr(
        dashboard, "_run_gnlse", lambda **kw: pytest.fail("must not run")
    )
    stub = _CaptureApp()
    dashboard.register_gnlse_callbacks(
        stub,
        dcc=dcc,
        html=html,
        Input=lambda *a, **k: None,
        Output=lambda *a, **k: None,
        State=lambda *a, **k: None,
    )
    out = stub.fn(1, 1550, 50, 1000, 0.1, -20.0, 0.01, False, 10**9)
    assert "between 1 and" in out.children
