"""
Interactive Pulse Visualization Dashboard
==========================================

FastAPI + HTMX powered interactive dashboard for exploring pulse parameters.
Uses the Envelope class from photonics_helper.pulse for all computation.

Usage:
    python examples/pulse_visualization.py
"""

from __future__ import annotations
import asyncio
import webbrowser
from typing import Optional
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

from photonics_helper.base import Time
from photonics_helper.pulse import Envelope

# Create FastAPI app
app = FastAPI(title="Interactive Pulse Visualization")

# Default parameters
DEFAULT_SHAPE = "gaussian"
DEFAULT_T0 = 50  # fs
DEFAULT_CHIRP = 0.0
DEFAULT_AMP = 1.0


def parse_params(shape: str, t0: str, chirp: str, amplitude: str, theme: str = "light"):
    """Parse and validate parameters from form data."""
    try:
        t0_val = float(t0)
        chirp_val = float(chirp)
        amp_val = float(amplitude)
        
        # Validate ranges
        if not (10 <= t0_val <= 500):
            raise ValueError(f"T0 must be between 10 and 500, got {t0_val}")
        if not (-10 <= chirp_val <= 10):
            raise ValueError(f"Chirp must be between -10 and 10, got {chirp_val}")
        if not (0.01 <= amp_val <= 5):
            raise ValueError(f"Amplitude must be between 0.01 and 5, got {amp_val}")
        
        # Validate shape
        valid_shapes = ["gaussian", "sech", "lorentzian", "rectangular", 
                       "super-gaussian", "triangular", "parabolic", "cosine",
                       "exponential", "airy"]
        if shape not in valid_shapes:
            raise ValueError(f"Invalid shape: {shape}")
        
        return shape, t0_val, chirp_val, amp_val
    except ValueError as e:
        raise ValueError(str(e))


def _build_pulse_figure(params: PulseParameters) -> go.Figure:
    """Build a 2x2 subplot figure for the given pulse parameters."""
    # Create pulse
    pulse = Envelope(
        shape=params.shape,
        peak_amplitude=params.amplitude,
        pulse_width=Time(params.t0, "fs"),
        chirp=params.chirp,
    )
    
    # Use the existing visualize_2d method which we know works
    fig = pulse.visualize_2d(backend="plotly", theme=params.theme)
    
    # Update title
    fig.update_layout(title_text=f"{params.shape.capitalize()} Pulse")
    
    return fig


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    # Read and return the index template
    template_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(template_path.read_text())


@app.post("/pulse", response_class=HTMLResponse)
async def pulse_update(request: Request, 
                       shape: str = Form(default=DEFAULT_SHAPE),
                       t0: str = Form(default=str(DEFAULT_T0)),
                       chirp: str = Form(default=str(DEFAULT_CHIRP)),
                       amplitude: str = Form(default=str(DEFAULT_AMP)),
                       theme: str = Form(default="light")):
    """Update pulse figure based on slider parameters."""
    try:
        shape, t0_val, chirp_val, amp_val = parse_params(shape, t0, chirp, amplitude, theme)
    except ValueError as e:
        return HTMLResponse(f"<div style='color:red; padding:20px;'>Error: {e}</div>")
    
    # Create a temporary PulseParameters object for _build_pulse_figure
    from pydantic import BaseModel, Field
    class PulseParameters(BaseModel):
        shape: str = Field(default=DEFAULT_SHAPE)
        t0: float = Field(default=DEFAULT_T0, ge=10, le=500)
        chirp: float = Field(default=DEFAULT_CHIRP, ge=-10, le=10)
        amplitude: float = Field(default=DEFAULT_AMP, ge=0.01, le=5.0)
        theme: str = Field(default="light")
    
    params = PulseParameters(shape=shape, t0=t0_val, chirp=chirp_val, 
                            amplitude=amp_val, theme=theme)
    
    fig = _build_pulse_figure(params)
    fig_html = fig.to_html(full_html=False, include_plotlyjs=True)
    
    # Add parameter display
    param_text = (
        f"<div style='margin-bottom: 10px; font-family: monospace; font-size: 14px;'>"
        f"Shape: {params.shape} | "
        f"T₀: {params.t0} fs | "
        f"Chirp: {params.chirp:.2f} | "
        f"Amplitude: {params.amplitude:.2f}"
        f"</div>"
    )
    
    # Read and return the pulse update template
    template_path = Path(__file__).parent / "templates" / "pulse_update.html"
    template_content = template_path.read_text()
    
    # Replace placeholders with actual values
    template_content = template_content.replace("{{ param_text }}", param_text)
    template_content = template_content.replace("{{ fig_html }}", fig_html)
    template_content = template_content.replace("{{ params.t0 }}", str(params.t0))
    template_content = template_content.replace("{{ params.chirp }}", str(params.chirp))
    template_content = template_content.replace("{{ params.amplitude }}", str(params.amplitude))
    
    return HTMLResponse(template_content)


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    """Handle exceptions and return 422 with error details."""
    print(f"Exception: {exc}")
    from fastapi import status
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Find an available port
    port = 8000
    for candidate_port in [8000, 8001, 8002, 8003, 8004, 8005]:
        try:
            import socket as _socket
            _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            _sock.bind(("0.0.0.0", candidate_port))
            _sock.close()
            port = candidate_port
            break
        except OSError:
            continue
    
    url = f"http://localhost:{port}"
    print(f"Starting server on {url}")
    
    # Auto-open browser
    asyncio.get_event_loop().run_in_executor(
        None, lambda: webbrowser.open(url)
    )
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=port)
