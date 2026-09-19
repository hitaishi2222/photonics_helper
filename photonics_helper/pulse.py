"""Pulse envelopes, temporal grids, and pulse trains.

Envelope-field convention
-------------------------
An :class:`Envelope` describes a *normalized* complex envelope ``A(t)``. The
absolute scale of ``A`` is whatever ``peak_amplitude`` the user chose, so
``Wave.envelope_intensity`` (``|A|²``) and the metrics derived from it
(``pulse_energy``, ``peak_power``) are in normalized envelope units, **not**
W/m², joules, or watts, unless a physical effective mode area is attached with
:meth:`Wave.with_effective_area`. Without one, ``peak_power``/``pulse_energy``
emit a one-time :class:`UserWarning` and the visualization summary labels the
value as normalized. With an effective area ``A_eff`` the library uses
``P = ½·n·c·ε₀·A_eff·|A|²`` (see :class:`~photonics_helper.base.PeakPower`).
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Callable
from functools import cached_property, lru_cache
from math import acosh, log, pi, sqrt
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass
from scipy.special import airy
from scipy.special import hermite as hermite_poly

from photonics_helper.base import C_MS, EPS_0, Area, Frequency, Time, Wavelength
from photonics_helper.core.grids import TemporalGrid

logger = logging.getLogger(__name__)

SHAPE_FACTORS: dict[str, float] = {
    "gaussian": 2 * sqrt(log(2)),
    "sech": 2 * acosh(sqrt(2)),
    "lorentzian": 2 * sqrt(sqrt(2) - 1),
    "rectangular": 2.0,
}


@lru_cache(maxsize=1)
def _airy_fwhm_roots() -> tuple[float, float]:
    """Half-maximum crossings of the Airy-pulse main lobe |Ai(−x)|².

    Ai(−x) peaks at x ≈ 1.0188 (first zero of Ai′), not at x = 0, and the
    pulse is asymmetric, so the FWHM spans two distinct roots on either
    side of the peak. Returns ``(x_left, x_right)`` in units of T0.
    """
    from scipy.optimize import brentq

    x_peak = brentq(lambda x: airy(-x)[1], 0.5, 2.0)
    half = airy(-x_peak)[0] / sqrt(2.0)
    x_left = brentq(lambda x: airy(-x)[0] - half, 0.0, x_peak)
    x_right = brentq(lambda x: airy(-x)[0] - half, x_peak, 2.33)
    return x_left, x_right


def _crossing_width(t: NDArray, intensity: NDArray, level: float = 0.5) -> float:
    """Width between the outermost crossings of ``level · peak`` intensity.

    Crossings are located by linear interpolation between grid samples.
    Returns 0.0 when fewer than two crossings are found.
    """
    peak = np.max(intensity)
    if peak == 0:
        return 0.0

    target = intensity - level * peak
    # Find sign changes (crossings)
    sign_changes = np.where(np.diff(np.sign(target)))[0]
    if len(sign_changes) < 2:
        return 0.0

    # Linear interpolation at each crossing
    roots = []
    for idx in sign_changes:
        t0, t1 = t[idx], t[idx + 1]
        y0, y1 = target[idx], target[idx + 1]
        denom = y1 - y0
        if abs(denom) < 1e-30:
            continue
        roots.append(t0 - y0 * (t1 - t0) / denom)

    if len(roots) < 2:
        return 0.0

    return float(np.max(roots)) - float(np.min(roots))


@dataclass
class Envelope:
    shape: Literal[
        "gaussian",
        "sech",
        "lorentzian",
        "rectangular",
        "super-gaussian",
        "triangular",
        "parabolic",
        "cosine",
        "exponential",
        "gauss-hermite",
        "airy",
        "custom",
    ]
    peak_amplitude: float
    pulse_width: Time  # T0
    chirp: float = 0.0

    # Shape-specific extra parameters
    super_gaussian_order: int = 2  # for "super-gaussian"
    beam_waist: Time | None = None  # for "gauss-hermite"
    hg_mode: int = 0  # Hermite polynomial mode index m
    func: Callable | None = None  # for "custom": func(t, T0, A0)
    phase_func: Callable | None = None  # for "custom": phase_func(t, T0, chirp)

    @property
    def fwhm(self) -> Time:
        """Full width at half maximum of the envelope (s)."""
        T0 = self.pulse_width.as_s
        if self.shape == "super-gaussian":
            val = 2.0 * T0 * (log(2) / 2) ** (1.0 / (2 * self.super_gaussian_order))
        elif self.shape == "triangular":
            val = 2.0 * T0 * (1.0 - 1.0 / sqrt(2))
        elif self.shape == "cosine":
            val = T0
        elif self.shape == "exponential":
            # I(t) = A0²·exp(−2|t|/T0) → FWHM_intensity = T0·ln2
            val = T0 * log(2)
        elif self.shape == "airy":
            x_left, x_right = _airy_fwhm_roots()
            val = (x_right - x_left) * T0
        elif self.shape in SHAPE_FACTORS:
            val = SHAPE_FACTORS[self.shape] * T0
        else:
            val = T0
        return Time(val, "s")

    @classmethod
    def from_fwhm(
        cls,
        shape: Literal[
            "gaussian",
            "sech",
            "lorentzian",
            "rectangular",
            "super-gaussian",
            "triangular",
            "cosine",
            "exponential",
            "airy",
        ],
        peak_amplitude: float,
        fwhm: Time,
    ) -> Self:
        """Construct an Envelope from a desired full-width at half-maximum."""
        f = fwhm.as_s
        if shape == "super-gaussian":
            T0 = Time(
                f / (2.0 * (log(2) / 2) ** (1.0 / (2 * 2))), "s"
            )  # default order=2
            return cls(
                shape="super-gaussian",
                peak_amplitude=peak_amplitude,
                pulse_width=T0,
                super_gaussian_order=2,
            )
        elif shape == "triangular":
            T0 = Time(f / (2.0 * (1.0 - 1.0 / sqrt(2))), "s")
            return cls(
                shape="triangular", peak_amplitude=peak_amplitude, pulse_width=T0
            )
        elif shape == "cosine":
            T0 = Time(f, "s")  # FWHM = T0
            return cls(shape="cosine", peak_amplitude=peak_amplitude, pulse_width=T0)
        elif shape == "exponential":
            T0 = Time(f / log(2), "s")
            return cls(
                shape="exponential", peak_amplitude=peak_amplitude, pulse_width=T0
            )
        elif shape == "airy":
            x_left, x_right = _airy_fwhm_roots()
            T0 = Time(f / (x_right - x_left), "s")

            return cls(shape="airy", peak_amplitude=peak_amplitude, pulse_width=T0)
        else:
            T0 = Time(f / SHAPE_FACTORS[shape], "s")
            return cls(
                shape=shape,
                peak_amplitude=peak_amplitude,
                pulse_width=T0,
            )

    def field(self, t: NDArray) -> NDArray:
        """Returns complex envelope A(t)."""
        T0 = self.pulse_width.as_s
        A0 = self.peak_amplitude

        amp: Any  # dtype varies by shape (bool/int/float/complex)
        match self.shape:
            case "gaussian":
                amp = A0 * np.exp(-(t**2) / (2 * T0**2))
            case "sech":
                x = np.clip(t / T0, -700, 700)  # prevent cosh overflow
                amp = A0 / np.cosh(x)
            case "lorentzian":
                amp = A0 / (1 + (t / T0) ** 2)
            case "rectangular":
                amp = A0 * (np.abs(t) <= T0)
            case "super-gaussian":
                N = self.super_gaussian_order
                amp = A0 * np.exp(-(abs(t / T0) ** (2 * N)))
            case "triangular":
                amp = A0 * np.maximum(0.0, 1.0 - abs(t) / T0)
            case "parabolic":
                x = t / T0
                amp = A0 * np.where(abs(x) <= 1.0, 1.0 - x**2, 0.0)
            case "cosine":
                x = t / T0
                amp = A0 * np.where(abs(x) <= 1.0, np.cos(pi * x / 2), 0.0)
            case "exponential":
                amp = A0 * np.exp(-abs(t) / T0)
            case "gauss-hermite":
                w = self.beam_waist.as_s if self.beam_waist is not None else T0
                x = sqrt(2) * t / w
                H_m = hermite_poly(self.hg_mode)
                amp = A0 * H_m(x) * np.exp(-(x**2) / 2)
            case "airy":
                # Standard Airy pulse: Ai(-(t-t0)/T0), t0=0, accelerating towards +t
                amp = A0 * airy(-t / T0)[0]
            case "custom":
                if self.func is None:
                    raise ValueError("'custom' shape requires 'func' to be set")
                amp = self.func(t, T0, A0)
            case _:
                raise ValueError(f"Unknown shape: {self.shape}")

        # Normalise integer/bool dtypes to float64, keep complex dtypes intact
        amp = np.asarray(amp)
        if not np.issubdtype(amp.dtype, np.complexfloating):
            amp = np.asarray(amp, dtype=np.result_type(amp.dtype, np.float64))

        # Apply phase
        if self.shape == "parabolic":
            # Parabolic chirp: phase = chirp * (t/T0)^2, active only where amp != 0
            x = t / T0
            phase = np.where(abs(x) <= 1.0, self.chirp * x**2, 0.0)
        elif self.shape == "custom":
            if self.phase_func is not None:
                phase = self.phase_func(t, T0, self.chirp)
            else:
                phase = 0.5 * self.chirp * (t / T0) ** 2
        else:
            phase = 0.5 * self.chirp * (t / T0) ** 2

        return np.asarray(amp * np.exp(1j * phase))

    def intensity(self, t: NDArray) -> NDArray:
        """Optical intensity |A|² (W)."""
        A = self.field(t)
        return np.asarray(np.abs(A) ** 2)

    def calc_width(self, level: float = 0.5) -> Time:
        """Calculate the pulse width using linear interpolation at crossing points.

        Finds the width between the widest pair of crossings where the
        intensity envelope drops to ``level * peak_intensity``.

        Parameters
        ----------
        level : float — fraction of peak to calculate width at.
            0.5 gives FWHM, 1/e ≈ 0.368, 1/e² ≈ 0.135. Default 0.5.

        Returns
        -------
        width : Time — pulse width in seconds (use ``.as_ps`` etc. for display units).
        """
        grid = self._make_grid()
        return Time(_crossing_width(grid.t, self.intensity(grid.t), level), "s")

    def apply_dispersion(
        self,
        GDD: float = 0.0,
        TOD: float = 0.0,
        FOD: float = 0.0,
        N: int = 2**12,
    ) -> Envelope:
        """Apply group-delay dispersion (GDD), TOD, FOD in the frequency domain.

        Multiplies the spectral amplitude by ``exp(−i·φ(ω))`` where
        ``φ(ω) = ½·GDD·Ω² + ⅙·TOD·Ω³ + ¹⁄₂₄·FOD·Ω⁴`` and ``Ω`` is the
        offset from the central angular frequency.  Uses the same sign as
        :meth:`~photonics_helper.gnlse.SplitStepEngine._linear_step`.

        Parameters
        ----------
        GDD : float — group-delay dispersion (ps²). Default 0.
        TOD : float — third-order dispersion (ps³). Default 0.
        FOD : float — fourth-order dispersion (ps⁴). Default 0.
        N : int — number of points for the internal grid (default 2¹²).

        Returns
        -------
        Envelope — a new Envelope with the dispersion-applied field.

        Notes
        -----
        The returned envelope has ``shape="custom"`` and owns the numerical
        field computed on an internal grid sized to contain the broadened
        pulse (estimated from the input bandwidth and the requested GDD,
        TOD, FOD; ``N`` controls the sampling resolution). Requests outside
        that window are clamped to the edge values. The analytic parameters
        of the input envelope no longer describe the broadened pulse, so
        ``peak_amplitude`` and ``pulse_width`` are re-measured numerically
        and the chirp is reset to zero (any input chirp is already baked
        into the dispersed field).
        """
        # Size the computation window so it can actually hold the broadened
        # pulse: estimate the temporal and spectral RMS widths of the input
        # on a provisional grid, then the group-delay spread implied by the
        # requested dispersion orders.
        T0 = self.pulse_width.as_s
        prov = TemporalGrid(N=N, Tmax=Time(10.0 * T0, "s"))
        A_prov = self.field(prov.t)
        I_prov = np.abs(A_prov) ** 2
        E_t = float(np.sum(I_prov))
        dt_rms = (
            sqrt(np.sum((prov.t - np.sum(prov.t * I_prov) / E_t) ** 2 * I_prov) / E_t)
            if E_t > 0.0
            else T0
        )
        S_prov = np.abs(prov.fft(A_prov)) ** 2
        E_s = float(np.sum(S_prov))
        w_ps = prov.w * 1e-12  # rad/s → rad/ps
        dw_rms = (
            sqrt(np.sum((w_ps - np.sum(w_ps * S_prov) / E_s) ** 2 * S_prov) / E_s)
            if E_s > 0.0
            else 0.0
        )
        # Characteristic group-delay spread (ps): τ ≈ GDD·Δω, ½·TOD·Δω², ⅙·FOD·Δω³
        delay_ps = (
            abs(GDD) * dw_rms
            + 0.5 * abs(TOD) * dw_rms**2
            + (1.0 / 6.0) * abs(FOD) * dw_rms**3
        )
        delay_s = delay_ps * 1e-12  # ps → s

        if GDD == 0.0 and TOD == 0.0 and FOD == 0.0:
            # Identity transform: keep the input field without an FFT round trip
            grid = prov
            disp_field = A_prov.copy()
        else:
            # Build a grid wide enough to hold the broadened pulse, growing
            # N if needed so the input pulse stays resolved (dt ≲ T0/8).
            # Capped at 2²² points to bound memory; extremely large
            # dispersion ratios may require passing a larger N explicitly.
            half_span = 8.0 * sqrt(dt_rms**2 + delay_s**2) + 5.0 * T0
            n_needed = int(np.ceil((2.0 * half_span) / (T0 / 8.0)))
            if n_needed > 1:
                n_fft = max(N, 1 << (n_needed - 1).bit_length())
            else:
                n_fft = N
            n_fft = min(n_fft, 2**22)
            grid = TemporalGrid(N=n_fft, Tmax=Time(2.0 * half_span, "s"))
            t = grid.t

            # Original field
            A_t = self.field(t)

            # FFT to frequency domain
            A_w = grid.fft(A_t)

            # Build dispersion phase: φ(Ω) = ½·GDD·Ω² + ⅙·TOD·Ω³ + ¹⁄₂₄·FOD·Ω⁴
            # GDD, TOD, FOD are in ps², ps³, ps⁴. Convert omega from rad/s to rad/ps.
            omega_ps = grid.w * 1e-12  # rad/s → rad/ps

            phase = np.zeros_like(omega_ps, dtype=float)
            if GDD != 0.0:
                phase += 0.5 * GDD * omega_ps**2
            if TOD != 0.0:
                phase += (1.0 / 6.0) * TOD * omega_ps**3
            if FOD != 0.0:
                phase += (1.0 / 24.0) * FOD * omega_ps**4

            # Apply dispersion phase (same sign convention as GNLSE linear step)
            A_w_disp = A_w * np.exp(-1j * phase)

            # IFFT back to time domain
            disp_field = grid.ifft(A_w_disp)

        # Create a new Envelope that holds the modified field. The shape must
        # be "custom" so that Envelope.field() actually consults ``func``;
        # for any built-in shape the analytic formula would silently discard
        # the dispersed field. The callable interpolates onto whatever time
        # axis the caller requests.
        def _disp_field(t, _T0, _A0):
            t = np.asarray(t, dtype=float)
            real = np.interp(t, grid.t, disp_field.real)
            imag = np.interp(t, grid.t, disp_field.imag)
            return real + 1j * imag

        measured_width = _crossing_width(grid.t, np.abs(disp_field) ** 2)

        new_env = Envelope(
            shape="custom",
            peak_amplitude=float(np.max(np.abs(disp_field))),
            pulse_width=Time(measured_width, "s")
            if measured_width > 0.0
            else self.pulse_width,
            chirp=0.0,
            func=_disp_field,
            phase_func=None,
        )
        return new_env

    def _make_grid(self, N: int = 2**12) -> TemporalGrid:
        """Create a TemporalGrid sized for this envelope."""
        # Window must cover ~10x pulse width for tails to decay
        Tmax = 10.0 * self.pulse_width.as_s
        return TemporalGrid(N=N, Tmax=Time(Tmax, "s"))

    def visualize_2d(
        self,
        backend: Literal["plotly", "matplotlib"] = "plotly",
        N: int = 2**12,
        show_phase: bool = True,
        show_fwhm: bool = True,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        theme: Literal["light", "dark"] = "light",
    ):
        """Plot temporal intensity, spectral intensity, and phase.

        Parameters
        ----------
        backend : "plotly" or "matplotlib" (default "plotly")
        N : number of time points (default 2^12)
        show_phase : show instantaneous phase overlay (default True)
        show_fwhm : show FWHM markers (default True)
        figsize : figure size for matplotlib backend (default None)
        title : optional title override (default uses shape name)
        theme : "light" or "dark" (default "light")
        """
        if backend not in ("plotly", "matplotlib"):
            raise ValueError("backend must be 'plotly' or 'matplotlib'")

        grid = self._make_grid(N)
        t = grid.t
        A = self.field(t)
        intensity_t = np.abs(A) ** 2
        spectral = grid.fft(A)
        intensity_w = np.abs(spectral) ** 2
        phase = np.unwrap(np.angle(A))

        if title is None:
            title = f"{self.shape.title()} Pulse Envelope (T₀={self.pulse_width.as_s * 1e15:.1f} fs)"

        if theme not in ("light", "dark"):
            raise ValueError("theme must be 'light' or 'dark'")

        if backend == "plotly":
            return self._visualize_2d_plotly(
                t,
                A,
                intensity_t,
                intensity_w,
                phase,
                grid,
                show_phase,
                show_fwhm,
                title,
                theme,
            )
        else:
            return self._visualize_2d_matplotlib(
                t,
                A,
                intensity_t,
                intensity_w,
                phase,
                grid,
                show_phase,
                show_fwhm,
                figsize,
                title,
                theme,
            )

    def _visualize_2d_plotly(
        self,
        t,
        A,
        intensity_t,
        intensity_w,
        phase,
        grid,
        show_phase,
        show_fwhm,
        title,
        theme,
    ):
        """Plotly implementation of 2D visualization."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Temporal Intensity",
                "Spectral Intensity",
                "Phase",
                "Polar Plot",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
            shared_xaxes=False,
        )

        # Temporal intensity
        fig.add_trace(
            go.Scatter(
                x=t,
                y=intensity_t,
                mode="lines",
                name="Intensity",
                line=dict(color="#00d4ff"),
            ),
            row=1,
            col=1,
        )
        if show_fwhm:
            fwhm_val = self.fwhm.as_s
            fig.add_vrect(
                x0=-fwhm_val / 2,
                x1=fwhm_val / 2,
                fillcolor="orange",
                opacity=0.1,
                line_width=0,
            )
            fig.add_hline(
                y=np.max(intensity_t) / 2,
                line_dash="dot",
                line_color="orange",
                opacity=0.5,
            )
            # Pulse width markers at ±T₀
            fig.add_vline(
                x=-self.pulse_width.as_s,
                line_dash="dot",
                line_color="#fbbf24",
                opacity=0.7,
                row=1,
                col=1,
            )
            fig.add_vline(
                x=self.pulse_width.as_s,
                line_dash="dot",
                line_color="#fbbf24",
                opacity=0.7,
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=self.pulse_width.as_s,
                y=0,
                text="T₀",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#fbbf24",
                ax=20,
                ay=-30,
                font=dict(size=12, color="#fbbf24"),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=-self.pulse_width.as_s,
                y=0,
                text="T₀",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#fbbf24",
                ax=-20,
                ay=-30,
                font=dict(size=12, color="#fbbf24"),
                row=1,
                col=1,
            )
        # Parameters text box
        params_text = (
            f"Shape: {self.shape}<br>"
            f"T₀ = {self.pulse_width.as_s:.3g} s ({self.pulse_width.as_s * 1e15:.1f} fs)<br>"
            f"FWHM = {self.fwhm.as_s:.3g} s ({self.fwhm.as_s * 1e15:.1f} fs)<br>"
            f"Chirp = {self.chirp:.2f}"
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=params_text,
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            valign="top",
            bgcolor="rgba(255, 235, 180, 0.85)",
            bordercolor="#d4a017",
            borderwidth=1.5,
            borderpad=6,
            xanchor="left",
            yanchor="top",
        )

        # Spectral intensity
        w = grid.w
        fig.add_trace(
            go.Scatter(
                x=w,
                y=intensity_w,
                mode="lines",
                name="Spectrum",
                line=dict(color="#a78bfa"),
            ),
            row=1,
            col=2,
        )

        # Phase
        if show_phase:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=phase,
                    mode="lines",
                    name="Phase",
                    line=dict(color="#ff6b6b", dash="dot"),
                ),
                row=2,
                col=1,
            )

        # Polar plot (Re vs Im)
        fig.add_trace(
            go.Scatter(
                x=np.real(A),
                y=np.imag(A),
                mode="lines",
                name="Polar",
                line=dict(color="#34d399"),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=title,
            height=700,
            showlegend=False,
            template="plotly_white" if theme == "light" else "plotly_dark",
        )
        return fig

    def _visualize_2d_matplotlib(
        self,
        t,
        A,
        intensity_t,
        intensity_w,
        phase,
        grid,
        show_phase,
        show_fwhm,
        figsize,
        title,
        theme,
    ):
        """Matplotlib implementation of 2D visualization."""
        import matplotlib.pyplot as plt

        if theme == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 8))
        fig.suptitle(title, fontsize=12, fontweight="bold")

        # Temporal intensity
        ax_t = axes[0, 0]
        ax_t.plot(t, intensity_t, color="#00d4ff", linewidth=1.5)
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Intensity")
        ax_t.set_title("Temporal Profile")
        if show_fwhm:
            fwhm_val = self.fwhm.as_s
            ax_t.axvspan(-fwhm_val / 2, fwhm_val / 2, alpha=0.1, color="orange")
            ax_t.axhline(
                y=np.max(intensity_t) / 2, color="orange", linestyle="--", alpha=0.5
            )
            # Pulse width markers at ±T₀
            ax_t.axvline(
                x=-self.pulse_width.as_s,
                color="#fbbf24",
                linewidth=1.0,
                linestyle="-.",
                alpha=0.7,
            )
            ax_t.axvline(
                x=self.pulse_width.as_s,
                color="#fbbf24",
                linewidth=1.0,
                linestyle="-.",
                alpha=0.7,
            )
            ax_t.annotate(
                "T₀",
                xy=(self.pulse_width.as_s, 0),
                xytext=(self.pulse_width.as_s, 0.15),
                color="#fbbf24",
                fontsize=10,
                fontweight="bold",
                ha="left",
                arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.0),
            )
            ax_t.annotate(
                "T₀",
                xy=(-self.pulse_width.as_s, 0),
                xytext=(-self.pulse_width.as_s, 0.15),
                color="#fbbf24",
                fontsize=10,
                fontweight="bold",
                ha="right",
            )
        # Parameters text box
        params_text = (
            f"Shape: {self.shape}\n"
            f"T₀ = {self.pulse_width.as_s:.3g} s ({self.pulse_width.as_s * 1e15:.1f} fs)\n"
            f"FWHM = {self.fwhm.as_s:.3g} s ({self.fwhm.as_s * 1e15:.1f} fs)\n"
            f"Chirp = {self.chirp:.2f}"
        )
        ax_t.text(
            0.02,
            0.98,
            params_text,
            transform=ax_t.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85),
        )

        # Spectral intensity
        w = grid.w
        ax_w = axes[0, 1]
        ax_w.plot(w, intensity_w, color="#a78bfa", linewidth=1.5)
        ax_w.set_xlabel("Angular frequency (rad/s)")
        ax_w.set_ylabel("Intensity")
        ax_w.set_title("Spectral Profile")

        # Phase
        ax_ph = axes[1, 0]
        if show_phase:
            ax_ph.plot(t, phase, color="#ff6b6b", linewidth=1.0, linestyle="--")
            ax_ph.set_ylabel("Phase (rad)")
        ax_ph.set_xlabel("Time (s)")
        ax_ph.set_title("Instantaneous Phase")

        # Polar plot
        ax_pol = axes[1, 1]
        ax_pol.plot(np.real(A), np.imag(A), color="#34d399", linewidth=1.0)
        ax_pol.set_xlabel("Re(A)")
        ax_pol.set_ylabel("Im(A)")
        ax_pol.set_title("Polar Plot")
        ax_pol.axis("equal")

        plt.tight_layout()
        return fig

    def visualize_3d(
        self,
        N: int = 2**12,
        title: str | None = None,
        theme: Literal["light", "dark"] = "light",
    ):
        """Plot 3D spectrogram (time vs frequency vs intensity).

        Shows how the frequency content evolves over time.
        Useful for visualizing chirp and time-frequency structure.

        Parameters
        ----------
        N : number of time points (default 2^12)
        title : optional title override
        theme : "light" or "dark" (default "light")
        """
        import plotly.graph_objects as go
        from scipy.signal import spectrogram as scipy_spectrogram

        grid = self._make_grid(N)
        t = grid.t
        A = self.field(t)

        # Compute spectrogram using scipy
        # Window size ~1/10 of pulse width for good time-frequency resolution
        win_size = max(32, int(self.pulse_width.as_s / grid.dt / 10))
        win_size = min(win_size, N // 4)
        win_size = win_size if win_size % 2 == 0 else win_size + 1

        f_sg, t_sg, Sxx = scipy_spectrogram(
            A,
            fs=1.0 / grid.dt,
            window="hann",
            nperseg=win_size,
            noverlap=win_size * 3 // 4,
            mode="magnitude",
        )

        # Frequency in rad/s
        f_sg_rad = f_sg * 2 * np.pi

        if title is None:
            title = f"3D Spectrogram  [{self.shape}]"

        # 3D surface: Time vs Frequency vs Power
        fig = go.Figure(
            data=[
                go.Surface(
                    x=t_sg,
                    y=f_sg_rad,
                    z=Sxx**2,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Power"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Time (s)",
                yaxis_title="Angular frequency (rad/s)",
                zaxis_title="Power",
            ),
            template="plotly_white" if theme == "light" else "plotly_dark",
        )
        return fig

    @classmethod
    def from_parabolic_asymptotic(
        cls,
        peak_amplitude: float,
        pulse_width: Time,
        gain: float,
        length: float,
        chirp: float = 0.0,
    ) -> Self:
        """Construct a parabolic pulse with asymptotic amplifier chirp.

        The chirp coefficient follows the asymptotic parabolic solution:
        ``α ≈ 0.2726 × z × gain``.

        Parameters
        ----------
        peak_amplitude : A₀
        pulse_width : T₀
        gain : Small-signal gain coefficient (unitless or per-length as used in the amplifier model)
        length : Propagation length in the amplifier
        chirp : Additional chirp on top of the asymptotic value (default 0)
        """
        alpha_asym = 0.2726 * length * gain + chirp
        logger.info(
            "Using asymptotic parabolic chirp: α ≈ 0.2726 × z × gain = %.4f",
            alpha_asym,
        )
        return cls(
            shape="parabolic",
            peak_amplitude=peak_amplitude,
            pulse_width=pulse_width,
            chirp=alpha_asym,
        )


@dataclass
class Wave:
    grid: TemporalGrid
    envelope: Envelope

    central_wavelength: Wavelength
    _pulse_train_field: Any = None
    refractive_index: float = 1.0
    repetition_rate: Frequency | None = None
    _effective_area: Area | None = None
    _warned_normalized: bool = False

    @cached_property
    def central_frequency(self) -> float:
        """Carrier angular frequency at the central wavelength (rad/s)."""
        return self.central_wavelength.to_omega().as_rad_s

    @property
    def wavelength_nm(self) -> NDArray:
        """Absolute wavelength grid (nm) for each frequency sample on ``grid.w``."""
        omega_abs = self.central_frequency + self.grid.w
        return np.asarray(2 * np.pi * C_MS / omega_abs * 1e9)

    @property
    def electric_field(self):
        """Physical electric field (V/m) reconstructed from the envelope."""
        A = self.envelope_field
        return np.real(A * np.exp(-1j * self.central_frequency * self.grid.t))

    def instantaneous_intensity(self):
        """Instantaneous intensity in physical units (W)."""
        E = self.electric_field
        return np.abs(E) ** 2

    @property
    def envelope_intensity(self):
        """Intensity of the envelope |A|² (W)."""
        return np.abs(self.envelope_field) ** 2

    @property
    def _power_scale(self) -> float | None:
        """Intensity→power factor ``½·n·c·ε₀·A_eff`` (W per unit ``|A|²``).

        Returns ``None`` when no effective mode area has been attached, in
        which case the power and energy metrics stay in normalized envelope
        units.
        """
        if self._effective_area is None:
            return None
        return (
            0.5 * self.refractive_index * C_MS * EPS_0 * self._effective_area.as_m2
        )

    def _warn_normalized(self, method: str) -> None:
        """Emit a one-time warning that ``method`` is in normalized units."""
        if self._warned_normalized:
            return
        self._warned_normalized = True
        warnings.warn(
            f"Wave.{method}() is in normalized envelope units, not physical "
            "watts/joules. Attach an effective mode area with "
            "Wave.with_effective_area(A_eff) to obtain physical units.",
            UserWarning,
            stacklevel=3,
        )

    def pulse_energy(self) -> float:
        """Pulse energy (or normalized integral ``∫|A|²dt``).

        Returns
        -------
        float
            Joules when an effective mode area has been attached with
            :meth:`with_effective_area`; otherwise the normalized integral
            ``∫|A|²dt`` in field-units²·seconds and a one-time
            :class:`UserWarning` is emitted.
        """
        energy = float(np.sum(self.envelope_intensity) * self.grid.dt)
        scale = self._power_scale
        if scale is None:
            self._warn_normalized("pulse_energy")
            return energy
        return scale * energy

    def calc_width(self, level: float = 0.5) -> Time:
        """Calculate the pulse width using linear interpolation at crossing points.

        Finds the width between the widest pair of crossings where the
        intensity envelope drops to ``level * peak_intensity``.

        Parameters
        ----------
        level : float — fraction of peak to calculate width at.
            0.5 gives FWHM, 1/e ≈ 0.368, 1/e² ≈ 0.135. Default 0.5.

        Returns
        -------
        width : Time — pulse width in seconds (use ``.as_ps`` etc. for display units).
        """
        return Time(_crossing_width(self.grid.t, self.envelope_intensity, level), "s")

    def peak_power(self) -> float:
        """Peak power (or normalized peak intensity ``max|A|²``).

        Returns
        -------
        float
            Watts when an effective mode area has been attached with
            :meth:`with_effective_area` (using
            ``P = ½·n·c·ε₀·A_eff·|A|²``); otherwise ``max|A|²`` in normalized
            envelope units and a one-time :class:`UserWarning` is emitted.
        """
        A = self.envelope_field
        peak_intensity = float(np.max(np.abs(A) ** 2))
        scale = self._power_scale
        if scale is None:
            self._warn_normalized("peak_power")
            return peak_intensity
        return scale * peak_intensity

    def with_effective_area(self, A_eff: Area) -> Self:
        """Return a copy of this ``Wave`` bound to a physical mode area.

        Once an effective area is attached, :meth:`peak_power` and
        :meth:`pulse_energy` return physical watts and joules via the
        intensity relation ``P = ½·n·c·ε₀·A_eff·|A|²`` (with
        :attr:`refractive_index` as ``n``). Without one they stay in
        normalized envelope units.

        Parameters
        ----------
        A_eff : Area — effective mode area (e.g. ``Area(80, "um^2")``).

        Returns
        -------
        Wave — a copy sharing this wave's field with the area attached.
        """
        scaled = copy.copy(self)
        scaled._effective_area = A_eff
        return scaled

    def average_power(self, repetition_rate: Frequency) -> float:
        """Average power = pulse energy × repetition_rate.

        Parameters
        ----------
        repetition_rate : Hz
        """
        return float(self.pulse_energy() * repetition_rate.as_Hz)

    @classmethod
    def from_pulse_train(
        cls,
        envelope: Envelope,
        central_wavelength: Wavelength,
        grid: TemporalGrid,
        repetition_rate: Frequency,
        n_pulses: int = 10,
        refractive_index: float = 1.0,
    ) -> Self:
        """Construct a pulse train Wave from a single-envelope shape.

        Parameters
        ----------
        envelope : The single-pulse envelope shape to repeat
        central_wavelength : Central wavelength of the carrier
        grid : TemporalGrid covering the full window (all pulses + padding)
        repetition_rate : Hz — spacing between consecutive pulses
        n_pulses : number of pulses (default 10)
        refractive_index : background refractive index (default 1.0)
        """
        # Build the multi-pulse envelope field, centered at t=0
        full_field = np.zeros_like(grid.t, dtype=complex)
        spacing = 1.0 / repetition_rate.as_Hz
        for k in range(n_pulses):
            t_centered = grid.t - (k - (n_pulses - 1) / 2) * spacing
            full_field += envelope.field(t_centered)

        # Create a temporary "virtual" envelope that stores the train flag
        train_env = Envelope(
            shape=envelope.shape,
            peak_amplitude=envelope.peak_amplitude,
            pulse_width=envelope.pulse_width,
            chirp=envelope.chirp,
            super_gaussian_order=envelope.super_gaussian_order,
            beam_waist=envelope.beam_waist,
            hg_mode=envelope.hg_mode,
            func=envelope.func,
            phase_func=envelope.phase_func,
        )

        wave = cls(
            grid=grid,
            envelope=train_env,
            central_wavelength=central_wavelength,
            refractive_index=refractive_index,
            repetition_rate=repetition_rate,
        )
        return wave.with_field(full_field)

    def with_field(self, field: NDArray) -> Self:
        """Return this ``Wave`` with an explicit envelope-field override.

        Used by :meth:`from_pulse_train` for precomputed multi-pulse fields.
        """
        self._pulse_train_field = np.asarray(field)
        return self

    @property
    def envelope_field(self):
        """Complex envelope field (√W)."""
        if hasattr(self, "_pulse_train_field") and self._pulse_train_field is not None:
            return self._pulse_train_field
        return self.envelope.field(self.grid.t)

    @cached_property
    def spectrum(self) -> NDArray:
        """Frequency-domain envelope obtained via the configured FFT backend."""
        return np.asarray(self.grid.fft(self.envelope_field))

    def time_bandwidth_product(self) -> float:
        """RMS time-bandwidth product. ≈0.707 for transform-limited Gaussian."""
        intensity = self.envelope_intensity
        spectral_intensity = np.abs(self.spectrum) ** 2

        t, w = self.grid.t, self.grid.w
        dt, dw = self.grid.dt, self.grid.dw

        E_t = self.pulse_energy()
        t_mean = np.sum(t * intensity) * dt / E_t
        rms_t = sqrt(np.sum((t - t_mean) ** 2 * intensity) * dt / E_t)

        E_w = np.sum(spectral_intensity) * dw
        w_mean = np.sum(w * spectral_intensity) * dw / E_w
        rms_w = sqrt(np.sum((w - w_mean) ** 2 * spectral_intensity) * dw / E_w)

        return rms_t * rms_w

    def visualize(
        self,
        t_unit: str = "s",
        w_unit: str = "rad/s",
        t_scale: float = 1.0,
        w_scale: float = 1.0,
        show_electric_field: bool = False,
        show_phase: bool = True,
        show_spectrogram: bool = False,
        figsize: tuple[float, float] | None = None,
        save_path: str | None = None,
    ):
        """
        Plot a comprehensive overview of the pulse.

        Parameters
        ----------
        t_unit : label for the time axis (e.g. "ps", "fs")
        w_unit : label for the frequency axis (e.g. "THz", "rad/ps")
        t_scale : multiply grid.t by this before plotting (e.g. 1e12 for ps)
        w_scale : multiply grid.w by this before plotting
        show_electric_field : add a panel with the real electric field
        show_phase : overlay instantaneous phase on the temporal panel
        show_spectrogram : add a spectrogram (STFT) panel
        figsize : passed to plt.figure()
        save_path : if given, save figure to this path
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        t = self.grid.t * t_scale
        w = self.grid.w * w_scale
        intensity = self.envelope_intensity
        spectral_intensity = np.abs(self.spectrum) ** 2

        # Normalise for clean plotting
        norm_t = intensity / np.max(intensity)
        norm_w = spectral_intensity / np.max(spectral_intensity)

        # Instantaneous phase (unwrapped)
        phase = np.unwrap(np.angle(self.envelope_field))

        # ---- figure layout ----------------------------------------
        n_rows = 2
        if show_electric_field:
            n_rows += 1
        if show_spectrogram:
            n_rows += 1

        fig = plt.figure(
            figsize=figsize or (10, 3.5 * n_rows),
            facecolor="#0f0f0f",
        )
        gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.45, left=0.1, right=0.95)

        COLORS = {
            "intensity": "#00d4ff",
            "phase": "#ff6b6b",
            "spectrum": "#a78bfa",
            "efield": "#34d399",
            "grid": "#2a2a2a",
            "text": "#e0e0e0",
            "fwhm": "#fbbf24",
        }

        def _style_ax(ax, xlabel, ylabel, title):
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors=COLORS["text"], labelsize=9)
            ax.xaxis.label.set_color(COLORS["text"])
            ax.yaxis.label.set_color(COLORS["text"])
            ax.title.set_color(COLORS["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
            ax.grid(True, color=COLORS["grid"], linewidth=0.5, linestyle="--")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10, pad=6, fontweight="bold")

        row = 0

        # ---- 1. Temporal intensity (+ optional phase) ---------------
        ax_t = fig.add_subplot(gs[row])
        row += 1

        ax_t.fill_between(t, norm_t, alpha=0.25, color=COLORS["intensity"])
        ax_t.plot(
            t, norm_t, color=COLORS["intensity"], linewidth=1.8, label="Intensity"
        )

        # FWHM marker
        fwhm_val = self.envelope.fwhm.as_s * t_scale
        ax_t.axvspan(-fwhm_val / 2, fwhm_val / 2, alpha=0.08, color=COLORS["fwhm"])
        ax_t.axhline(0.5, color=COLORS["fwhm"], linewidth=0.8, linestyle=":")
        ax_t.annotate(
            f"FWHM = {fwhm_val:.3g} {t_unit}",
            xy=(fwhm_val / 2, 0.5),
            xytext=(fwhm_val / 2 + (t[-1] - t[0]) * 0.03, 0.55),
            color=COLORS["fwhm"],
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=COLORS["fwhm"], lw=0.8),
        )

        if show_phase:
            ax_ph = ax_t.twinx()
            ax_ph.plot(
                t,
                phase,
                color=COLORS["phase"],
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label="Phase",
            )
            ax_ph.set_ylabel("Phase (rad)", color=COLORS["phase"], fontsize=9)
            ax_ph.tick_params(colors=COLORS["phase"], labelsize=9)
            ax_ph.spines["right"].set_edgecolor(COLORS["phase"])

        _style_ax(
            ax_t,
            f"Time ({t_unit})",
            "Normalised Intensity",
            f"Temporal Profile  [{self.envelope.shape}]",
        )

        tbp = self.time_bandwidth_product()
        ax_t.text(
            0.02,
            0.92,
            f"TBP = {tbp:.3f}   |   Peak power = {self.peak_power():.3g} "
            + ("W" if self._effective_area is not None else "(normalized units)"),
            transform=ax_t.transAxes,
            color=COLORS["text"],
            fontsize=8,
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333", boxstyle="round,pad=0.3"),
        )

        # ---- 2. Spectral intensity -----------------------------------
        ax_w = fig.add_subplot(gs[row])
        row += 1

        ax_w.fill_between(w, norm_w, alpha=0.25, color=COLORS["spectrum"])
        ax_w.plot(w, norm_w, color=COLORS["spectrum"], linewidth=1.8)

        # Spectral FWHM (numerical)
        half_max_mask = norm_w >= 0.5
        if half_max_mask.any():
            w_fwhm = w[half_max_mask]
            spec_fwhm = w_fwhm[-1] - w_fwhm[0]
            ax_w.axvspan(w_fwhm[0], w_fwhm[-1], alpha=0.08, color=COLORS["fwhm"])
            ax_w.text(
                0.02,
                0.92,
                f"Spectral FWHM ≈ {spec_fwhm:.3g} {w_unit}",
                transform=ax_w.transAxes,
                color=COLORS["text"],
                fontsize=8,
                bbox=dict(
                    facecolor="#1a1a1a", edgecolor="#333", boxstyle="round,pad=0.3"
                ),
            )

        _style_ax(
            ax_w, f"Angular frequency ({w_unit})", "Normalised PSD", "Spectral Profile"
        )

        # ---- 3. Electric field (optional) ---------------------------
        if show_electric_field:
            ax_e = fig.add_subplot(gs[row])
            row += 1
            E = self.electric_field
            E_norm = E / np.max(np.abs(E))
            ax_e.plot(t, E_norm, color=COLORS["efield"], linewidth=0.8, alpha=0.9)
            ax_e.plot(
                t,
                norm_t,
                color=COLORS["intensity"],
                linewidth=1.2,
                linestyle="--",
                alpha=0.6,
                label="Envelope",
            )
            ax_e.plot(
                t,
                -norm_t,
                color=COLORS["intensity"],
                linewidth=1.2,
                linestyle="--",
                alpha=0.6,
            )
            _style_ax(ax_e, f"Time ({t_unit})", "Normalised E-field", "Electric Field")

        # ---- 4. Spectrogram (optional) ------------------------------
        if show_spectrogram:
            ax_sg = fig.add_subplot(gs[row])
            row += 1

            # Use a window ~1/5 of pulse width for STFT
            win_size = max(16, int(self.envelope.pulse_width.as_s / self.grid.dt / 5))
            win_size = min(win_size, self.grid.N // 4)
            # Make even
            win_size = win_size if win_size % 2 == 0 else win_size + 1

            from scipy.signal import spectrogram as scipy_spectrogram

            f_sg, t_sg, Sxx = scipy_spectrogram(
                self.envelope_field,
                fs=1.0 / self.grid.dt,
                window="hann",
                nperseg=win_size,
                noverlap=win_size * 3 // 4,
                mode="complex",
            )
            t_sg_scaled = (t_sg - self.grid.Tmax.as_s / 2) * t_scale
            f_sg_scaled = np.fft.fftshift(f_sg) * 2 * np.pi * w_scale

            im = ax_sg.pcolormesh(
                t_sg_scaled,
                np.fft.fftshift(f_sg_scaled),
                np.fft.fftshift(np.abs(Sxx) ** 2, axes=0),
                shading="gouraud",
                cmap="inferno",
            )
            fig.colorbar(im, ax=ax_sg, label="Power", fraction=0.03, pad=0.02)
            _style_ax(
                ax_sg, f"Time ({t_unit})", f"Frequency ({w_unit})", "Spectrogram (STFT)"
            )

        fig.suptitle(
            f"Pulse Analysis  —  λ-equivalent  |  chirp = {self.envelope.chirp}",
            color=COLORS["text"],
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        if save_path:
            fig.savefig(
                save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
            )

        return fig


# ─── SHG-FROG ────────────────────────────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class FROGTrace:
    """Represents a FROG trace I(ω, τ).

    Attributes
    ----------
    trace : 2D array of shape (N_omega, N_tau)
        The FROG trace (normalized if normalize=True in from_field).
    unnormalized_trace : 2D array of shape (N_omega, N_tau)
        The raw trace before normalization (used for retrieval).
    omega : angular frequency axis (rad/s), centered at 0.
    tau : delay axis (s), centered at 0.
    dt : time step (s).
    dw : frequency step (rad/s).
    field : reconstructed field E(t) (set after retrieval).
    """

    trace: NDArray
    unnormalized_trace: NDArray
    omega: NDArray
    tau: NDArray
    dt: float
    dw: float
    field: NDArray | None = None

    @classmethod
    def from_field(
        cls,
        E_field: NDArray,
        dt: float,
        normalize: bool = True,
    ) -> FROGTrace:
        """Generate a FROG trace from a complex electric field E(t).

        Parameters
        ----------
        E_field : complex 1D array, the electric field on a uniform time grid.
        dt : time step in seconds.
        normalize : normalize trace to [0, 1] (default True).

        Returns
        -------
        FROGTrace
        """
        N = len(E_field)
        # Create delay axis: tau = [-N/2*dt, ..., N/2*dt - dt]
        tau = np.arange(-N // 2, N // 2) * dt

        # Build the gated SHG signal. The (periodic) delay roll realizes
        # G(t, tau) = E(t)·E(t + tau); the delay axis is symmetric, so this is
        # the same trace as the usual E(t)·E(t - tau) definition.
        E_shifted = np.zeros((len(tau), N), dtype=complex)
        for i, t_val in enumerate(tau):
            shift = int(round(t_val / dt))
            E_shifted[i] = np.roll(E_field, -shift)

        # G(t, tau) = E(t)·E(t + tau)
        G = E_field[np.newaxis, :] * E_shifted  # shape: (N_tau, N)

        # FFT along time axis
        E_hat = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G, axes=1), axis=1), axes=1)

        # FROG trace: I(omega, tau) = |E_hat|²
        trace = np.abs(E_hat) ** 2

        # Frequency axis
        omega = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(N, d=dt))

        raw_trace = trace.copy()
        if normalize and trace.max() > 0:
            trace = trace / trace.max()

        dw = omega[1] - omega[0] if len(omega) > 1 else 1.0

        return cls(
            trace=trace,
            unnormalized_trace=raw_trace,
            omega=omega,
            tau=tau,
            dt=dt,
            dw=dw,
        )

    def visualize(
        self,
        retrieved: FROGTrace | None = None,
        figsize: tuple[float, float] | None = None,
        save_path: str | None = None,
    ):
        """Plot the FROG trace and optionally the retrieved pulse.

        Parameters
        ----------
        retrieved : optional FROGTrace with .field set (from retrieve()).
        figsize : figure size.
        save_path : if given, save figure to this path.
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (14, 6) if retrieved is None else (16, 7)

        if retrieved is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()

        # Trace heatmap
        ax_trace = axes[0]
        omega_THz = self.omega / (2 * np.pi * 1e12)
        tau_ps = self.tau * 1e12

        im = ax_trace.pcolormesh(
            tau_ps,
            omega_THz,
            self.trace,
            shading="gouraud",
            cmap="inferno",
        )
        ax_trace.set_xlabel("Delay (ps)")
        ax_trace.set_ylabel("Frequency (THz)")
        ax_trace.set_title("FROG Trace")
        fig.colorbar(im, ax=ax_trace, label="Normalized intensity")

        if retrieved is not None and retrieved.field is not None:
            E = retrieved.field
            t = (np.arange(len(E)) - len(E) // 2) * self.dt * 1e12  # ps

            # Intensity
            ax_int = axes[1]
            intensity = np.abs(E) ** 2
            intensity /= intensity.max() if intensity.max() > 0 else 1
            ax_int.plot(t, intensity, color="#00d4ff", linewidth=1.2)
            ax_int.set_xlabel("Time (ps)")
            ax_int.set_ylabel("Intensity (arb.)")
            ax_int.set_title("Retrieved Intensity")
            ax_int.grid(True, alpha=0.3)

            # Phase
            ax_phase = axes[2]
            phase = np.unwrap(np.angle(E))
            ax_phase.plot(t, phase, color="#ff6b6b", linewidth=1.2)
            ax_phase.set_xlabel("Time (ps)")
            ax_phase.set_ylabel("Phase (rad)")
            ax_phase.set_title("Retrieved Phase")
            ax_phase.grid(True, alpha=0.3)

            # Retrieved trace vs original trace comparison (axes[3])
            ax_comp = axes[3]
            # Integrated trace (over frequency) for comparison
            orig_integrated = np.sum(self.trace, axis=0)
            if orig_integrated.max() > 0:
                orig_integrated = orig_integrated / orig_integrated.max()
            if retrieved.trace is not None and retrieved.trace.size > 0:
                ret_integrated = np.sum(retrieved.trace, axis=0)
                if ret_integrated.max() > 0:
                    ret_integrated = ret_integrated / ret_integrated.max()
                tau_ps = self.tau * 1e12
                ax_comp.plot(
                    tau_ps,
                    orig_integrated,
                    color="#a78bfa",
                    linewidth=1.2,
                    label="Original",
                    alpha=0.7,
                )
                ax_comp.plot(
                    tau_ps,
                    ret_integrated,
                    color="#ff6b6b",
                    linewidth=1.2,
                    linestyle="--",
                    label="Retrieved",
                    alpha=0.7,
                )
                ax_comp.legend(fontsize=8)
            ax_comp.set_xlabel("Delay (ps)")
            ax_comp.set_ylabel("Integrated intensity (arb.)")
            ax_comp.set_title("Trace Comparison (integrated over ω)")
            ax_comp.grid(True, alpha=0.3)
        else:
            # Hide extra subplots when no retrieved field is available
            # (axes has 1 element when retrieved is None, 4 otherwise).
            for extra in axes[1:]:
                extra.axis("off")

        fig.suptitle(
            "FROG Analysis" + (" — Retrieved" if retrieved is not None else ""),
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


def generate_trace(
    E_field: NDArray,
    dt: float,
    normalize: bool = True,
) -> FROGTrace:
    """Generate a SHG-FROG trace from an electric field.

    Parameters
    ----------
    E_field : complex 1D array, electric field on uniform time grid.
    dt : time step in seconds.
    normalize : normalize trace to [0, 1] (default True).

    Returns
    -------
    FROGTrace
    """
    return FROGTrace.from_field(E_field, dt=dt, normalize=normalize)


def fidelity(trace1: FROGTrace, trace2: FROGTrace) -> float:
    """Compute FROG fidelity between two traces.

    fidelity = 1 - ||I1 - I2|| / ||I1||

    Returns
    -------
    float in [0, 1], where 1 = perfect match.
    """
    t1 = trace1.trace
    t2 = trace2.trace

    # Crop or pad to same shape
    min_rows = min(t1.shape[0], t2.shape[0])
    min_cols = min(t1.shape[1], t2.shape[1])
    t1 = t1[:min_rows, :min_cols]
    t2 = t2[:min_rows, :min_cols]

    diff = t1 - t2
    norm_diff = np.linalg.norm(diff)
    norm_ref = np.linalg.norm(t1)

    if norm_ref == 0:
        return 1.0 if norm_diff == 0 else 0.0

    return float(1.0 - norm_diff / norm_ref)


def _pcgpa_update(G_time: NDArray, E: NDArray, shifts: list[int]) -> NDArray:
    """Generalized-projections (GP) extraction step for SHG-FROG.

    The SHG signal model is ``E_sig(t, τ) = E(t)·E(t+τ)``. Given the
    measured-trace-constrained signal ``G'(t, τ)``, the least-squares
    projection back onto the set of signals of this form is obtained by
    minimizing ``Σ_{t,τ} |G'(t,τ) − E(t)E_old(t+τ)|²`` w.r.t. ``E``, which gives

        E_new(t) = Σ_τ G'(t, τ)·E_old*(t+τ) / Σ_τ |E_old(t+τ)|²

    (the denominator is the delay-autocorrelation of the current estimate).
    This is the material-constraint projection of the generalized-projections
    algorithm for SHG-FROG, in the form used by PCGPA. The conjugate and the
    pointwise denominator are essential: dropping them is not the least-squares
    solution and stalls on chirped pulses.

    References
    ----------
    - DeLong, Trebino, Hunter & White, *J. Opt. Soc. Am. B* **11**, 2206 (1994)
      (generalized projections for SHG-FROG).
    - Reid, Dantus & Zewail, *Opt. Commun.* **181**, 73 (2000)
      (generalized phase retrieval for SHG-FROG).
    - Kane, *IEEE J. Quantum Electron.* **35**, 421 (1999); Trebino,
      *Frequency-Resolved Optical Gating* (Kluwer, 2000) (PCGPA).
    """
    num = np.zeros(len(E), dtype=complex)
    den = np.zeros(len(E), dtype=float)
    for i, shift in enumerate(shifts):
        E_shift = np.roll(E, -shift)
        num += G_time[i] * np.conj(E_shift)
        den += np.abs(E_shift) ** 2
    out = np.zeros_like(num)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    norm_out = float(np.linalg.norm(out))
    return out / norm_out if norm_out > 0 else E


def retrieve(
    trace: FROGTrace,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = True,
    *,
    seed: int | None = 0,
    n_restarts: int = 5,
) -> FROGTrace:
    """Retrieve the electric field E(t) from a FROG trace using PCGPA.

    Implements the generalized-projections / PCGPA material-constraint step:
    after replacing the trace magnitude with the measurement (keeping the
    phase), the new field is the least-squares solution

        E_new(t) = Σ_τ G'(t, τ)·E*(t+τ) / Σ_τ |E(t+τ)|²

    for the bilinear SHG signal ``G(t, τ) = E(t)E(t+τ)``. Retrieval is
    initialized from the dominant SVD component of the trace plus one or more
    random complex fields (seeded for reproducibility); the lowest-error result
    is returned.

    Parameters
    ----------
    trace : FROGTrace — the measured (or generated) trace.
    max_iter : maximum iterations per restart (default 100).
    tol : convergence tolerance on the change in the normalized FROG error
        (default 1e-4).
    verbose : print iteration progress (default True).
    seed : RNG seed for the random initialization (default 0). ``None`` uses
        fresh entropy.
    n_restarts : number of random restarts (default 5). Together with the
        SVD initialization these are ranked by FROG error and the lowest-error
        result is returned.

    Returns
    -------
    FROGTrace with ``.field`` set to the retrieved E(t).

    Notes
    -----
    SHG-FROG cannot distinguish ``E(t)`` from ``E*(−t)`` or a global phase, so
    compare retrieved and reference fields up to those symmetries (comparing
    traces is ambiguity-free).
    """
    dt = trace.dt

    # Use unnormalized trace if available (preserves amplitude info)
    measured_trace = (
        trace.unnormalized_trace
        if trace.unnormalized_trace is not None
        else trace.trace
    )
    measured_mag = np.sqrt(measured_trace)
    measured_max = float(measured_trace.max())
    if measured_max <= 0:
        raise ValueError("Cannot retrieve from an all-zero FROG trace.")
    # Scale-invariant error reference (a normalized FROG trace carries no
    # absolute amplitude information, so retrieval cannot recover the scale).
    measured_trace_norm = measured_trace / measured_max
    measured_norm_unit = float(np.linalg.norm(measured_trace_norm)) or 1.0

    shifts = [int(round(tau_val / dt)) for tau_val in trace.tau]
    N = measured_trace.shape[1]

    rng = np.random.default_rng(seed)
    best_E: NDArray | None = None
    best_err = np.inf

    # Initial guesses: the dominant SVD component of the trace (reliable for
    # symmetric pulses) plus `n_restarts` random complex fields (needed to
    # escape the local minimum that traps chirped retrieval). The lowest-error
    # result across all candidates is returned.
    _, singular_values, Vt = np.linalg.svd(measured_trace, full_matrices=False)
    svd_guess = np.sqrt(singular_values[0]) * Vt[0]
    candidates = [svd_guess]
    candidates += [
        rng.standard_normal(N) + 1j * rng.standard_normal(N)
        for _ in range(max(1, n_restarts))
    ]

    for restart, init_E in enumerate(candidates):
        E = init_E / np.linalg.norm(init_E)
        prev_err: float | None = None
        err = np.inf

        for iteration in range(max(1, max_iter)):
            # a. Gated signal G(t, τ) = E(t)·E(t + τ)  (roll realizes E(t+τ))
            G = np.empty((len(shifts), N), dtype=complex)
            for i, shift in enumerate(shifts):
                G[i] = E * np.roll(E, -shift)

            # b. FFT along the time axis
            G_hat = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(G, axes=1), axis=1), axes=1
            )

            # c. Replace magnitude with sqrt(I_meas), keep the phase
            G_new = measured_mag * np.exp(1j * np.angle(G_hat))

            # d. Inverse FFT back to the time domain
            G_time = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(G_new, axes=1), axis=1), axes=1
            )

            # e. Principal-component projection over ALL delays
            E_new = _pcgpa_update(G_time, E, shifts)

            # f. Scale-invariant normalized FROG error of the updated estimate
            G_check = np.empty((len(shifts), N), dtype=complex)
            for i, shift in enumerate(shifts):
                G_check[i] = E_new * np.roll(E_new, -shift)
            G_check_hat = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(G_check, axes=1), axis=1), axes=1
            )
            calc_trace = np.abs(G_check_hat) ** 2
            calc_max = float(calc_trace.max())
            if calc_max > 0:
                err = float(
                    np.linalg.norm(calc_trace / calc_max - measured_trace_norm)
                    / measured_norm_unit
                )
            else:
                err = np.inf

            if verbose and (iteration % 10 == 0 or iteration == max(1, max_iter) - 1):
                print(
                    f"  Restart {restart} iter {iteration:3d}: FROG error = {err:.6e}"
                )

            E = E_new
            if err < tol:
                break
            if prev_err is not None and iteration > 5 and abs(prev_err - err) < tol:
                break
            prev_err = err

        if err < best_err:
            best_err = err
            best_E = E

    E = best_E if best_E is not None else E

    # Build result FROGTrace with retrieved field
    # Regenerate the trace from the retrieved field for accurate fidelity comparison
    retrieved_trace = FROGTrace.from_field(E, dt=dt, normalize=True)

    result = FROGTrace(
        trace=retrieved_trace.trace,
        unnormalized_trace=retrieved_trace.unnormalized_trace,
        omega=trace.omega,
        tau=trace.tau,
        dt=trace.dt,
        dw=trace.dw,
        field=E,
    )

    return result
