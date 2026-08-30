from __future__ import annotations
from pydantic.dataclasses import dataclass
from math import sqrt, log, acosh, pi
from typing import Any, Callable, Dict, Literal, Self
from functools import cached_property, lru_cache
import logging
from matplotlib import gridspec
from numpy.typing import NDArray
from photonics_helper.base import Wavelength, Frequency, Time, C_MS
from photonics_helper._fftw import fft as _fft_backend, ifft as _ifft_backend

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy, hermite as hermite_poly

logger = logging.getLogger(__name__)

SHAPE_FACTORS: Dict[str, float] = {
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

        return amp * np.exp(1j * phase)

    def intensity(self, t: NDArray) -> NDArray:
        A = self.field(t)
        return np.abs(A) ** 2

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
    ) -> "Envelope":
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
                phase += 0.5 * GDD * omega_ps ** 2
            if TOD != 0.0:
                phase += (1.0 / 6.0) * TOD * omega_ps ** 3
            if FOD != 0.0:
                phase += (1.0 / 24.0) * FOD * omega_ps ** 4

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
            pulse_width=Time(measured_width, "s") if measured_width > 0.0 else self.pulse_width,
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
        backend: Literal["plotly", "matplotlib", "xy"] = "plotly",
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
        backend : "plotly", "matplotlib", or "xy" (default "plotly")
        N : number of time points (default 2^12)
        show_phase : show instantaneous phase overlay (default True)
        show_fwhm : show FWHM markers (default True)
        figsize : figure size for matplotlib backend (default None)
        title : optional title override (default uses shape name)
        theme : "light" or "dark" (default "light")
        """
        if backend not in ("plotly", "matplotlib", "xy"):
            raise ValueError("backend must be 'plotly', 'matplotlib', or 'xy'")

        grid = self._make_grid(N)
        t = grid.t
        A = self.field(t)
        intensity_t = np.abs(A) ** 2
        spectral = grid.fft(A)
        intensity_w = np.abs(spectral) ** 2
        phase = np.unwrap(np.angle(A))

        if title is None:
            title = f"{self.shape.title()} Pulse Envelope (T₀={self.pulse_width.as_s*1e15:.1f} fs)"

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
        elif backend == "xy":
            return self._visualize_2d_xy(
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
            f"T₀ = {self.pulse_width.as_s:.3g} s ({self.pulse_width.as_s*1e15:.1f} fs)<br>"
            f"FWHM = {self.fwhm.as_s:.3g} s ({self.fwhm.as_s*1e15:.1f} fs)<br>"
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
            f"T₀ = {self.pulse_width.as_s:.3g} s ({self.pulse_width.as_s*1e15:.1f} fs)\n"
            f"FWHM = {self.fwhm.as_s:.3g} s ({self.fwhm.as_s*1e15:.1f} fs)\n"
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
            np.abs(A),
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


class _XyHtmlView:
    """Lightweight wrapper so xy backends return an object with .to_html()."""

    def __init__(self, html: str) -> None:
        self._html = html

    def to_html(self, full_html: bool = False, include_plotlyjs: str | None = None) -> str:
        return self._html


def _visualize_2d_xy(
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
    """XY backend implementation of 2D visualization."""
    import re

    import xy

    # Convert to fs and THz for readable axes
    t_fs = t * 1e15
    w_THz = grid.w / (2 * np.pi * 1e12)  # rad/s -> THz

    # Theme colors
    colors = {
        "dark": {
            "bg": "#1a1a2e",
            "intensity": "#00d4ff",
            "spectrum": "#a78bfa",
            "phase": "#ff6b6b",
            "polar": "#34d399",
            "text": "#e0e0e0",
        },
        "light": {
            "bg": "#ffffff",
            "intensity": "#0088cc",
            "spectrum": "#7c3aed",
            "phase": "#dc2626",
            "polar": "#059669",
            "text": "#1a1a1a",
        },
    }
    c = colors[theme]

    # Temporal intensity chart
    t_children = [xy.line(x=t_fs, y=intensity_t, color=c["intensity"])]
    if show_fwhm:
        t0_fs = self.pulse_width.as_s * 1e15
        t_children.extend([xy.vline(x=-t0_fs, color="#fbbf24"), xy.vline(x=t0_fs, color="#fbbf24")])
    t_chart = xy.line_chart(*t_children, title="Temporal Intensity", width=400, height=280)

    # Spectral intensity chart
    s_chart = xy.line_chart(
        xy.line(x=w_THz, y=intensity_w, color=c["spectrum"]),
        title="Spectral Intensity",
        width=400,
        height=280,
    )

    # Phase chart
    if show_phase:
        p_chart = xy.line_chart(
            xy.line(x=t_fs, y=phase, color=c["phase"]),
            title="Instantaneous Phase",
            width=400,
            height=280,
        )
    else:
        p_chart = xy.line_chart(
            xy.line(x=t_fs[:0], y=np.array([])),
            title="Instantaneous Phase",
            width=400,
            height=280,
        )

    # Polar plot (Re vs Im)
    polar_chart = xy.line_chart(
        xy.line(x=np.real(A), y=np.imag(A), color=c["polar"]),
        title="Polar Plot",
        width=400,
        height=280,
    )

    # Combine into grid HTML
    charts = [t_chart, s_chart, p_chart, polar_chart]
    chart_htmls = []
    for ch in charts:
        full_html = ch.to_html()
        body_match = re.search(r"<body>(.*?)</body>", full_html, re.DOTALL)
        if body_match:
            chart_htmls.append(body_match.group(1))

    grid_html = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px;background:'
        + ("#1a1a2e" if theme == "dark" else "#ffffff")
        + '">\n'
        + "".join(chart_htmls)
        + "\n</div>"
    )
    return _XyHtmlView(grid_html)


Envelope._visualize_2d_xy = _visualize_2d_xy


@dataclass
class TemporalGrid:
    N: int
    Tmax: Time  # total time window

    @cached_property
    def dt(self):
        return self.Tmax.as_s / self.N

    @cached_property
    def t(self):
        dt = self.dt
        return np.linspace(-self.Tmax.as_s / 2, self.Tmax.as_s / 2 - dt, self.N)

    @cached_property
    def w(self):
        return np.fft.fftshift(2 * np.pi * np.fft.fftfreq(self.N, d=self.dt))

    @cached_property
    def dw(self):
        w = self.w
        return w[1] - w[0]

    def fft(self, A_t):
        """Forward FFT with photonics_helper convention: ``FFT(A)·dt``.

        The result is fftshifted and scaled by ``dt`` so that Parseval's
        theorem holds with the companion :meth:`ifft`. Raw FFT magnitudes
        differ from laserfun (which uses unshifted ``fft`` without ``dt``);
        compare normalized spectra or time-domain intensities across tools.

        Executes on the FFTW3 backend (:mod:`photonics_helper._fftw`) when
        ``pyfftw`` is installed, falling back to ``numpy.fft`` otherwise.
        """
        return _fft_backend(A_t) * self.dt

    def ifft(self, A_w):
        """Inverse FFT paired with :meth:`fft` (includes ``1/dt`` scaling).

        Executes on the FFTW3 backend when ``pyfftw`` is installed.
        """
        return _ifft_backend(A_w) / self.dt

    @property
    def omega_max(self):
        return np.max(np.abs(self.w))

    @property
    def time_window(self):
        return self.N * self.dt

    @classmethod
    def for_pulse_train(
        cls,
        repetition_rate: Frequency,
        n_pulses: int,
        pulse_width: Time,
        N: int = 2**12,
    ) -> Self:
        """Compute the right Tmax to cover a pulse train.

        Parameters
        ----------
        repetition_rate : Frequency — pulse spacing = 1 / repetition_rate
        n_pulses : number of pulses
        pulse_width : T₀ — characteristic width, used to estimate needed padding
        N : number of time points (default 2¹²)
        """
        spacing = 1.0 / repetition_rate.as_Hz
        # Window must cover all pulses + padding for tails
        Tmax = Time(n_pulses * spacing + 10 * pulse_width.as_s, "s")
        return cls(N=N, Tmax=Tmax)


@dataclass
class Wave:
    grid: TemporalGrid
    envelope: Envelope

    central_wavelength: Wavelength
    _pulse_train_field: Any = None
    refractive_index: float = 1.0
    repetition_rate: Frequency | None = None

    @cached_property
    def central_frequency(self) -> float:
        return self.central_wavelength.to_omega().as_rad_s

    @property
    def wavelength_nm(self) -> NDArray:
        """Absolute wavelength grid (nm) for each frequency sample on ``grid.w``."""
        omega_abs = self.central_frequency + self.grid.w
        return 2 * np.pi * C_MS / omega_abs * 1e9

    @property
    def electric_field(self):
        A = self.envelope_field
        return np.real(A * np.exp(-1j * self.central_frequency * self.grid.t))

    def instantaneous_intensity(self):
        E = self.electric_field
        return np.abs(E) ** 2

    @property
    def envelope_intensity(self):
        return np.abs(self.envelope_field) ** 2

    def pulse_energy(self):
        return np.sum(self.envelope_intensity) * self.grid.dt

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

    def peak_power(self):
        A = self.envelope_field
        return np.max(np.abs(A) ** 2)

    def average_power(self, repetition_rate: Frequency) -> float:
        """Average power = pulse energy × repetition_rate.

        Parameters
        ----------
        repetition_rate : Hz
        """
        return self.pulse_energy() * repetition_rate.as_Hz

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
        # Override the envelope_field property via a wrapper
        wave._pulse_train_field = full_field
        return wave

    @property
    def envelope_field(self):
        if hasattr(self, "_pulse_train_field") and self._pulse_train_field is not None:
            return self._pulse_train_field
        return self.envelope.field(self.grid.t)

    @cached_property
    def spectrum(self) -> NDArray:
        return self.grid.fft(self.envelope_field)

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
            f"TBP = {tbp:.3f}   |   Peak power = {self.peak_power():.3g} W",
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
    ) -> "FROGTrace":
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

        # Build the gated signal G(t, tau) = E(t) * E(t - tau)
        E_shifted = np.zeros((len(tau), N), dtype=complex)
        for i, t_val in enumerate(tau):
            shift = int(round(t_val / dt))
            E_shifted[i] = np.roll(E_field, -shift)

        # G(t, tau) = E(t) * E(t - tau)
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
        retrieved: "FROGTrace | None" = None,
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
                    tau_ps, orig_integrated, color="#a78bfa",
                    linewidth=1.2, label="Original", alpha=0.7
                )
                ax_comp.plot(
                    tau_ps, ret_integrated, color="#ff6b6b",
                    linewidth=1.2, linestyle="--", label="Retrieved", alpha=0.7
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


def retrieve(
    trace: FROGTrace,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = True,
) -> FROGTrace:
    """Retrieve the electric field E(t) from a FROG trace using PCGPA.

    Parameters
    ----------
    trace : FROGTrace — the measured (or generated) trace.
    max_iter : maximum iterations (default 100).
    tol : convergence tolerance on fidelity change (default 1e-4).
    verbose : print iteration progress (default True).

    Returns
    -------
    FROGTrace with .field set to the retrieved E(t).
    """
    N_tau, N_omega = trace.trace.shape
    dt = trace.dt

    # Use unnormalized trace if available (preserves amplitude info)
    measured_trace = (
        trace.unnormalized_trace
        if trace.unnormalized_trace is not None
        else trace.trace
    )

    # --- Step 1: SVD initialization ---
    U, S, Vt = np.linalg.svd(measured_trace, full_matrices=False)

    # Take the singular vector with largest singular value
    E_init = np.sqrt(S[0]) * Vt[0]

    # Normalize
    E_init = E_init / np.linalg.norm(E_init)

    # --- Step 2: PCGPA iteration ---
    E = E_init.copy()

    prev_fidelity = 0.0

    for iteration in range(max_iter):
        N = len(E)
        G = np.zeros((N_tau, N), dtype=complex)

        for i, tau_val in enumerate(trace.tau):
            shift = int(round(tau_val / dt))
            E_shifted = np.roll(E, -shift)
            G[i] = E * E_shifted

        # b. FFT along time axis
        G_hat = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G, axes=1), axis=1), axes=1)

        # c. Replace magnitude with sqrt(I_meas), keep phase
        measured_mag = np.sqrt(measured_trace)
        G_new = measured_mag * np.exp(1j * np.angle(G_hat))

        # d. Inverse FFT along frequency axis
        G_new_time = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(G_new, axes=1), axis=1), axes=1
        )

        # e. Extract new E from G_new_time at tau = 0 (center column)
        E_new = G_new_time[:, N_tau // 2].copy()

        # Normalize
        E_new = E_new / np.linalg.norm(E_new)

        # f. Check convergence
        G_check = np.zeros((N_tau, N), dtype=complex)
        for i, tau_val in enumerate(trace.tau):
            shift = int(round(tau_val / dt))
            E_shifted = np.roll(E, -shift)
            G_check[i] = E * E_shifted

        G_check_hat = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(G_check, axes=1), axis=1), axes=1
        )
        trace_calc = np.abs(G_check_hat) ** 2
        if trace_calc.max() > 0:
            trace_calc /= trace_calc.max()

        curr_fidelity = fidelity(
            trace,
            FROGTrace(
                trace=trace_calc,
                unnormalized_trace=trace_calc,
                omega=trace.omega,
                tau=trace.tau,
                dt=trace.dt,
                dw=trace.dw,
            ),
        )

        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            print(f"  Iter {iteration:3d}: fidelity = {curr_fidelity:.6f}")

        if abs(curr_fidelity - prev_fidelity) < tol and iteration > 5:
            if verbose:
                print(
                    f"  Converged at iteration {iteration}, fidelity = {curr_fidelity:.6f}"
                )
            break

        prev_fidelity = curr_fidelity
        E = E_new

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
