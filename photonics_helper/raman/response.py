"""Raman response functions: time domain, frequency domain, pulse interaction."""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from .._fftw import convolve_full as _fftw_convolve_full
from ..base import (
    Time,
)
from ..pulse import TemporalGrid, Wave

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


from .spec import RamanSpec

# ─── RamanResponse (Layer 2) ─────────────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanResponse:
    """Time-domain Raman response function h_R(t).

    Implements the standard Silica model (Agrawal):
        h_R(t) = (τ1² + τ2²) / (τ1·τ2²) · exp(-t/τ2) · sin(t/τ1)   for t ≥ 0
        h_R(t) = 0                                                    for t < 0

    Combined response: R(t) = (1 - fR)·δ(t) + fR·h_R(t)
    where δ(t) is approximated as a narrow Gaussian.

    Attributes
    ----------
    spec : RamanSpec
        Material Raman properties.
    fR : float
        Raman response fraction. Overrides spec.fR if set.
    tau1 : float
        Oscillation period (s). Auto-derived from Raman shift if None.
    tau2 : float
        Damping time (s). Auto-derived from linewidth if None.
    grid : TemporalGrid
        Time grid for computations.
    """

    spec: RamanSpec
    fR: float | None = None
    tau1: float | None = None
    tau2: float | None = None
    grid: TemporalGrid | None = None

    @model_validator(mode="after")
    def _derive_tau(self) -> RamanResponse:
        """Auto-derive τ1, τ2 from Raman shift + linewidth if not provided."""
        if self.fR is None:
            self.fR = self.spec.fR or 0.0

        if self.tau1 is None and self.spec.raman_shift_Hz > 0:
            self.tau1 = 1.0 / (2 * np.pi * self.spec.raman_shift_Hz)

        if self.tau2 is None and self.spec.linewidth_Hz > 0:
            self.tau2 = 1.0 / (np.pi * self.spec.linewidth_Hz)
            warnings.warn(
                f"Auto-derived τ2 = {self.tau2 * 1e15:.1f} fs from linewidth. "
                f"This approximation (τ2 = 1/(π·linewidth)) assumes weak damping "
                f"and may be inaccurate for materials like Silica where τ2/τ1 is small. "
                f"Consider providing τ1 and τ2 explicitly for accurate Raman responses.",
                stacklevel=2,
            )

        if self.grid is None:
            # Default grid: cover ~20 τ2 for damped oscillation to decay
            from ..base import Time

            tau2_val = self.tau2 or 1e-12
            tmax = max(10e-12, 20 * tau2_val)
            self.grid = TemporalGrid(N=2**14, Tmax=Time(tmax, unit="s"))

        return self

    @property
    def _delta_width(self) -> float:
        """Width parameter ε for the narrow Gaussian approximation of δ(t)."""
        # Use grid.dt * 5 as the delta width — narrow enough to look like a spike
        return self.grid.dt * 5  # type: ignore

    def _h_R(self, t: NDArray) -> NDArray:
        """Raw delayed response h_R(t) (without fR scaling).

        Standard Agrawal exponential-damped form:
        h_R(t) = (τ1² + τ2²) / (τ1·τ2²) · exp(-t/τ2) · sin(t/τ1)   for t ≥ 0
        h_R(t) = 0                                                    for t < 0
        """
        result = np.zeros_like(t, dtype=float)
        mask = t >= 0
        t_pos = t[mask]

        prefactor = (self.tau1**2 + self.tau2**2) / (self.tau1 * self.tau2**2)  # type: ignore
        exponential = np.exp(-t_pos / self.tau2)
        oscillation = np.sin(t_pos / self.tau1)

        result[mask] = prefactor * exponential * oscillation

        integral = np.trapezoid(result[mask], t_pos)
        if integral > 0:
            result[mask] /= integral
        return result

    def instantaneous_response(self, t: NDArray | None = None) -> NDArray:
        """Electronic Kerr response: (1 - fR)·δ(t).

        δ(t) approximated as a narrow Gaussian: exp(-t²/(2ε²)) / (ε·√(2π)).

        Parameters
        ----------
        t : array of time values (s). If None, uses self.grid.t.

        Returns
        -------
        NDArray — instantaneous response values.
        """
        if t is None:
            t = self.grid.t  # type: ignore
        t = np.asarray(t, dtype=float)
        eps = self._delta_width
        delta = np.exp(-(t**2) / (2 * eps**2)) / (eps * np.sqrt(2 * np.pi))
        return (1.0 - self.fR) * delta  # type: ignore

    def delayed_response(self, t: NDArray | None = None) -> NDArray:
        """Lattice oscillation: fR·h_R(t).

        Parameters
        ----------
        t : array of time values (s). If None, uses self.grid.t.

        Returns
        -------
        NDArray — delayed (Raman) response values.
        """
        if t is None:
            t = self.grid.t  # type: ignore
        t = np.asarray(t, dtype=float)
        return self.fR * self._h_R(t)  # type: ignore

    def combined_response(self, t: NDArray | None = None) -> NDArray:
        """Total Raman response R(t) = (1-fR)δ(t) + fR·h_R(t).

        Parameters
        ----------
        t : array of time values (s). If None, uses self.grid.t.

        Returns
        -------
        NDArray — combined response values.
        """
        if t is None:
            t = self.grid.t  # type: ignore
        t = np.asarray(t, dtype=float)
        return self.instantaneous_response(t) + self.delayed_response(t)  # type: ignore

    def plot_components(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
        t_range_ps: tuple[float, float] | None = None,
    ):
        """3-panel plot: instantaneous, delayed, combined response.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        figsize : Figure size for matplotlib
        t_range_ps : Time range in picoseconds as (t_min, t_max). Overrides grid.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_components_plotly(figsize, t_range_ps)
        else:
            return self._plot_components_matplotlib(figsize, t_range_ps)

    def _plot_components_matplotlib(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Matplotlib 3-panel plot."""
        import matplotlib.pyplot as plt

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = self.grid.t  # type: ignore

        inst = self.instantaneous_response(t)
        delayed = self.delayed_response(t)
        combined = self.combined_response(t)

        fig, axes = plt.subplots(3, 1, figsize=figsize or (10, 9), sharex=True)
        fig.suptitle(
            f"Raman Response: {self.spec.name}  "
            f"(fR={self.fR:.2f}, τ1={self.tau1 * 1e15:.2f} fs, τ2={self.tau2 * 1e15:.2f} fs)",  # type: ignore
            fontsize=13,
            fontweight="bold",
        )

        # Panel 1: Instantaneous
        ax1 = axes[0]
        ax1.plot(t * 1e12, inst, color="#00d4ff", linewidth=1.5)
        ax1.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax1.set_title("Instantaneous (Electronic Kerr) Response", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="k", linewidth=0.5)

        # Panel 2: Delayed
        ax2 = axes[1]
        ax2.plot(t * 1e12, delayed, color="#a78bfa", linewidth=1.5)
        ax2.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax2.set_title("Delayed (Lattice Oscillation) Response fR·h_R(t)", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="k", linewidth=0.5)
        # Shade the oscillation envelope
        envelope = np.abs(delayed) if np.any(delayed != 0) else delayed
        ax2.fill_between(t * 1e12, 0, envelope, alpha=0.1, color="#a78bfa")

        # Panel 3: Combined
        ax3 = axes[2]
        ax3.plot(
            t * 1e12, combined, color="#34d399", linewidth=1.5, label="Combined R(t)"
        )
        ax3.set_xlabel("Time (ps)", fontsize=10)
        ax3.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax3.set_title("Combined Response R(t) = (1-fR)δ(t) + fR·h_R(t)", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color="k", linewidth=0.5)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        return fig

    def _plot_components_plotly(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Plotly 3-panel plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = self.grid.t  # type: ignore

        inst = self.instantaneous_response(t)
        delayed = self.delayed_response(t)
        combined = self.combined_response(t)

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Instantaneous (Electronic Kerr) Response",
                "Delayed (Lattice Oscillation) Response fR·h_R(t)",
                "Combined Response R(t) = (1-fR)δ(t) + fR·h_R(t)",
            ),
            shared_xaxes=True,
            vertical_spacing=0.08,
        )

        t_ps = t * 1e12

        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=inst,
                mode="lines",
                name="Instant.",
                line=dict(color="#00d4ff", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=delayed,
                mode="lines",
                name="Delayed",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=combined,
                mode="lines",
                name="Combined",
                line=dict(color="#34d399", width=1.5),
            ),
            row=3,
            col=1,
        )

        for i in range(1, 4):
            fig.add_hline(
                y=0, line_dash="dot", line_color="gray", opacity=0.5, row=i, col=1
            )

        fig.update_yaxes(title_text="Amplitude (arb.)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (arb.)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude (arb.)", row=3, col=1)
        fig.update_xaxes(title_text="Time (ps)", row=3, col=1)

        fig.update_layout(
            title_text=f"Raman Response: {self.spec.name}  "
            f"(fR={self.fR:.2f}, τ1={self.tau1 * 1e15:.2f} fs, τ2={self.tau2 * 1e15:.2f} fs)",  # type: ignore
            height=750,
            showlegend=False,
        )
        return fig


# ─── RamanFrequencyResponse (Layer 3) ─────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanFrequencyResponse:
    """Frequency-domain Raman response H(Ω) = F{h_R(t)}.

    Connects time-domain physics to measured Raman spectra via FFT.

    Attributes
    ----------
    response : RamanResponse
        Layer 2 time-domain response.
    grid : TemporalGrid
        Frequency grid (from the temporal grid's fft).
    """

    response: RamanResponse
    grid: TemporalGrid | None = None

    @model_validator(mode="after")
    def _ensure_grid(self) -> RamanFrequencyResponse:
        if self.grid is None:
            self.grid = self.response.grid
        return self

    @property
    def H(self) -> NDArray:
        """Complex frequency response H(Ω)."""
        h_R_t = (
            self.response.delayed_response(self.grid.t) / self.response.fR  # type: ignore
            if self.response.fR > 0  # type: ignore
            else np.zeros_like(self.grid.t)  # type: ignore
        )
        return self.grid.fft(h_R_t)  # type: ignore

    @property
    def H_real(self) -> NDArray:
        """Real part Re(H(Ω))."""
        return np.real(self.H)

    @property
    def H_imag(self) -> NDArray:
        """Imaginary part Im(H(Ω))."""
        return np.imag(self.H)

    @property
    def H_magnitude(self) -> NDArray:
        """Magnitude |H(Ω)|."""
        return np.asarray(np.abs(self.H))

    @property
    def H_phase(self) -> NDArray:
        """Phase ∠H(Ω)."""
        return np.asarray(np.angle(self.H))

    @property
    def resonance_frequency_THz(self) -> float:
        """Resonance frequency in THz (peak of |Im(H)|)."""
        # Find peak of |Im(H)| in positive frequencies
        positive_mask = self.grid.w > 0  # type: ignore
        w_pos = self.grid.w[positive_mask]  # type: ignore
        imag_pos = self.H_imag[positive_mask]
        peak_idx = np.argmax(np.abs(imag_pos))
        return float(w_pos[peak_idx] / (2 * np.pi * 1e12))  # rad/s → THz

    @property
    def resonance_FWHM_THz(self) -> float:
        """FWHM of the resonance in THz."""
        mag = self.H_magnitude
        w = self.grid.w / (2 * np.pi * 1e12)  # type: ignore  # rad/s → THz

        # Find peak
        peak_idx = np.argmax(mag)
        peak_val = mag[peak_idx]
        half_max = peak_val / 2

        # Find left and right crossings
        left_idx = np.where(mag[:peak_idx] < half_max)[0]
        right_idx = np.where(mag[peak_idx:] < half_max)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            w_left = w[left_idx[-1]]
            w_right = w[peak_idx + right_idx[0]]
            return float(w_right - w_left)

        # Fallback: use grid resolution
        return float(self.grid.dw / (2 * np.pi * 1e12))  # type: ignore

    @property
    def quality_factor(self) -> float:
        """Quality factor Q = f_res / FWHM."""
        fwhm = self.resonance_FWHM_THz
        if fwhm == 0:
            return float("inf")
        return self.resonance_frequency_THz / fwhm

    def plot_real(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Re(H(Ω))."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Real Part Re(H)", self.H_real, "Re(H(Ω))", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Real Part Re(H)", self.H_real, "Re(H(Ω))", figsize
        )

    def plot_imag(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Im(H(Ω)) — the Raman gain spectrum."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Imaginary Part Im(H)", self.H_imag, "Im(H(Ω))", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Imaginary Part Im(H)", self.H_imag, "Im(H(Ω))", figsize
        )

    def plot_magnitude(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot |H(Ω)|."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Magnitude |H(Ω)|", self.H_magnitude, "|H(Ω)|", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Magnitude |H(Ω)|", self.H_magnitude, "|H(Ω)|", figsize
        )

    def plot_phase(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot ∠H(Ω)."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Phase ∠H(Ω)", self.H_phase, "∠H(Ω) (rad)", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Phase ∠H(Ω)", self.H_phase, "∠H(Ω) (rad)", figsize
        )

    def plot_all(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """4-panel plot: Re, Im, |H|, phase."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_all_plotly(figsize)
        return self._plot_all_matplotlib(figsize)

    def _plot_dispersion_matplotlib(
        self,
        title: str,
        data: NDArray,
        ylabel: str,
        figsize: tuple[float, float] | None,
    ):
        """Single-panel dispersion plot (matplotlib)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (10, 5))
        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        ax.plot(w_THz, data, linewidth=1.5, color="#00d4ff")
        ax.set_xlabel("Angular frequency (THz)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

        # Annotate resonance
        if "Im" in title or "Magnitude" in title:
            f_res = self.resonance_frequency_THz
            ax.axvline(
                f_res,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Resonance: {f_res:.2f} THz",
            )
            ax.legend()

        plt.tight_layout()
        return fig

    def _plot_dispersion_plotly(
        self,
        title: str,
        data: NDArray,
        ylabel: str,
        figsize: tuple[float, float] | None,
    ):
        """Single-panel dispersion plot (plotly)."""
        import plotly.graph_objects as go

        fig = go.Figure()
        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        fig.add_trace(
            go.Scatter(
                x=w_THz,
                y=data,
                mode="lines",
                name="H(Ω)",
                line=dict(color="#00d4ff", width=1.5),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Angular frequency (THz)",
            yaxis_title=ylabel,
            height=500,
        )
        return fig

    def _plot_all_matplotlib(self, figsize: tuple[float, float] | None) -> plt.Figure:
        """4-panel matplotlib plot: Re, Im, |H|, phase."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 9))
        fig.suptitle(
            f"Raman Frequency Response: {self.response.spec.name}  "
            f"(f_res={self.resonance_frequency_THz:.2f} THz, Q={self.quality_factor:.1f})",
            fontsize=13,
            fontweight="bold",
        )

        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        colors = ["#00d4ff", "#a78bfa", "#34d399", "#fbbf24"]
        titles = ["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω)"]
        data = [self.H_real, self.H_imag, self.H_magnitude, self.H_phase]
        ylabels = ["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω) (rad)"]

        for idx, ax in enumerate(axes.flatten()):
            ax.plot(w_THz, data[idx], linewidth=1.5, color=colors[idx])
            ax.set_xlabel("Angular frequency (THz)", fontsize=9)
            ax.set_ylabel(ylabels[idx], fontsize=9)
            ax.set_title(titles[idx], fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", linewidth=0.5)

            # Annotate resonance on Im and |H| panels
            if idx in (1, 2):
                f_res = self.resonance_frequency_THz
                ax.axvline(
                    f_res,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Resonance: {f_res:.2f} THz",
                )
                ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    def _plot_all_plotly(self, figsize: tuple[float, float] | None) -> go.Figure:
        """4-panel plotly plot: Re, Im, |H|, phase."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω)"],
            shared_xaxes=True,
            shared_yaxes=False,
        )

        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        colors = ["#00d4ff", "#a78bfa", "#34d399", "#fbbf24"]
        data = [self.H_real, self.H_imag, self.H_magnitude, self.H_phase]
        ylabels = ["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω) (rad)"]

        for idx in range(4):
            row, col = divmod(idx, 2)
            row += 1
            col += 1
            fig.add_trace(
                go.Scatter(
                    x=w_THz,
                    y=data[idx],
                    mode="lines",
                    name=ylabels[idx],
                    line=dict(color=colors[idx], width=1.5),
                ),
                row=row,
                col=col,
            )
            fig.add_hline(
                y=0, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col
            )

        fig.update_layout(
            title_text=f"Raman Frequency Response: {self.response.spec.name}",
            height=700,
            showlegend=False,
        )
        return fig


# ─── RamanPulseInteraction (Layer 4) ──────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanPulseInteraction:
    """Pulse interaction with Raman-active medium.

    Computes the nonlinear polarization P_NL(t) = n₂ · (R(t) ⊗ I(t))
    where I(t) = |E(t)|² is the pulse intensity and R(t) is the combined
    Raman response from Layer 2.

    Attributes
    ----------
    pulse : Wave
        Input pulse (from pulse.py).
    response : RamanResponse
        Layer 2 Raman response function.
    spec : RamanSpec
        Layer 1 material properties (provides n₂).
    n2 : float
        Nonlinear refractive index n₂ (m²/W). Overrides spec.n2 if set.
    grid : TemporalGrid
        Time grid. Defaults to pulse.grid if not provided.
    """

    pulse: Wave
    response: RamanResponse
    spec: RamanSpec
    n2: float | None = None
    grid: TemporalGrid | None = None

    @model_validator(mode="after")
    def _ensure_defaults(self) -> RamanPulseInteraction:
        if self.grid is None:
            self.grid = self.pulse.grid
        if self.n2 is None:
            self.n2 = self.spec.n2 or 0.0
        return self

    @property
    def nonlinear_polarization(self) -> NDArray:
        """Nonlinear polarization P_NL(t) = n₂ · (R(t) ⊗ I(t)).

        Computed via FFT-based convolution on the FFTW backend
        (:func:`photonics_helper._fftw.convolve_full`), which reproduces
        ``scipy.signal.fftconvolve(..., mode='full')``.
        The result is cropped to the central N points to match the grid.

        Returns
        -------
        NDArray — nonlinear polarization values.
        """
        I_t = self.pulse.envelope_intensity
        R_t = self.response.combined_response(self.grid.t)  # type: ignore

        # Full convolution, then extract central N points
        P_full = _fftw_convolve_full(I_t, R_t)

        N = self.grid.N  # type: ignore
        if len(P_full) >= N:
            start = (len(P_full) - N) // 2
            P_NL = P_full[start : start + N]
        else:
            P_NL = np.zeros(N)
            P_NL[: len(P_full)] = P_full

        return self.n2 * P_NL

    def plot_interaction(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
        t_range_ps: tuple[float, float] | None = None,
    ):
        """4-panel plot: input pulse, Raman response, delayed polarization, output.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        figsize : Figure size for matplotlib
        t_range_ps : Time range in picoseconds as (t_min, t_max). Overrides grid.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_interaction_plotly(figsize, t_range_ps)
        else:
            return self._plot_interaction_matplotlib(figsize, t_range_ps)

    def _plot_interaction_matplotlib(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Matplotlib 4-panel plot."""
        import matplotlib.pyplot as plt

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
            I_t = (
                np.abs(
                    self.pulse.envelope_field[:1]
                    if False
                    else self.pulse.envelope_field
                )
                ** 2
            )
            # For custom time range, recompute intensity on the fly
            from photonics_helper.pulse import TemporalGrid

            custom_grid = TemporalGrid(N=len(t), Tmax=Time(t_max * 1e-12, "s"))
            custom_grid.t = t  # type: ignore[reportAttributeAccessIssue]  # override cached_property via instance dict
            I_t = np.abs(self.pulse.envelope_field) ** 2
            # Use grid-based computation for consistency
            t = self.grid.t  # type: ignore
        else:
            t = self.grid.t  # type: ignore

        I_t = self.pulse.envelope_intensity
        R_t = self.response.combined_response(t)
        P_NL = self.nonlinear_polarization

        # Simplified output: E_out = E_in + α·P_NL (where α is a coupling constant)
        # For visualization, we show the envelope of the output
        alpha_coupling = 0.1  # arbitrary coupling strength for visualization
        E_in = self.pulse.envelope_field
        E_out = E_in + alpha_coupling * P_NL
        I_out = np.abs(E_out) ** 2

        fig, axes = plt.subplots(4, 1, figsize=figsize or (10, 12), sharex=True)
        fig.suptitle(
            f"Pulse-Raman Interaction: {self.spec.name}  "
            f"(n₂={self.n2:.2e} m²/W, fR={self.response.fR:.2f})",
            fontsize=13,
            fontweight="bold",
        )

        t_ps = t * 1e12

        # Panel 1: Input pulse intensity
        ax1 = axes[0]
        ax1.plot(t_ps, I_t, color="#00d4ff", linewidth=1.5, label="|E(t)|²")
        ax1.set_ylabel("Intensity (arb.)", fontsize=10)
        ax1.set_title("Input Pulse Intensity |E(t)|²", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # Panel 2: Raman response
        ax2 = axes[1]
        ax2.plot(t_ps, R_t, color="#a78bfa", linewidth=1.5, label="R(t)")
        ax2.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax2.set_title("Combined Raman Response R(t)", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.legend(loc="upper right")

        # Panel 3: Nonlinear polarization
        ax3 = axes[2]
        ax3.plot(t_ps, P_NL, color="#34d399", linewidth=1.5, label="P_NL(t)")
        ax3.set_ylabel("P_NL (arb.)", fontsize=10)
        ax3.set_title("Nonlinear Polarization P_NL(t) = n₂·(R⊗I)", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color="k", linewidth=0.5)
        ax3.legend(loc="upper right")

        # Panel 4: Output pulse
        ax4 = axes[3]
        ax4.plot(
            t_ps, I_t, color="#00d4ff", linewidth=1.0, alpha=0.5, label="Input |E|²"
        )
        ax4.plot(t_ps, I_out, color="#fbbf24", linewidth=1.5, label="Output |E_out|²")
        ax4.set_xlabel("Time (ps)", fontsize=10)
        ax4.set_ylabel("Intensity (arb.)", fontsize=10)
        ax4.set_title(f"Output Pulse (α={alpha_coupling})", fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper right")

        plt.tight_layout()
        return fig

    def _plot_interaction_plotly(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Plotly 4-panel plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        t = self.grid.t  # type: ignore
        t_ps = t * 1e12
        I_t = self.pulse.envelope_intensity
        R_t = self.response.combined_response(t)
        P_NL = self.nonlinear_polarization

        alpha_coupling = 0.1
        E_in = self.pulse.envelope_field
        E_out = E_in + alpha_coupling * P_NL
        I_out = np.abs(E_out) ** 2

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Input Pulse Intensity |E(t)|²",
                "Combined Raman Response R(t)",
                "Nonlinear Polarization P_NL(t) = n₂·(R⊗I)",
                f"Output Pulse (α={alpha_coupling})",
            ),
            shared_xaxes=True,
            vertical_spacing=0.08,
        )

        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_t,
                mode="lines",
                name="|E|²",
                line=dict(color="#00d4ff", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=R_t,
                mode="lines",
                name="R(t)",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=P_NL,
                mode="lines",
                name="P_NL(t)",
                line=dict(color="#34d399", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_t,
                mode="lines",
                name="Input",
                opacity=0.5,
                line=dict(color="#00d4ff", width=1.0),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_out,
                mode="lines",
                name="Output",
                line=dict(color="#fbbf24", width=1.5),
            ),
            row=4,
            col=1,
        )

        for i in range(1, 5):
            fig.add_hline(
                y=0, line_dash="dot", line_color="gray", opacity=0.3, row=i, col=1
            )

        fig.update_yaxes(title_text="Intensity (arb.)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (arb.)", row=2, col=1)
        fig.update_yaxes(title_text="P_NL (arb.)", row=3, col=1)
        fig.update_yaxes(title_text="Intensity (arb.)", row=4, col=1)
        fig.update_xaxes(title_text="Time (ps)", row=4, col=1)

        fig.update_layout(
            title_text=f"Pulse-Raman Interaction: {self.spec.name}  "
            f"(n₂={self.n2:.2e} m²/W, fR={self.response.fR:.2f})",
            height=800,
            showlegend=True,
        )
        return fig

    def animate(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        frames: int = 20,
        figsize: tuple[float, float] | None = None,
    ):
        """Animation: pulse enters → instantaneous response → lattice oscillation → pulse exits.

        Shows the time evolution of the pulse and the induced nonlinear polarization.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        frames : number of animation frames (default 20)
        figsize : figure size for matplotlib
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._animate_plotly(frames, figsize)
        else:
            return self._animate_matplotlib(frames, figsize)

    def _animate_matplotlib(
        self,
        frames: int,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib animation."""
        import matplotlib.pyplot as plt
        from matplotlib import animation

        t = self.grid.t  # type: ignore
        t_ps = t * 1e12
        I_t = self.pulse.envelope_intensity
        P_NL = self.nonlinear_polarization

        fig, axes = plt.subplots(2, 1, figsize=figsize or (10, 8), sharex=True)
        fig.suptitle(
            f"Pulse-Raman Interaction Animation: {self.spec.name}",
            fontsize=13,
            fontweight="bold",
        )

        # Line objects for animation
        (line_pulse,) = axes[0].plot(t_ps, I_t, color="#00d4ff", linewidth=1.5)
        (line_PNL,) = axes[1].plot(
            t_ps, np.zeros_like(t_ps), color="#34d399", linewidth=1.5
        )

        axes[0].set_ylabel("Intensity (arb.)", fontsize=10)
        axes[0].set_title("Input Pulse", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, np.max(I_t) * 1.1)

        axes[1].set_ylabel("P_NL (arb.)", fontsize=10)
        axes[1].set_title("Nonlinear Polarization", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        pNL_max = np.max(np.abs(P_NL)) if np.max(np.abs(P_NL)) > 0 else 1
        axes[1].set_ylim(-pNL_max * 1.1, pNL_max * 1.1)

        axes[1].set_xlabel("Time (ps)", fontsize=10)

        # Animation frame function
        def update(frame):
            # Shift the pulse by a fraction of the grid
            shift = frame / frames
            # Create a shifted version of the intensity
            I_shifted = np.roll(I_t, int(shift * len(I_t)))
            PNL_shifted = np.roll(P_NL, int(shift * len(P_NL)))

            line_pulse.set_ydata(I_shifted)
            line_PNL.set_ydata(PNL_shifted)

            # Update title with frame info
            fig.suptitle(
                f"Pulse-Raman Interaction: {self.spec.name}  (frame {frame}/{frames})",
                fontsize=13,
                fontweight="bold",
            )
            return line_pulse, line_PNL

        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=100, blit=False
        )
        fig._animation = anim  # type: ignore[attr-defined]  # keep reference alive to prevent GC warning

        plt.tight_layout()
        return fig

    def _animate_plotly(
        self,
        frames: int,
        figsize: tuple[float, float] | None,
    ):
        """Plotly animation."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        t = self.grid.t  # type: ignore
        t_ps = t * 1e12
        I_t = self.pulse.envelope_intensity
        P_NL = self.nonlinear_polarization

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Input Pulse", "Nonlinear Polarization"),
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_t,
                mode="lines",
                name="Intensity",
                line=dict(color="#00d4ff", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=P_NL,
                mode="lines",
                name="P_NL",
                line=dict(color="#34d399", width=1.5),
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="Intensity (arb.)", row=1, col=1)
        fig.update_yaxes(title_text="P_NL (arb.)", row=2, col=1)
        fig.update_xaxes(title_text="Time (ps)", row=2, col=1)

        fig.update_layout(
            title_text=f"Pulse-Raman Interaction: {self.spec.name}",
            height=600,
            showlegend=False,
        )
        return fig
