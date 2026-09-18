"""PumpWavelengthExplorer and MaterialComparison visualisation helpers."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass

from ..base import (
    C_MS,
    Time,
    Wavelength,
)
from ..pulse import TemporalGrid
from .spec import RamanSpec

#: Whether the optional plotly backend is importable.  The plotly figures
#: themselves import ``plotly.graph_objects`` locally inside each method.
HAS_PLOTLY = find_spec("plotly") is not None

# ─── PumpWavelengthExplorer (Layer 5) ─────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class PumpWavelengthExplorer:
    """Stokes/Anti-Stokes analysis vs pump wavelength.

    Demonstrates that the Raman frequency shift is constant, but the
    corresponding wavelength shift depends on the pump wavelength.

    This is the key educational insight of Layer 5: equal frequency shifts
    produce unequal wavelength shifts because λ = c/ν (hyperbolic relationship).

    Attributes
    ----------
    spec : RamanSpec
        Layer 1 material properties.
    """

    spec: RamanSpec

    def pump_to_stokes(self, pump_wl: Wavelength) -> Wavelength:
        """Stokes wavelength for a given pump wavelength.

        ν_stokes = ν_pump − ν_Raman
        λ_stokes = c / ν_stokes

        Parameters
        ----------
        pump_wl : Pump wavelength.

        Returns
        -------
        Wavelength — Stokes wavelength.
        """
        return self.spec.stokes_wavelength(pump_wl)

    def pump_to_anti_stokes(self, pump_wl: Wavelength) -> Wavelength:
        """Anti-Stokes wavelength for a given pump wavelength.

        ν_anti_stokes = ν_pump + ν_Raman
        λ_anti_stokes = c / ν_anti_stokes

        Parameters
        ----------
        pump_wl : Pump wavelength.

        Returns
        -------
        Wavelength — Anti-Stokes wavelength.
        """
        return self.spec.anti_stokes_wavelength(pump_wl)

    def plot_frequency_axis(
        self,
        pump_wl: Wavelength,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        freq_range_THz: float = 50,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot frequency axis: Anti-Stokes | Pump | Stokes (equidistant).

        The frequency axis shows equal spacing between Anti-Stokes, Pump,
        and Stokes because the Raman shift is a constant frequency.

        Parameters
        ----------
        pump_wl : Pump wavelength.
        backend : "matplotlib" or "plotly"
        freq_range_THz : Range in THz to display on each side of pump.
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_freq_plotly(pump_wl, freq_range_THz, figsize)
        return self._plot_freq_matplotlib(pump_wl, freq_range_THz, figsize)

    def plot_wavelength_axis(
        self,
        pump_wl: Wavelength,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        wl_range_nm: float = 200,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot wavelength axis: Stokes — Pump — Anti-Stokes (unequal spacing).

        Demonstrates that equal frequency shifts produce unequal wavelength
        shifts due to the hyperbolic λ = c/ν relationship.

        Parameters
        ----------
        pump_wl : Pump wavelength.
        backend : "matplotlib" or "plotly"
        wl_range_nm : Range in nm to display around pump.
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_wl_plotly(pump_wl, wl_range_nm, figsize)
        return self._plot_wl_matplotlib(pump_wl, wl_range_nm, figsize)

    def plot_both(
        self,
        pump_wl: Wavelength,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        freq_range_THz: float = 50,
        wl_range_nm: float = 200,
        figsize: tuple[float, float] | None = None,
    ):
        """Side-by-side: frequency axis vs wavelength axis.

        Parameters
        ----------
        pump_wl : Pump wavelength.
        backend : "matplotlib" or "plotly"
        freq_range_THz : Frequency range in THz.
        wl_range_nm : Wavelength range in nm.
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_both_plotly(pump_wl, freq_range_THz, wl_range_nm, figsize)
        return self._plot_both_matplotlib(pump_wl, freq_range_THz, wl_range_nm, figsize)

    def plot_vs_pump(
        self,
        pump_range_um: tuple[float, float] = (0.4, 2.0),
        n_points: int = 100,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Sweep pump wavelength: show how Stokes/Anti-Stokes shift changes.

        Parameters
        ----------
        pump_range_um : (min, max) pump wavelength in μm.
        n_points : Number of pump wavelengths to sweep.
        backend : "matplotlib" or "plotly"
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_vs_pump_plotly(pump_range_um, n_points, figsize)
        return self._plot_vs_pump_matplotlib(pump_range_um, n_points, figsize)

    # ── Matplotlib implementations ──

    def _plot_freq_matplotlib(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        figsize: tuple[float, float] | None,
    ):
        """Frequency axis plot (equidistant markers)."""
        import matplotlib.pyplot as plt

        nu_pump = C_MS / pump_wl.as_m / 1e12  # THz
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        fig, ax = plt.subplots(figsize=figsize or (10, 5))

        # Draw a frequency axis line
        ax.hlines(
            0,
            nu_pump - freq_range_THz,
            nu_pump + freq_range_THz,
            colors="gray",
            linewidth=2,
        )

        # Mark Anti-Stokes (blue), Pump (red), Stokes (green)
        ax.vlines(
            nu_anti,
            -0.1,
            0.15,
            colors="#3b82f6",
            linewidth=3,
            label=f"Anti-Stokes ({C_MS / nu_anti * 1e12:.0f} nm)",
        )
        ax.vlines(
            nu_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({pump_wl.as_nm:.0f} nm)",
        )
        ax.vlines(
            nu_stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"Stokes ({C_MS / nu_stokes * 1e12:.0f} nm)",
        )

        # Annotations
        ax.annotate(
            "Anti-Stokes\n(higher ν)",
            xy=(nu_anti, 0.18),
            ha="center",
            fontsize=9,
            color="#3b82f6",
        )
        ax.annotate(
            "Pump", xy=(nu_pump, -0.18), ha="center", fontsize=9, color="#ef4444"
        )
        ax.annotate(
            "Stokes\n(lower ν)",
            xy=(nu_stokes, 0.18),
            ha="center",
            fontsize=9,
            color="#22c55e",
        )

        # Mark the equal shift
        ax.annotate(
            f"Δν = {self.spec.raman_shift_THz:.2f} THz",
            xy=((nu_pump + nu_stokes) / 2, -0.05),
            ha="center",
            fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )

        ax.set_xlim(nu_pump - freq_range_THz, nu_pump + freq_range_THz)
        ax.set_ylim(-0.3, 0.35)
        ax.set_xlabel("Frequency (THz)", fontsize=12)
        ax.set_title(
            f"Frequency Axis: {self.spec.name}  "
            f"(Δν = {self.spec.raman_shift_THz:.2f} THz = {self.spec.raman_shift_cm:.0f} cm⁻¹)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_yticks([])
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def _plot_wl_matplotlib(
        self,
        pump_wl: Wavelength,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Wavelength axis plot (unequal spacing)."""
        import matplotlib.pyplot as plt

        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        fig, ax = plt.subplots(figsize=figsize or (10, 5))

        # Draw a wavelength axis line
        ax.hlines(0, anti - 50, stokes + 50, colors="gray", linewidth=2)

        # Mark Anti-Stokes (blue), Pump (red), Stokes (green)
        ax.vlines(
            anti,
            -0.1,
            0.15,
            colors="#3b82f6",
            linewidth=3,
            label=f"Anti-Stokes ({anti:.0f} nm)",
        )
        ax.vlines(
            wl_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({wl_pump:.0f} nm)",
        )
        ax.vlines(
            stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"Stokes ({stokes:.0f} nm)",
        )

        # Annotations
        ax.annotate(
            "Anti-Stokes\n(shorter λ)",
            xy=(anti, 0.18),
            ha="center",
            fontsize=9,
            color="#3b82f6",
        )
        ax.annotate(
            "Pump", xy=(wl_pump, -0.18), ha="center", fontsize=9, color="#ef4444"
        )
        ax.annotate(
            "Stokes\n(longer λ)",
            xy=(stokes, 0.18),
            ha="center",
            fontsize=9,
            color="#22c55e",
        )

        # Mark unequal shifts
        delta_stokes = stokes - wl_pump
        delta_anti = wl_pump - anti
        ax.annotate(
            f"Δλ_S = {delta_stokes:.1f} nm",
            xy=((wl_pump + stokes) / 2, -0.05),
            ha="center",
            fontsize=8,
            color="#22c55e",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )
        ax.annotate(
            f"Δλ_AS = {delta_anti:.1f} nm",
            xy=((anti + wl_pump) / 2, -0.12),
            ha="center",
            fontsize=8,
            color="#3b82f6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

        ax.set_xlim(anti - 50, stokes + 50)
        ax.set_ylim(-0.3, 0.35)
        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_title(
            f"Wavelength Axis: {self.spec.name}  "
            f"(Pump = {wl_pump:.0f} nm — unequal spacing!)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_yticks([])
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def _plot_both_matplotlib(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Side-by-side frequency and wavelength axis plots."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize or (16, 5), gridspec_kw={"width_ratios": [1, 1]}
        )
        fig.suptitle(
            f"Pump Wavelength Explorer: {self.spec.name}  "
            f"(Pump = {pump_wl.as_nm:.0f} nm)",
            fontsize=13,
            fontweight="bold",
        )

        # Frequency axis (left)
        nu_pump = C_MS / pump_wl.as_m / 1e12
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        ax1.hlines(
            0,
            nu_pump - freq_range_THz,
            nu_pump + freq_range_THz,
            colors="gray",
            linewidth=2,
        )
        ax1.vlines(
            nu_anti,
            -0.1,
            0.15,
            colors="#3b82f6",
            linewidth=3,
            label=f"AS ({C_MS / nu_anti * 1e12:.0f} nm)",
        )
        ax1.vlines(
            nu_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({pump_wl.as_nm:.0f} nm)",
        )
        ax1.vlines(
            nu_stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"S ({C_MS / nu_stokes * 1e12:.0f} nm)",
        )
        ax1.set_xlabel("Frequency (THz)", fontsize=11)
        ax1.set_title("Frequency Axis (equidistant)", fontsize=11)
        ax1.set_yticks([])
        ax1.set_xlim(nu_pump - freq_range_THz, nu_pump + freq_range_THz)
        ax1.set_ylim(-0.3, 0.35)
        ax1.grid(True, alpha=0.3, axis="x")
        ax1.legend(fontsize=8, loc="upper right")

        # Wavelength axis (right)
        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        ax2.hlines(0, anti - 50, stokes + 50, colors="gray", linewidth=2)
        ax2.vlines(
            anti, -0.1, 0.15, colors="#3b82f6", linewidth=3, label=f"AS ({anti:.0f} nm)"
        )
        ax2.vlines(
            wl_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({wl_pump:.0f} nm)",
        )
        ax2.vlines(
            stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"S ({stokes:.0f} nm)",
        )
        ax2.set_xlabel("Wavelength (nm)", fontsize=11)
        ax2.set_title("Wavelength Axis (unequal spacing!)", fontsize=11)
        ax2.set_yticks([])
        ax2.set_xlim(anti - 50, stokes + 50)
        ax2.set_ylim(-0.3, 0.35)
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        return fig

    def _plot_vs_pump_matplotlib(
        self,
        pump_range_um: tuple[float, float],
        n_points: int,
        figsize: tuple[float, float] | None,
    ):
        """Sweep pump wavelength: Stokes/Anti-Stokes vs pump."""
        import matplotlib.pyplot as plt

        pump_wls_um = np.linspace(pump_range_um[0], pump_range_um[1], n_points)
        stokes_wls_um = []
        anti_wls_um = []

        for wl_um in pump_wls_um:
            pump = Wavelength(wl_um, "um")
            stokes_wls_um.append(self.pump_to_stokes(pump).as_um)
            anti_wls_um.append(self.pump_to_anti_stokes(pump).as_um)

        fig, ax = plt.subplots(figsize=figsize or (10, 6))

        ax.plot(
            pump_wls_um, stokes_wls_um, color="#22c55e", linewidth=2, label="Stokes"
        )
        ax.plot(
            pump_wls_um,
            pump_wls_um,
            color="#ef4444",
            linewidth=1.5,
            linestyle="--",
            label="Pump = Stokes (identity)",
        )
        ax.plot(
            pump_wls_um, anti_wls_um, color="#3b82f6", linewidth=2, label="Anti-Stokes"
        )

        # Shade the Raman shift regions
        ax.fill_between(
            pump_wls_um,
            pump_wls_um,
            stokes_wls_um,
            alpha=0.15,
            color="#22c55e",
            label="Stokes shift",
        )

        ax.set_xlabel("Pump Wavelength (μm)", fontsize=12)
        ax.set_ylabel("Signal Wavelength (μm)", fontsize=12)
        ax.set_title(
            f"Stokes/Anti-Stokes vs Pump: {self.spec.name}  "
            f"(Δν̃ = {self.spec.raman_shift_cm:.0f} cm⁻¹)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_aspect("auto")

        plt.tight_layout()
        return fig

    # ── Plotly implementations ──

    def _plot_freq_plotly(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        figsize: tuple[float, float] | None,
    ):
        """Frequency axis plot (plotly)."""
        import plotly.graph_objects as go

        nu_pump = C_MS / pump_wl.as_m / 1e12
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        fig = go.Figure()

        # Use add_shape for the axis line (xrange not supported)
        fig.add_shape(
            type="line",
            x0=nu_pump - freq_range_THz,
            x1=nu_pump + freq_range_THz,
            y0=0,
            y1=0,
            line=dict(color="gray", width=3),
        )

        fig.add_vline(
            x=nu_anti,
            y1=0.15,
            line=dict(color="#3b82f6", width=3),
            annotation_text=f"AS ({C_MS / nu_anti * 1e12:.0f} nm)",
        )
        fig.add_vline(
            x=nu_pump,
            y1=0.15,
            line=dict(color="#ef4444", width=3),
            annotation_text=f"Pump ({pump_wl.as_nm:.0f} nm)",
        )
        fig.add_vline(
            x=nu_stokes,
            y1=0.15,
            line=dict(color="#22c55e", width=3),
            annotation_text=f"S ({C_MS / nu_stokes * 1e12:.0f} nm)",
        )

        fig.update_layout(
            title=f"Frequency Axis: {self.spec.name} (Δν = {self.spec.raman_shift_THz:.2f} THz)",
            xaxis_title="Frequency (THz)",
            yaxis=dict(showticklabels=False, showgrid=False),
            height=300,
            showlegend=False,
        )
        return fig

    def _plot_wl_plotly(
        self,
        pump_wl: Wavelength,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Wavelength axis plot (plotly)."""
        import plotly.graph_objects as go

        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        fig = go.Figure()

        fig.add_shape(
            type="line",
            x0=anti - 50,
            x1=stokes + 50,
            y0=0,
            y1=0,
            line=dict(color="gray", width=3),
        )
        fig.add_vline(
            x=anti,
            y1=0.15,
            line=dict(color="#3b82f6", width=3),
            annotation_text=f"AS ({anti:.0f} nm)",
        )
        fig.add_vline(
            x=wl_pump,
            y1=0.15,
            line=dict(color="#ef4444", width=3),
            annotation_text=f"Pump ({wl_pump:.0f} nm)",
        )
        fig.add_vline(
            x=stokes,
            y1=0.15,
            line=dict(color="#22c55e", width=3),
            annotation_text=f"S ({stokes:.0f} nm)",
        )

        fig.update_layout(
            title=f"Wavelength Axis: {self.spec.name} (Pump = {wl_pump:.0f} nm)",
            xaxis_title="Wavelength (nm)",
            yaxis=dict(showticklabels=False, showgrid=False),
            height=300,
            showlegend=False,
        )
        return fig

    def _plot_both_plotly(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Side-by-side frequency and wavelength axis (plotly)."""
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Frequency Axis: {self.spec.name}",
                f"Wavelength Axis: {self.spec.name}",
            ),
        )

        # Frequency axis
        nu_pump = C_MS / pump_wl.as_m / 1e12
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        fig.add_shape(
            type="line",
            x0=nu_pump - freq_range_THz,
            x1=nu_pump + freq_range_THz,
            y0=0,
            y1=0,
            line=dict(color="gray", width=2),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=nu_anti, y1=0.15, line=dict(color="#3b82f6", width=3), row=1, col=1
        )
        fig.add_vline(
            x=nu_pump, y1=0.15, line=dict(color="#ef4444", width=3), row=1, col=1
        )
        fig.add_vline(
            x=nu_stokes, y1=0.15, line=dict(color="#22c55e", width=3), row=1, col=1
        )

        # Wavelength axis
        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        fig.add_shape(
            type="line",
            x0=anti - 50,
            x1=stokes + 50,
            y0=0,
            y1=0,
            line=dict(color="gray", width=2),
            row=1,
            col=2,
        )
        fig.add_vline(
            x=anti, y1=0.15, line=dict(color="#3b82f6", width=3), row=1, col=2
        )
        fig.add_vline(
            x=wl_pump, y1=0.15, line=dict(color="#ef4444", width=3), row=1, col=2
        )
        fig.add_vline(
            x=stokes, y1=0.15, line=dict(color="#22c55e", width=3), row=1, col=2
        )

        fig.update_layout(height=350, showlegend=False)
        return fig

    def _plot_vs_pump_plotly(
        self,
        pump_range_um: tuple[float, float],
        n_points: int,
        figsize: tuple[float, float] | None,
    ):
        """Sweep pump wavelength (plotly)."""
        import plotly.graph_objects as go

        pump_wls_um = np.linspace(pump_range_um[0], pump_range_um[1], n_points)
        stokes_wls_um = []
        anti_wls_um = []

        for wl_um in pump_wls_um:
            pump = Wavelength(wl_um, "um")
            stokes_wls_um.append(self.pump_to_stokes(pump).as_um)
            anti_wls_um.append(self.pump_to_anti_stokes(pump).as_um)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pump_wls_um,
                y=stokes_wls_um,
                mode="lines",
                name="Stokes",
                line=dict(color="#22c55e", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pump_wls_um,
                y=pump_wls_um,
                mode="lines",
                name="Pump = Stokes (identity)",
                line=dict(color="#ef4444", width=1.5, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pump_wls_um,
                y=anti_wls_um,
                mode="lines",
                name="Anti-Stokes",
                line=dict(color="#3b82f6", width=2),
            )
        )

        fig.update_layout(
            title=f"Stokes/Anti-Stokes vs Pump: {self.spec.name}  "
            f"(Δν̃ = {self.spec.raman_shift_cm:.0f} cm⁻¹)",
            xaxis_title="Pump Wavelength (μm)",
            yaxis_title="Signal Wavelength (μm)",
            height=500,
        )
        return fig


# ─── MaterialComparison (Layer 6) ─────────────────────────────────────────────


# Preset comparison groups for quick multi-material visualization
COMMON_COMPARISONS = {
    "glass_vs_chalcogenide": ["Silica", "As2Se3"],
    "semiconductor": ["CdS", "GaAs", "Si", "Ge"],
    "high_n2": ["Silica", "As2Se3", "Diamond"],
    "high_shift": ["Silica", "Diamond", "Si"],
    "nitride_semiconductor": ["GaN", "AlN", "Si3N4"],
    "nonlinear_crystal": ["LiNbO3", "KTP", "LBO", "BaTiO3"],
    "high_gain": ["Diamond", "As2Se3", "YAG"],
    "wide_bandgap": ["Diamond", "GaN", "SiC_4H", "AlN", "Ga2O3"],
    "iii_v": ["GaAs", "InP", "AlGaAs", "InGaAs"],
    "laser_host": ["YAG", "Al2O3", "YLF"],
    "nlo_crystal": ["LBO", "KTP", "AgGaS2", "AgGaSe2", "LiNbO3"],
    "chalcogenide": ["As2S3", "As2Se3"],
    "ferroelectric": ["LiNbO3", "LiTaO3", "BaTiO3", "KTP"],
    "negative_n2": ["ZnO", "CdTe"],
}


# Color palette for overlay plots (8 distinct colors)
_OVERLAY_COLORS = [
    "#00d4ff",  # cyan
    "#a78bfa",  # purple
    "#34d399",  # green
    "#fbbf24",  # amber
    "#f87171",  # red
    "#fb923c",  # orange
    "#38bdf8",  # sky blue
    "#c084fc",  # violet
]


@dataclass(config={"arbitrary_types_allowed": True})
class MaterialComparison:
    """Multi-material Raman comparison overlay.

    Overlays Raman spectra, time-domain responses, and frequency-domain
    responses for multiple materials on shared axes for direct comparison.

    Attributes
    ----------
    materials : list[RamanSpec]
        List of RamanSpec instances to compare.
    """

    materials: list[RamanSpec] = Field(default_factory=list)

    def add(self, spec: RamanSpec) -> None:
        """Add a material, replacing if a material with the same name exists.

        Parameters
        ----------
        spec : RamanSpec
            Material to add.
        """
        # Remove existing material with the same name
        self.materials = [m for m in self.materials if m.name != spec.name]
        self.materials.append(spec)

    def remove(self, name: str) -> None:
        """Remove a material by name.

        Parameters
        ----------
        name : str
            Material name to remove.
        """
        self.materials = [m for m in self.materials if m.name != name]

    def clear(self) -> None:
        """Remove all materials."""
        self.materials.clear()

    def _color_for_index(self, idx: int) -> str:
        """Get a color for a material index from the palette."""
        return _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]

    def plot_spectra_overlay(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        shift_range_cm: float = 200,
        n_points: int = 1000,
        figsize: tuple[float, float] | None = None,
    ):
        """Overlay Raman spectra for all materials.

        Each material is plotted as a Lorentzian lineshape centered at its
        Raman shift, normalized to unit peak height.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        shift_range_cm : Range around 0 to plot (cm⁻¹)
        n_points : Number of points
        figsize : Figure size for matplotlib

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_spectra_plotly(shift_range_cm, n_points)
        return self._plot_spectra_matplotlib(shift_range_cm, n_points, figsize)

    def _plot_spectra_matplotlib(
        self,
        shift_range_cm: float,
        n_points: int,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib spectra overlay."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue

            color = self._color_for_index(idx)

            # Use PhononResponse for multi-mode materials
            if spec.phonon_modes:
                pr = spec.phonon_response
                if pr is None:
                    continue
                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = pr.frequency_domain(shift)
                ax.plot(
                    shift,
                    intensity,
                    linewidth=2,
                    color=color,
                    label=f"{spec.name} ({len(spec.phonon_modes)} modes)",
                )
            else:
                center = spec.raman_shift_cm
                width = spec.raman_linewidth_cm / 2

                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
                intensity /= np.max(intensity)

                ax.plot(shift, intensity, linewidth=2, color=color, label=spec.name)

            name_labels.append(spec.name)

        ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (arb.)", fontsize=12)
        ax.set_title(
            f"Raman Spectra Comparison: {', '.join(name_labels)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _plot_spectra_plotly(self, shift_range_cm: float, n_points: int):
        """Plotly spectra overlay."""
        import plotly.graph_objects as go

        fig = go.Figure()
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue

            color = self._color_for_index(idx)

            # Use PhononResponse for multi-mode materials
            if spec.phonon_modes:
                pr = spec.phonon_response
                if pr is None:
                    continue
                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = pr.frequency_domain(shift)
                label = f"{spec.name} ({len(spec.phonon_modes)} modes)"
            else:
                center = spec.raman_shift_cm
                width = spec.raman_linewidth_cm / 2

                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
                intensity /= np.max(intensity)
                label = spec.name

            fig.add_trace(
                go.Scatter(
                    x=shift,
                    y=intensity,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                )
            )
            name_labels.append(spec.name)

        fig.update_layout(
            title=f"Raman Spectra Comparison: {', '.join(name_labels)}",
            xaxis_title="Raman shift (cm⁻¹)",
            yaxis_title="Intensity (arb.)",
            height=500,
        )
        return fig

    def plot_response_overlay(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        grid: TemporalGrid | None = None,
        figsize: tuple[float, float] | None = None,
        t_range_ps: tuple[float, float] | None = None,
    ):
        """Overlay h_R(t) for all materials.

        Computes the delayed Raman response for each material and overlays
        them on a shared time axis.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        grid : TemporalGrid — shared time grid. Auto-derived if None.
        figsize : Figure size for matplotlib
        t_range_ps : Time range in picoseconds as (t_min, t_max). Overrides grid.

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if grid is None:
            # Use the longest τ2 to determine grid size
            max_tau2 = max(
                (1.0 / (np.pi * s.linewidth_Hz) if s.linewidth_Hz > 0 else 1e-12)
                for s in self.materials
            )
            grid = TemporalGrid(N=2**14, Tmax=Time(max(10e-12, 20 * max_tau2), "s"))

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_response_plotly(grid, t_range_ps)
        return self._plot_response_matplotlib(grid, figsize, t_range_ps)

    @staticmethod
    def _compute_h_R(spec: RamanSpec, t: NDArray) -> NDArray:
        """Compute delayed Raman response fR·h_R(t) for a single material.

        Standard Agrawal exponential-damped form:
        h_R(t) = (τ₁² + τ₂²)/(τ₁·τ₂²) · exp(-t/τ₂) · sin(t/τ₁)   for t ≥ 0
        h_R(t) = 0                                                  for t < 0

        Parameters
        ----------
        spec : RamanSpec
            Material with raman_shift_Hz, linewidth_Hz, fR.
        t : NDArray
            Time array.

        Returns
        -------
        NDArray — delayed response fR·h_R(t).
        """
        tau1 = 1.0 / (2 * np.pi * spec.raman_shift_Hz)
        tau2 = 1.0 / (np.pi * spec.linewidth_Hz) if spec.linewidth_Hz > 0 else 1e-12
        prefactor = (tau1**2 + tau2**2) / (tau1 * tau2**2)
        exponential = np.exp(-t / tau2)
        oscillation = np.sin(t / tau1)
        mask = t >= 0
        delayed = np.zeros_like(t)
        raw_h = prefactor * exponential[mask] * oscillation[mask]
        integral = np.trapezoid(raw_h, t[mask])
        if integral > 0:
            raw_h /= integral
        delayed[mask] = float(spec.fR or 0.0) * raw_h
        return delayed

    def _plot_response_matplotlib(
        self,
        grid: TemporalGrid,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Matplotlib response overlay."""
        import matplotlib.pyplot as plt

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = grid.t

        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            delayed = self._compute_h_R(spec, t)

            # Normalize to peak for visibility
            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak

            ax.plot(t * 1e12, delayed, linewidth=1.5, color=color, label=spec.name)
            name_labels.append(spec.name)

        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_ylabel("Delayed response h_R(t) (normalized)", fontsize=11)
        ax.set_title(
            f"Raman Response Overlay: {', '.join(name_labels)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        return fig

    def _plot_response_plotly(
        self, grid: TemporalGrid, t_range_ps: tuple[float, float] | None
    ):
        """Plotly response overlay."""
        import plotly.graph_objects as go

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = grid.t

        fig = go.Figure()
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            delayed = self._compute_h_R(spec, t)

            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak

            fig.add_trace(
                go.Scatter(
                    x=t * 1e12,
                    y=delayed,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                )
            )
            name_labels.append(spec.name)

        fig.update_layout(
            title=f"Raman Response Overlay: {', '.join(name_labels)}",
            xaxis_title="Time (ps)",
            yaxis_title="Delayed response h_R(t) (normalized)",
            height=500,
        )
        return fig

    def plot_frequency_overlay(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        grid: TemporalGrid | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Overlay Im(H(Ω)) for all materials.

        The imaginary part of the Fourier transform gives the Raman gain
        spectrum. Overlapping multiple materials shows how their gain
        profiles compare.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        grid : TemporalGrid — shared time grid. Auto-derived if None.
        figsize : Figure size for matplotlib

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if grid is None:
            max_tau2 = max(
                (1.0 / (np.pi * s.linewidth_Hz) if s.linewidth_Hz > 0 else 1e-12)
                for s in self.materials
            )
            grid = TemporalGrid(N=2**14, Tmax=Time(max(10e-12, 20 * max_tau2), "s"))

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_frequency_plotly(grid)
        return self._plot_frequency_matplotlib(grid, figsize)

    def _plot_frequency_matplotlib(
        self,
        grid: TemporalGrid,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib frequency overlay."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            # Compute h_R(t) and FFT
            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)

            w_THz = grid.w / (2 * np.pi * 1e12)
            ax.plot(w_THz, H_imag, linewidth=1.5, color=color, label=spec.name)
            name_labels.append(spec.name)

            # Annotate resonance frequency
            if np.max(np.abs(H_imag)) > 0:
                f_res = w_THz[np.argmax(np.abs(H_imag))]
                ax.axvline(f_res, color=color, linestyle=":", alpha=0.5, linewidth=0.8)

        ax.set_xlabel("Angular frequency (THz)", fontsize=12)
        ax.set_ylabel("Im(H(Ω))", fontsize=12)
        ax.set_title(
            f"Raman Gain Spectrum: {', '.join(name_labels)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        return fig

    def _plot_frequency_plotly(
        self,
        grid: TemporalGrid,
    ):
        """Plotly frequency overlay."""
        import plotly.graph_objects as go

        fig = go.Figure()
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)

            w_THz = grid.w / (2 * np.pi * 1e12)
            fig.add_trace(
                go.Scatter(
                    x=w_THz,
                    y=H_imag,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                )
            )
            name_labels.append(spec.name)

        fig.update_layout(
            title=f"Raman Gain Spectrum: {', '.join(name_labels)}",
            xaxis_title="Angular frequency (THz)",
            yaxis_title="Im(H(Ω))",
            height=500,
        )
        return fig

    def comparison_table(self) -> str:
        """Text table comparing Raman properties across all materials.

        Returns
        -------
        str — formatted comparison table.
        """
        if not self.materials:
            return "No materials in comparison."

        # Header
        header = (
            f"{'Material':<12} {'n₂ (m²/W)':>14} {'fR':>6} "
            f"{'Shift (cm⁻¹)':>13} {'FWHM (cm⁻¹)':>13} "
            f"{'τ1 (fs)':>9} {'τ2 (fs)':>9} {'Q':>8}"
        )
        separator = "─" * len(header)

        lines = [header, separator]

        for spec in self.materials:
            n2_str = f"{spec.n2:.2e}" if spec.n2 else "N/A"
            fr_str = f"{spec.fR:.2f}" if spec.fR is not None else "N/A"
            shift_str = f"{spec.raman_shift_cm:.0f}" if spec.raman_shift_cm else "N/A"
            fwhm_str = (
                f"{spec.raman_linewidth_cm:.0f}" if spec.raman_linewidth_cm else "N/A"
            )

            if spec.raman_shift_Hz > 0:
                tau1_fs = (1.0 / spec.raman_shift_Hz) * 1e15
                tau1_str = f"{tau1_fs:.1f}"
            else:
                tau1_str = "N/A"

            if spec.linewidth_Hz > 0:
                tau2_fs = (1.0 / (np.pi * spec.linewidth_Hz)) * 1e15
                tau2_str = f"{tau2_fs:.1f}"
            else:
                tau2_str = "N/A"

            if spec.quality_factor != float("inf") and spec.quality_factor != 0:
                q_str = f"{spec.quality_factor:.1f}"
            else:
                q_str = "∞" if spec.raman_shift_Hz > 0 else "N/A"

            lines.append(
                f"{spec.name:<12} {n2_str:>14} {fr_str:>6} "
                f"{shift_str:>13} {fwhm_str:>13} "
                f"{tau1_str:>9} {tau2_str:>9} {q_str:>8}"
            )

        return "\n".join(lines)

    def plot_all(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        grid: TemporalGrid | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Full comparison: spectra + response + frequency for all materials.

        3-panel layout:
        1. Raman spectra overlay
        2. Time-domain response overlay
        3. Frequency-domain gain spectrum overlay

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        grid : TemporalGrid — shared time grid. Auto-derived if None.
        figsize : Figure size for matplotlib

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if grid is None:
            max_tau2 = max(
                (1.0 / (np.pi * s.linewidth_Hz) if s.linewidth_Hz > 0 else 1e-12)
                for s in self.materials
            )
            grid = TemporalGrid(N=2**14, Tmax=Time(max(10e-12, 20 * max_tau2), "s"))

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_all_plotly(grid)
        return self._plot_all_matplotlib(grid, figsize)

    def _plot_all_matplotlib(
        self,
        grid: TemporalGrid,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib 3-panel comparison."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=figsize or (12, 12), sharex=False)
        fig.suptitle(
            f"Raman Material Comparison: {', '.join(s.name for s in self.materials)}",
            fontsize=14,
            fontweight="bold",
        )

        # Panel 1: Spectra
        axes[0].set_title("Raman Spectra", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
        axes[0].set_ylabel("Intensity (arb.)", fontsize=10)
        # Reuse spectra logic inline for the subplot
        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue
            color = self._color_for_index(idx)
            center = spec.raman_shift_cm
            width = spec.raman_linewidth_cm / 2
            shift = np.linspace(-200, 200, 1000)
            intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
            intensity /= np.max(intensity)
            axes[0].plot(shift, intensity, linewidth=2, color=color, label=spec.name)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Response
        axes[1].set_title("Time-Domain Response h_R(t)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Time (ps)", fontsize=10)
        axes[1].set_ylabel("Amplitude (normalized)", fontsize=10)
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            delayed = self._compute_h_R(spec, grid.t)
            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak
            axes[1].plot(
                grid.t * 1e12, delayed, linewidth=1.5, color=color, label=spec.name
            )
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color="k", linewidth=0.5)
        axes[1].legend(fontsize=9)

        # Panel 3: Frequency
        axes[2].set_title(
            "Raman Gain Spectrum Im(H(Ω))", fontsize=12, fontweight="bold"
        )
        axes[2].set_xlabel("Angular frequency (THz)", fontsize=10)
        axes[2].set_ylabel("Im(H(Ω))", fontsize=10)
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)
            w_THz = grid.w / (2 * np.pi * 1e12)
            axes[2].plot(w_THz, H_imag, linewidth=1.5, color=color, label=spec.name)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color="k", linewidth=0.5)
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        return fig

    def _plot_all_plotly(self, grid: TemporalGrid):
        """Plotly 3-panel comparison."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Raman Spectra",
                "Time-Domain Response h_R(t)",
                "Raman Gain Spectrum Im(H(Ω))",
            ),
            shared_xaxes=False,
            vertical_spacing=0.1,
        )

        # Panel 1: Spectra
        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue
            color = self._color_for_index(idx)
            center = spec.raman_shift_cm
            width = spec.raman_linewidth_cm / 2
            shift = np.linspace(-200, 200, 1000)
            intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
            intensity /= np.max(intensity)
            fig.add_trace(
                go.Scatter(
                    x=shift,
                    y=intensity,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=2),
                ),
                row=1,
                col=1,
            )

        # Panel 2: Response
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            delayed = self._compute_h_R(spec, grid.t)
            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak
            fig.add_trace(
                go.Scatter(
                    x=grid.t * 1e12,
                    y=delayed,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                ),
                row=2,
                col=1,
            )

        # Panel 3: Frequency
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)
            w_THz = grid.w / (2 * np.pi * 1e12)
            fig.add_trace(
                go.Scatter(
                    x=w_THz,
                    y=H_imag,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                ),
                row=3,
                col=1,
            )

        fig.update_layout(height=900, showlegend=True)
        return fig
