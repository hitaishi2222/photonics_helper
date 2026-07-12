"""Phonon module — multi-mode Raman-active phonon modeling.

Provides PhononMode dataclass and PhononResponse class for combining
multiple Raman-active phonon modes into a unified response function.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from pydantic.dataclasses import dataclass

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ─── PhononMode ───────────────────────────────────────────────────────────────

@dataclass
class PhononMode:
    """A single Raman-active phonon mode.

    Attributes
    ----------
    shift_cm : float
        Raman shift in cm⁻¹.
    linewidth_cm : float
        Full width at half maximum in cm⁻¹.
    symmetry : str or None
        Symmetry label (e.g. "A₁g", "E_g", "A₁(TO)").
    relative_strength : float
        Relative Raman cross-section (default 1.0).
    lo_phonon_cm : float or None
        LO phonon frequency in cm⁻¹ (if different from Raman shift).
    to_phonon_cm : float or None
        TO phonon frequency in cm⁻¹.
    note : str or None
        Annotation or source reference.
    """

    shift_cm: float
    linewidth_cm: float
    symmetry: str | None = None
    relative_strength: float = 1.0
    lo_phonon_cm: float | None = None
    to_phonon_cm: float | None = None
    note: str | None = None


# ─── PhononResponse ───────────────────────────────────────────────────────────

@dataclass(config={"arbitrary_types_allowed": True})
class PhononResponse:
    """Multi-mode Raman response function.

    Combines multiple PhononMode instances into a unified response by summing
    individual mode responses weighted by their relative strengths.

    Each mode produces a Lorentzian lineshape in frequency domain:
        L(ω) = (γ/2) / [(ω - ω₀)² + (γ/2)²]
    where ω₀ is the mode shift and γ is the linewidth.

    The time-domain response is the inverse FFT of the frequency-domain response.

    Attributes
    ----------
    modes : list[PhononMode]
        List of phonon modes.
    fR : float
        Total Raman fraction. If None, computed as sum of relative strengths.
    """

    modes: list[PhononMode]
    fR: float | None = None

    def __post_init__(self):
        if self.fR is None:
            object.__setattr__(self, "fR", sum(m.relative_strength for m in self.modes))

    def _lorentzian(self, shift_cm: float, gamma_cm: float, w_cm: np.ndarray) -> np.ndarray:
        """Single Lorentzian lineshape.

        Parameters
        ----------
        shift_cm : float
            Center shift in cm⁻¹.
        gamma_cm : float
            FWHM in cm⁻¹.
        w_cm : np.ndarray
            Frequency axis in cm⁻¹.

        Returns
        -------
        np.ndarray — Lorentzian values (unnormalized).
        """
        half_gamma = gamma_cm / 2.0
        return (half_gamma / np.pi) / ((w_cm - shift_cm) ** 2 + half_gamma ** 2)

    def frequency_domain(self, w_cm: np.ndarray) -> np.ndarray:
        """Frequency-domain response: sum of Lorentzian mode lineshapes.

        Parameters
        ----------
        w_cm : np.ndarray
            Frequency axis in cm⁻¹.

        Returns
        -------
        np.ndarray — Total response (sum of weighted Lorentzians).
        """
        if not self.modes:
            return np.zeros_like(w_cm)

        total = np.zeros_like(w_cm, dtype=float)
        for mode in self.modes:
            lorentzian = self._lorentzian(mode.shift_cm, mode.linewidth_cm, w_cm)
            total += mode.relative_strength * lorentzian

        # Normalize to peak = 1
        if np.max(total) > 0:
            total /= np.max(total)

        return total

    def time_domain(self, t: np.ndarray) -> np.ndarray:
        """Time-domain response via inverse FFT of frequency-domain response.

        The time-domain response is the inverse FFT of the frequency-domain
        response, representing the delayed Raman response in the time domain.

        Parameters
        ----------
        t : np.ndarray
            Time array in seconds.

        Returns
        -------
        np.ndarray — Time-domain response (sum of damped oscillations).
        """
        if not self.modes:
            return np.zeros_like(t)

        # Use a broad frequency range to capture all modes
        w_max = max(abs(m.shift_cm) + 3 * m.linewidth_cm for m in self.modes)
        n_points = 2 ** int(np.ceil(np.log2(len(t))))
        w_cm = np.linspace(-w_max, w_max, n_points)

        # Compute frequency-domain response
        freq_resp = self.frequency_domain(w_cm)

        # Inverse FFT to time domain
        # Shift so that w=0 is at the center (zero frequency)
        freq_resp_shifted = np.fft.fftshift(freq_resp)
        t_domain = np.fft.ifft(freq_resp_shifted).real

        # Normalize
        if np.max(np.abs(t_domain)) > 0:
            t_domain /= np.max(np.abs(t_domain))

        # Interpolate to requested time array
        t_fine = np.linspace(t[0], t[-1], len(t_domain))
        from scipy.interpolate import interp1d
        interp = interp1d(t_fine, t_domain, kind='linear', fill_value=0, bounds_error=False)
        return interp(t)

    def plot_modes(self, backend: Literal["matplotlib", "plotly"] = "matplotlib"):
        """Bar chart of all phonon modes.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"

        Returns
        -------
        fig or None — Plot figure, or None if no modes.
        """
        if not self.modes:
            return None

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_modes_plotly()
        return self._plot_modes_matplotlib()

    def _plot_modes_matplotlib(self):
        """Matplotlib bar chart of modes."""
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = [f"{m.shift_cm:.0f} cm⁻¹" for m in self.modes]
        strengths = [m.relative_strength for m in self.modes]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.modes)))  # type: ignore[attr-defined]

        ax.barh(range(len(self.modes)), strengths, color=colors)
        ax.set_yticks(range(len(self.modes)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Relative Strength", fontsize=12)
        ax.set_title(f"Phonon Modes ({len(self.modes)} modes)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        return fig

    def _plot_modes_plotly(self):
        """Plotly bar chart of modes."""
        fig = go.Figure()
        labels = [f"{m.shift_cm:.0f} cm⁻¹" for m in self.modes]
        fig.add_trace(go.Bar(
            y=labels,
            x=[m.relative_strength for m in self.modes],
            orientation='h',
            marker_color=plt.cm.viridis(np.linspace(0, 0.8, len(self.modes))).tolist(),  # type: ignore[attr-defined]
        ))
        fig.update_layout(
            title=f"Phonon Modes ({len(self.modes)} modes)",
            xaxis_title="Relative Strength",
            height=max(400, len(self.modes) * 40),
        )
        return fig

    def plot_spectrum(
        self,
        shift_range_cm: float = 200,
        n_points: int = 1000,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
    ):
        """Overlay of all Lorentzian mode lineshapes.

        Parameters
        ----------
        shift_range_cm : float
            Range around 0 to plot (cm⁻¹).
        n_points : int
            Number of points.
        backend : "matplotlib" or "plotly"

        Returns
        -------
        fig or None — Plot figure, or None if no modes.
        """
        if not self.modes:
            return None

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_spectrum_plotly(shift_range_cm, n_points)
        return self._plot_spectrum_matplotlib(shift_range_cm, n_points)

    def _plot_spectrum_matplotlib(self, shift_range_cm: float, n_points: int):
        """Matplotlib spectrum overlay."""
        fig, ax = plt.subplots(figsize=(10, 6))

        w_cm = np.linspace(-shift_range_cm, shift_range_cm, n_points)

        # Plot individual modes
        for mode in self.modes:
            lorentzian = self._lorentzian(mode.shift_cm, mode.linewidth_cm, w_cm)
            ax.plot(w_cm, lorentzian, linewidth=1, alpha=0.5,
                    label=f"{mode.shift_cm:.0f} cm⁻¹ ({mode.symmetry or '?'})")

        # Plot total response
        total = self.frequency_domain(w_cm)
        ax.plot(w_cm, total, linewidth=2, color='black', label='Total')

        ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (arb.)", fontsize=12)
        ax.set_title(f"Phonon Spectrum ({len(self.modes)} modes)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _plot_spectrum_plotly(self, shift_range_cm: float, n_points: int):
        """Plotly spectrum overlay."""
        import plotly.graph_objects as go

        fig = go.Figure()
        w_cm = np.linspace(-shift_range_cm, shift_range_cm, n_points)

        for mode in self.modes:
            lorentzian = self._lorentzian(mode.shift_cm, mode.linewidth_cm, w_cm)
            fig.add_trace(go.Scatter(
                x=w_cm, y=lorentzian, mode='lines', name=f"{mode.shift_cm:.0f} cm⁻¹",
                line=dict(width=1, opacity=0.5),
            ))

        total = self.frequency_domain(w_cm)
        fig.add_trace(go.Scatter(
            x=w_cm, y=total, mode='lines', name='Total',
            line=dict(width=2, color='black'),
        ))

        fig.update_layout(
            title=f"Phonon Spectrum ({len(self.modes)} modes)",
            xaxis_title="Raman shift (cm⁻¹)",
            yaxis_title="Intensity (arb.)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def demo(self):
        """Self-check: verify basic functionality.

        Returns
        -------
        dict — Test results with pass/fail status.
        """
        results = {}

        # Test 1: Empty modes
        empty = PhononResponse([])
        w_test = np.linspace(-100, 100, 100)
        results['empty_freq'] = np.allclose(empty.frequency_domain(w_test), 0)

        # Test 2: Single mode
        single = PhononResponse([PhononMode(shift_cm=254, linewidth_cm=14, relative_strength=1.0)])
        w_single = np.linspace(200, 300, 1000)
        freq_single = single.frequency_domain(w_single)
        peak_idx = np.argmax(freq_single)
        peak_shift = w_single[peak_idx]
        results['single_peak'] = abs(peak_shift - 254) < 5  # Peak near 254 cm⁻¹

        # Test 3: Multi-mode sum
        multi = PhononResponse([
            PhononMode(shift_cm=200, linewidth_cm=10, relative_strength=1.0),
            PhononMode(shift_cm=400, linewidth_cm=20, relative_strength=0.5),
        ])
        w_multi = np.linspace(150, 450, 10000)
        freq_multi = multi.frequency_domain(w_multi)
        # Should have two peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(freq_multi, distance=50)
        results['multi_peaks'] = len(peaks) == 2

        # Test 4: Normalization
        results['normalized'] = abs(np.max(freq_multi) - 1.0) < 0.01

        return results


# ─── PHONON_MATERIALS ─────────────────────────────────────────────────────────

# Multi-mode phonon data for key crystalline materials.
# Sources: see spec for full references.

PHONON_MATERIALS: dict[str, list[PhononMode]] = {
    "LiNbO3": [
        PhononMode(254, 14, "A₁", 1.0, note="Barker & Loudon (1967)"),
        PhononMode(162, 8, "A₁", 0.8, note="Ridah et al. (1997)"),
        PhononMode(701, 6, "A₁", 0.6, note="Barker & Loudon (1967)"),
        PhononMode(637, 5, "E", 0.7, note="Ridah et al. (1997)"),
        PhononMode(576, 4, "E", 0.5, note="Barker & Loudon (1967)"),
        PhononMode(277, 10, "E", 0.4, note="Ridah et al. (1997)"),
        PhononMode(819, 3, "E", 0.3, note="Barker & Loudon (1967)"),
    ],
    "LiTaO3": [
        PhononMode(255, 12, "A₁", 1.0, note="Raptis (1988)"),
        PhononMode(143, 6, "A₁", 0.7, note="Margueron et al. (2012)"),
        PhononMode(740, 5, "A₁", 0.5, note="Raptis (1988)"),
        PhononMode(690, 4, "E", 0.6, note="Raptis (1988)"),
        PhononMode(327, 8, "E", 0.4, note="Margueron et al. (2012)"),
    ],
    "BaTiO3": [
        PhononMode(258, 5, "A₁", 1.0, note="Scalabrin et al. (1977)"),
        PhononMode(181, 3, "A₁", 0.8, note="Chaves et al. (1974)"),
        PhononMode(142, 2, "A₁", 0.6, note="Scalabrin et al. (1977)"),
        PhononMode(520, 45, "A₁", 0.9, note="Scalabrin et al. (1977)"),
        PhononMode(306, 4, "E", 0.7, note="Chaves et al. (1974)"),
        PhononMode(280, 3, "E", 0.5, note="Scalabrin et al. (1977)"),
        PhononMode(720, 6, "E", 0.4, note="Chaves et al. (1974)"),
        PhononMode(890, 8, "B₁", 0.3, note="Scalabrin et al. (1977)"),
    ],
    "YAG": [
        PhononMode(784, 8, "T₂g", 1.0, note="Hurrell et al. (1968)"),
        PhononMode(360, 5, "E_g", 0.6, note="Hurrell et al. (1968)"),
        PhononMode(260, 4, "E_g", 0.5, note="Hurrell et al. (1968)"),
        PhononMode(1230, 6, "T₂g", 0.7, note="Lamaignere et al. (2020)"),
        PhononMode(1500, 10, "A₁g", 0.4, note="Hurrell et al. (1968)"),
    ],
    "Al2O3": [
        PhononMode(418, 5, "E_g", 1.0, note="Watson et al. (1981)"),
        PhononMode(378, 4, "E_g", 0.8, note="Major et al. (2004)"),
        PhononMode(636, 6, "A₁g", 0.6, note="Watson et al. (1981)"),
        PhononMode(750, 8, "E_g", 0.5, note="Watson et al. (1981)"),
        PhononMode(1150, 10, "A₁g", 0.4, note="Major et al. (2004)"),
    ],
    "KTP": [
        PhononMode(270, 20, "A", 1.0, note="Neufeld et al. (2023) — dominant"),
        PhononMode(500, 15, "A", 0.8, note="Neufeld et al. (2023)"),
        PhononMode(620, 12, "B", 0.7, note="Neufeld et al. (2023)"),
        PhononMode(750, 10, "A", 0.6, note="Neufeld et al. (2023)"),
        PhononMode(890, 8, "B", 0.5, note="Neufeld et al. (2023)"),
    ],
    "GaN": [
        PhononMode(568, 4, "E₂(high)", 1.0, note="Zeng et al. (2020)"),
        PhononMode(144, 3, "A₁(LO)", 0.6, note="Zeng et al. (2020)"),
        PhononMode(532, 5, "E₁(LO)", 0.5, note="Almeida et al. (2019)"),
        PhononMode(736, 6, "A₁(LO)", 0.4, note="Zeng et al. (2020)"),
    ],
    "AlN": [
        PhononMode(658, 1, "E₂(high)", 1.0, note="Jung & Tang (2016)"),
        PhononMode(615, 2, "E₁(LO)", 0.7, note="Pandit et al. (2007)"),
        PhononMode(895, 3, "A₁(LO)", 0.5, note="Jung & Tang (2016)"),
        PhononMode(330, 2, "A₁(TO)", 0.4, note="Pandit et al. (2007)"),
    ],
    "SiC_4H": [
        PhononMode(777, 5, "E₂(TO)", 1.0, note="Feldman et al. (1968)"),
        PhononMode(983, 8, "A₁(LO)", 0.6, note="Li et al. (2023)"),
        PhononMode(750, 4, "A₁(TO)", 0.5, note="Feldman et al. (1968)"),
        PhononMode(250, 3, "Folded TA", 0.3, note="Feldman et al. (1968)"),
    ],
    "YLF": [
        PhononMode(262, 5, "B_g", 1.0, note="Miller et al. (1970)"),
        PhononMode(170, 4, "A_g", 0.7, note="Salaun et al. (1997)"),
        PhononMode(380, 6, "E_g", 0.6, note="Miller et al. (1970)"),
        PhononMode(520, 8, "B_g", 0.5, note="Miller et al. (1970)"),
        PhononMode(640, 5, "E_g", 0.4, note="Salaun et al. (1997)"),
    ],
}
