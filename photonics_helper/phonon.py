"""Phonon module — multi-mode Raman-active phonon modeling.

Provides PhononMode dataclass and PhononResponse class for combining
multiple Raman-active phonon modes into a unified response function.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pydantic import model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs

from .base import Wavenumber, WavenumberArray

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
    shift_cm : Wavenumber
        Raman shift (e.g. ``Wavenumber(254, "1/cm")``).
    linewidth_cm : Wavenumber
        Full width at half maximum (e.g. ``Wavenumber(14, "1/cm")``).
    symmetry : str or None
        Symmetry label (e.g. "A₁g", "E_g", "A₁(TO)").
    relative_strength : float
        Relative Raman cross-section (default 1.0).
    lo_phonon_cm : Wavenumber or None
        LO phonon frequency (if different from Raman shift).
    to_phonon_cm : Wavenumber or None
        TO phonon frequency.
    note : str or None
        Annotation or source reference.
    """

    shift_cm: Wavenumber
    linewidth_cm: Wavenumber
    symmetry: str | None = None
    relative_strength: float = 1.0
    lo_phonon_cm: Wavenumber | None = None
    to_phonon_cm: Wavenumber | None = None
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_wavenumbers(cls, values):
        """Accept bare numbers in cm⁻¹ and wrap them in :class:`Wavenumber`."""
        keys = ("shift_cm", "linewidth_cm", "lo_phonon_cm", "to_phonon_cm")

        def coerce(v):
            if v is None or isinstance(v, Wavenumber):
                return v
            return Wavenumber(v, "1/cm")

        if isinstance(values, ArgsKwargs):
            args = list(values.args)
            kwargs = dict(values.kwargs) if values.kwargs else {}
            # positional order: shift(0), linewidth(1), symmetry(2), strength(3), lo(4), to(5), note(6)
            for i in (0, 1, 4, 5):
                if len(args) > i:
                    args[i] = coerce(args[i])
            for k in keys:
                if k in kwargs:
                    kwargs[k] = coerce(kwargs[k])
            return ArgsKwargs(tuple(args), kwargs)
        if isinstance(values, dict):
            return {k: (coerce(v) if k in keys else v) for k, v in values.items()}
        return values


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
        # fR is the Raman fraction, NOT the sum of mode strengths.  Relative
        # strengths only distribute the fraction among modes, so inferring fR
        # from them produced values > 1 (e.g. 4.3 for LiNbO3).
        if self.fR is not None and not (0.0 <= self.fR <= 1.0):
            raise ValueError(
                f"fR must be a fraction in [0, 1], got {self.fR!r}"
            )

    def _lorentzian(self, shift_cm: float, gamma_cm: float, w_cm: NDArray) -> NDArray:
        """Single Lorentzian lineshape.

        Parameters
        ----------
        shift_cm : float
            Center shift in cm⁻¹.
        gamma_cm : float
            FWHM in cm⁻¹.
        w_cm : NDArray
            Frequency axis in cm⁻¹.

        Returns
        -------
        NDArray — Lorentzian values (unnormalized).
        """
        half_gamma = gamma_cm / 2.0
        return (half_gamma / np.pi) / ((w_cm - shift_cm) ** 2 + half_gamma ** 2)

    def frequency_domain(self, w_cm: WavenumberArray | NDArray) -> NDArray:
        """Frequency-domain response: sum of Lorentzian mode lineshapes.

        Parameters
        ----------
        w_cm : WavenumberArray | NDArray
            Frequency axis in cm⁻¹ (a :class:`WavenumberArray` or a bare array).

        Returns
        -------
        NDArray — Total response (sum of weighted Lorentzians).
        """
        if isinstance(w_cm, WavenumberArray):
            w_arr = np.asarray(w_cm.as_1_cm, dtype=float)
        else:
            w_arr = np.asarray(w_cm, dtype=float)

        if not self.modes:
            return np.zeros_like(w_arr)

        total = np.zeros_like(w_arr, dtype=float)
        for mode in self.modes:
            lorentzian = self._lorentzian(
                mode.shift_cm.as_1_cm, mode.linewidth_cm.as_1_cm, w_arr
            )
            total += mode.relative_strength * lorentzian

        # Normalize the lineshape to peak = 1 (shape only; independent of fR).
        if np.max(total) > 0:
            total /= np.max(total)

        return total

    def time_domain(self, t: NDArray) -> NDArray:
        """Time-domain response via inverse FFT of frequency-domain response.

        The time-domain response is the inverse FFT of the frequency-domain
        response, representing the delayed Raman response in the time domain.

        Parameters
        ----------
        t : NDArray
            Time array in seconds.

        Returns
        -------
        NDArray — Time-domain response (sum of damped oscillations).
        """
        if not self.modes:
            return np.zeros_like(t)

        # Use a broad frequency range to capture all modes
        w_max = max(abs(m.shift_cm.as_1_cm) + 3 * m.linewidth_cm.as_1_cm for m in self.modes)
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
        return np.asarray(interp(t))

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

        labels = [f"{m.shift_cm.as_1_cm:.0f} cm⁻¹" for m in self.modes]
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
        labels = [f"{m.shift_cm.as_1_cm:.0f} cm⁻¹" for m in self.modes]
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
            lorentzian = self._lorentzian(
                mode.shift_cm.as_1_cm, mode.linewidth_cm.as_1_cm, w_cm
            )
            ax.plot(w_cm, lorentzian, linewidth=1, alpha=0.5,
                    label=f"{mode.shift_cm.as_1_cm:.0f} cm⁻¹ ({mode.symmetry or '?'})")

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
            lorentzian = self._lorentzian(
                mode.shift_cm.as_1_cm, mode.linewidth_cm.as_1_cm, w_cm
            )
            fig.add_trace(go.Scatter(
                x=w_cm, y=lorentzian, mode='lines', name=f"{mode.shift_cm.as_1_cm:.0f} cm⁻¹",
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
        single = PhononResponse([PhononMode(shift_cm=Wavenumber(254, "1/cm"), linewidth_cm=Wavenumber(14, "1/cm"), relative_strength=1.0)])
        w_single = np.linspace(200, 300, 1000)
        freq_single = single.frequency_domain(w_single)
        peak_idx = np.argmax(freq_single)
        peak_shift = w_single[peak_idx]
        results['single_peak'] = abs(peak_shift - 254) < 5  # Peak near 254 cm⁻¹

        # Test 3: Multi-mode sum
        multi = PhononResponse([
            PhononMode(shift_cm=Wavenumber(200, "1/cm"), linewidth_cm=Wavenumber(10, "1/cm"), relative_strength=1.0),
            PhononMode(shift_cm=Wavenumber(400, "1/cm"), linewidth_cm=Wavenumber(20, "1/cm"), relative_strength=0.5),
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
        PhononMode(Wavenumber(254, "1/cm"), Wavenumber(14, "1/cm"), "A₁", 1.0, note="Barker & Loudon (1967)"),
        PhononMode(Wavenumber(162, "1/cm"), Wavenumber(8, "1/cm"), "A₁", 0.8, note="Ridah et al. (1997)"),
        PhononMode(Wavenumber(701, "1/cm"), Wavenumber(6, "1/cm"), "A₁", 0.6, note="Barker & Loudon (1967)"),
        PhononMode(Wavenumber(637, "1/cm"), Wavenumber(5, "1/cm"), "E", 0.7, note="Ridah et al. (1997)"),
        PhononMode(Wavenumber(576, "1/cm"), Wavenumber(4, "1/cm"), "E", 0.5, note="Barker & Loudon (1967)"),
        PhononMode(Wavenumber(277, "1/cm"), Wavenumber(10, "1/cm"), "E", 0.4, note="Ridah et al. (1997)"),
        PhononMode(Wavenumber(819, "1/cm"), Wavenumber(3, "1/cm"), "E", 0.3, note="Barker & Loudon (1967)"),
    ],
    "LiTaO3": [
        PhononMode(Wavenumber(255, "1/cm"), Wavenumber(12, "1/cm"), "A₁", 1.0, note="Raptis (1988)"),
        PhononMode(Wavenumber(143, "1/cm"), Wavenumber(6, "1/cm"), "A₁", 0.7, note="Margueron et al. (2012)"),
        PhononMode(Wavenumber(740, "1/cm"), Wavenumber(5, "1/cm"), "A₁", 0.5, note="Raptis (1988)"),
        PhononMode(Wavenumber(690, "1/cm"), Wavenumber(4, "1/cm"), "E", 0.6, note="Raptis (1988)"),
        PhononMode(Wavenumber(327, "1/cm"), Wavenumber(8, "1/cm"), "E", 0.4, note="Margueron et al. (2012)"),
    ],
    "BaTiO3": [
        PhononMode(Wavenumber(258, "1/cm"), Wavenumber(5, "1/cm"), "A₁", 1.0, note="Scalabrin et al. (1977)"),
        PhononMode(Wavenumber(181, "1/cm"), Wavenumber(3, "1/cm"), "A₁", 0.8, note="Chaves et al. (1974)"),
        PhononMode(Wavenumber(142, "1/cm"), Wavenumber(2, "1/cm"), "A₁", 0.6, note="Scalabrin et al. (1977)"),
        PhononMode(Wavenumber(520, "1/cm"), Wavenumber(45, "1/cm"), "A₁", 0.9, note="Scalabrin et al. (1977)"),
        PhononMode(Wavenumber(306, "1/cm"), Wavenumber(4, "1/cm"), "E", 0.7, note="Chaves et al. (1974)"),
        PhononMode(Wavenumber(280, "1/cm"), Wavenumber(3, "1/cm"), "E", 0.5, note="Scalabrin et al. (1977)"),
        PhononMode(Wavenumber(720, "1/cm"), Wavenumber(6, "1/cm"), "E", 0.4, note="Chaves et al. (1974)"),
        PhononMode(Wavenumber(890, "1/cm"), Wavenumber(8, "1/cm"), "B₁", 0.3, note="Scalabrin et al. (1977)"),
    ],
    "YAG": [
        PhononMode(Wavenumber(784, "1/cm"), Wavenumber(8, "1/cm"), "T₂g", 1.0, note="Hurrell et al. (1968)"),
        PhononMode(Wavenumber(360, "1/cm"), Wavenumber(5, "1/cm"), "E_g", 0.6, note="Hurrell et al. (1968)"),
        PhononMode(Wavenumber(260, "1/cm"), Wavenumber(4, "1/cm"), "E_g", 0.5, note="Hurrell et al. (1968)"),
        PhononMode(Wavenumber(1230, "1/cm"), Wavenumber(6, "1/cm"), "T₂g", 0.7, note="Lamaignere et al. (2020)"),
        PhononMode(Wavenumber(1500, "1/cm"), Wavenumber(10, "1/cm"), "A₁g", 0.4, note="Hurrell et al. (1968)"),
    ],
    "Al2O3": [
        PhononMode(Wavenumber(418, "1/cm"), Wavenumber(5, "1/cm"), "E_g", 1.0, note="Watson et al. (1981)"),
        PhononMode(Wavenumber(378, "1/cm"), Wavenumber(4, "1/cm"), "E_g", 0.8, note="Major et al. (2004)"),
        PhononMode(Wavenumber(636, "1/cm"), Wavenumber(6, "1/cm"), "A₁g", 0.6, note="Watson et al. (1981)"),
        PhononMode(Wavenumber(750, "1/cm"), Wavenumber(8, "1/cm"), "E_g", 0.5, note="Watson et al. (1981)"),
        PhononMode(Wavenumber(1150, "1/cm"), Wavenumber(10, "1/cm"), "A₁g", 0.4, note="Major et al. (2004)"),
    ],
    "KTP": [
        PhononMode(Wavenumber(270, "1/cm"), Wavenumber(20, "1/cm"), "A", 1.0, note="Neufeld et al. (2023) — dominant"),
        PhononMode(Wavenumber(500, "1/cm"), Wavenumber(15, "1/cm"), "A", 0.8, note="Neufeld et al. (2023)"),
        PhononMode(Wavenumber(620, "1/cm"), Wavenumber(12, "1/cm"), "B", 0.7, note="Neufeld et al. (2023)"),
        PhononMode(Wavenumber(750, "1/cm"), Wavenumber(10, "1/cm"), "A", 0.6, note="Neufeld et al. (2023)"),
        PhononMode(Wavenumber(890, "1/cm"), Wavenumber(8, "1/cm"), "B", 0.5, note="Neufeld et al. (2023)"),
    ],
    "GaN": [
        PhononMode(Wavenumber(568, "1/cm"), Wavenumber(4, "1/cm"), "E₂(high)", 1.0, note="Zeng et al. (2020)"),
        PhononMode(Wavenumber(144, "1/cm"), Wavenumber(3, "1/cm"), "A₁(LO)", 0.6, note="Zeng et al. (2020)"),
        PhononMode(Wavenumber(532, "1/cm"), Wavenumber(5, "1/cm"), "E₁(LO)", 0.5, note="Almeida et al. (2019)"),
        PhononMode(Wavenumber(736, "1/cm"), Wavenumber(6, "1/cm"), "A₁(LO)", 0.4, note="Zeng et al. (2020)"),
    ],
    "AlN": [
        PhononMode(Wavenumber(658, "1/cm"), Wavenumber(1, "1/cm"), "E₂(high)", 1.0, note="Jung & Tang (2016)"),
        PhononMode(Wavenumber(615, "1/cm"), Wavenumber(2, "1/cm"), "E₁(LO)", 0.7, note="Pandit et al. (2007)"),
        PhononMode(Wavenumber(895, "1/cm"), Wavenumber(3, "1/cm"), "A₁(LO)", 0.5, note="Jung & Tang (2016)"),
        PhononMode(Wavenumber(330, "1/cm"), Wavenumber(2, "1/cm"), "A₁(TO)", 0.4, note="Pandit et al. (2007)"),
    ],
    "SiC_4H": [
        PhononMode(Wavenumber(777, "1/cm"), Wavenumber(5, "1/cm"), "E₂(TO)", 1.0, note="Feldman et al. (1968)"),
        PhononMode(Wavenumber(983, "1/cm"), Wavenumber(8, "1/cm"), "A₁(LO)", 0.6, note="Li et al. (2023)"),
        PhononMode(Wavenumber(750, "1/cm"), Wavenumber(4, "1/cm"), "A₁(TO)", 0.5, note="Feldman et al. (1968)"),
        PhononMode(Wavenumber(250, "1/cm"), Wavenumber(3, "1/cm"), "Folded TA", 0.3, note="Feldman et al. (1968)"),
    ],
    "YLF": [
        PhononMode(Wavenumber(262, "1/cm"), Wavenumber(5, "1/cm"), "B_g", 1.0, note="Miller et al. (1970)"),
        PhononMode(Wavenumber(170, "1/cm"), Wavenumber(4, "1/cm"), "A_g", 0.7, note="Salaun et al. (1997)"),
        PhononMode(Wavenumber(380, "1/cm"), Wavenumber(6, "1/cm"), "E_g", 0.6, note="Miller et al. (1970)"),
        PhononMode(Wavenumber(520, "1/cm"), Wavenumber(8, "1/cm"), "B_g", 0.5, note="Miller et al. (1970)"),
        PhononMode(Wavenumber(640, "1/cm"), Wavenumber(5, "1/cm"), "E_g", 0.4, note="Salaun et al. (1997)"),
    ],
}
