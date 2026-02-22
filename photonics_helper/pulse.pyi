# -*- coding: utf-8 -*-
"""
Type‑stub file for :mod:`photonics_helper.pulse`.

This module provides three public classes that model ultrashort optical
pulses:

* :class:`Envelope` – the analytic envelope (amplitude, phase, shape).
* :class:`TemporalGrid` – a uniformly sampled time/frequency grid and the
  FFT helpers needed to switch between domains.
* :class:`Wave` – a high‑level object that ties an :class:`Envelope` to a
  :class:`TemporalGrid` and offers convenience methods (energy, TBP, plotting,
  …).
"""

from __future__ import annotations

from typing import Literal, Optional, Self, Tuple

import numpy as np
from matplotlib.figure import Figure

from photonics_helper.base import Wavelength

class Envelope:
    """
    Analytic description of an optical pulse envelope.

    Parameters
    ----------
    shape:
        The functional shape of the pulse. Accepted values are
        ``"gaussian"``, ``"sech"``, ``"lorentzian"``, and ``"rectangular"``.
    peak_amplitude:
        Peak amplitude of the electric‑field envelope *A(t)*.
    pulse_width:
        Characteristic width *T₀* (the “1/e” width for a Gaussian, etc.).
    chirp:
        Linear chirp coefficient *C* (default ``0.0``). The instantaneous
        phase added to the envelope is ``0.5*C*(t/T₀)**2``.
    """

    shape: Literal["gaussian", "sech", "lorentzian", "rectangular"]
    peak_amplitude: float
    pulse_width: float  # T0
    chirp: float

    def __init__(
        self,
        shape: Literal["gaussian", "sech", "lorentzian", "rectangular"],
        peak_amplitude: float,
        pulse_width: float,
        chirp: float = ...,
    ) -> None: ...
    @property
    def fwhm(self) -> float:
        """
        Full‑width at half‑maximum of the intensity profile.

        The conversion factor depends on ``self.shape`` and is stored in
        :data:`photonics_helper.pulse.SHAPE_FACTORS`.
        """
        ...

    @classmethod
    def from_fwhm(
        cls,
        shape: Literal["gaussian", "sech", "lorentzian", "rectangular"],
        peak_amplitude: float,
        fwhm: float,
    ) -> Self:
        """
        Construct an :class:`Envelope` from a desired full‑width at half‑maximum.

        Parameters
        ----------
        shape, peak_amplitude, fwhm:
            See the class documentation. ``fwhm`` is converted to the internal
            ``pulse_width`` (``T0``) using the appropriate shape factor.
        """
        ...

    def field(self, t: np.ndarray) -> np.ndarray:
        """
        Complex envelope :math:`A(t)` evaluated on a time array ``t``.

        Returns
        -------
        np.ndarray
            Complex‑valued array of the same shape as ``t``.
        """
        ...

    def intensity(self, t: np.ndarray) -> np.ndarray:
        """
        Intensity :math:`|A(t)|^2` (the squared magnitude of the envelope).

        Returns
        -------
        np.ndarray
            Real‑valued intensity evaluated on ``t``.
        """
        ...

class TemporalGrid:
    """
    Uniformly sampled temporal/frequency grid and related FFT utilities.

    The grid is defined by the number of points ``N`` and the total time
    window ``Tmax``.  All derived quantities (``dt``, ``t``, ``w`` …) are
    cached after first use.
    """

    N: int
    Tmax: float

    def __init__(self, N: int, Tmax: float) -> None: ...
    @property
    def dt(self) -> float:
        """Time step ``Δt = Tmax / N``."""
        ...

    @property
    def t(self) -> np.ndarray:
        """Array of time points centred around zero, shape ``(N,)``."""
        ...

    @property
    def w(self) -> np.ndarray:
        """Angular‑frequency axis (rad·s⁻¹) corresponding to ``t``."""
        ...

    @property
    def dw(self) -> float:
        """Frequency step ``Δω`` derived from the ``w`` axis."""
        ...

    def fft(self, A_t: np.ndarray) -> np.ndarray:
        """
        Forward FFT that respects the ``dt`` scaling used internally.

        Parameters
        ----------
        A_t:
            Time‑domain array (normally the envelope field).

        Returns
        -------
        np.ndarray
            Frequency‑domain representation, shifted so that zero frequency is
            centred.
        """
        ...

    def ifft(self, A_w: np.ndarray) -> np.ndarray:
        """
        Inverse FFT that undoes :meth:`fft`.

        Parameters
        ----------
        A_w:
            Frequency‑domain array.

        Returns
        -------
        np.ndarray
            Time‑domain array with the same scaling as the original input to
            :meth:`fft`.
        """
        ...

    @property
    def omega_max(self) -> float:
        """Largest absolute angular frequency present on the grid."""
        ...

    @property
    def time_window(self) -> float:
        """Total simulated time window ``N * dt`` (should equal ``Tmax``)."""
        ...

    def check_aliasing(self, T0: float) -> None:
        """
        Emit a warning if the total time window is too short to contain the
        pulse comfortably (heuristic: ``time_window < 10 * T0``).

        The function prints a message; it does not raise an exception.
        """
        ...

class Wave:
    """
    Complete description of a propagating optical pulse.

    Combines a :class:`TemporalGrid` with an :class:`Envelope` and a central
    wavelength.  Provides methods for energy, peak power, time‑bandwidth
    product, and a high‑level visualisation routine.
    """

    grid: TemporalGrid
    envelope: Envelope

    central_wavelength: Wavelength
    refractive_index: float
    repetition_rate: Optional[float]

    def __init__(
        self,
        grid: TemporalGrid,
        envelope: Envelope,
        central_wavelength: Wavelength,
        refractive_index: float = ...,
        repetition_rate: Optional[float] = ...,
    ) -> None: ...
    @property
    def central_frequency(self) -> float:
        """
        Central angular frequency :math:`\\omega_0` (rad·s⁻¹) derived from the
        supplied :class:`~photonics_helper.base.Wavelength`.
        """
        ...

    @property
    def envelope_field(self) -> np.ndarray:
        """Complex envelope :math:`A(t)` sampled on ``self.grid.t``."""
        ...

    @property
    def electric_field(self) -> np.ndarray:
        """
        Real electric field :math:`E(t)=\\Re\\{A(t)\\exp(-j\\omega_0 t)\\}`.
        """
        ...

    def instantaneous_intensity(self) -> np.ndarray:
        """Instantaneous intensity :math:`|E(t)|^2`."""
        ...

    @property
    def envelope_intensity(self) -> np.ndarray:
        """Intensity of the envelope :math:`|A(t)|^2`."""
        ...

    def pulse_energy(self) -> float:
        """
        Energy of the pulse (integrated intensity) in arbitrary units.
        """
        ...

    def peak_power(self) -> float:
        """Maximum envelope intensity (peak power) in the same units."""
        ...

    @property
    def spectrum(self) -> np.ndarray:
        """
        Frequency‑domain representation of the envelope ``A(ω)`` obtained via
        :meth:`TemporalGrid.fft`.
        """
        ...

    def time_bandwidth_product(self) -> float:
        """
        Compute the product :math:`Δt·Δω` (root‑mean‑square widths) of the
        intensity and spectral intensity.  For a transform‑limited Gaussian
        pulse the value is ≈ 0.44.
        """
        ...

    def visualize(
        self,
        t_unit: str = ...,
        w_unit: str = ...,
        t_scale: float = ...,
        w_scale: float = ...,
        show_electric_field: bool = ...,
        show_phase: bool = ...,
        show_spectrogram: bool = ...,
        figsize: Optional[Tuple[float, float]] = ...,
        save_path: Optional[str] = ...,
    ) -> Figure:
        """
        Produce a multi‑panel figure summarising the pulse.

        Parameters
        ----------
        t_unit, w_unit :
            Labels for the plotted axes (e.g. ``"ps"``, ``"THz"``).
        t_scale, w_scale :
            Multiplicative factors applied to the time and angular‑frequency
            axes before plotting (convenient for unit conversion).
        show_electric_field :
            If ``True`` add a panel with the real electric field.
        show_phase :
            Overlay the instantaneous phase on the temporal intensity plot.
        show_spectrogram :
            Add a spectrogram (STFT) panel.
        figsize :
            ``(width, height)`` passed to :func:`matplotlib.pyplot.figure`.  If
            ``None`` a size based on the number of panels is used.
        save_path :
            If provided, the resulting figure is saved to this path (format
            inferred from the filename).

        Returns
        -------
        matplotlib.figure.Figure
            The created figure object, allowing further user customisation.
        """
        ...
