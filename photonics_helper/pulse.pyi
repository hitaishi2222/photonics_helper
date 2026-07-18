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

from functools import cached_property
from typing import Callable, Literal, Self

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure

from photonics_helper.base import Wavelength, Frequency, Time

class Envelope:
    """
    Analytic description of an optical pulse envelope.

    Parameters
    ----------
    shape:
        The functional shape of the pulse. Accepted values are
        ``"gaussian"``, ``"sech"``, ``"lorentzian"``, ``"rectangular"``,
        ``"super-gaussian"``, ``"triangular"``, ``"parabolic"``,
        ``"cosine"``, ``"exponential"``, ``"gauss-hermite"``,
        ``"airy"``, ``"custom"``.
    peak_amplitude:
        Peak amplitude of the electric‑field envelope *A(t)*.
    pulse_width:
        Characteristic width *T₀* (the "1/e" width for a Gaussian, etc.).
    chirp:
        Linear chirp coefficient *C* (default ``0.0``). The instantaneous
        phase added to the envelope is ``0.5*C*(t/T₀)**2``.
    """

    shape: Literal[
        "gaussian", "sech", "lorentzian", "rectangular",
        "super-gaussian", "triangular", "parabolic", "cosine",
        "exponential", "gauss-hermite", "airy", "custom",
    ]
    peak_amplitude: float
    pulse_width: Time  # T0
    chirp: float
    super_gaussian_order: int
    beam_waist: Time | None
    hg_mode: int
    func: Callable | None
    phase_func: Callable | None

    def __init__(
        self,
        shape: Literal[
            "gaussian", "sech", "lorentzian", "rectangular",
            "super-gaussian", "triangular", "parabolic", "cosine",
            "exponential", "gauss-hermite", "airy", "custom",
        ],
        peak_amplitude: float,
        pulse_width: float,
        chirp: float = ...,
        super_gaussian_order: int = ...,
        beam_waist: Time | None = ...,
        hg_mode: int = ...,
        func: Callable | None = ...,
        phase_func: Callable | None = ...,
    ) -> None: ...

    @property
    def fwhm(self) -> Time:
        """
        Full‑width at half‑maximum of the intensity profile.

        The conversion factor depends on ``self.shape`` and is stored in
        :data:`photonics_helper.pulse.SHAPE_FACTORS`.
        """
        ...

    @classmethod
    def from_fwhm(
        cls,
        shape: Literal[
            "gaussian", "sech", "lorentzian", "rectangular",
            "super-gaussian", "triangular", "cosine", "exponential", "airy",
        ],
        peak_amplitude: float,
        fwhm: Time,
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

    def field(self, t: NDArray) -> NDArray:
        """
        Complex envelope :math:`A(t)` evaluated on a time array ``t``.

        Returns
        -------
        NDArray
            Complex‑valued array of the same shape as ``t``.
        """
        ...

    def intensity(self, t: NDArray) -> NDArray:
        """
        Intensity :math:`|A(t)|^2` (the squared magnitude of the envelope).

        Returns
        -------
        NDArray
            Real‑valued intensity evaluated on ``t``.
        """
        ...

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
        """
        Plot temporal intensity, spectral intensity, phase, and polar plot.

        Parameters
        ----------
        backend : "plotly" or "matplotlib" (default "plotly")
        N : number of time points (default 2^12)
        show_phase : show instantaneous phase overlay (default True)
        show_fwhm : show FWHM markers (default True)
        figsize : figure size for matplotlib backend (default None)
        title : optional title override (default uses shape name)
        theme : "light" or "dark" (default "light")

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
        """
        ...

    def visualize_3d(
        self,
        N: int = 2**12,
        title: str | None = None,
        theme: Literal["light", "dark"] = "light",
    ):
        """
        Plot 3D surface of temporal intensity |A(t)|².

        Parameters
        ----------
        N : number of time points (default 2^12)
        title : optional title override
        theme : "light" or "dark" (default "light")

        Returns
        -------
        plotly.graph_objects.Figure
        """
        ...

    @classmethod
    def from_parabolic_asymptotic(
        cls,
        peak_amplitude: float,
        pulse_width: Time,
        gain: float,
        length: float,
        chirp: float = ...,
    ) -> Self:
        """
        Construct a parabolic pulse with asymptotic amplifier chirp.

        The chirp coefficient follows the asymptotic parabolic solution:
        ``α ≈ 0.2726 × z × gain``.

        Parameters
        ----------
        peak_amplitude : A₀
        pulse_width : T₀
        gain : Small‑signal gain coefficient
        length : Propagation length in the amplifier
        chirp : Additional chirp on top of the asymptotic value (default 0)
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
    def t(self) -> NDArray:
        """Array of time points centred around zero, shape ``(N,)``."""
        ...

    @property
    def w(self) -> NDArray:
        """Angular‑frequency axis (rad·s⁻¹) corresponding to ``t``."""
        ...

    @property
    def dw(self) -> float:
        """Frequency step ``Δω`` derived from the ``w`` axis."""
        ...

    def fft(self, A_t: NDArray) -> NDArray:
        """
        Forward FFT that respects the ``dt`` scaling used internally.

        Parameters
        ----------
        A_t:
            Time‑domain array (normally the envelope field).

        Returns
        -------
        NDArray
            Frequency‑domain representation, shifted so that zero frequency is
            centred.
        """
        ...

    def ifft(self, A_w: NDArray) -> NDArray:
        """
        Inverse FFT that undoes :meth:`fft`.

        Parameters
        ----------
        A_w:
            Frequency‑domain array.

        Returns
        -------
        NDArray
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

    @classmethod
    def for_pulse_train(
        cls,
        repetition_rate: Frequency,
        n_pulses: int,
        pulse_width: Time,
        N: int = ...,
    ) -> Self:
        """
        Compute the right Tmax to cover a pulse train.

        Parameters
        ----------
        repetition_rate : Frequency — pulse spacing = 1 / repetition_rate
        n_pulses : number of pulses
        pulse_width : T₀ — characteristic width, used to estimate needed padding
        N : number of time points (default 2¹²)
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
    repetition_rate: Frequency | None

    def __init__(
        self,
        grid: TemporalGrid,
        envelope: Envelope,
        central_wavelength: Wavelength,
        refractive_index: float = ...,
        repetition_rate: Frequency | None = ...,
    ) -> None: ...

    @cached_property
    def central_frequency(self) -> float:
        """
        Central angular frequency :math:`\\omega_0` (rad·s⁻¹) derived from the
        supplied :class:`~photonics_helper.base.Wavelength`.
        """
        ...

    @property
    def envelope_field(self) -> NDArray:
        """Complex envelope :math:`A(t)` sampled on ``self.grid.t``."""
        ...

    @property
    def electric_field(self) -> NDArray:
        """
        Real electric field :math:`E(t)=\\Re\\{A(t)\\exp(-j\\omega_0 t)\\}`.
        """
        ...

    def instantaneous_intensity(self) -> NDArray:
        """Instantaneous intensity :math:`|E(t)|^2`."""
        ...

    @property
    def envelope_intensity(self) -> NDArray:
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

    def average_power(self, repetition_rate: Frequency) -> float:
        """
        Average power = pulse energy × repetition_rate.

        Parameters
        ----------
        repetition_rate : Frequency
        """
        ...

    @classmethod
    def from_pulse_train(
        cls,
        envelope: Envelope,
        central_wavelength: Wavelength,
        grid: TemporalGrid,
        repetition_rate: Frequency,
        n_pulses: int = ...,
        refractive_index: float = ...,
    ) -> Self:
        """
        Construct a pulse train Wave from a single-envelope shape.

        Parameters
        ----------
        envelope : The single-pulse envelope shape to repeat
        central_wavelength : Central wavelength of the carrier
        grid : TemporalGrid covering the full window (all pulses + padding)
        repetition_rate : Frequency — spacing between consecutive pulses
        n_pulses : number of pulses (default 10)
        refractive_index : background refractive index (default 1.0)
        """
        ...

    @cached_property
    def spectrum(self) -> NDArray:
        """
        Frequency‑domain representation of the envelope ``A(ω)`` obtained via
        :meth:`TemporalGrid.fft`.
        """
        ...

    def time_bandwidth_product(self) -> float:
        """
        Compute the product :math:`Δt·Δω` (root‑mean‑square widths) of the
        intensity and spectral intensity.  For a transform‑limited Gaussian
        pulse the value is ≈ 0.707.
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
        figsize: tuple[float, float] | None = ...,
        save_path: str | None = ...,
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
