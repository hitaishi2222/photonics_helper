"""Temporal grids — a foundation primitive.

`TemporalGrid` defines the time/frequency sampling and the FFT convention used
throughout the library (``FFT(A)·dt`` forward, ``/dt`` inverse, both
fftshifted), so Parseval holds with the paired transforms. It is deliberately
free of plotting dependencies: :mod:`photonics_helper.pulse` re-exports this
same class object for backwards compatibility.

The FFTs execute on the configured backend (:mod:`photonics_helper._fftw`) —
FFTW3 when ``pyfftw`` is installed, otherwise numpy.
"""

from __future__ import annotations

from functools import cached_property
from typing import Self

import numpy as np
from pydantic.dataclasses import dataclass

from .._fftw import fft as _fft_backend
from .._fftw import ifft as _ifft_backend
from ..base import Frequency, Time

__all__ = ["TemporalGrid"]


@dataclass
class TemporalGrid:
    """Uniform temporal grid with FFT helpers.

    Parameters
    ----------
    N : number of samples.
    Tmax : total time window.
    """

    N: int
    Tmax: Time  # total time window

    @cached_property
    def dt(self):
        """Temporal grid step (s)."""
        return self.Tmax.as_s / self.N

    @cached_property
    def t(self):
        return np.linspace(-self.Tmax.as_s / 2, self.Tmax.as_s / 2 - self.dt, self.N)

    @cached_property
    def w(self):
        return np.fft.fftshift(2 * np.pi * np.fft.fftfreq(self.N, d=self.dt))

    @cached_property
    def dw(self):
        """Frequency grid spacing (rad/ps)."""
        w = self.w
        return w[1] - w[0]

    def fft(self, A_t):
        """Forward FFT — analysis kernel ``e^{+iΩt}``, scaled by ``dt``.

        **Convention swap 2026-09 (openspec ``fix-audit-issues-batch``,
        ``ISSUES.md #0``)**: the analysis kernel is now ``e^{+iΩt}`` and the
        companion :meth:`ifft` synthesis kernel ``e^{−iΩt}``, the standard
        Agrawal pairing (A(z,T) = ∫Ã(Ω)·e^{−iΩT}; the carrier factor is
        ``e^{−iω₀t}``). Implemented as ``conj(DFT(conj(·)))`` on the same
        backend primitive (the DFT pair mirrored). Bin ``w > 0`` under the
        λ map ``λ = c/(ω₀ + grid.w)`` is the blue side, and Parseval still
        holds with the paired transforms.
        """
        return np.conj(_fft_backend(np.conj(np.asarray(A_t)))) * self.dt

    def ifft(self, A_w):
        """Inverse FFT paired with :meth:`fft` (includes ``1/dt`` scaling).

        Synthesis kernel ``e^{−iΩt}`` after the #0 convention swap; see
        :meth:`fft`.
        """
        return np.conj(_ifft_backend(np.conj(np.asarray(A_w)))) / self.dt

    @property
    def omega_max(self):
        """Maximum angular frequency on the grid (rad/ps)."""
        return np.max(np.abs(self.w))

    @property
    def time_window(self):
        """Total simulated time window (s)."""
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

    @classmethod
    def for_gnlse(
        cls,
        pulse_width: Time,
        length_m: float,
        beta2_pskm: float,
        *,
        gamma: float | None = None,
        peak_power: float | None = None,
        N: int = 2**13,
        margin: float = 8.0,
    ) -> Self:
        """Size a grid that holds soliton-fission products without wrapping.

        The plain ``envelope._make_grid`` window (≈10× T0) fits the input
        pulse but not what the fiber makes of it: higher-order solitons shed
        dispersive waves that walk off by ``~|β₂|·Δω·L`` and wrap around the
        periodic FFT boundary, interfering with themselves (speckly spectral
        fringes + diagonal streaks + percent-level energy drift). This sizes
        the window from the dispersive spread ``|β₂|·L/T0``, inflated by the
        soliton order ``N = sqrt(γ·P·T0²/|β₂|)`` when ``gamma`` and
        ``peak_power`` are known::

            grid = TemporalGrid.for_gnlse(pulse.pulse_width, fiber.length.as_m,
                                          betas[0], gamma=0.045, peak_power=10e3)

        Parameters
        ----------
        pulse_width : T₀ of the input pulse.
        length_m : fiber length in metres.
        beta2_pskm : β₂ in ps²/m (engine units, i.e. ``betas[0]``).
        gamma : nonlinear coefficient γ in 1/(W·m), if known.
        peak_power : input peak power in watts, if known.
        N : samples (default 2¹³ — wide windows need resolving power).
        margin : safety multiplier on the estimated spread (default 8).
        """
        from math import sqrt

        T0 = pulse_width.as_s
        beta2_si = abs(float(beta2_pskm)) * 1e-24  # ps²/m → s²/m
        if beta2_si > 0.0:
            spread = beta2_si * float(length_m) / T0
            if gamma is not None and peak_power is not None and gamma > 0.0 and peak_power > 0.0:
                order_sq = gamma * peak_power * T0**2 / beta2_si
                if order_sq > 1.0:
                    spread *= sqrt(order_sq)
            spread *= margin
        else:
            spread = 0.0
        Tmax = max(10.0 * T0, spread)
        return cls(N=N, Tmax=Time(Tmax, "s"))
