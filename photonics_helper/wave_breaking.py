"""Optical wave breaking in the normal-dispersion regime.

When self-phase modulation (SPM) chirps a pulse in the **normal-dispersion**
regime (β₂ > 0), group-velocity dispersion converts the chirp into a time shift
and the pulse develops steep (shock) edges, then a flat top, and finally
oscillations.  For a Gaussian input ``A(0,T) = sqrt(P0) exp(-T²/2T₀²)`` the
SPM chirp after a distance ``z`` is

.. math::

    \\delta\\omega(T) = -\\frac{2\\gamma P_0 z}{T_0^2}\\,
        T e^{-T^2/T_0^2},

so each spectral component is displaced to ``T' = T + β₂ z δω(T)``.  Wave
breaking is the point where this map becomes non-monotonic, which gives

.. math::

    z_{WB} = \\frac{T_0}{\\sqrt{2\\beta_2\\gamma P_0}}
           = \\frac{e^{3/4}}{2}\\sqrt{L_D L_{NL}},
    \\qquad
    L_D = \\frac{T_0^2}{\\beta_2}, \\quad
    L_{NL} = \\frac{1}{\\gamma P_0}.

The dimensionless edge steepness

.. math::

    S = \\frac{\\max_T |\\partial I/\\partial T|\\, T_0}{I_{peak}}

grows from the Gaussian value ``sqrt(2) e^{-1/2} ≈ 0.8578``; its departure marks
the wave-breaking onset.

References
----------
- W. J. Tomlinson, R. H. Stolen, A. M. Johnson, *Opt. Lett.* **10**, 457 (1985).
- G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Sec. 4.1.3.
- D. Anderson, M. Desaix, M. Lisak, M. L. Quiroga-Teixeiro, *JOSA B* **9**, 1358 (1992).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "WaveBreaking",
    "detect_oscillation_onset",
    "detect_steepening_onset",
    "dispersion_length",
    "edge_steepness",
    "gaussian_edge_steepness",
    "nonlinear_length",
    "wave_breaking_distance",
]

#: Peak dimensionless edge steepness of a transform-limited Gaussian profile.
gaussian_edge_steepness = float(np.sqrt(2.0) * np.exp(-0.5))


def dispersion_length(beta2: float, T0: float) -> float:
    """Dispersion length ``L_D = T₀²/β₂`` in m (``β₂`` in s²/m)."""
    if beta2 == 0.0:
        raise ValueError("dispersion_length is undefined for beta2 = 0.")
    return float(T0**2 / abs(beta2))


def nonlinear_length(gamma: float, P0: float) -> float:
    """Nonlinear length ``L_NL = 1/(γP₀)`` in m."""
    if gamma <= 0.0 or P0 <= 0.0:
        raise ValueError("gamma and P0 must be positive.")
    return float(1.0 / (gamma * P0))


def wave_breaking_distance(beta2: float, gamma: float, P0: float, T0: float) -> float:
    """Analytic optical wave-breaking distance ``z_WB`` in m.

    Valid for the normal-dispersion regime (``β₂ > 0``).  Derived from the
    chirp-folding criterion ``1 + β₂ z ∂δω/∂T = 0`` for a Gaussian pulse.

    Parameters
    ----------
    beta2 : float — group-velocity dispersion in s²/m (> 0, normal dispersion).
    gamma : float — nonlinear coefficient in 1/(W·m).
    P0 : float — peak power in W.
    T0 : float — Gaussian half-width (1/e intensity radius) in s.
    """
    if beta2 <= 0.0:
        raise ValueError(
            "Optical wave breaking requires normal dispersion (beta2 > 0); "
            f"got beta2={beta2!r} s^2/m."
        )
    L_D = T0**2 / beta2
    L_NL = nonlinear_length(gamma, P0)
    return float(np.exp(0.75) / 2.0 * np.sqrt(L_D * L_NL))


def edge_steepness(
    field: NDArray | None = None,
    t: NDArray | None = None,
    T0: float = 1.0,
    *,
    intensity: NDArray | None = None,
) -> float:
    """Normalised edge steepness ``max|∂I/∂T|·T₀ / I_peak``.

    Pass either a complex ``field`` (intensity is ``|field|²``) or a precomputed
    ``intensity`` array.  ``t`` is the time grid and ``T₀`` the input pulse
    width.  Returns the Gaussian value ``0.8578`` for a transform-limited
    Gaussian.
    """
    if intensity is None:
        if field is None:
            raise ValueError("Provide either field or intensity.")
        intensity = np.abs(np.asarray(field)) ** 2
    if t is None:
        raise ValueError("t (time grid) is required.")
    intensity = np.asarray(intensity, dtype=float)
    peak = float(intensity.max())
    if peak <= 0.0:
        return 0.0
    return float(np.max(np.abs(np.gradient(intensity, t))) * T0 / peak)


def detect_steepening_onset(
    z_array: NDArray,
    steepness: NDArray,
    *,
    factor: float = 1.10,
    reference: float = gaussian_edge_steepness,
) -> float:
    """First distance where the edge steepness exceeds ``factor × reference``."""
    z_array = np.asarray(z_array, dtype=float)
    steepness = np.asarray(steepness, dtype=float)
    mask = steepness > factor * reference
    if not np.any(mask):
        return float("nan")
    return float(z_array[int(np.argmax(mask))])


def detect_oscillation_onset(
    z_array: NDArray,
    fields: Sequence[NDArray],
    *,
    prominence: float = 0.01,
) -> float:
    """First distance where an intensity profile develops more than one local maximum."""
    from scipy.signal import find_peaks

    z_array = np.asarray(z_array, dtype=float)
    for z, field in zip(z_array, fields):
        intensity = np.abs(np.asarray(field)) ** 2
        peak = float(intensity.max())
        if peak <= 0.0:
            continue
        peaks, _ = find_peaks(intensity, prominence=prominence * peak)
        if len(peaks) >= 2:
            return float(z)
    return float("nan")


@dataclass(frozen=True)
class WaveBreaking:
    """`Wave-breaking` diagnostics for a pulse in a normal-dispersion fibre.

    Parameters
    ----------
    beta2 : float — GVD in s²/m (> 0).
    gamma : float — nonlinear coefficient in 1/(W·m).
    P0 : float — peak power in W.
    T0 : float — Gaussian half-width in s.
    """

    beta2: float
    gamma: float
    P0: float
    T0: float

    def __post_init__(self) -> None:
        if self.beta2 <= 0.0:
            raise ValueError(
                "WaveBreaking requires normal dispersion (beta2 > 0); "
                f"got beta2={self.beta2!r} s^2/m."
            )

    @property
    def L_D(self) -> float:
        return float(self.T0**2 / self.beta2)

    @property
    def L_NL(self) -> float:
        return nonlinear_length(self.gamma, self.P0)

    @property
    def sqrt_LD_LNL(self) -> float:
        return float(np.sqrt(self.L_D * self.L_NL))

    @property
    def z_WB(self) -> float:
        """Analytic wave-breaking distance in m."""
        return float(np.exp(0.75) / 2.0 * self.sqrt_LD_LNL)

    def steepness(self, field: NDArray, t: NDArray) -> float:
        """Edge steepness of a single field (see :func:`edge_steepness`)."""
        return edge_steepness(field=field, t=t, T0=self.T0)

    def steepness_curve(self, fields: Sequence[NDArray], t: NDArray) -> NDArray:
        return np.array([self.steepness(f, t) for f in fields], dtype=float)

    def analyze(
        self,
        z_array: NDArray,
        fields: Sequence[NDArray],
        t: NDArray,
        *,
        steepening_factor: float = 1.10,
    ) -> dict:
        """Detect the steepening and oscillation onsets along a propagation.

        Parameters
        ----------
        z_array : distances in m for each stored field.
        fields : sequence of complex envelope fields.
        t : time grid in s.
        steepening_factor : threshold relative to the Gaussian steepness.

        Returns
        -------
        dict with ``z_onset_m``, ``z_oscillation_m``, ``peak_steepness``,
        ``z_WB_m``, ``L_D_m``, ``L_NL_m`` and the steepness array under
        ``"steepness"``.
        """
        steepness = self.steepness_curve(fields, t)
        return {
            "L_D_m": self.L_D,
            "L_NL_m": self.L_NL,
            "sqrt_LD_LNL_m": self.sqrt_LD_LNL,
            "z_WB_m": self.z_WB,
            "z_onset_m": detect_steepening_onset(
                z_array, steepness, factor=steepening_factor
            ),
            "z_oscillation_m": detect_oscillation_onset(z_array, fields),
            "peak_steepness": float(steepness.max()),
            "steepness": steepness,
        }
