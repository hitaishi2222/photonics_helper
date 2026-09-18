"""Exact soliton-on-finite-background (breather) solutions of the NLSE.

Analytic solutions of the focusing nonlinear Schrödinger equation on a
non-zero background.  These describe modulation-instability growth and decay
(Akhmediev breather), the extreme-event limit localised in both dimensions
(Peregrine soliton), and the periodically breathing solution
(Kuznetsov–Ma soliton).  They are the structures seen in spontaneous MI and
rogue-wave experiments and are the ground truth used by the
``narhi_2016_mi_breathers`` and ``kuznetsov_ma_2012_breather`` reproductions.

Convention
----------
The dimensionless focusing NLSE used throughout this module is

.. math::

    i\\,\\partial_\\xi\\psi + \\tfrac{1}{2}\\partial_\\tau^2\\psi
        + |\\psi|^2\\psi = 0,

which is the normalisation of the fibre equation
``i A_z = (β₂/2) A_TT − γ|A|²A`` in the anomalous-dispersion regime
(``β₂ < 0``) under

.. math::

    A(z,T) = \\sqrt{P_0}\\,\\psi(\\xi,\\tau), \\quad
    T = \\tau T_0, \\quad z = \\xi L_{NL}, \\quad
    L_{NL} = \\frac{1}{\\gamma P_0}, \\quad
    T_0 = \\sqrt{|\\beta_2|\\,L_{NL}}.

The general solution (Akhmediev–Korneev) is

.. math::

    \\psi(\\xi,\\tau) = e^{i\\xi}\\left[1 +
        \\frac{2(1-2a)\\cosh(b\\xi) + i b\\sinh(b\\xi)}
             {\\sqrt{2a}\\cos(\\nu\\tau) - \\cosh(b\\xi)}\\right],

with :math:`b = \\sqrt{8a(1-2a)}` and :math:`\\nu = 2\\sqrt{1-2a}`.

* :math:`a < 1/2` — **Akhmediev breather** (:math:`b,\\nu` real): periodic in
  time, grows and decays once in :math:`\\xi`.
* :math:`a = 1/2` — **Peregrine soliton** (limiting case).
* :math:`a > 1/2` — **Kuznetsov–Ma soliton** (:math:`b,\\nu` imaginary):
  localised in time, periodic in :math:`\\xi`.

References
----------
- N. Akhmediev & V. I. Korneev, *Theor. Math. Phys.* **69**, 1089 (1986).
- D. H. Peregrine, *J. Austral. Math. Soc. Ser. B* **25**, 16 (1983).
- B. Kibler et al., *Nat. Phys.* **6**, 790 (2010); *Sci. Rep.* **2**, 463 (2012).
- M. Närhi et al., *Nat. Commun.* **7**, 13675 (2016).
- J. M. Dudley et al., *Nat. Photon.* **8**, 755 (2014).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Time, Wavelength

if TYPE_CHECKING:
    from .pulse import TemporalGrid, Wave

__all__ = [
    "peregrine_soliton",
    "general_sfb",
    "akhmediev_breather",
    "kuznetsov_ma",
    "sfb_peak_ratio",
    "sfb_spatial_period",
    "sfb_temporal_period",
    "SolitonOnBackground",
]

_PERE = 1e-9  # tolerance on a for the Peregrine limit


def _asarray(value: NDArray | float) -> NDArray:
    return np.asarray(value, dtype=float)


def peregrine_soliton(xi: float, tau: NDArray | float) -> NDArray | complex:
    """Exact Peregrine soliton of the focusing NLSE.

    ``psi(xi, tau) = e^{i xi} [1 - 4(1 + 2 i xi)/(1 + 4 tau^2 + 4 xi^2)]``.
    Peak-to-background intensity ratio is exactly ``|psi(0, 0)|^2 = 9``.
    """
    tau = _asarray(tau)
    return np.asarray(
        np.exp(1j * xi)
        * (1.0 - 4.0 * (1.0 + 2j * xi) / (1.0 + 4.0 * tau**2 + 4.0 * xi**2)),
        dtype=complex,
    )


def general_sfb(xi: float, tau: NDArray | float, a: float) -> NDArray | complex:
    """General soliton-on-finite-background solution (Akhmediev–Korneev).

    Parameters
    ----------
    xi : float — dimensionless propagation distance.
    tau : array-like — dimensionless retarded time.
    a : float — governing parameter in ``(0, 1)``; ``a < 1/2`` gives the
        Akhmediev breather, ``a > 1/2`` the Kuznetsov–Ma soliton and
        ``a -> 1/2`` the Peregrine soliton.
    """
    tau = _asarray(tau)
    if not (0.0 < a < 1.0):
        raise ValueError(f"a must lie in (0, 1), got {a!r}")
    if abs(a - 0.5) < _PERE:
        return peregrine_soliton(xi, tau)

    b = complex(np.emath.sqrt(8.0 * a * (1.0 - 2.0 * a)))
    nu = complex(2.0 * np.emath.sqrt(1.0 - 2.0 * a))
    num = 2.0 * (1.0 - 2.0 * a) * np.cosh(b * xi) + 1j * b * np.sinh(b * xi)
    den = np.sqrt(2.0 * a) * np.cos(nu * tau) - np.cosh(b * xi)
    return np.asarray(np.exp(1j * xi) * (1.0 + num / den), dtype=complex)


def akhmediev_breather(xi: float, tau: NDArray | float, a: float) -> NDArray | complex:
    """Akhmediev breather (``a < 1/2``): periodic in time, single growth/decay."""
    if not (0.0 < a < 0.5):
        raise ValueError(f"Akhmediev breather requires 0 < a < 1/2, got {a!r}")
    return general_sfb(xi, tau, a)


def kuznetsov_ma(xi: float, tau: NDArray | float, a: float) -> NDArray | complex:
    """Kuznetsov–Ma soliton (``a > 1/2``): localised in time, periodic in :math:`\\xi`."""
    if not (0.5 < a < 1.0):
        raise ValueError(f"Kuznetsov-Ma soliton requires 1/2 < a < 1, got {a!r}")
    return general_sfb(xi, tau, a)


def sfb_peak_ratio(a: float) -> float:
    """Maximum ``|psi|^2`` (relative to the unit background) of the SFB family.

    Gives ``5.828…`` for ``a = 0.25`` (Akhmediev), ``9`` for the Peregrine
    limit and ``10.876…`` for ``a = 0.66`` (Kuznetsov–Ma).
    """
    if abs(a - 0.5) < _PERE:
        return 9.0
    if not (0.0 < a < 1.0):
        raise ValueError(f"a must lie in (0, 1), got {a!r}")
    return float((1.0 + 2.0 * (1.0 - 2.0 * a) / (np.sqrt(2.0 * a) - 1.0)) ** 2)


def sfb_spatial_period(a: float) -> float:
    """Dimensionless :math:`\\xi` period of the Kuznetsov–Ma soliton (``a > 1/2``)."""
    if not (0.5 < a < 1.0):
        raise ValueError(f"spatial period is defined for 1/2 < a < 1, got {a!r}")
    return float(2.0 * np.pi / np.sqrt(8.0 * a * (2.0 * a - 1.0)))


def sfb_temporal_period(a: float) -> float:
    """Dimensionless :math:`\\tau` period of the Akhmediev breather (``a < 1/2``)."""
    if not (0.0 < a < 0.5):
        raise ValueError(f"temporal period is defined for 0 < a < 1/2, got {a!r}")
    return float(2.0 * np.pi / (2.0 * np.sqrt(1.0 - 2.0 * a)))


@dataclass(frozen=True)
class SolitonOnBackground:
    """Physical mapping of the SFB family onto a fibre/waveguide.

    Parameters
    ----------
    beta2 : float — group-velocity dispersion β₂ in **s²/m**. Must be negative
        (anomalous/focusing) for the breather solutions.
    gamma : float — nonlinear coefficient γ in 1/(W·m).
    P0 : float — background (plane-wave) power in W.
    """

    beta2: float
    gamma: float
    P0: float

    def __post_init__(self) -> None:
        if self.beta2 >= 0.0:
            raise ValueError(
                "Soliton-on-background solutions require anomalous dispersion "
                f"(beta2 < 0), got beta2={self.beta2!r} s^2/m"
            )
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma!r}")
        if self.P0 <= 0.0:
            raise ValueError(f"P0 must be positive, got {self.P0!r}")

    @property
    def L_NL(self) -> float:
        """Nonlinear length ``1/(γ P₀)`` in m."""
        return 1.0 / (self.gamma * self.P0)

    @property
    def T0(self) -> float:
        """Time scale ``sqrt(|β₂| L_NL)`` in s."""
        return float(np.sqrt(abs(self.beta2) * self.L_NL))

    def xi(self, z: float) -> float:
        return z / self.L_NL

    def tau(self, t: NDArray | float) -> NDArray | float:
        return np.asarray(t) / self.T0

    def psi(self, z: float, t: NDArray | float, a: float) -> NDArray:
        """Dimensionless SFB field at physical ``(z, t)``."""
        return np.asarray(general_sfb(self.xi(z), self.tau(t), a), dtype=complex)

    def field(self, z: float, t: NDArray | float, a: float) -> NDArray:
        """Physical envelope ``A(z, T) = sqrt(P0) psi`` in W^{1/2}."""
        return np.asarray(np.sqrt(self.P0) * self.psi(z, t, a), dtype=complex)

    def peak_ratio(self, a: float) -> float:
        """Maximum ``|A|²/P0`` of the SFB."""
        return sfb_peak_ratio(a)

    def peak_power(self, a: float) -> float:
        """Maximum instantaneous power ``P0 · |psi|²`` in W."""
        return self.P0 * sfb_peak_ratio(a)

    def spatial_period_m(self, a: float) -> float:
        """Kuznetsov–Ma breathing period in m (``a > 1/2``)."""
        return sfb_spatial_period(a) * self.L_NL

    def temporal_period_s(self, a: float) -> float:
        """Akhmediev temporal period in s (``a < 1/2``)."""
        return sfb_temporal_period(a) * self.T0

    def initial_wave(
        self,
        grid: "TemporalGrid",
        a: float,
        z0: float = 0.0,
        wavelength: Wavelength | None = None,
    ) -> "Wave":
        """Build a :class:`~photonics_helper.pulse.Wave` with the exact SFB field at ``z0``.

        Parameters
        ----------
        grid : TemporalGrid
            Time grid; for Akhmediev breathers choose a window containing an
            integer number of :meth:`temporal_period_s` periods.
        a : float — SFB parameter.
        z0 : float — physical starting distance in m (e.g. ``-3 L_NL`` for a
            Peregrine soliton so that the peak falls at ``z = 0``).
        wavelength : Wavelength — carrier; defaults to 1550 nm.
        """
        from .pulse import Envelope, Wave

        def func(t_in: NDArray, _T0: float, _A0: float) -> NDArray:
            del _T0, _A0
            return np.asarray(self.field(z0, t_in, a), dtype=complex)

        env = Envelope(
            shape="custom",
            peak_amplitude=float(np.sqrt(self.P0)),
            pulse_width=Time(self.T0, "s"),
            func=func,
        )
        return Wave(
            grid=grid,
            envelope=env,
            central_wavelength=wavelength or Wavelength(1550e-9, "m"),
        )
