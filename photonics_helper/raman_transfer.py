"""Raman RIN transfer analytics: closed forms and the continuous-wave cascade.

This module answers a narrow question with as little machinery as possible: *how does
relative intensity noise from a pump laser appear at the signal of a Raman amplifier, and
how does that transfer function depend on fiber loss, gain, length, dispersion, and
geometry?*

Three published results pin the answer down exactly, and all three are implemented here so
that a regression in the physics fails the test suite rather than the paper.

**Keita et al. 2006, JOSA B 23(12) 2479, doi 10.1364/JOSAB.23.002479**

.. math::

    \\sigma_{\\mathrm{dB}} = \\frac{20}{\\ln 10}
        \\ln\\!\\left[\\frac{\\ln\\!\\left(10^{g_{\\mathrm{net}}/20}\\right)}{\\alpha_p}
        + \\frac{\\alpha_s L}{2}\\right]

is the low-frequency transfer for a single pump, Keita Eq. 4.7. Note that the paper defines
:math:`\\sigma_{\\mathrm{dB}} = 10\\ln(\\sigma)/\\ln 10`, a **power** ratio, which differs
from the amplitude convention :math:`20\\log_{10}` used elsewhere in this library. The
loss-coefficient units must therefore be consistent with the paper. Their Eq. 4.9 and
Eq. 4.10 give the full co- and counter-propagating expressions including the walk-off term
:math:`\\Delta_k`.

**Zhu, Zhang, Zhang 2007, JLT 25(6) 1458, doi 10.1109/JLT.2007.895532**

.. math::

    |H(f)|^2 = \\frac{[\\ln G_R]^2\\,(V_s/L_{\\mathrm{eff}})^2}
                    {(\\alpha_p V_s)^2 + (4\\pi f)^2}

is the single-pump frequency-dependent transfer, Zhu Eq. 7, implying a DC level of
:math:`\\ln G_R/(\\alpha_p L_{\\mathrm{eff}})` and a corner at :math:`\\alpha_p V_s/(4\\pi)`.

**Mermelstein, Brar, Headley 2003, JLT 21(6) 1518, doi 10.1109/JLT.2003.812461**

solve the full dual-order continuous-wave cascade: three coupled steady-state power
equations plus six complex modulation-index equations, integrated as nine ODEs, with the
sign of the walk-off term selecting co- or counter-propagating geometry.

Units
-----
Every public function takes quantities in documented SI units and performs **no** implicit
conversion. The loss-coefficient normalization is the single most likely transcription error
in this physics, and implicit conversion hides it rather than surfacing it.

Benchmark fixtures
------------------
Published targets live under ``benchmarks/raman_noise/<paper>/fixture.json`` and are loaded
with :func:`load_benchmark`. Every numeric entry carries its source, DOI, unit, and whether it
was read from a table, a figure, or running text, so a reader can judge its precision.
Figure-transcribed values are marked ``kind: "figure"`` and should not be held to table
precision.

Scope
-----
**Amplifier, not laser.** Cavity boundaries, above-threshold dynamics, and distributed
Rayleigh feedback are out of scope here and are modelled in the later cascade stages. The
RIN transfer physics tested by every benchmark in this module lives in the amplifier.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import C_MS, H_PLANCK, Length, Wavelength, WavelengthArray

__all__ = [
    "BenchmarkFixture",
    "CWWCascade",
    "ProvenanceError",
    "RamanChannel",
    "Geometry",
    "build_cascade",
    "channel_wavelengths",
    "db_from_linear",
    "NoiseResponse",
    "double_pole_fit",
    "effective_length",
    "group_index",
    "load_benchmark",
    "photon_flux",
    "photon_flux_drift",
    "rin_transfer_dispersion",
    "rin_transfer_from_net_gain",
    "rin_transfer_low_frequency",
    "rin_transfer_monochromatic_pump",
    "single_pump_corner_frequency",
    "single_pump_transfer",
    "PropagationDirection",
    "walk_off",
    "walk_off_parameter",
]


class ProvenanceError(ValueError):
    """A benchmark fixture entry is missing required provenance metadata."""


_REQUIRED_PROVENANCE_FIELDS = ("value", "unit", "source", "doi", "kind")
_VALID_KINDS = frozenset({"table", "figure", "text", "equation"})


@dataclass(frozen=True, slots=True)
class BenchmarkFixture:
    """A loaded benchmark fixture with its provenance metadata intact."""

    key: str
    citation: str
    doi: str
    path: Path
    data: dict[str, Any]

    def entry(self, *keys: str) -> dict[str, Any]:
        """Return one provenance-tagged entry, validating it on the way out.

        Parameters
        ----------
        *keys
            Nested key path, for example ``("parameters", "loss_1375nm")``.
        """
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"{self.key}: no entry at {keys!r}")
            node = node[key]
        _validate_entry(self.key, keys, node)
        return dict(node)

    def value(self, *keys: str) -> float:
        """Return the numeric value of one provenance-tagged entry."""
        return float(self.entry(*keys)["value"])

    def is_figure_transcribed(self, *keys: str) -> bool:
        """Whether the entry was read by eye from a figure rather than a table."""
        return str(self.entry(*keys)["kind"]) == "figure"


def _validate_entry(key: str, keys: tuple[str, ...], node: Any) -> None:
    if not isinstance(node, dict):
        raise ProvenanceError(
            f"{key}: entry at {keys!r} is not a provenance record "
            f"(got {type(node).__name__}); every numeric value must be an object "
            f"with {sorted(_REQUIRED_PROVENANCE_FIELDS)}"
        )
    missing = [f for f in _REQUIRED_PROVENANCE_FIELDS if f not in node]
    if missing:
        raise ProvenanceError(
            f"{key}: entry at {keys!r} is missing required provenance field(s) "
            f"{missing}; the loader fails loudly rather than silently trusting "
            f"an unsourced number"
        )
    if not str(node["doi"]).strip():
        raise ProvenanceError(
            f"{key}: entry at {keys!r} has an empty DOI; every benchmark value must "
            f"be traceable to a resolvable source"
        )
    if node["kind"] not in _VALID_KINDS:
        raise ProvenanceError(
            f"{key}: entry at {keys!r} has kind {node['kind']!r}, expected one of "
            f"{sorted(_VALID_KINDS)}"
        )


# ---------------------------------------------------------------------------
# Unit conventions
# ---------------------------------------------------------------------------
#
# Keita 2006 works in inverse-length units throughout Eqs. 4.5 to 4.10. Its Fig. 1
# inset lists a loss of 0.046 /km, which is 0.2 dB/km, which fixes the convention.
# Every function below therefore takes ``alpha`` and ``gain_coefficient`` in inverse
# length and ``length`` in any consistent length unit. Where a loss is quoted in
# dB/km, convert with :func:`np.log(10) / 10`.
#
# Keita 2006 also defines rho_dB = 10 ln(rho) / ln(10), a POWER ratio, which differs
# from the amplitude convention 20 log10 used by :func:`db_from_linear` below. The
# transfer values in this module follow the paper.


class Geometry:
    """Propagation geometry, which sets the sign of the walk-off term.

    Keita 2006 writes Delta_k = [(1/v_p) -/+ (1/v_s)] nu_s with the upper sign for
    co-propagating and the lower sign for counter-propagating beams. Using the class
    rather than a raw sign keeps the choice visible at every call site, because an
    inverted geometry is silent: it produces a plausible curve with the corner
    frequency wrong by orders of magnitude.
    """

    CO_PROPAGATING = +1
    COUNTER_PROPAGATING = -1


class PropagationDirection:
    """Per-channel propagation direction of a multi-pump cascade.

    A channel descriptor carries its own direction so that an arbitrary pump set can be
    forward, backward, or a mixture, rather than the whole cascade sharing one global
    geometry. Zhu 2007's six pumps are backward and its signal is forward; Ma 2012's
    laser is co-propagating. Keeping the direction on the channel is what lets the same
    :class:`RamanChannel` describe both.

    The values are ``+1`` for forward and ``-1`` for backward, the same sign convention
    :class:`Geometry` uses, so a per-channel direction maps onto the solver's global
    geometry without a second translation table.
    """

    FORWARD = +1
    BACKWARD = -1


def db_from_linear(rho: float | NDArray[np.float64], *, amplitude: bool = True) -> float | NDArray[np.float64]:
    """Convert a linear power ratio to decibels.

    Parameters
    ----------
    rho
        Linear ratio. For RIN transfer this is a variance ratio, so pass
        ``amplitude=False`` to get the power convention Keita 2006 uses in Eq. 4.7.
    amplitude
        ``True`` uses the amplitude convention ``20 log10``; ``False`` uses the power
        convention ``10 log10``.
    """
    scale = 20.0 if amplitude else 10.0
    with np.errstate(divide="ignore"):
        return scale * np.log10(rho)


def effective_length(alpha_per_m: float, length: Length) -> Length:
    """Effective interaction length ``(1 - exp(-alpha L)) / alpha``, as a :class:`Length`.

    Parameters
    ----------
    alpha_per_m
        Loss in inverse metres.
    length
        Physical length.

    Notes
    -----
    Computed as ``-expm1(-alpha L) / alpha`` rather than
    ``(1 - exp(-alpha L)) / alpha``, because ``1 - exp(-x)`` cancels
    catastrophically for small ``x`` and a low-loss Raman amplifier is exactly that case.
    """
    l_m = length.as_m
    x = alpha_per_m * l_m
    if x == 0.0:
        return Length(l_m, "m")
    return Length(float(-np.expm1(-x) / alpha_per_m), "m")


# ---------------------------------------------------------------------------
# Keita 2006 closed forms
# ---------------------------------------------------------------------------


def walk_off_parameter(
    inv_v_pump: float,
    inv_v_signal: float,
    offset_hz: float,
    geometry: int = Geometry.COUNTER_PROPAGATING,
) -> float:
    r"""First-order walk-off term ``Delta_k`` of Keita 2006.

    .. math::

        \Delta_k^{(1)} = \left[\frac{1}{v_p} \mp \frac{1}{v_s}\right]\nu_s

    Parameters
    ----------
    inv_v_pump, inv_v_signal
        Inverse group velocities, in inverse velocity.
    offset_hz
        The modulation frequency :math:`\nu_s`.
    geometry
        :data:`Geometry.CO_PROPAGATING` for the upper sign, which cancels for equal
        group velocities, or :data:`Geometry.COUNTER_PROPAGATING` for the lower sign,
        which adds.
    """
    return float((inv_v_pump - geometry * inv_v_signal) * offset_hz)


def rin_transfer_monochromatic_pump(
    gain_coefficient: float,
    alpha_p: float,
    delta_k: float,
    length: float,
    geometry: int = Geometry.COUNTER_PROPAGATING,
) -> float:
    r"""RIN transfer for a modulated monochromatic pump, Keita 2006 Eq. 4.5.

    .. math::

        \rho = \frac{G^2}{\Delta_k^2 + \alpha_p^2}
               \left\{1 - 2\cos(\Delta_k L)\exp(-\alpha_p L)
                     + \exp(-2\alpha_p L)\right\}

    Parameters
    ----------
    gain_coefficient
        Raman gain coefficient :math:`G`, in inverse length. **Not** a linear gain
        ratio: the paper's Fig. 1 inset lists ``G = 4.74 /km`` with ``L = 1 km``, so
        ``G*L = 4.74``.
    alpha_p
        Pump loss in inverse length.
    delta_k
        Walk-off term from :func:`walk_off_parameter`, in inverse length.
    length
        Length, matching the inverse-length units above.
    geometry
        Unused. The geometry is carried entirely by the sign of ``delta_k``, obtained
        from :func:`walk_off_parameter`. Keita 2006 Eq. 4.5 has no separate geometry
        branch, unlike Eq. 4.9 and Eq. 4.10.

    Returns
    -------
    float
        Linear transfer ratio :math:`\\rho`, a power ratio. As ``delta_k`` tends to zero
        this reduces exactly to :func:`rin_transfer_low_frequency`.
    """
    del geometry  # sign of delta_k carries the geometry
    value = gain_coefficient**2 / (delta_k**2 + alpha_p**2) * (
        1.0
        - 2.0 * np.cos(delta_k * length) * np.exp(-alpha_p * length)
        + np.exp(-2.0 * alpha_p * length)
    )
    return float(value)


def rin_transfer_low_frequency(
    gain_coefficient: float, alpha_p: float, length: float
) -> float:
    r"""Low-frequency RIN transfer, Keita 2006 Eq. 4.6.

    This is the exact limit of :func:`rin_transfer_monochromatic_pump` as
    :math:`\Delta_k \to 0`, obtained by writing
    :math:`1 - 2 e^{-\alpha_p L} + e^{-2\alpha_p L} = (1 - e^{-\alpha_p L})^2`.

    .. math::

        \rho = \left(\frac{G}{\alpha_p}\right)^2
               \left[1 - \exp(-\alpha_p L)\right]^2
    """
    value = (gain_coefficient / alpha_p) ** 2 * (1.0 - np.exp(-alpha_p * length)) ** 2
    return float(value)


def rin_transfer_from_net_gain(g_net_db: float, alpha_p: float, length: float) -> float:
    r"""Low-frequency RIN transfer against net gain, Keita 2006 Eq. 4.7.

    .. math::

        \rho_{\mathrm{dB}} = \frac{20}{\ln 10}
            \ln\!\left[\frac{\ln 10}{20}\,g_{\mathrm{net}} + \frac{\alpha_p L}{2}\right]

    Parameters
    ----------
    g_net_db
        The paper's **net** gain in dB, which includes pump depletion. Note this is
        not ``20 log10(G*L)``: the Fig. 1 inset gives ``g_net = 40 dB`` alongside
        ``G = 4.74 /km`` and ``L = 1 km``, for which ``20 log10(G*L)`` is 13.5 dB. The
        mapping between the two is the paper's Eq. 3.7, which is not transcribed here,
        so this function is gated only at the single point the paper states.
    alpha_p
        Pump loss in inverse length.
    length
        Length, matching ``alpha_p``.

    Returns
    -------
    float
        The transfer in decibels, using the paper's power convention ``10 log10``.
    """
    bracket = (np.log(10.0) / 20.0) * g_net_db + alpha_p * length / 2.0
    if bracket <= 0.0:
        raise ValueError(
            f"Eq. 4.7 bracket is {bracket!r}, which is not positive; check that "
            f"g_net_db is large enough and that alpha_p * length is in inverse-length "
            f"units as Keita 2006 uses them"
        )
    return float(20.0 / np.log(10.0) * np.log(bracket))


def rin_transfer_dispersion(
    gain_coefficient: float,
    alpha_p: float,
    delta_k: float,
    length: float,
    geometry: int = Geometry.COUNTER_PROPAGATING,
) -> float:
    """Alias selecting Eq. 4.9 or Eq. 4.10 by geometry.

    Keita 2006 prints the two geometries as separate equations, Eq. 4.9 for
    co-propagating and Eq. 4.10 for counter-propagating. These describe the
    **large-bandwidth** pump case and are distinct from Eq. 4.5, which covers a
    modulated monochromatic pump. Here ``gain_coefficient`` is the Raman gain
    coefficient in inverse length, as printed.

    .. note::

       Eq. 4.9 and Eq. 4.10 describe the response when the pump bandwidth is much
       smaller than the Raman linewidth, which is the regime where the walk-off term
       :math:`\\Delta_k` alone shapes the response. Keita 2006 Eq. 4.8 states the
       approximation requires a propagation distance much greater than
       ``2 pi / ((1/v_p -/+ 1/v_s) Delta_nu_p)``, which is 1e-7 km for
       counter-propagating beams at a 10 nm pump linewidth but 3e-3 km for
       co-propagating beams even at ``D = 2 ps/(nm km)``. It is **not** satisfied for
       a 1 km fiber pumped by a 0.1 nm bandwidth source.
    """
    if geometry == Geometry.CO_PROPAGATING:
        denom = (gain_coefficient - alpha_p) ** 2 + delta_k**2
        bracket = (
            np.exp(-2.0 * alpha_p * length)
            - 2.0 * np.cos(delta_k * length)
            * np.exp(-(gain_coefficient + alpha_p) * length)
            + np.exp(-2.0 * gain_coefficient * length)
        )
    else:
        denom = (gain_coefficient + alpha_p) ** 2 + delta_k**2
        bracket = (
            np.exp(-2.0 * (gain_coefficient + alpha_p) * length)
            - 2.0 * np.cos(delta_k * length)
            * np.exp(-(gain_coefficient + alpha_p) * length)
            + 1.0
        )
    value = gain_coefficient**2 / denom * bracket
    return float(value)


# ---------------------------------------------------------------------------
# Zhu 2007 closed form
# ---------------------------------------------------------------------------


def single_pump_transfer(
    f_hz: NDArray[np.float64],
    g_on_off_db: float,
    alpha_p: float,
    v_signal: float,
    l_eff: float,
) -> NDArray[np.float64]:
    r"""Analytical RIN transfer for a single-pump amplifier, Zhu 2007 Eq. 7.

    .. math::

        |H(f)|^2 = \frac{[\ln G_R]^2\,(V_s/L_{\mathrm{eff}})^2}
                        {(\alpha_p V_s)^2 + (4\pi f)^2}

    Parameters
    ----------
    f_hz
        Offset frequency array, in Hz.
    g_on_off_db
        On-off Raman gain in dB, as a *power* ratio, so the linear gain is
        ``10**(g_on_off_db/10)``. The paper quotes 9.1 dB for its six-pump case,
        which is ``ln G = 2.10``, not ``1.04``. Using ``/20`` here (the amplitude
        convention) halves ``ln G`` and costs 6 dB of DC.
    alpha_p
        Fiber loss at the pump wavelength, in inverse length.
    v_signal
        Group velocity at the signal wavelength.
    l_eff
        Effective fiber length at the pump wavelength, from :func:`effective_length`.

    Notes
    -----
    This is the squared magnitude :math:`|H(f)|^2`, so the DC level of the *amplitude*
    transfer is :math:`\\ln(G_R)/(\\alpha_p L_{\\mathrm{eff}})` and that of
    :math:`|H|^2` is its square. Take the square root before converting to dB if you
    want amplitude decibels. Use :func:`single_pump_corner_frequency` for the corner
    rather than re-deriving it.
    """
    g_linear = 10.0 ** (g_on_off_db / 10.0)
    numerator = np.log(g_linear) ** 2 * (v_signal / l_eff) ** 2
    denominator = (alpha_p * v_signal) ** 2 + (4.0 * np.pi * f_hz) ** 2
    result: NDArray[np.float64] = numerator / denominator
    return result


def single_pump_corner_frequency(alpha_p: float, v_signal: float) -> float:
    """Corner frequency of Zhu 2007 Eq. 7, ``alpha_p * V_s / (4 pi)``, in Hz."""
    return float(alpha_p * v_signal / (4.0 * np.pi))


def group_index(
    wavelength_nm: float, group_index_zero: float, slope_ps_per_km_nm2: float, lambda0_nm: float
) -> float:
    r"""Group index from the zero-dispersion wavelength and the dispersion slope.

    Uses :math:`dn_g/d\lambda = c\,D(\lambda)` with
    :math:`D(\lambda) = S_0(\lambda-\lambda_0)`, which integrates to

    .. math::

        n_g(\lambda) = n_{g,0} + \frac{c S_0}{2}(\lambda-\lambda_0)^2

    .. note::

       Mermelstein 2003 prints ``n_i = n_0 + (S_0/8) (lambda_i - lambda_0^2/lambda_i)^2``.
       That form is dimensionally inconsistent, because the bracket carries units of
       length squared while the dispersion slope carries time per length cubed, so an
       implicit conversion must be hiding in the printed expression. Substituting
       Table I directly gives a group index of 2293, which is not a physical value.
       The relation implemented here is the one derived from
       :math:`D = -(\lambda/c)\,d^2n/d\lambda^2` and gives 1.46681 at 1560 nm,
       consistent with the standard single-mode value near 1.468 at 1550 nm.

    Parameters
    ----------
    wavelength_nm
        Evaluation wavelength, in nm.
    group_index_zero
        Group index at the zero-dispersion wavelength.
    slope_ps_per_km_nm2
        Dispersion slope at the zero-dispersion wavelength, in ps/(km nm^2).
    lambda0_nm
        Zero-dispersion wavelength, in nm.
    """
    delta = wavelength_nm - lambda0_nm
    return float(group_index_zero + 1.5e-7 * slope_ps_per_km_nm2 * delta**2)


@dataclass(frozen=True, slots=True)
class RamanChannel:
    """One channel of a continuous-wave Raman cascade.

    Attributes
    ----------
    index
        Position in the cascade, 1-based, ordered from highest to lowest frequency.
    label
        Human-readable name, for example ``"second_order_pump"``.
    wavelength
        Channel wavelength, as a :class:`base.Wavelength`. Used for the group index and
        the photon-flux weight.
    loss_db_per_km
        Fiber loss at this wavelength, in the usual dB/km convention. Converted to
        inverse metres internally.
    gains
        Raman gain coefficient to each other channel, in ``(W km)^-1``, indexed by that
        channel's ``index - 1``. Must satisfy ``gains[i][j] == gains[j][i]``, because
        Raman gain is a symmetric pair interaction.
    group_index
        Group index at this wavelength. Derived from the fiber Sellmeier parameters when
        left as ``None``. Mermelstein 2003 Eq. 2 defines the group velocity as
        ``v_i = c / n_i``, so the group index is the natural input and no separate
        velocity type is needed.
    """

    index: int
    label: str
    wavelength: Wavelength
    power_w: float
    loss_db_per_km: float
    gains: tuple[float, ...]
    group_index: float | None = None
    direction: int = PropagationDirection.FORWARD

    @property
    def wavelength_nm(self) -> float:
        """Channel wavelength in nm, for the Sellmeier group-index relation."""
        return float(self.wavelength.as_nm)

    @property
    def alpha_per_km(self) -> float:
        """Fiber loss in inverse kilometres, converted from dB/km."""
        return float(self.loss_db_per_km * np.log(10.0) / 10.0)

    @property
    def alpha_per_m(self) -> float:
        """Fiber loss in inverse metres, the internal length unit."""
        return self.alpha_per_km * 1e-3

    def velocity_m_s(self) -> float:
        """Group velocity in m/s, from ``v_i = c / n_i`` (Mermelstein 2003 Eq. 2)."""
        if self.group_index is None:
            raise ValueError(
                f"Channel {self.label!r} has no group_index; build the cascade with "
                f"build_cascade() so the Sellmeier parameters are applied"
            )
        return C_MS / self.group_index


def build_cascade(
    channels: Sequence[RamanChannel],
    *,
    group_index_zero: float,
    slope_ps_per_km_nm2: float,
    lambda0_nm: float,
) -> tuple[RamanChannel, ...]:
    """Fill in group velocities from Sellmeier parameters and validate the gain matrix.

    Parameters
    ----------
    channels
        Channels ordered by ``index``. Every channel's ``gains`` must have the same
        length as ``channels`` with a zero on its own diagonal.
    group_index_zero, slope_ps_per_km_nm2, lambda0_nm
        The fiber's zero-dispersion wavelength, dispersion slope, and group index there,
        following Mermelstein 2003. See :func:`group_index` for why the printed
        Mermelstein form is not used.

    Returns
    -------
    tuple of RamanChannel
        New channels with ``group_index`` populated.

    Raises
    ------
    ValueError
        If the gain matrix is not square with a zero diagonal, if it is asymmetric, or if
        the channel indices are not ``1..N`` in order.
    """
    n = len(channels)
    if n == 0:
        raise ValueError("a cascade needs at least one channel")
    for expected, ch in enumerate(channels, start=1):
        if ch.index != expected:
            raise ValueError(
                f"channel indices must be 1..{n} in order; got {ch.index} at position "
                f"{expected} (label {ch.label!r})"
            )
        if len(ch.gains) != n:
            raise ValueError(
                f"channel {ch.label!r} has {len(ch.gains)} gains but there are {n} "
                f"channels; the gain matrix must be square"
            )
        if ch.gains[ch.index - 1] != 0.0:
            raise ValueError(
                f"channel {ch.label!r} has a nonzero self-gain "
                f"{ch.gains[ch.index - 1]}; the diagonal must be zero"
            )
    for i, ch_i in enumerate(channels):
        for j in range(i + 1, n):
            a, b = ch_i.gains[j], channels[j].gains[i]
            if not np.isclose(a, b, rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"Raman gain is a symmetric pair interaction but channel "
                    f"{ch_i.label!r}->{channels[j].label!r} is {a} and "
                    f"{channels[j].label!r}->{ch_i.label!r} is {b}"
                )

    return tuple(
        RamanChannel(
            index=ch.index,
            label=ch.label,
            wavelength=ch.wavelength,
            power_w=ch.power_w,
            loss_db_per_km=ch.loss_db_per_km,
            gains=ch.gains,
            direction=ch.direction,
            group_index=(
                ch.group_index
                if ch.group_index is not None
                else group_index(
                    ch.wavelength_nm, group_index_zero, slope_ps_per_km_nm2, lambda0_nm
                )
            ),
        )
        for ch in channels
    )


def walk_off(
    observer: RamanChannel, source: RamanChannel, geometry: int = Geometry.COUNTER_PROPAGATING
) -> float:
    r"""Walk-off parameter ``d_ij`` of Mermelstein 2003 Eq. 2, in km^-1.

    .. math::

        d_{ij} = \frac{1}{v_j} \mp \frac{1}{v_i}

    with the upper sign for co-propagating and the lower for counter-propagating beams,
    in the observer's travelling frame.

    Parameters
    ----------
    observer
        The channel whose modulation is being propagated, ``i``.
    source
        The channel whose power fluctuation averages the observer, ``j``.
    geometry
        :data:`Geometry.CO_PROPAGATING` or :data:`Geometry.COUNTER_PROPAGATING`.

    Notes
    -----
    The paper writes ``d_31`` and ``d_32``, i.e. always with the signal as ``i``. The
    geometry sign multiplies the source term, so for co-propagating channels with equal
    group velocity the walk-off vanishes, which is the physical statement that
    co-propagating waves with matched group velocity do not separate.
    """
    inv_obs = 1.0 / observer.velocity_m_s()
    inv_src = 1.0 / source.velocity_m_s()
    return float(inv_src - geometry * inv_obs)


def photon_flux(power_w: float, wavelength: Wavelength) -> float:
    r"""Photon flux of a channel in photons per second, ``P * lambda / (h c)``.

    Real stimulated Raman scattering is photon-count preserving: a 1375 nm photon becomes
    a 1465 nm photon and the energy difference goes into a phonon. Use this when you want
    the photon picture, for example when comparing channel budgets across orders of very
    different wavelength.

    .. note::

       Note the ``P * lambda`` weighting. A photon count is ``P / E`` with
       ``E = hc / lambda``, so it grows with wavelength at fixed power; ``P / lambda``
       weights by photon *energy* and is conserved by nothing physical.

       Do **not** use this as a solver diagnostic. The Mermelstein 2003 power equations
       transfer equal power between a coupled pair rather than equal photon number, so
       their conserved quantity is power and the photon flux drifts by a few percent of
       the transferred power. Build the cascade with
       ``CWWCascade(..., photon_consistent=True)`` for a photon-conserving one, and then
       :func:`photon_flux_drift` is the solver check.
    """
    return power_w * wavelength.as_m / (H_PLANCK * C_MS)


def channel_wavelengths(channels: Sequence[RamanChannel]) -> WavelengthArray:
    """Collect the channel wavelengths into a :class:`base.WavelengthArray`."""
    return WavelengthArray.from_wavelengths([ch.wavelength for ch in channels])


def photon_flux_drift(
    powers_w: NDArray[np.float64],
    z_m: NDArray[np.float64],
    alpha_per_m: NDArray[np.float64],
    lambda_m: NDArray[np.float64],
) -> float:
    r"""Relative drift of the photon flux ``sum_i P_i lambda_i`` along a solved span.

    The weighting is ``P * lambda``, not ``P / lambda``: a photon count is
    ``P lambda / (h c)``, which a power-coupled pair conserves only once the Stokes power
    gain is reduced by ``lambda_pump / lambda_stokes``. That reduction is what
    ``CWWCascade(..., photon_consistent=True)`` applies. The fiber-loss contribution is
    subtracted analytically, so what remains is the part of the flux change the coupled
    equations do not account for.

    Parameters
    ----------
    powers_w
        Powers with shape ``(n_positions, n_channels)``, as :attr:`CascadeResult.powers_w`.
    z_m
        Ascending sample positions in metres.
    alpha_per_m, lambda_m
        Per-channel loss in inverse metres and wavelength in metres.

    Returns
    -------
    float
        Maximum absolute residual over the span, normalised by the flux entering at
        ``z = 0``.

    Notes
    -----
    This is near zero **only** for a cascade built with ``photon_consistent=True``.
    Mermelstein 2003 Eqs. 5a to 5c move equal power rather than equal photon number, so
    this diagnostic reports a few percent of the transferred power for them, and that is
    a statement about the model rather than about the integrator. Use
    :attr:`CascadeResult.power_balance_drift` as the solver check for that model.
    """
    return _balance_drift(
        powers_w, z_m, np.asarray(alpha_per_m, dtype=float), np.asarray(lambda_m, dtype=float), False
    )


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """Steady-state powers along the cascade."""

    z_m: NDArray[np.float64]
    """Sample positions in metres, always ascending from 0 to the span length."""

    @property
    def z_km(self) -> NDArray[np.float64]:
        """Sample positions in km."""
        result: NDArray[np.float64] = self.z_m * 1e-3
        return result

    powers_w: NDArray[np.float64]
    """Powers with shape ``(n_positions, n_channels)``, in channel index order."""

    labels: tuple[str, ...]
    """Channel labels, matching the column order of ``powers_w``."""

    power_balance_drift: float
    """Residual of power balance over the span, with the known fiber loss subtracted."""

    photon_flux_drift: float = 0.0
    """Residual of photon-flux balance, ``sum(P / lambda)``, with fiber loss subtracted.

    Zero to integration tolerance only when the cascade was built with
    ``photon_consistent=True``. In the default Mermelstein form this drifts by a few
    percent of the transferred power, which is a property of those equations rather than
    a solver error; see :func:`photon_flux_drift`.
    """

    signal_reversed: bool = False
    """Whether the signal column runs against the propagation direction."""

    length_m: float = 0.0
    """Span length in metres, used for the on-off gain loss correction."""

    @property
    def output_w(self) -> NDArray[np.float64]:
        """Power of each channel where it leaves the span.

        Only the signal is reversed in a counter-propagating cascade; the pumps always
        enter at ``z = 0`` and leave at ``z = L``.
        """
        result: NDArray[np.float64] = self.powers_w[-1].copy()
        if self.signal_reversed:
            result[-1] = self.powers_w[0, -1]
        return result

    @property
    def input_w(self) -> NDArray[np.float64]:
        """Power of each channel where it enters the span."""
        result: NDArray[np.float64] = self.powers_w[0].copy()
        if self.signal_reversed:
            result[-1] = self.powers_w[-1, -1]
        return result

    def raw_gain_db(self) -> NDArray[np.float64]:
        """Raw power gain ``10 log10(out/in)``, including fiber loss.

        This is **not** the figure the literature reports. A channel fed at 780 mW and
        leaving at 4.7e-4 W looks like -32 dB here even though it converted almost
        everything it could into the next order; most of the missing power went into
        fiber attenuation, not into the cascade.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            g = 10.0 * np.log10(self.output_w / self.input_w)
        result: NDArray[np.float64] = np.where(self.input_w > 0.0, g, np.nan)
        return result

    def on_off_gain_db(self, loss_per_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """On-off gain in dB, that is the raw gain with the fiber loss divided out.

        Parameters
        ----------
        loss_per_m
            Loss of each channel in inverse metres, in the same order as
            :attr:`labels`. Use :attr:`CWWCascade.loss_per_m`.

        Notes
        -----
        This is the quantity Mermelstein 2003 quotes as "approximately 13 dB of on/off
        gain". For the signal that is 13.41 dB here, against 2.67 dB of raw gain, so the
        distinction is not cosmetic: quoting raw gain would understate the result by more
        than 10 dB and would not match the paper.

        The loss correction uses the channel's own loss over the span it actually
        traverses, which for a counter-propagating signal is the same span length.
        """
        attenuation = np.exp(-np.asarray(loss_per_m, dtype=float) * self.length_m)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = 10.0 * np.log10((self.output_w / self.input_w) / attenuation)
        result: NDArray[np.float64] = np.where(self.input_w > 0.0, g, np.nan)
        return result


class CWWCascade:
    """Continuous-wave Raman cascade with a linearized noise response.

    Implements the dual-order model of Mermelstein, Brar, Headley 2003, JLT 21(6) 1518,
    doi 10.1109/JLT.2003.812461: three coupled steady-state power equations (their Eqs. 5a
    to 5c) plus six complex modulation-index equations for the noise response (their Eqs.
    6a to 6c), with the walk-off parameter of their Eq. 2 selecting co- or
    counter-propagating geometry.

    **Integration order.** Mermelstein integrates all nine equations together. This class
    splits them into passes, because the counter-propagating geometry makes the signal
    travel against the pump direction, which is a two-point problem rather than an
    initial-value problem:

    1. pumps forward from ``z = 0`` to ``z = L``;
    2. the signal steady state, forward for co-propagating and backward from ``z = L`` for
       counter-propagating;
    3. the modulation indices, with ``m1`` and ``m2`` following the pumps forward and
       ``m3`` following the signal.

    **Eq. 5b correction.** Mermelstein's Eq. 5b as printed depletes ``P2`` by
    ``gamma_23 * P1 * P3``, while their Eq. 5c gains the signal by ``gamma_23 * P2 * P3``.
    The same Raman interaction cannot be ``P1 P3`` on one side and ``P2 P3`` on the other,
    and their Eqs. 5a and 5c are self-consistent for both other gain pairs. This class
    implements the self-consistent ``- gamma_23 * P2 * P3``, which is also the only version
    under which the photon-flux conservation diagnostic is meaningful. See the
    ``eq_5_steady_state`` block in the Mermelstein fixture.

    **Scope.** Amplifier only. No cavity boundaries, no threshold, no distributed Rayleigh
    feedback.
    """

    def __init__(
        self,
        channels: Sequence[RamanChannel],
        length: Length,
        *,
        geometry: int = Geometry.COUNTER_PROPAGATING,
        n_points: int = 601,
        rtol: float = 1e-9,
        atol: float = 1e-14,
        photon_consistent: bool = False,
        noise_max_iter: int = 60,
    ) -> None:
        if len(channels) < 2:
            raise ValueError("a cascade needs at least two channels")
        self.channels = tuple(channels)
        self.length = length
        self.length_m = length.as_m
        self.geometry = geometry
        self.n_points = int(n_points)
        self.rtol = rtol
        self.atol = atol
        self.photon_consistent = bool(photon_consistent)
        self.noise_max_iter = int(noise_max_iter)
        self._gains = self._gain_array()
        self._alpha = np.array([ch.alpha_per_m for ch in self.channels])
        self._lambda = np.array([ch.wavelength.as_m for ch in self.channels])
        self._z_grid = np.asarray(
            np.linspace(0.0, self.length_m, self.n_points), dtype=np.float64
        )
        self._forward_sol: Any = None
        self._reverse_sol: Any = None
        self._result: CascadeResult | None = None

    @property
    def result(self) -> CascadeResult:
        """The most recent :meth:`solve` result.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been run.
        """
        if self._result is None:
            raise RuntimeError("call solve() before reading the result")
        return self._result

    @property
    def loss_per_m(self) -> NDArray[np.float64]:
        """Loss of each channel in inverse metres, in channel order."""
        result: NDArray[np.float64] = self._alpha.copy()
        return result

    def _gain_array(self) -> NDArray[np.float64]:
        """Raman gain matrix in ``(W m)^-1``, the internal inverse-length unit.

        ``RamanChannel.gains`` is in ``(W km)^-1``, which is the convention in the
        literature and in fiber data sheets, so it is scaled here rather than at every
        call site. Mixing per-kilometre gains with a per-metre integration variable makes
        the gain term 1000 times too large, which drives the whole cascade to zero within
        a few kilometres and is not obvious from the output.
        """
        n = len(self.channels)
        g = np.zeros((n, n), dtype=float)
        for ch in self.channels:
            g[ch.index - 1, :] = ch.gains
        return g * 1e-3

    def _signal_index(self) -> int:
        """The highest-index channel, which the model treats as the signal."""
        return len(self.channels) - 1

    def _backward_indices(self) -> NDArray[np.int64]:
        """Indices of channels launched at the far end, ascending.

        The counter-propagating geometry is confirmed by a contiguous leading block that
        shares the first channel's direction (the pumps, integrated forward from z=0)
        and every subsequent channel travels the other way: those are integrated
        backward from z=L. In a single-signal legacy cascade that block is exactly the
        probe channel; Zhu 2007's Fig. 1 case launches a whole WDM comb there, and each
        member both gains from the pumps and transfers power to its longer-wavelength
        neighbours. Integrating only one of them would miss the pump depletion that pins
        the published on-off gain.
        """
        from photonics_helper.raman_transfer import Geometry

        if self.geometry != Geometry.COUNTER_PROPAGATING:
            return np.array([], dtype=np.int64)
        d0 = self.channels[0].direction
        backward = np.array(
            [i for i, ch in enumerate(self.channels) if ch.direction != d0],
            dtype=np.int64,
        )
        if backward.size:
            return backward
        # Fallback for a legacy channel set where every direction is the same but the
        # geometry is counter-propagating: only the highest index is launched at L.
        return np.array([self._signal_index()], dtype=np.int64)

    def _energy_factor(self, donor: int, acceptor: int) -> float:
        """Power delivered to ``acceptor`` per watt removed from ``donor``.

        In the default mode this is exactly 1, which is what Mermelstein 2003 Eqs. 5a to
        5c do: they move equal power between a coupled pair. Real stimulated Raman
        scattering moves equal **photon number**, so the Stokes power gain is reduced by
        the ratio of wavelengths and the difference goes into a phonon. ``donor`` must
        be the higher-frequency channel, so ``lambda_donor < lambda_acceptor`` and the
        factor is below 1.
        """
        if not self.photon_consistent:
            return 1.0
        return float(self._lambda[donor] / self._lambda[acceptor])

    def _pump_rhs(
        self,
        z: float,
        y: NDArray[np.float64],
        n_pumps: int,
        signal: Callable[[float], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Steady-state power derivatives for the pump channels only.

        The signal power is supplied by ``signal(z)``, a callable returning the power of
        **every** backward channel at ``z``, because in the counter-propagating geometry
        the signal block is a two-point problem and cannot live in the same state vector
        as the pumps. In a single-signal cascade the array has one entry, which keeps
        the historical call sites working. It is built from the previous pass's dense
        output rather than from resampled grid points: piecewise-linear resampling
        carries a discretization error of order the grid spacing squared, which showed up
        as a 5e-4 power-balance residual that looked like a physics error and was not.
        """
        p = y[:n_pumps]
        p_sig = np.atleast_1d(np.asarray(signal(z), dtype=float))
        out = np.empty(n_pumps, dtype=float)
        for i in range(n_pumps):
            d = -self._alpha[i] * p[i]
            for j in range(n_pumps):
                if i == j:
                    continue
                if j < i:
                    # i is the lower-frequency member of the pair, so it gains.
                    d += self._energy_factor(j, i) * self._gains[i, j] * p[i] * p[j]
                else:
                    # i is the higher-frequency member, so it loses.
                    d -= self._gains[i, j] * p[i] * p[j]
            # Every pump sits above the signals in frequency, so it loses to each of
            # them. With a single signal this is the historical single term.
            for k, s_index in enumerate(self._signal_indices_vec):
                if i >= n_pumps or int(s_index) == i:
                    continue
                d -= self._gains[i, int(s_index)] * p[i] * p_sig[k]
            out[i] = d
        return out

    def _signal_rhs(
        self,
        z: float,
        y: NDArray[np.float64],
        pumps: Callable[[float], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Steady-state derivative for the backward (signal) channels at ``z``.

        With a single backward channel this is the historical one-element signal
        equation; with the WDM comb of Zhu 2007 it integrates every member at once, each
        gaining from the pumps and transferring power to its longer-wavelength
        neighbours exactly like the forward cascade does, because all backward channels
        travel with the signal and see the pumps at the same ``z``. ``pumps(z)`` returns
        the pump powers at an arbitrary position, a callable rather than positional
        indexing so that this works in the backward pass, where ``z`` runs against the
        integration direction.
        """
        s = self._signal_index()
        p_pumps = np.asarray(pumps(z), dtype=float)
        p_sig = np.atleast_1d(np.asarray(y, dtype=float))
        sig_idx = self._signal_indices_vec
        d = np.empty(p_sig.size, dtype=float)
        for a, ia in enumerate(sig_idx):
            i = int(ia)
            d[a] = -self._alpha[i] * p_sig[a]
            for j in range(min(s, p_pumps.size)):
                # pumps are all higher in frequency; every pump donates to this signal
                d[a] += self._energy_factor(j, i) * self._gains[i, j] * float(p_pumps[j]) * p_sig[a]
        # Mutual transfer inside the signal block: member a loses to longer-wavelength
        # members b > a and gains from b < a, exactly as the forward cascade does.
        for a in range(p_sig.size):
            ia = int(sig_idx[a])
            for b in range(p_sig.size):
                if a == b:
                    continue
                ib = int(sig_idx[b])
                g = self._gains[ia, ib]
                if g == 0.0:
                    continue
                if ia < ib:
                    # a is the higher-frequency member: it donates to b.
                    d[a] -= g * p_sig[a] * p_sig[b]
                else:
                    # a is the lower-frequency member: it gains from b.
                    d[a] += self._energy_factor(ib, ia) * g * p_sig[b] * p_sig[a]
        return d

    @property
    def _signal_indices_vec(self) -> NDArray[np.int64]:
        """Cached backward-channel index vector; the probe is always included."""
        idx = self._backward_indices()
        if idx.size == 0:
            idx = np.array([self._signal_index()], dtype=np.int64)
        return idx

    def _all_forward_rhs(self, _z: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Co-propagating steady state: all channels in one forward pass, Mermelstein 5a to 5c.

        Raman gain transfers power from the higher-frequency member of a pair to the
        lower-frequency one. For the pair ``(i, j)`` with ``i < j``, that means
        ``dP_i -= g_ij P_i P_j`` and ``dP_j += g_ij P_i P_j``, which is what Mermelstein's
        Eqs. 5a to 5c show: ``P1`` loses to both ``P2`` and ``P3``, ``P2`` gains from
        ``P1`` and loses to ``P3``, and ``P3`` gains from both. Getting this sign wrong
        makes both members of every pair lose, which annihilates the chain instead of
        cascading it.
        """
        n = len(self.channels)
        out = -self._alpha * y
        for i in range(n):
            for j in range(i + 1, n):
                g = self._gains[i, j]
                transfer = g * y[i] * y[j]
                out[i] -= transfer
                out[j] += self._energy_factor(i, j) * transfer
        return out

    def solve(self, *, max_iter: int = 200, tol: float = 1e-6) -> CascadeResult:
        """Integrate the steady state and return powers versus distance.

        Parameters
        ----------
        max_iter
            Iteration cap for the counter-propagating fixed point. Unused for
            co-propagating, which is a single forward pass.
        tol
            Convergence tolerance on the maximum relative change in signal power. The
            iteration converges geometrically with a rate set by the fraction of pump
            power drained into the signal, about 0.1 per pass for this configuration. It
            then stalls near 6e-8, which is the noise floor of the underlying ODE
            integrations at ``rtol=1e-9`` rather than a property of the fixed point, so
            the default sits just above that floor. Tightening ``tol`` below it without
            also tightening ``rtol`` will exhaust ``max_iter`` and raise.

        Returns
        -------
        CascadeResult
            Powers sampled on an ascending grid from ``0`` to the span length. In the
            counter-propagating geometry the signal is injected at ``z = L`` and read out
            at ``z = 0``, so ``signal_reversed`` is set and the ``input_w`` and
            ``output_w`` helpers stay orientation-aware.

        Raises
        ------
        RuntimeError
            If an integration fails or the counter-propagating fixed point does not
            converge. An unconverged cascade is never returned, because it would look
            like a physical result.

        Notes
        -----
        The counter-propagating case is a two-point boundary value problem: pumps are
        specified at ``z = 0`` and the signal at ``z = L``. It is solved by fixed-point
        iteration between a forward pump pass and a backward signal pass. The iteration
        is well conditioned here because the signal is roughly four orders of magnitude
        weaker than the pumps, so the loop gain is the fraction of pump power drained into
        the signal and is small.
        """
        from scipy.integrate import solve_ivp

        n = len(self.channels)
        s = self._signal_index()
        bwd = self._backward_indices()
        # Pumps are every channel before the first backward one; the probe stays the
        # highest-index channel and is always inside the backward block.
        n_pumps = int(bwd[0]) if bwd.size else s
        z_grid = np.asarray(
            np.linspace(0.0, self.length_m, self.n_points), dtype=np.float64
        )
        self._z_grid = z_grid
        p_sig_in = self.channels[s].power_w
        counter = self.geometry == Geometry.COUNTER_PROPAGATING

        if not counter:
            y0 = np.array([ch.power_w for ch in self.channels], dtype=float)
            sol = solve_ivp(
                self._all_forward_rhs,
                (0.0, self.length_m),
                y0,
                t_eval=z_grid,
                dense_output=True,
                rtol=self.rtol,
                atol=self.atol,
            )
            if not sol.success:
                raise RuntimeError(f"steady-state integration failed: {sol.message}")
            powers = sol.y.T
            self._forward_sol = sol
            self._reverse_sol = None
        else:
            y0 = np.array([ch.power_w for ch in self.channels[:n_pumps]], dtype=float)
            # The first pass assumes a flat signal, then each pass replaces the signal
            # callable with the previous pass's dense interpolant. The convergence test
            # compares the new profile against the old one on the sample grid, since that
            # is where the result is read back.
            signal_sol: Any = None
            delta = np.inf
            sig_launch = np.array(
                [self.channels[int(i)].power_w for i in self._signal_indices_vec],
                dtype=float,
            )
            for _ in range(max_iter):
                if signal_sol is None:
                    signal_at: Any = lambda _z, _p=p_sig_in: np.full(  # noqa: E731
                        self._signal_indices_vec.size, _p
                    )
                else:
                    signal_at = lambda z, _s=signal_sol: np.asarray(_s.sol(z))  # noqa: E731
                sol = solve_ivp(
                    self._pump_rhs,
                    (0.0, self.length_m),
                    y0,
                    t_eval=z_grid,
                    dense_output=True,
                    args=(n_pumps, signal_at),
                    rtol=self.rtol,
                    atol=self.atol,
                )
                if not sol.success:
                    raise RuntimeError(f"pump integration failed: {sol.message}")
                pumps = sol.y.T
                sig_sol = solve_ivp(
                    self._signal_rhs,
                    (self.length_m, 0.0),
                    sig_launch,
                    t_eval=z_grid[::-1],
                    dense_output=True,
                    args=(lambda z: sol.sol(z)[:n_pumps],),
                    rtol=self.rtol,
                    atol=self.atol,
                )
                if not sig_sol.success:
                    raise RuntimeError(f"signal integration failed: {sig_sol.message}")
                new_signal = sig_sol.sol(z_grid)
                if signal_sol is None:
                    delta = np.inf
                else:
                    delta = float(
                        np.max(np.abs(new_signal - signal_sol.sol(z_grid)))
                        / max(float(np.max(np.abs(sig_launch))), 1e-300)
                    )
                signal_sol = sig_sol
                if delta < tol:
                    break
            else:
                raise RuntimeError(
                    f"counter-propagating fixed point did not converge in {max_iter} "
                    f"iterations (last relative change {delta:.3e}); the signal drains a "
                    f"non-negligible fraction of the pumps"
                )
            # The loop exits with the pumps integrated against the *previous* signal, so
            # the two columns describe marginally different states. One further pass in
            # each direction makes them a matched pair, which is what the conservation
            # diagnostic and any reader of powers_w are entitled to assume.
            sol = solve_ivp(
                self._pump_rhs,
                (0.0, self.length_m),
                y0,
                t_eval=z_grid,
                dense_output=True,
                args=(n_pumps, lambda z: np.asarray(signal_sol.sol(z))),
                rtol=self.rtol,
                atol=self.atol,
            )
            if not sol.success:
                raise RuntimeError(f"final pump integration failed: {sol.message}")
            sig_sol = solve_ivp(
                self._signal_rhs,
                (self.length_m, 0.0),
                sig_launch,
                t_eval=z_grid[::-1],
                dense_output=True,
                args=(lambda z: sol.sol(z)[:n_pumps],),
                rtol=self.rtol,
                atol=self.atol,
            )
            if not sig_sol.success:
                raise RuntimeError(f"final signal integration failed: {sig_sol.message}")
            pumps = sol.y.T
            powers = np.zeros((self.n_points, n), dtype=float)
            powers[:, :n_pumps] = pumps
            for a, ia in enumerate(self._signal_indices_vec):
                powers[:, int(ia)] = sig_sol.sol(z_grid)[a]
            # Any channel that is neither a pump nor backward defaults to its launch
            # power (the historical single-signal path had no others).
        # Kept for the linearized noise response, which is integrated along the same
        # converged steady state rather than re-solving it per modulation frequency.
        self._forward_sol = sol
        self._reverse_sol = sig_sol if counter else None

        # Conservation is evaluated on a grid far finer than the returned one. The
        # diagnostic integrates the fiber-loss rate numerically, and at 601 points over
        # 60 km that integration alone contributes about 5e-4, which would swamp the
        # 1e-6 target and read as a physics error.
        z_fine = np.asarray(
            np.linspace(0.0, self.length_m, self.n_points * 100), dtype=np.float64
        )
        powers_fine = (
            sol.sol(z_fine).T
            if not counter
            else np.column_stack(
                (
                    sol.sol(z_fine)[:n_pumps].T,
                    *
                    [
                        sig_sol.sol(z_fine)[a]
                        for a in range(sig_launch.size)
                    ],
                )
            )
        )
        power_drift = _balance_drift(powers_fine, z_fine, self._alpha, None)
        flux_drift = _balance_drift(powers_fine, z_fine, self._alpha, self._lambda)
        conserved = flux_drift if self.photon_consistent else power_drift
        if conserved > 1e-6:
            warnings.warn(
                f"{'photon flux' if self.photon_consistent else 'power'} balance drift is "
                f"{conserved:.3e} relative, above the 1e-6 target; the conservative "
                f"quantity of this model is "
                f"{'photon flux' if self.photon_consistent else 'power'}, so a drift "
                f"above 1e-6 points at the integration tolerances "
                f"(rtol={self.rtol}, atol={self.atol}) rather than at the physics",
                RuntimeWarning,
                stacklevel=2,
            )

        self._result = CascadeResult(
            z_m=z_grid,
            powers_w=powers,
            labels=tuple(ch.label for ch in self.channels),
            power_balance_drift=power_drift,
            photon_flux_drift=flux_drift,
            signal_reversed=counter,
            length_m=self.length_m,
        )
        return self._result

    def powers_at(self, z_m: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Steady-state channel powers at arbitrary positions, shape ``(..., n_channels)``.

        Requires :meth:`solve` to have run. This reads the dense interpolant of the
        solved system rather than resampling the output grid, so the linearized noise
        response sees the same profiles the conservation diagnostics were computed from.
        """
        if self._forward_sol is None:
            raise RuntimeError("call solve() before asking for the steady state")
        z = np.atleast_1d(np.asarray(z_m, dtype=float))
        n = len(self.channels)
        if self._forward_sol is None:  # pragma: no cover - guarded by the check above
            raise RuntimeError("call solve() before asking for the steady state")
        if self._reverse_sol is None:
            forward_out: NDArray[np.float64] = np.asarray(
                self._forward_sol.sol(z), dtype=float
            ).T
            return forward_out
        forward = np.asarray(self._forward_sol.sol(z), dtype=float)
        signals = np.asarray(self._reverse_sol.sol(z), dtype=float)
        out = np.zeros((z.size, n), dtype=float)
        for a, ia in enumerate(self._signal_indices_vec):
            out[:, int(ia)] = signals[a]
        # The pump integration carries only the pump block in the counter-propagating
        # geometry; every remaining column is a backward channel solved above.
        n_pumps = n - signals.shape[0]
        out[:, :n_pumps] = forward.T
        return out

    def _modulation_rhs(self, omega: float) -> tuple[Any, Any]:
        r"""Build the complex modulation-index right-hand side and launch condition.

        Parameters
        ----------
        omega
            Modulation angular frequency ``2 pi f``, in inverse seconds.

        Returns
        -------
        tuple
            ``(rhs, m0)``. ``rhs(z, m)`` gives the complex derivative of the ``n``
            modulation indices and ``m0`` has 1.0 on the perturbed pump and 0 elsewhere.

        Notes
        -----
        **This is Mermelstein's printed Eqs. 6, with no self terms, and that is exact.**
        Linearizing a power equation for the *relative* modulation index M = dP / Pbar
        makes the steady-state net gain cancel identically: it multiplies Pbar on the
        gain side and dP = Pbar M on the fluctuation side by the same factor, so only
        the cross terms dP_j Pbar_i / Pbar_i survive. Adding a diagonal
        ``-(alpha_i + sum_j gamma_ij P_j) m_i``, as an earlier revision did as a
        "necessary correction", is the *absolute* perturbation's equation and is wrong
        for a relative one: it re-amplifies the pumps' own noise by the whole Raman net
        gain, which collapsed the published 15 dB gap between the two pumps down to
        under 1 dB (observed 17.8 dB vs 0.04 dB and 18.6 dB against published 15.6 dB
        and 0.04 dB for the counter-propagating case). The printed model reproduces
        the gap at 14.3 dB instead.

        **Retardation.** Mermelstein's Eqs. 6a and 6b carry a term ``i Omega d_31 m1``
        and ``i Omega d_32 m2``: the walk-off between a pump and the signal multiplies
        that pump's **own** modulation index. That is the right place for it and it is
        what produces the observed behaviour. Rotating ``m_i`` against the gain envelope
        turns the transfer into a Fourier-type integral whose magnitude falls as
        ``1 / |A_i - i Omega d_i|``, i.e. a clean single pole with a 6 dB corner at
        ``A_i / (2 pi d_i)``. The corner therefore scales as the inverse of the walk-off,
        which is why the co-propagating corner sits four orders of magnitude above the
        counter-propagating one: the group velocities nearly cancel in one geometry and
        add in the other. Putting the retardation on the coupling term instead, as a
        ``z``-weighted phase, produces a resonance rather than a pole, because the phase
        then oscillates inside the integration and averages out. An inverted geometry sign
        swaps the two silently, so the geometry is asserted on the result rather than
        trusted.
        """
        n = len(self.channels)
        signal = n - 1
        counter = self.geometry == Geometry.COUNTER_PROPAGATING
        gains = self._gains
        # d_i = 1/v_signal -/+ 1/v_i, upper sign co-propagating, as in Mermelstein's
        # Eqs. 2a and 2b. Vanishes for co-propagating channels of equal group velocity.
        d_walkoff = np.array(
            [
                1.0 / self.channels[signal].velocity_m_s()
                - (1.0 if not counter else -1.0) / self.channels[i].velocity_m_s()
                for i in range(n)
            ]
        )
        # The signal index carries no retardation term; Mermelstein's Eq. 6c has none.
        d_walkoff[signal] = 0.0

        def rhs(z: float, m: NDArray[np.complex128]) -> NDArray[np.complex128]:
            # The gain envelope is the local steady state, which varies along the span;
            # the walk-off term is a constant, since it is a property of the group
            # velocities alone.
            p = self.powers_at(z)[0]
            coupling = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i gains from a lower-index (higher-frequency) source and loses to a
                    # higher-index one, matching the sign of the pair term in Eqs. 5.
                    coupling[i, j] = (1.0 if j < i else -1.0) * gains[i, j] * p[j]
            # Relative modulation index: no self term (see the docstring above). The
            # walk-off retardation stays on the diagonal, where Mermelstein has it too.
            diagonal = 1j * omega * d_walkoff
            vector: NDArray[np.complex128] = np.diag(-diagonal) + coupling
            result: NDArray[np.complex128] = vector @ np.asarray(m, dtype=complex)
            return result

        m0 = np.zeros(n, dtype=complex)
        return rhs, m0

    def noise_response(
        self, frequencies_hz: NDArray[np.float64], source: int = 1
    ) -> NoiseResponse:
        r"""Complex RIN transfer from one perturbed pump to the signal.

        Linearizes Eqs. 5 about the solved steady state and integrates the complex
        modulation indices with the source pump launched at ``m_source = 1``. The
        returned transfer is ``m_signal(output) / m_source(launch)``, the dimensionless
        amplitude ratio a 100 percent pump modulation produces at the signal.

        Parameters
        ----------
        frequencies_hz
            Offset frequencies in Hz, at least two points.
        source
            1-based index of the perturbed pump. Mermelstein 2003 Figs. 5 and 6 report
            the second-order (1) and first-order (2) pump responses separately.

        Returns
        -------
        NoiseResponse
            Frequencies, complex transfer, and amplitude in dB.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been run or an integration fails.
        """
        from scipy.integrate import solve_ivp

        f = np.asarray(frequencies_hz, dtype=float)
        if f.size < 2:
            raise ValueError("a transfer function needs at least two frequencies")
        if not 1 <= source <= len(self.channels) - 1:
            raise ValueError(
                f"source {source} is not a pump; only channels 1..{len(self.channels) - 1} "
                f"drive the signal"
            )
        n = len(self.channels)
        signal = n - 1
        z_grid = np.asarray(
            np.linspace(0.0, self.length_m, self.n_points), dtype=np.float64
        )
        transfer = np.empty(f.size, dtype=complex)

        for k, freq in enumerate(f):
            rhs, m0 = self._modulation_rhs(2.0 * np.pi * float(freq))
            m0[source - 1] = 1.0
            if self._reverse_sol is None:
                sol = solve_ivp(
                    rhs,
                    (0.0, self.length_m),
                    m0,
                    t_eval=z_grid,
                    dense_output=True,
                    rtol=self.rtol,
                    atol=1e-13,
                )
                if not sol.success:
                    raise RuntimeError(f"noise integration failed: {sol.message}")
                m_sig = complex(sol.sol(self.length_m)[signal])
            else:
                # Two-point problem: the pump perturbation is launched at z = 0 while the
                # signal perturbation is free at the far end, so the two cannot march in
                # one pass. They are marched alternately. A direct boundary-value solve is
                # the obvious alternative and was tried first; it does not converge here,
                # because the walk-off rotates m_i by omega * d_i * L radians across the
                # span, which makes the two-point system resonant. The fixed point has the
                # same structure as the steady-state cascade and converges geometrically.
                m_signal: Any = None
                for _ in range(self.noise_max_iter):
                    previous: Any = m_signal
                    seed: Any = (lambda _z: 0.0 + 0.0j) if previous is None else previous
                    fwd = solve_ivp(
                        _forward_noise_rhs(rhs, signal, seed),
                        (0.0, self.length_m),
                        m0,
                        t_eval=z_grid,
                        dense_output=True,
                        rtol=self.rtol,
                        atol=1e-13,
                    )
                    if not fwd.success:
                        raise RuntimeError(f"noise pump integration failed: {fwd.message}")
                    back = solve_ivp(
                        _reverse_noise_rhs(rhs, signal, n, fwd.sol),
                        (self.length_m, 0.0),
                        np.zeros(1, dtype=complex),
                        t_eval=z_grid[::-1],
                        dense_output=True,
                        rtol=self.rtol,
                        atol=1e-13,
                    )
                    if not back.success:
                        raise RuntimeError(f"noise signal integration failed: {back.message}")
                    m_signal = lambda z, _b=back.sol: complex(_b(z)[0])  # noqa: E731
                    if previous is not None and _noise_delta(previous, m_signal, z_grid) < 1e-10:
                        break
                else:
                    raise RuntimeError(
                        f"counter-propagating noise response did not converge in "
                        f"{self.noise_max_iter} iterations"
                    )
                m_sig = complex(m_signal(0.0))

            transfer[k] = m_sig / m0[source - 1]

        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.abs(transfer))
        return NoiseResponse(frequencies_hz=f, transfer=transfer, db=db, source=source)

    def modulation_indices(
        self,
        z_m: NDArray[np.float64],
        frequency_hz: float,
        source: int = 1,
    ) -> NDArray[np.complex128]:
        r"""Complex modulation index of every channel along the span.

        This is the same linearized solution :meth:`noise_response` reads at one point, kept
        as a profile so that the spatial build-up of a perturbation can be inspected. It
        is what Mermelstein 2003 Fig. 7 plots, and what the interaction-length check
        measures.

        Parameters
        ----------
        z_m
            Ascending positions in metres.
        frequency_hz
            Modulation frequency in Hz.
        source
            1-based index of the perturbed pump.

        Returns
        -------
        NDArray of complex
            One row per position, one column per channel, in channel index order.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been run or an integration fails.
        """
        from scipy.integrate import solve_ivp

        z = np.asarray(z_m, dtype=float)
        if not 1 <= source <= len(self.channels) - 1:
            raise ValueError(
                f"source {source} is not a pump; only channels 1..{len(self.channels) - 1} "
                f"drive the signal"
            )
        n = len(self.channels)
        signal = n - 1
        rhs, m0 = self._modulation_rhs(2.0 * np.pi * float(frequency_hz))
        m0[source - 1] = 1.0
        m_signal = None
        if self._reverse_sol is None:
            sol = solve_ivp(
                rhs,
                (0.0, self.length_m),
                m0,
                t_eval=z,
                dense_output=True,
                rtol=self.rtol,
                atol=1e-13,
            )
            if not sol.success:
                raise RuntimeError(f"noise integration failed: {sol.message}")
            return np.asarray(sol.sol(z), dtype=complex).T
        for _ in range(self.noise_max_iter):
            previous = m_signal
            seed: Any = (lambda _z: 0.0 + 0.0j) if previous is None else previous
            fwd = solve_ivp(
                _forward_noise_rhs(rhs, signal, seed),
                (0.0, self.length_m),
                m0,
                t_eval=z,
                dense_output=True,
                rtol=self.rtol,
                atol=1e-13,
            )
            if not fwd.success:
                raise RuntimeError(f"noise pump integration failed: {fwd.message}")
            back = solve_ivp(
                _reverse_noise_rhs(rhs, signal, n, fwd.sol),
                (self.length_m, 0.0),
                np.zeros(1, dtype=complex),
                t_eval=z[::-1],
                dense_output=True,
                rtol=self.rtol,
                atol=1e-13,
            )
            if not back.success:
                raise RuntimeError(f"noise signal integration failed: {back.message}")
            m_signal = lambda zi, _b=back.sol: complex(_b(zi)[0])  # noqa: E731
            if previous is not None and _noise_delta(previous, m_signal, z) < 1e-10:
                break
        else:
            raise RuntimeError(
                f"counter-propagating modulation profile did not converge in "
                f"{self.noise_max_iter} iterations"
            )
        out = np.asarray(fwd.sol(z), dtype=complex).T
        out[:, signal] = np.array([m_signal(zi) for zi in z])
        return out

    def interaction_length_km(
        self,
        frequency_hz: float,
        *,
        indirect: bool,
        threshold: float = 0.37,
    ) -> float:
        r"""Full width at ``1/e`` of the first-order pump's power fluctuation, in km.

        Under **direct** modulation the perturbed pump is the first-order pump itself;
        under **indirect** modulation the second-order pump is perturbed and the
        first-order pump's perturbation is induced through the cascade, which is what
        Mermelstein 2003 Fig. 7 compares.

        Parameters
        ----------
        frequency_hz
            Modulation frequency in Hz.
        indirect
            ``False`` to modulate the first-order pump directly, ``True`` to modulate the
            second-order pump and measure the induced first-order response.
        threshold
            Fraction of the peak the fluctuation must fall to, ``1/e`` by default.

        Returns
        -------
        float
            Distance in kilometres between the rising and falling crossings of
            ``threshold`` times the peak of the **absolute** power fluctuation
            ``dP_2(z) = |m_2(z)| P_2(z)``.

        Notes
        -----
        Two details matter and both were wrong in the first implementation.

        **Absolute, not relative.** Mermelstein's Fig. 7 plots the 1465 nm pump power
        fluctuation ``dP_2`` in milliwatts, not the relative modulation index ``m_2``.
        The first-order pump is amplified along the span, so ``dP_2 = m_2 P_2`` rises as
        the pump grows even where ``m_2`` is flat, and the two have their peaks at
        different positions. Measuring the relative index puts the direct peak at
        ``z = 0`` and gives 17.4 km; measuring the absolute fluctuation puts it at the
        pump-power maximum and gives the published 20.5 km.

        **Full width, not distance from the peak.** The figure's two arrows span both the
        rising and the falling ``1/e`` crossing. For the indirect case the fluctuation is
        still above ``1/e`` of its peak at ``z = L``, so a distance-from-peak measurement
        never terminates; the full width does.

        This is a mechanism check, not a fitted quantity. It is worth having because a
        model that gets the corner frequencies right by construction, rather than through
        walk-off and spatial averaging, produces the two lengths in the wrong order.
        """
        z = np.asarray(np.linspace(0.0, self.length_m, self.n_points), dtype=np.float64)
        source = 1 if indirect else 2
        observer = 1
        profile = self.modulation_indices(z, frequency_hz, source=source)
        pump_power = np.asarray(self.powers_at(z), dtype=float)[:, observer]
        magnitude = np.abs(profile[:, observer]) * pump_power
        if not np.any(magnitude > 0.0):
            raise ValueError(
                f"the first-order pump carries no modulation at {frequency_hz} Hz under "
                f"{'indirect' if indirect else 'direct'} modulation; there is no "
                f"interaction length to measure"
            )
        peak = int(np.argmax(magnitude))
        above = np.flatnonzero(magnitude >= threshold * magnitude[peak])
        return float((z[above[-1]] - z[above[0]]) * 1e-3)

    def _power_balance_drift(self, powers: NDArray[np.float64]) -> float:
        """Residual of power balance on the returned grid, after subtracting fiber loss.

        See :func:`_balance_drift`, which does the work and carries the reasoning.
        """
        return _balance_drift(powers, self._z_grid, self._alpha, None)


def _balance_drift(
    powers: NDArray[np.float64],
    z: NDArray[np.float64],
    alpha: NDArray[np.float64],
    weights: NDArray[np.float64] | None,
    last_channel_reversed: bool = False,
) -> float:
    r"""Residual of a weighted channel sum after subtracting fiber loss.

    With ``weights = None`` this sums raw power, which is the conserved quantity of
    Mermelstein 2003 Eqs. 5a to 5c: they move **equal power** between a coupled pair,
    ``dP_i = -g P_i P_j`` and ``dP_j = +g P_i P_j``. With ``weights = lambda`` it sums
    the photon flux ``P * lambda``, the conserved quantity of the same equations once
    ``CWWCascade(..., photon_consistent=True)`` rescales the Stokes gain.

    Real stimulated Raman scattering converts photons and dumps the energy difference
    into a phonon, so the two conserved quantities genuinely differ by a few percent of
    the transferred power. Neither is a solver error, which is why both are reported
    rather than one being silently chosen.

    The counter-propagating signal needs no special handling here, which is worth being
    explicit about because it is easy to get backwards. A signal travelling against ``z``
    satisfies ``dP_s/dz = rhs`` where ``rhs`` is the same right-hand side as a
    co-propagating channel, and ``rhs`` carries ``-alpha_s P_s`` and a positive gain
    term; integrating it over a decreasing ``z`` makes the power grow towards ``-z`` on
    its own. So the budget is the plain sum over all channels, and only the input and
    output bookkeeping in :attr:`CascadeResult.input_w` and
    :attr:`CascadeResult.output_w` needs to know about the reversal.

    Returns
    -------
    float
        Maximum absolute residual over the span, normalised by the launch value of the
        weighted sum. A correct solver gives a value set by the integration tolerance.
    """
    from scipy.integrate import cumulative_trapezoid

    scaled = powers if weights is None else powers * weights
    total = scaled.sum(axis=1)
    removal_rate = -(scaled * alpha).sum(axis=1)
    removed = cumulative_trapezoid(removal_rate, z, initial=0.0)
    residual = (total - total[0]) - removed
    scale = abs(total[0])
    if scale == 0.0:
        return 0.0
    return float(np.max(np.abs(residual)) / scale)


@dataclass(frozen=True, slots=True)
class NoiseResponse:
    """Frequency-dependent RIN transfer from one perturbed pump to the signal."""

    frequencies_hz: NDArray[np.float64]
    """Offset frequencies in Hz."""

    transfer: NDArray[np.complex128]
    """Complex modulation-index ratio ``m_signal / m_source`` at the signal output."""

    db: NDArray[np.float64]
    """``20 log10 |transfer|``, the amplitude convention used by Mermelstein 2003."""

    source: int
    """1-based index of the perturbed pump channel."""

    def double_pole(self) -> tuple[float, float]:
        """Fit :func:`double_pole_fit` to this response and return ``(dc_db, corner_hz)``."""
        return double_pole_fit(self.frequencies_hz, self.db)





def _forward_noise_rhs(rhs: Any, signal: int, m_signal_at: Any) -> Any:
    """Pump-side modulation ODE, taking the signal perturbation from ``m_signal_at``."""

    def wrapped(z: float, m: NDArray[np.complex128]) -> NDArray[np.complex128]:
        full = np.asarray(m, dtype=complex).copy()
        full[signal] = complex(m_signal_at(z))
        out: NDArray[np.complex128] = np.asarray(rhs(z, full), dtype=complex)
        return out

    return wrapped


def _reverse_noise_rhs(rhs: Any, signal: int, n: int, pumps_at: Any) -> Any:
    """Signal-side modulation ODE, marched backward against the current pump solution.

    The signal carries no boundary condition at ``z = 0`` in the counter-propagating
    geometry: it is launched clean at the far end and read out at ``z = 0``. This pass
    therefore marches the signal backward against the pump modulation profiles from the
    forward pass, which is the other half of the two-point problem.
    """

    def wrapped(z: float, m: NDArray[np.complex128]) -> NDArray[np.complex128]:
        full = np.zeros(n, dtype=complex)
        full[:signal] = np.asarray(pumps_at(z)[:signal], dtype=complex)
        full[signal] = m[0]
        return np.array([rhs(z, full)[signal]], dtype=complex)

    return wrapped


def _noise_delta(
    old: Callable[[float], complex], new: Callable[[float], complex], z: NDArray[np.float64]
) -> float:
    """Max relative change between two signal-perturbation curves sampled on ``z``."""
    a = np.array([complex(old(zi)) for zi in z])
    b = np.array([complex(new(zi)) for zi in z])
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(b - a)) / scale)



def double_pole_fit(
    f_hz: NDArray[np.float64], transfer_db: NDArray[np.float64]
) -> tuple[float, float]:
    """Fit a double-pole response and return ``(dc_db, corner_hz)``.

    This is the trial function of Mermelstein 2003 Eq. 10, whose dc level and 6 dB corner
    frequency they report for every measured transfer function.

    Parameters
    ----------
    f_hz
        Offset frequency array, in Hz, sorted ascending.
    transfer_db
        Transfer magnitude in dB at each frequency.

    Returns
    -------
    tuple of float
        The dc level in dB and the 6 dB corner frequency in Hz.

    Notes
    -----
    The fit is performed on the linear amplitude ``10**(db/20)`` by nonlinear least
    squares over ``(dc, corner)``, not on the decibels, because the trial function is
    rational and a fit in log space biases the corner.
    """
    from scipy.optimize import curve_fit

    f = np.asarray(f_hz, dtype=float)
    amp = 10.0 ** (np.asarray(transfer_db, dtype=float) / 20.0)

    def model(freq: NDArray[np.float64], dc: float, corner: float) -> NDArray[np.float64]:
        return dc / np.sqrt(1.0 + (freq / corner) ** 2)

    guess = [float(amp[0]), float(f[np.argmin(np.abs(amp - amp[0] / np.sqrt(2.0)))])]
    guess[1] = max(guess[1], f[1] if len(f) > 1 else f[0])
    popt, _ = curve_fit(model, f, amp, p0=guess, maxfev=20_000)
    dc, corner = float(popt[0]), float(abs(popt[1]))
    return float(20.0 * np.log10(abs(dc))), corner


def _walk(node: Any, key: str, path: tuple[str, ...] = ()) -> None:
    """Validate every provenance record reachable from ``node``."""
    if isinstance(node, dict):
        if _REQUIRED_PROVENANCE_FIELDS[0] in node:
            _validate_entry(key, path, node)
            return
        for sub, child in node.items():
            _walk(child, key, (*path, sub))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            _walk(child, key, (*path, str(i)))


def load_benchmark(key: str, root: Path | None = None) -> BenchmarkFixture:
    """Load and validate a Raman noise benchmark fixture.

    Parameters
    ----------
    key
        Paper key, matching the directory name under ``benchmarks/raman_noise/``, for
        example ``"mermelstein2003"``.
    root
        Repository root. Defaults to the package's parent directory.

    Returns
    -------
    BenchmarkFixture
        The fixture with every provenance record validated.

    Raises
    ------
    FileNotFoundError
        If the fixture file does not exist.
    ProvenanceError
        If any numeric entry lacks a source, a non-empty DOI, a unit, or a valid kind.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    path = base / "benchmarks" / "raman_noise" / key / "fixture.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Raman noise benchmark {key!r} not found at {path}. Expected one of the "
            f"fixture directories under benchmarks/raman_noise/."
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    schema = data.get("schema")
    expected = "photonics-helper/raman-noise-fixture/1"
    if schema != expected:
        raise ProvenanceError(
            f"{key}: fixture schema is {schema!r}, expected {expected!r}"
        )

    paper = data.get("paper")
    if not isinstance(paper, dict) or not paper.get("doi"):
        raise ProvenanceError(f"{key}: fixture is missing paper DOI metadata")

    _walk(data, key)
    return BenchmarkFixture(
        key=key,
        citation=str(paper.get("citation", "")),
        doi=str(paper["doi"]),
        path=path,
        data=data,
    )