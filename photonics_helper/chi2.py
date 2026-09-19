"""χ⁽²⁾ (second-order) nonlinear optics: coupled SHG / SFG / DFG solver.

**Where is phase matching?** (χ³ vs χ² map)
- ``photonics_helper.chi2`` (this module): χ⁽²⁾ — SHG / SFG / DFG phase
  matching (``delta_k_shg``, ``Lambda_qpm``, QPM grating) and the coupling
  constants (``shg_coupling``, ``shg_coupling_overlap``, ``pgln_overlap``).
- ``photonics_helper.phase_matching``: χ⁽³⁾ — FWM / modulation instability /
  dispersive-wave roots.

Scalar, long-pulse (CW-like) three-wave mixing in a waveguide. The envelopes
are normalized so that ``|A|²`` is an optical power in watts; the coupled
equations are integrated in ``z`` with a classical fourth-order Runge–Kutta
step in the interaction picture (RK4IP), which removes the fast
``exp(±iΔk z)`` phase from the state vector.

Conventions
-----------
For a non-degenerate three-wave process ``ω₃ = ω₁ + ω₂`` with phase mismatch
``Δk = β(ω₃) − β(ω₁) − β(ω₂)`` and a single effective coupling ``σ``
[``1/(√W·m)``] the physical envelopes obey

.. math::

    \\frac{dA_1}{dz} &= i\\sigma A_3 A_2^* e^{-i\\Delta k z} \\\\
    \\frac{dA_2}{dz} &= i\\sigma A_3 A_1^* e^{-i\\Delta k z} \\\\
    \\frac{dA_3}{dz} &= i\\sigma A_1 A_2 e^{+i\\Delta k z}

Substituting ``A_3 = a_3 e^{iΔk z}`` gives the autonomous interaction-picture
system that is actually integrated::

    da₁/dz = iσ a₃ a₂*
    da₂/dz = iσ a₃ a₁*
    da₃/dz = iσ a₁ a₂ − iΔk a₃

SHG is the degenerate case ``A₁ = A₂ = A_f``, ``A₃ = A_sh``. With
``Δk = 0`` the undepleted-pump solution is exact:

.. math::

    \\eta(z) = \\frac{|A_\\mathrm{sh}(z)|^2}{|A_f(0)|^2}
            = \\tanh^2\\!\\left(\\sigma \\sqrt{P_0}\\, z\\right),
    \\qquad \\kappa = \\sigma\\sqrt{P_0}

which is the regression ground truth used by the reproductions and tests.

The physical coupling for SHG in a waveguide of effective area ``A_eff`` and
index ``n`` is (Boyd convention, fields power-normalised, ``Z₀ = 1/(c ε₀)``)::

    σ = (ω d_eff / (n c)) · sqrt( 2 Z₀ / (n³ A_eff) )

   (v0.1.1: this is 2× smaller than the pre-0.1.1 ``σ`` — the earlier
   formula double-counted the SHG degeneracy factor; conversion
   efficiencies change by 4×.)

Quasi-phase-matching
--------------------
Periodic poling flips the sign of ``d_eff`` every half period. The square-wave
grating ``g(z) = sign[cos(2πz/Λ)]`` is applied to ``σ`` directly. The
first-order QPM period that compensates a mismatch ``Δk`` is
``Λ = 2π/|Δk|`` (see :func:`Lambda_qpm`); the effective coupling is reduced by
``2/π``.

Scope
-----
Scalar envelopes, no dispersion / group-velocity mismatch / walk-off, no
loss. Suitable for CW or long-pulse conversion-efficiency estimates and for
validating χ⁽²⁾ bookkeeping; broadband ultrafast χ⁽²⁾+χ⁽³⁾ coupling is out of
scope.

Public API
----------
delta_k_shg        — Δk = β(2ω) − 2β(ω) for SHG
Lambda_qpm         — first-order QPM poling period from Δk
qpm_grating        — square-wave poling sign g(z)
shg_coupling       — σ from d_eff, n, A_eff, λ
Chi2Result         — z-resolved envelopes and powers
solve_shg          — degenerate SHG
solve_three_wave   — generic ω₃ = ω₁ + ω₂ system
solve_sfg          — sum-frequency generation wrapper
solve_dfg          — difference-frequency generation wrapper
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.typing import NDArray
from scipy.special import jn as _jn

from .base import C_MS, EPS_0, Z0, Area, Wavelength

# ============================================================================
# Phase matching / QPM helpers
# ============================================================================


def delta_k_shg(beta_fn: Callable[[float], float], omega: float) -> float:
    """Phase mismatch for SHG: ``Δk = β(2ω) − 2β(ω)``.

    Parameters
    ----------
    beta_fn : callable — ``β(ω)`` in 1/m (e.g. a
        :class:`~photonics_helper.phase_matching.PropagationConstantAdaptor`).
    omega : float — fundamental angular frequency in rad/s.

    Returns
    -------
    float — Δk in 1/m.
    """
    return float(beta_fn(2.0 * omega) - 2.0 * beta_fn(omega))


def Lambda_qpm(delta_k: float, order: int = 1) -> float:
    """Quasi-phase-matching poling period ``Λ = 2π·order / |Δk|``.

    Parameters
    ----------
    delta_k : float — phase mismatch in 1/m (non-zero).
    order : int — QPM order (default 1). Odd orders are the useful ones for a
        50 % duty-cycle square grating.

    Returns
    -------
    float — poling period in metres.
    """
    if order < 1:
        raise ValueError(f"QPM order must be >= 1, got {order}")
    if delta_k == 0.0:
        raise ValueError("delta_k must be non-zero to define a QPM period")
    return float(2.0 * pi * order / abs(delta_k))


def qpm_grating(
    z: float | NDArray, period: float, duty_cycle: float = 0.5
) -> NDArray:
    """Square-wave poling sign ``g(z) = ±1`` with period ``period``.

    Parameters
    ----------
    z : float or array — propagation coordinate(s) in metres.
    period : float — poling period Λ in metres (positive).
    duty_cycle : float — fraction of the period with ``+1`` (default 0.5).

    Returns
    -------
    NDArray — ``+1`` or ``-1``, same shape as ``z``.
    """
    if period <= 0:
        raise ValueError(f"QPM period must be positive, got {period}")
    if not 0.0 < duty_cycle < 1.0:
        raise ValueError(f"duty_cycle must be in (0, 1), got {duty_cycle}")
    z_arr = np.asarray(z, dtype=float)
    phase = np.mod(z_arr, period) / period
    return np.where(phase < duty_cycle, 1.0, -1.0)


def shg_coupling(
    wavelength: Wavelength,
    d_eff: float,
    *,
    n: float = 1.0,
    A_eff: Area | None = None,
) -> float:
    """SHG coupling coefficient ``κ = σ`` in ``1/(√W·m)`` (Boyd convention).

    Uses the Maxwell/Boyd plane-wave normalisation with the field amplitudes
    scaled so that ``|A|²`` is power:

    .. math::

        σ = κ/√P₀ = (ω d_eff / (n c)) · sqrt( 2 Z₀ / (n³ A_eff) ),

    giving ``κ = σ√P₀`` and the undepleted-pump limit ``η = tanh²(σ√P₀ z)``
    at perfect phase matching.  This is the coupling obtained directly from
    Boyd's *Nonlinear Optics* (3rd ed., Ch. 2) coupled-wave equations and the
    Maxwell derivation with ``I = ½ n c ε₀ |E|²`` and ``P = I·A_eff``; the
    degeneracy (factor 2) of the SHG process is already carried by ``d_eff``
    (``d_eff = χ⁽²⁾/2``), so no extra factor of 2 belongs here.

    **Breaking change (v0.1.1)**: earlier releases used
    ``σ = (2ω d_eff/(n c))·√(2/(n c ε₀ A_eff))``, which is exactly 2× this
    value (efficiencies 4× too large). Re-derive any reference values from
    before this change.

    Parameters
    ----------
    wavelength : Wavelength — fundamental (pump) vacuum wavelength.
    d_eff : float — effective second-order coefficient in m/V (i.e.
        ``χ⁽²⁾/2``; for LiNbO₃ ``d_eff = d33 = 27 pm/V``).
    n : float — refractive index at the fundamental (default 1.0). The
        second-harmonic index is taken equal (non-dispersive approximation).
    A_eff : Area — effective mode area; defaults to 1 µm².

    Returns
    -------
    float — σ in ``1/(√W·m)``.
    """
    if A_eff is None:
        A_eff = Area(1.0, "um^2")
    if n <= 0:
        raise ValueError(f"refractive index must be positive, got {n}")
    if d_eff == 0.0:
        raise ValueError("d_eff must be non-zero")
    omega = 2.0 * pi * C_MS / wavelength.as_m
    prefactor = omega * d_eff / (n * C_MS)
    return float(prefactor * np.sqrt(2.0 * Z0 / (n * A_eff.as_m2)))


# ============================================================================
# Result container
# ============================================================================


@dataclass
class Chi2Result:
    """Z-resolved χ⁽²⁾ mixing result.

    Attributes
    ----------
    z : 1-D array — propagation coordinate (m), ``n_steps + 1`` points.
    A : 2-D complex array — shape ``(n_fields, n_z)``; ``|A|²`` is power (W).
    labels : tuple[str, ...] — field names, aligned with ``A`` rows.
    sigma : float — coupling used, ``1/(√W·m)``.
    delta_k : float — phase mismatch used, 1/m.
    qpm_period : float or None — poling period if QPM was enabled.
    loss_alpha : tuple (pump, sh) — Napierian power-loss coefficients (1/m)
        used by the solver; ``(0.0, 0.0)`` means lossless.
    """

    z: NDArray
    A: NDArray
    labels: tuple[str, ...]
    sigma: float
    delta_k: float
    qpm_period: float | None = None
    loss_alpha: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=float)
        self.A = np.asarray(self.A, dtype=complex)
        if self.A.ndim != 2 or self.A.shape[1] != len(self.z):
            raise ValueError(
                f"A must have shape (n_fields, {len(self.z)}), got {self.A.shape}"
            )
        if len(self.labels) != self.A.shape[0]:
            raise ValueError(
                f"labels ({len(self.labels)}) must match A rows ({self.A.shape[0]})"
            )

    @property
    def powers(self) -> NDArray:
        """Real power array ``|A|²`` in watts, shape ``(n_fields, n_z)``."""
        return np.asarray(np.abs(self.A) ** 2, dtype=float)

    def field(self, label: str) -> NDArray:
        """Complex envelope for ``label`` (shape ``(n_z,)``)."""
        return np.asarray(self.A[self.labels.index(label)])

    def power(self, label: str) -> NDArray:
        """Power in watts for ``label`` (shape ``(n_z,)``)."""
        return np.asarray(self.powers[self.labels.index(label)])

    def efficiency(self, signal: str = "sh", pump: str | None = None) -> NDArray:
        """Conversion efficiency ``P_signal(z) / P_pump(0)``.

        Parameters
        ----------
        signal : str — label of the generated field (default ``"sh"``).
        pump : str or None — label of the reference pump; defaults to the first
            field (``labels[0]``).
        """
        pump = pump if pump is not None else self.labels[0]
        p_signal = self.power(signal)
        p_pump0 = float(self.power(pump)[0])
        if p_pump0 == 0.0:
            raise ValueError(f"reference pump '{pump}' has zero input power")
        return p_signal / p_pump0

    @property
    def kappa(self) -> float:
        """Effective coupling ``κ = σ√P₀`` (1/m) for the first field."""
        p0 = float(self.powers[0, 0])
        return float(self.sigma * np.sqrt(p0))


# ============================================================================
# RK4IP integrator
# ============================================================================


def _grating_sign(
    z: float, qpm_period: float | None, duty_cycle: float
) -> float:
    if qpm_period is None:
        return 1.0
    return float(qpm_grating(z, qpm_period, duty_cycle))


def _rk4_step(
    rhs: Callable[[float, NDArray], NDArray],
    z: float,
    y: NDArray,
    h: float,
) -> NDArray:
    k1 = rhs(z, y)
    k2 = rhs(z + h / 2.0, y + h / 2.0 * k1)
    k3 = rhs(z + h / 2.0, y + h / 2.0 * k2)
    k4 = rhs(z + h, y + h * k3)
    return np.asarray(y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _integrate(
    rhs: Callable[[float, NDArray], NDArray],
    y0: NDArray,
    length: float,
    n_steps: int,
    to_physical: Callable[[float, NDArray], NDArray] | None = None,
) -> tuple[NDArray, NDArray]:
    """Fixed-step RK4 over ``[0, length]``; returns ``(z, A)``."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    z = np.linspace(0.0, length, n_steps + 1)
    h = length / n_steps
    y = np.asarray(y0, dtype=complex).copy()
    out = np.empty((len(y), n_steps + 1), dtype=complex)
    for i in range(n_steps + 1):
        out[:, i] = y if to_physical is None else to_physical(z[i], y)
        if i < n_steps:
            y = _rk4_step(rhs, z[i], y, h)
    return z, out


# ============================================================================
# Solvers
# ============================================================================


def solve_shg(
    *,
    length: float,
    P0: float,
    sigma: float,
    n_steps: int = 2000,
    delta_k: float = 0.0,
    qpm_period: float | None = None,
    qpm_duty_cycle: float = 0.5,
    loss_db_per_cm: tuple[float, float] | None = None,
) -> Chi2Result:
    """Integrate degenerate SHG ``ω + ω → 2ω`` with pump depletion.

    Parameters
    ----------
    length : float — interaction length L in metres.
    P0 : float — input fundamental power in watts (``|A_f(0)|²``).
    sigma : float — coupling coefficient ``σ`` in ``1/(√W·m)`` (see
        :func:`shg_coupling`).
    n_steps : int — number of RK4 steps (default 2000).
    delta_k : float — phase mismatch ``Δk = β(2ω) − 2β(ω)`` in 1/m.
    qpm_period : float or None — poling period Λ for QPM; ``None`` disables it.
    qpm_duty_cycle : float — QPM duty cycle (default 0.5).
    loss_db_per_cm : tuple (pump, sh) or None — power-loss coefficients in
        dB/cm; converted internally to Napierian per metre via
        ``α = ln(10)/10 · 10² · (dB/cm)`` and applied as ``−α/2·A`` in the
        coupled equations. ``None`` (default) is lossless and bitwise
        identical to the pre-loss solver.

    Returns
    -------
    Chi2Result — fields labelled ``("fundamental", "sh")``.
    """
    if P0 <= 0:
        raise ValueError(f"P0 must be positive, got {P0}")
    if loss_db_per_cm is not None:
        if len(loss_db_per_cm) != 2:
            raise ValueError(
                "loss_db_per_cm must be a (pump, sh) pair, got "
                f"{loss_db_per_cm!r}"
            )
        # dB/cm → Napierian per metre: α = ln(10)/10 · dB/cm · 100
        alpha_f = np.log(10.0) / 10.0 * 100.0 * float(loss_db_per_cm[0])
        alpha_sh = np.log(10.0) / 10.0 * 100.0 * float(loss_db_per_cm[1])
    else:
        alpha_f = alpha_sh = 0.0

    def rhs(z: float, y: NDArray) -> NDArray:
        a_f_env, a_sh_env = y
        s = sigma * _grating_sign(z, qpm_period, qpm_duty_cycle)
        da_f = 1j * s * a_sh_env * np.conj(a_f_env)
        da_sh = 1j * s * a_f_env * a_f_env - 1j * delta_k * a_sh_env
        if alpha_f != 0.0:
            da_f -= 0.5 * alpha_f * a_f_env
        if alpha_sh != 0.0:
            da_sh -= 0.5 * alpha_sh * a_sh_env
        return np.array([da_f, da_sh])

    def to_physical(z: float, y: NDArray) -> NDArray:
        # A_sh = a_sh·exp(iΔkz); A_f is already physical.
        return np.array([y[0], y[1] * np.exp(1j * delta_k * z)])

    z, A = _integrate(
        rhs,
        np.array([np.sqrt(P0) + 0j, 0j]),
        length,
        n_steps,
        to_physical=to_physical,
    )
    return Chi2Result(
        z=z,
        A=A,
        labels=("fundamental", "sh"),
        sigma=sigma,
        delta_k=delta_k,
        qpm_period=qpm_period,
        loss_alpha=(alpha_f, alpha_sh),
    )


def solve_three_wave(
    *,
    length: float,
    sigma: float,
    A1_0: complex,
    A2_0: complex,
    A3_0: complex = 0j,
    labels: tuple[str, str, str] = ("field1", "field2", "field3"),
    n_steps: int = 2000,
    delta_k: float = 0.0,
    qpm_period: float | None = None,
    qpm_duty_cycle: float = 0.5,
) -> Chi2Result:
    """Integrate a generic ``ω₃ = ω₁ + ω₂`` three-wave mixing process.

    Covers SFG (two inputs, sum output) and DFG (pump + signal → idler) by
    choosing the input amplitudes and ``Δk = β(ω₃) − β(ω₁) − β(ω₂)``.

    Parameters
    ----------
    length : float — interaction length in metres.
    sigma : float — coupling coefficient in ``1/(√W·m)``.
    A1_0, A2_0 : complex — input envelopes (``√W``).
    A3_0 : complex — input of the generated field (default 0).
    labels : tuple[str, str, str] — names for the three fields.
    n_steps, delta_k, qpm_period, qpm_duty_cycle — as in :func:`solve_shg`.

    Returns
    -------
    Chi2Result — three labelled fields.
    """

    def rhs(z: float, y: NDArray) -> NDArray:
        a1, a2, a3 = y
        s = sigma * _grating_sign(z, qpm_period, qpm_duty_cycle)
        da1 = 1j * s * a3 * np.conj(a2)
        da2 = 1j * s * a3 * np.conj(a1)
        da3 = 1j * s * a1 * a2 - 1j * delta_k * a3
        return np.array([da1, da2, da3])

    def to_physical(z: float, y: NDArray) -> NDArray:
        return np.array([y[0], y[1], y[2] * np.exp(1j * delta_k * z)])

    z, A = _integrate(
        rhs,
        np.array([A1_0, A2_0, A3_0], dtype=complex),
        length,
        n_steps,
        to_physical=to_physical,
    )
    return Chi2Result(
        z=z,
        A=A,
        labels=labels,
        sigma=sigma,
        delta_k=delta_k,
        qpm_period=qpm_period,
    )


def solve_sfg(
    *,
    length: float,
    P1: float,
    P2: float,
    sigma: float,
    n_steps: int = 2000,
    delta_k: float = 0.0,
    qpm_period: float | None = None,
    qpm_duty_cycle: float = 0.5,
) -> Chi2Result:
    """Sum-frequency generation ``ω₁ + ω₂ → ω₃`` from two real inputs.

    Returns a :class:`Chi2Result` labelled ``("signal", "pump", "sum")``.
    """
    if P1 <= 0 or P2 <= 0:
        raise ValueError("P1 and P2 must be positive")
    return solve_three_wave(
        length=length,
        sigma=sigma,
        A1_0=np.sqrt(P1) + 0j,
        A2_0=np.sqrt(P2) + 0j,
        labels=("signal", "pump", "sum"),
        n_steps=n_steps,
        delta_k=delta_k,
        qpm_period=qpm_period,
        qpm_duty_cycle=qpm_duty_cycle,
    )


def solve_dfg(
    *,
    length: float,
    Ppump: float,
    Psignal: float,
    sigma: float,
    n_steps: int = 2000,
    delta_k: float = 0.0,
    qpm_period: float | None = None,
    qpm_duty_cycle: float = 0.5,
) -> Chi2Result:
    """Difference-frequency generation ``ω_pump − ω_signal → ω_idler``.

    Returns a :class:`Chi2Result` labelled ``("pump", "signal", "idler")``.
    """
    if Ppump <= 0 or Psignal <= 0:
        raise ValueError("Ppump and Psignal must be positive")
    return solve_three_wave(
        length=length,
        sigma=sigma,
        A1_0=np.sqrt(Ppump) + 0j,
        A2_0=np.sqrt(Psignal) + 0j,
        labels=("pump", "signal", "idler"),
        n_steps=n_steps,
        delta_k=delta_k,
        qpm_period=qpm_period,
        qpm_duty_cycle=qpm_duty_cycle,
    )


# ============================================================================
# Mode-overlap coupling (Wang et al., Opt. Express 25(6), 6963 (2017))
# ============================================================================


def shg_coupling_overlap(
    E_pump: np.ndarray,
    E_sh: np.ndarray,
    dx: float,
    dz: float,
    *,
    wavelength: Wavelength,
    d: float | np.ndarray = 27e-12,
    n_pump: float | None = None,
    n_sh: float | None = None,
) -> float:
    """Modal nonlinear overlap coupling ``g`` in ``1/(√W·m)``.

    Implements the Wang et al. 2017 (Eq. 2) overlap integral with the
    scale-invariant, units-safe shape-factor form::

        g = (ω d₀/c)·√(2 Z₀/(n_p² n_s)) · O / (I_p √I_s)

    where ``I_p = ∬|E_ω|²dx dz``, ``I_s = ∬|E_2ω|²dx dz``, and
    ``O = ∬ (E_2ω)*·(d(x,z)/d₀)·(E_ω)² dx dz``. For a uniform field over
    area ``A_eff`` this reduces exactly to the Boyd plane-wave coupling of
    :func:`shg_coupling`; for higher-order / sign-alternating (e.g. TE₃) SH
    modes the partial cancelation shrinks ``g`` by the modal overlap factor,
    which a scalar ``A_eff`` cannot represent.

    Parameters
    ----------
    E_pump, E_sh : (nx, nz) complex arrays — transverse mode fields at ω and
        2ω; arbitrary absolute scaling (all ratios are scale-invariant).
    dx, dz : float — transverse grid spacings (m).
    wavelength : Wavelength — fundamental (pump) vacuum wavelength.
    d : float or (nx, nz) array — χ⁽²⁾ coefficient d(x,z) (m/V); a scalar
        broadcasts (e.g. LiNbO₃ d33 = 27 pm/V); spatial profiles carry
        groove/poling modulation.
    n_pump, n_sh : float or None — effective indices at ω / 2ω
        (default 2.0).

    Returns
    -------
    float — modal coupling ``g`` in ``1/(√W·m)``, same σ convention as
    :func:`shg_coupling`; directly usable in :func:`solve_shg`.
    """
    E_pump = np.asarray(E_pump, dtype=complex)
    E_sh = np.asarray(E_sh, dtype=complex)
    if E_pump.shape != E_sh.shape or E_pump.ndim != 2:
        raise ValueError(
            "E_pump and E_sh must be 2-D arrays of identical shape, got "
            f"{E_pump.shape} and {E_sh.shape}"
        )
    if dx <= 0 or dz <= 0:
        raise ValueError(f"grid spacings must be positive, got {dx}, {dz}")
    if n_pump is None:
        n_pump = 2.0
    if n_sh is None:
        n_sh = 2.0
    if n_pump <= 0 or n_sh <= 0:
        raise ValueError("effective indices must be positive")

    d_nom = float(np.asarray(d, dtype=float).ravel()[0])
    d_arr = np.broadcast_to(np.asarray(d, dtype=float), E_pump.shape)

    O_int = (
        float(np.real(np.sum(np.conj(E_sh) * (d_arr / d_nom) * E_pump**2)))
        * dx
        * dz
    )
    I_p = float(np.sum(np.abs(E_pump) ** 2)) * dx * dz
    I_s = float(np.sum(np.abs(E_sh) ** 2)) * dx * dz
    if I_p <= 0 or I_s <= 0 or O_int == 0.0:
        raise ValueError("mode fields must have non-zero overlap integrals")
    shape = abs(O_int) / (I_p * np.sqrt(I_s))          # unit: 1/m
    omega = 2.0 * pi * C_MS / wavelength.as_m
    return float(
        (omega * d_nom / C_MS)
        * np.sqrt(2.0 * Z0 / (n_pump**2 * n_sh))
        * shape
    )


def pgln_overlap(
    E_pump: np.ndarray,
    E_sh: np.ndarray,
    dx: float,
    dz: float,
    *,
    wavelength: Wavelength,
    d0: float | np.ndarray,
    d1: float | np.ndarray,
    delta_eps1_pump: float | np.ndarray,
    delta_eps1_sh: float | np.ndarray,
    delta_k: float,
    n_pump: float | None = None,
    n_sh: float | None = None,
) -> dict:
    """Quasi-phase-matched overlap of a periodically-grooved (PGLN) guide.

    Implements Wang et al. 2017 (`Opt. Express` **25**(6), 6963), Eqs (3)-(5),
    for the first Fourier order of a geometric (groove) modulation of period
    Λ. All values use the same proven overg convention as
    :func:`shg_coupling_overlap`::

        g_NL^(m)  = (2ω d₀/c)·√(2 Z₀/(n_p² n_s)) · D_m
        g_L^ω     = (ω ε₀/4)·∫Δε₁|E_ω|² modulated shape  (1-W fields)
        g_L^2ω    = (2ω ε₀/4)·∫Δε₁|E_2ω|² modulated shape
        φ         = 2 (g_L^2ω − g_L^ω)/Δk
        g'        = g_NL^(1)·(J₀(φ) + J₂(φ)) − g_NL^(0)·J₁(φ)

    ``D_m`` here are Fourier-harmonic-overlap shape factors, computed with
    the same scale-invariant ratio as :func:`shg_coupling_overlap`: the
    caller supplies (nx, nz) profiles of the *harmonic mean coefficient*
    ``d^(m)(x, z)`` (grooves set d = 0 there) and the index-perturbation
    profile ``Δε₁(x, z)``.

    Parameters
    ----------
    E_pump, E_sh : (nx, nz) complex arrays — transverse mode fields at ω and
        2ω; arbitrary absolute scaling.
    dx, dz : float — transverse grid spacings (m).
    wavelength : Wavelength — pump vacuum wavelength.
    d0 : float or (nx, nz) array — mean χ⁽²⁾ coefficient (m/V), e.g. 27 pm/V.
    d1 : float or (nx, nz) array — first-harmonic χ⁽²⁾ profile (0 for a pure
        groove-modulated waveguide with grooves that remove the LN).
    delta_eps1_pump, delta_eps1_sh : float or (nx, nz) arrays — first-harmonic
        permittivity perturbation Δε₁ seen by each harmonic (dimensionless
        e.g. ``n_clad² − n_LN²`` over the groove; harmonic-specific).
    delta_k : float — residual mismatch ``β(2ω) − 2β(ω) − 2π/Λ`` (1/m).
    n_pump, n_sh : float or None — effective indices; default 2.0/2.0.

    Returns
    -------
    dict with ``"g_nl_0"``, ``"g_nl_1"``, ``"g_L_w"``, ``"g_L_2w"``,
    ``"phi"``, ``"g_eff"`` (floats; coupling values in ``1/(√W·m)``,
    ``φ`` dimensionless).

    Notes
    -----
    For a zero-modulation check, ``d1 = 0`` gives ``g_NL^(1) = 0`` →
    ``g' = j1(φ)·g_NL^(0)·(∓1)``; the uniform-guide limit
    (``Δε₁ = 0``, helper) does NOT reduce to :func:`shg_coupling_overlap`
    because a groove with zero depth is not the uniform waveguide — callers
    verifying divisibility should set ``(Δε₁ = 0, d1 = d0·shape)`` (the
    article's duty-cycle square wave) and check ``g_eff → g`` there.
    """
    E_pump = np.asarray(E_pump, dtype=complex)
    E_sh = np.asarray(E_sh, dtype=complex)
    if E_pump.shape != E_sh.shape or E_pump.ndim != 2:
        raise ValueError(
            "E_pump and E_sh must be 2-D arrays of identical shape, got "
            f"{E_pump.shape} and {E_sh.shape}"
        )
    if dx <= 0 or dz <= 0:
        raise ValueError(f"grid spacings must be positive, got {dx}, {dz}")
    if delta_k == 0.0:
        raise ValueError(
            "delta_k must be non-zero (φ undefined); for the uniform guide "
            "use shg_coupling_overlap"
        )

    d0_nom = float(np.asarray(d0, dtype=float).ravel()[0])
    if n_pump is None:
        n_pump = 2.0
    if n_sh is None:
        n_sh = 2.0
    d0_arr = np.broadcast_to(np.asarray(d0, dtype=float), E_pump.shape)
    d1_arr = np.broadcast_to(np.asarray(d1, dtype=float), E_pump.shape)
    des_arr = np.broadcast_to(
        np.asarray(delta_eps1_sh, dtype=float), E_pump.shape
    )

    I_p = float(np.sum(np.abs(E_pump) ** 2)) * dx * dz
    I_s = float(np.sum(np.abs(E_sh) ** 2)) * dx * dz
    if I_p <= 0 or I_s <= 0:
        raise ValueError("mode fields must have non-zero norms")

    omega = 2.0 * pi * C_MS / wavelength.as_m

    # Fourier-overlap shape factors (complex sum → magnitude: coupling strength)

    O0 = (
        np.sum(np.conj(E_sh) * (d0_arr / d0_nom) * E_pump**2) * dx * dz
    )
    O1 = (
        np.sum(np.conj(E_sh) * (d1_arr / d0_nom) * E_pump**2) * dx * dz
    )
    # d^(1) carries the same units as d₀ → D1 is dimensionless × (1/m units
    # from the grid); take magnitudes for the coupling-strength convention
    D0 = abs(O0) / (I_p * np.sqrt(I_s))
    D1 = abs(O1) / (I_p * np.sqrt(I_s))
    g_nl0 = (2.0 * omega * d0_nom / C_MS) * np.sqrt(
        2.0 * Z0 / (n_pump**2 * n_sh)
    ) * D0
    g_nl1 = (2.0 * omega * d0_nom / C_MS) * np.sqrt(
        2.0 * Z0 / (n_pump**2 * n_sh)
    ) * D1

    g_L_w = (
        omega
        * EPS_0
        / 4.0
        * (2.0 * Z0 / n_pump)
        * float(np.sum(des_arr * np.abs(E_pump) ** 2))
        * dx
        * dz
        / I_p
    )
    g_L_2w = (
        2.0
        * omega
        * EPS_0
        / 4.0
        * (2.0 * Z0 / n_sh)
        * float(np.sum(des_arr * np.abs(E_sh) ** 2))
        * dx
        * dz
        / I_s
    )

    phi = 2.0 * (g_L_2w - g_L_w) / delta_k
    j0v, j1v, j2v = (float(_jn(m, phi)) for m in (0, 1, 2))

    g_eff = (j0v + j2v) * g_nl1 - j1v * g_nl0
    return {
        "g_nl_0": float(g_nl0),
        "g_nl_1": float(g_nl1),
        "g_L_w": float(g_L_w),
        "g_L_2w": float(g_L_2w),
        "phi": float(phi),
        "g_eff": float(g_eff),
    }
