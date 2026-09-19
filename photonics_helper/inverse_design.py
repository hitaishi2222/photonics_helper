"""Inverse-design layer (Phase 4 item 5, v1).

Deterministic least-squares parameter identification and device design over
the library's exact forward solvers — the practical sibling of the card-
PINN direction, with no heavy dependencies:

- :func:`fit_two_wave` — identify the (κ, Δk) pair of a ``χ⁽²⁾`` SHG
  interaction from measured conversion-efficiency samples ``η(z)``
  through the exact :func:`solve_shg` integrator. The individual
  ``(σ, P₀)`` pair is *not* identifiable from η(z) alone (they enter the
  quadratic-only observable through ``κ = σ√P₀``), so the fit takes the
  physically well-posed parameter set.
- :func:`design_efficiency` — bounded design inversion: solve
  :func:`solve_shg` for the *length* achieving a target efficiency, with
  a loud failure when the target saturates beyond ``tanh²`` reach.

Both return :class:`FitResult` (values + cost + convergence metadata).
Scope: scalar-parameter least squares over exact physics — no surrogate
training; the differentiable-PINN layer re-uses these forward calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["FitResult", "design_efficiency", "fit_two_wave"]


@dataclass
class FitResult:
    """Inverse-design outcome (parameters + convergence metadata).

    Attributes
    ----------
    values : dict[str, float]
        Fitted / designed parameter values.
    cost : float
        Final least-squares cost (``½ Σ residual²``); ``0`` for the design
        functions.
    converged : bool
        Optimizer success flag.
    n_evals : int
        Number of forward-solver evaluations.
    residual_norm : float
        ``‖residual‖₂`` at the returned parameters.
    """

    values: dict[str, float]
    cost: float
    converged: bool
    n_evals: int
    residual_norm: float


def fit_two_wave(
    *,
    z_samples: Sequence[float],
    ratios: Sequence[float],
    kappa_bounds: tuple[float, float] = (1e-3, 20.0),
    sigma_P0_bounds: tuple[float, float] | None = None,
    delta_k_bounds: tuple[float, float] | None = None,
    qpm_period: float | None = None,
    n_steps: int = 800,
) -> FitResult:
    """Fit (κ, σP₀, Δk) of a χ⁽²⁾ SHG run from measured ``η(z)`` data.

    The observable is the fundamental→SH conversion-efficiency ratio
    ``η(z) = P_SH(z)/P_f(z)`` along the interaction. In the exact
    :func:`solve_shg` integrator it depends on *two* independent quadratic
    invariants — the fundamental coupling ``κ = σ√P₀`` (drives the
    fundamental's cubic-rate equation) and the SH-source strength
    ``σP₀ = σ·|A_f(0)|²`` (drives the second-harmonic generation) — plus
    the phase mismatch ``Δk``. The individual ``(σ, P₀)`` pair is not
    identifiable from ``η(z)`` alone; the fit therefore returns the
    identifiable parameter set.

    Parameters
    ----------
    z_samples : sequence of float
        Propagation positions (m) of the measurements.
    ratios : sequence of float
        Measured conversion-efficiency samples ``η(z)``.
    kappa_bounds : tuple — bracket for the fundamental coupling ``κ = σ√P₀``
        in 1/m.
    sigma_P0_bounds : tuple or None — bracket for the SH-source invariant
        ``σP₀`` (``√W·m⁻¹·W = √(W³)/m⁻¹ес``); ``None`` (default) fits freely
        in ``[1e-4, 20]`` with the same magnitude scale as ``κ`` (which
        equals ``κ·√P₀``).
    delta_k_bounds : tuple or None — bracket for ``Δk`` in 1/m; ``None``
        (default) fits freely in ``[-200, 200]``.
    qpm_period : float or None — poling period (fixed in v1).
    n_steps : int — RK4 resolution of every forward evaluation.

    Identifiability contract
    ------------------------
    ``kappa`` and ``|delta_k|`` are recovered exactly by the fit (validated
    on synthetic data in ``tests/test_inverse_design.py``), but the
    ``(σ, P₀)`` pair inside them is a *degenerate direction* of the η(z)
    observable: the returned ``sigma_P0`` may drift with the optimizer
    basin while η(z) stays flat. Use the reported ``kappa`` and
    ``delta_k``; ``σ`` and ``P₀`` follow only if one extra independent
    observable (e.g. absolute power) is supplied.

    Returns
    -------
    FitResult
        ``values = {"kappa": …, "sigma_P0": …, "delta_k": …}`` plus cost /
        convergence metadata.
    """
    import scipy.optimize

    from .chi2 import solve_shg

    z = np.asarray(z_samples, dtype=float)
    ratios_a = np.asarray(ratios, dtype=float)
    if len(z) != len(ratios_a) or len(z) < 3:
        raise ValueError(
            "z_samples and ratios must be equal-length sequences of at "
            "least 3 samples."
        )
    if np.any(np.diff(z) <= 0):
        raise ValueError("z_samples must be strictly increasing.")
    length = float(z.max())
    if length <= 0:
        raise ValueError(
            f"z_samples must span a positive length, got max {length}"
        )

    spb = sigma_P0_bounds if sigma_P0_bounds is not None else (1e-4, 100.0)
    dkl, dku = delta_k_bounds if delta_k_bounds is not None else (-200.0, 200.0)
    lower = np.array([kappa_bounds[0], spb[0], dkl])
    upper = np.array([kappa_bounds[1], spb[1], dku])
    if np.any(lower >= upper):
        raise ValueError(
            f"bounds must satisfy lower < upper element-wise; got {lower} "
            f"vs {upper}."
        )

    n_evals = [0]

    def residual(p: "NDArray") -> "NDArray":
        kappa, sigma_P0, delta_k = p
        if kappa <= 1e-30 or sigma_P0 <= 1e-30:
            return np.full_like(z, 1e6)
        # invert the invariants: σ = κ²/(σP₀), P₀ = (σP₀/κ)²
        sigma_par = float(kappa**2 / sigma_P0)
        P0_par = float(sigma_P0**2 / kappa**2)
        n_evals[0] += 1
        result = solve_shg(
            length=length,
            P0=P0_par,
            sigma=sigma_par,
            n_steps=n_steps,
            delta_k=float(delta_k),
            qpm_period=qpm_period,
        )
        curve = np.abs(result.field("sh")) ** 2 / np.maximum(
            np.abs(result.field("fundamental")) ** 2, 1e-30
        )
        return np.interp(z, result.z, curve) - ratios_a

    # multi-start grid: the η(z) surface is oscillatory in Δk, so a single
    # start can land in a shallow basin. Seed κ across its bracket,
    # σP₀ at a modest value, and Δk over the full bracket.
    kappa_starts = [
        lower[0] * (upper[0] / lower[0]) ** fr for fr in (0.0, 0.42, 0.78, 1.0)
    ]
    dk_starts = [
        lower[2] + i * (upper[2] - lower[2]) / 8.0 for i in range(9)
    ]
    best = None
    for k0 in kappa_starts:
        for dk0 in dk_starts:
            try:
                fitres = scipy.optimize.least_squares(
                    residual,
                    np.array([k0, 1.0, dk0]),
                    bounds=(lower, upper),
                    # the three parameters carry very different magnitude
                    # scales; unscaled TRF mixes their sensitivities badly.
                    x_scale=np.array(
                        [
                            max(1.0, kappa_bounds[1] - kappa_bounds[0]),
                            max(1.0, spb[1] - spb[0]),
                            max(100.0, upper[2] - lower[2]),
                        ]
                    ),
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or fitres.cost < best.cost:
                best = fitres
    if best is None:
        raise RuntimeError(
            "fit_two_wave: every optimizer start failed to evaluate; "
            "check the data range and the κ / σP₀ / Δk bounds."
        )
    kappa, sigma_p0, delta_k = best.x
    return FitResult(
        values={
            "kappa": float(kappa),
            "sigma_P0": float(sigma_p0),
            "delta_k": float(delta_k),
        },
        cost=float(best.cost),
        converged=bool(best.success),
        n_evals=n_evals[0],
        residual_norm=float(np.linalg.norm(best.fun)),
    )


def design_efficiency(
    *,
    target: float,
    P0: float,
    sigma: float,
    delta_k: float = 0.0,
    lo: float = 1e-3,
    hi: float = 10.0,
    n_steps: int = 500,
    tolerance: float = 1e-6,
    qpm_period: float | None = None,
    qpm_duty_cycle: float = 0.5,
) -> FitResult:
    """Design the device *length* for a target conversion efficiency.

    Bounded bisection of :func:`solve_shg` over ``[lo, hi]`` metres; fails
    loudly when the target saturates beyond the ``tanh²`` maximum in the
    bracket (physical reachability guard).

    Parameters
    ----------
    target : float — target conversion efficiency, in ``(0, 1)``.
    P0 : float — pump power (W).
    sigma : float — quadratic coupling in ``1/(√W·m)``.
    delta_k : float — phase mismatch ``β(2ω) − 2β(ω)`` in 1/m.
    lo, hi : float — bisection bracket over length (m).
    n_steps : int — RK4 resolution of every forward evaluation.
    tolerance : float — bisection width tolerance.
    qpm_period, qpm_duty_cycle — poling parameters for QPM designs.

    Returns
    -------
    FitResult
        ``values = {"length": …, "efficiency": …}``.
    """
    from .chi2 import solve_shg

    if not 0 < target < 1:
        raise ValueError(f"target efficiency must be in (0, 1), got {target!r}")
    if P0 <= 0 or sigma <= 0:
        raise ValueError(
            f"P0 and sigma must be positive, got P0={P0!r}, sigma={sigma!r}"
        )

    n_evals = [0]

    def eta(L: float) -> float:
        n_evals[0] += 1
        result = solve_shg(
            length=L,
            P0=P0,
            sigma=sigma,
            n_steps=n_steps,
            delta_k=delta_k,
            qpm_period=qpm_period,
            qpm_duty_cycle=qpm_duty_cycle,
        )
        return float(result.efficiency()[-1])

    upper_eff = eta(hi)
    if upper_eff < target:
        raise ValueError(
            f"target efficiency {target:.4f} is unreachable in the length "
            f"range [{lo}, {hi}] m (max achievable η = {upper_eff:.3f}). "
            "Increase sigma, P0 or the bracket."
        )

    a = lo
    b = hi
    eta(a)  # reachability of the lower bracket (evaluated for completeness)
    while b - a > tolerance:
        mid = 0.5 * (a + b)
        if eta(mid) < target:
            a = mid
        else:
            b = mid
    L_star = 0.5 * (a + b)
    achieved = eta(L_star)
    return FitResult(
        values={"length": L_star, "efficiency": achieved},
        cost=0.0,
        converged=True,
        n_evals=n_evals[0],
        residual_norm=abs(achieved - target),
    )
