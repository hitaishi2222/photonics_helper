"""Adaptive step-size machinery for Heidt 2009 (JLT 27, 3984).

Self-contained implementation of exactly what the paper compares:

* two integration schemes for ``∂A/∂z = (D + N)A`` —
  **SSF** (symmetric split-step Fourier, local error order η = 3) and
  **RK4IP** (Hult 2007, local error order η = 5);
* two step-size controllers —
  the **local error method** of Sinkin et al. (step doubling + local
  extrapolation, paper Eq. (6)/(7)) and Heidt's **conservation quantity
  error (CQE)** method (paper Eq. (13), no step doubling);
* an exact **FFT counter**, so the paper's cost metric (ε versus "computational
  time, normalized by the time required to evaluate one FFT", Fig. 2/3) is
  reproduced without wall-clock noise.

Everything is written against the repository FFT convention
(:class:`photonics_helper.core.grids.TemporalGrid`, analysis kernel
``e^{+iΩt}``, synthesis ``e^{-iΩt}`` — ``ISSUES.md`` #0), so the results
comparable with the engine-based reproductions in this repo. The physics
operators are the paper's Eq. (2)/(3):

    D A = -α/2 A - Σ_{n≥2} β_n i^{n-1}/n! ∂^n A/∂t^n
    N A = iγ (1 + iτ_shock ∂/∂t) A ∫ R(t-t')|A(t')|² dt'

Linear loss is carried in ``D`` and, for the photon/energy invariant, the
exact loss correction of the paper's Eq. (14)/(15) is applied on the analysis
side so the CQE remains valid with α ≠ 0.

The module deliberately does *not* use ``GNLSESolver``: the paper's schemes
must be separable into a pure linear and a pure nonlinear operator, and the
FFT budget must be observable step by step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial

import numpy as np

from photonics_helper.core.grids import TemporalGrid
from photonics_helper.base import Time

__all__ = [
    "GNLSEOperator",
    "SSFIntegrator",
    "RK4IPIntegrator",
    "ConstantStepper",
    "DiagnosticStepper",
    "LocalErrorStepper",
    "CQEStepper",
    "PropagationResult",
    "global_average_error",
]

C_LIGHT = 2.99792458e8


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------


@dataclass
class GNLSEOperator:
    """Paper Eq. (2)/(3) operators, with an exact FFT budget counter."""

    n_points: int
    T_s: float
    omega0: float
    gamma: float
    betas_si: np.ndarray | None = None  # [β2, β3, ...] in s^k/m
    beta_fn: object | None = None  # callable ω -> β(ω), rad/m
    alpha: float = 0.0  # power loss coefficient, 1/m
    fR: float = 0.0  # delayed Raman fraction
    raman_tau: tuple[float, float] | None = None  # (tau1, tau2), s
    shock: bool = False
    tau_shock: float | None = None  # s; default 1/ω0
    invariant_kind: str = "photon"  # "photon" (GNLSE) | "energy" (NLSE)

    grid: TemporalGrid = field(init=False)
    n_fft: int = field(default=0, init=False)
    _phase: np.ndarray | None = field(default=None, init=False)
    _phase_from_D: np.ndarray | None = field(default=None, init=False)
    _h_R: np.ndarray | None = field(default=None, init=False)
    _shock_kernel: np.ndarray | None = field(default=None, init=False)
    _inv_omega: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.grid = TemporalGrid(N=self.n_points, Tmax=Time(self.T_s, "s"))
        if self.betas_si is None and self.beta_fn is None:
            raise ValueError("supply betas_si or beta_fn")
        if self.betas_si is not None:
            self.betas_si = np.atleast_1d(np.asarray(self.betas_si, dtype=float))
            w = self.grid.w
            phase = np.zeros_like(w)
            for k, b_k in enumerate(self.betas_si, start=2):
                phase = phase + b_k * w**k / factorial(k)
            self._phase = phase
        if self.fR > 0.0 and self.raman_tau is None:
            raise ValueError("fR > 0 requires raman_tau = (tau1, tau2)")
        if self.fR > 0.0:
            # Stolen & Lin (1978) step-like response, as used in the paper
            # (ref. [15]): h_R(t) = (1-fR)δ(t) + fR Σ_k a_k H(t)e^{-t/τ_k}
            # with the fractions normalised to unit total area.
            tau1, tau2 = self.raman_tau
            t = self.grid.t
            h = np.zeros_like(t)
            for tau, frac in ((tau1, 0.79), (tau2, 0.21)):
                # only evaluate the exponential for t >= 0 (the mask must come
                # first, otherwise exp(-t/tau) overflows on the negative part
                # of the window)
                h += (
                    frac
                    * np.where(t >= 0, np.exp(-np.maximum(t, 0.0) / tau), 0.0)
                    / tau
                )
            # Store the *spectrum* of h_R, not h_R itself: `nl_field`/`_P_of`
            # convolve as ``ifft( fft(I) * _h_R )`` with the same e^{+iΩt}
            # analysis kernel as the engine's ``_raman_polarization``
            # (``h_R_fft = grid.fft(h_R)``, photonics_helper/gnlse.py). Storing
            # the time-domain array here silently reduced the delayed Raman
            # term to a near-instantaneous (1-f_R+f_R) drive and switched the
            # self-frequency shift off entirely.
            self._h_R = self.grid.fft(h)
        if self.shock:
            tau = self.tau_shock if self.tau_shock is not None else 1.0 / self.omega0
            self._shock_kernel = 1j * self.gamma * tau * self.grid.w
        if self.invariant_kind == "photon":
            omega_abs = self.omega0 + self.grid.w
            if np.any(omega_abs <= 0):
                raise ValueError("photon invariant needs Ω_max < ω0")
            self._inv_omega = 1.0 / omega_abs

    # -- transform pair (counted) -------------------------------------------

    def fft(self, A: np.ndarray) -> np.ndarray:
        self.n_fft += 1
        return self.grid.fft(A)

    def ifft(self, A_w: np.ndarray) -> np.ndarray:
        self.n_fft += 1
        return self.grid.ifft(A_w)

    def reset_count(self) -> None:
        self.n_fft = 0

    # -- dispersion phase ----------------------------------------------------

    def dispersion_phase(self) -> np.ndarray:
        """φ(Ω) = β(ω0+Ω) − β(ω0), rad (Taylor, callable or D(λ) model)."""
        if self._phase is not None:
            return self._phase
        if self._phase_from_D is not None:
            return self._phase_from_D
        w = self.grid.w
        b0 = float(self.beta_fn(np.array([self.omega0]))[0])
        return np.asarray(self.beta_fn(self.omega0 + w)) - b0

    @classmethod
    def from_dispersion_D(cls, D_of_lambda, **kwargs) -> "GNLSEOperator":
        """Build the operator from a *group-delay-derivative* model.

        ``D_of_lambda(lam_m)`` returns ``dT/dλ`` in s/m² (the quantity quoted
        as ps/(nm·km) in the fiber literature; note the conversion
        ``D[ps/(nm·km)] = 1e6 · D[s/m²]``).  The propagation constant follows
        by two quadratures along the frequency grid —
        ``T(Ω) = ∫D(λ)dλ`` (the group delay) and then
        ``φ(Ω) = β(ω0+Ω) − β(ω0) = ∫T dΩ`` — i.e. the paper's ``β_n`` Taylor
        expansion is used in *untruncated* form.  This is how the deck-A PCF
        is specified here (the paper takes its coefficients from a reference
        that is not available offline).  Sanity identity: a flat
        ``D = −β₂2πc/λ²`` returns exactly ``φ = β₂Ω²/2``.
        """
        kwargs.setdefault(
            "beta_fn", lambda om: np.zeros_like(np.asarray(om, dtype=float))
        )
        op = cls(**kwargs)
        w = op.grid.w
        order = np.argsort(w)
        lam = C_LIGHT * 2 * np.pi / (op.omega0 + w)
        dT_dw = (
            -np.asarray(D_of_lambda(lam), dtype=float) * lam**2 / (2 * np.pi * C_LIGHT)
        )  # dT/dΩ
        dw = np.diff(w[order])
        T = np.zeros_like(w)
        T[order[1:]] = np.cumsum(0.5 * (dT_dw[order][1:] + dT_dw[order][:-1]) * dw)
        T -= np.interp(0.0, w[order], T[order])  # anchor the group delay at Ω=0
        phase = np.zeros_like(w)
        phase[order[1:]] = np.cumsum(0.5 * (T[order][1:] + T[order][:-1]) * dw)
        phase -= np.interp(0.0, w[order], phase[order])  # φ(0) = β(ω0) − β(ω0) = 0
        op._phase_from_D = phase
        return op

    # -- D and N -------------------------------------------------------------

    def D(self, A: np.ndarray, dz: float) -> np.ndarray:
        """Linear operator (loss + full Taylor/tabulated dispersion)."""
        A_w = self.fft(A)
        A_w = A_w * np.exp(1j * self.dispersion_phase() * dz)
        if self.alpha > 0.0:
            A_w = A_w * np.exp(-0.5 * self.alpha * dz)
        return np.asarray(self.ifft(A_w))

    def nl_field(self, A: np.ndarray) -> np.ndarray:
        """Nonlinear *vector field* N̂A of the paper's Eq. (3), per metre.

        ``N̂ = iγ(1 + iτ_shock∂_t) A·P_NL`` with
        ``P_NL = (1−f_R)I + f_R(h_R ⊛ I)``.  The shock term is the first-order
        expansion of the ``ω/ω₀`` factor of the multiplicative nonlinearity,
        ``A e^{iγP_NL dz}(1+iτ∂_t) ≈ A e^{iγP_NL dz} + iγτ dz ∂_t(A P_NL)``,
        and is kept purely imaginary (phase-like), as in
        ``GNLSESolver._nonlinear_step``; for a frozen ``P_NL`` the operator is
        an exact phase rotation, hence exactly photon-conserving.
        """
        intensity = np.abs(A) ** 2
        P = (1.0 - self.fR) * intensity
        if self.fR > 0.0:
            P = P + self.fR * np.real(self.ifft(self.fft(intensity) * self._h_R))
        out = 1j * self.gamma * P * A
        if self.shock:
            kernel = 1j * self.grid.w  # ∂_t in the frequency domain
            d_t = np.asarray(self.ifft(self.fft(A * P) * kernel), dtype=complex)
            out = out + 1j * self.gamma * self._tau * d_t
        return out

    def N(self, A: np.ndarray, dz: float) -> np.ndarray:
        """One nonlinear *step* exp(dz·N̂) — the SSF building block.

        For a frozen ``P_NL`` the vector field ``N̂A = iγP_NL A`` is diagonal in
        the amplitude, so the propagator is the exact phase rotation
        ``A e^{iγP_NL dz}`` (modulus preserving); the ``τ_shock ∂_t`` part is
        carried to first order, which is the standard split-step treatment and
        exactly the error the paper's adaptive scheme is there to control.
        """
        P = self._P_of(A)
        out = A * np.exp(1j * self.gamma * P * dz)
        if self.shock:
            kernel = 1j * self.grid.w  # ∂_t in the frequency domain
            d_t = np.asarray(self.ifft(self.fft(A * P) * kernel), dtype=complex)
            out = out + 1j * self.gamma * self._tau * dz * d_t
        return out

    def _P_of(self, A: np.ndarray) -> np.ndarray:
        intensity = np.abs(A) ** 2
        P = (1.0 - self.fR) * intensity
        if self.fR > 0.0:
            P = P + self.fR * np.real(self.ifft(self.fft(intensity) * self._h_R))
        return P

    @property
    def _tau(self) -> float:
        return self.tau_shock if self.tau_shock is not None else 1.0 / self.omega0

    # -- invariants ----------------------------------------------------------

    def invariant(self, A: np.ndarray) -> float:
        """Photon number (paper Eq. (8), with S(ω) = n_eff A_eff factored out)
        or the pulse energy for the NLSE."""
        spec = np.abs(self.fft(A)) ** 2
        if self.invariant_kind == "photon":
            return float(np.sum(spec * self._inv_omega) * self.grid.dw)
        return float(np.sum(spec) * self.grid.dw)

    def invariant_rate(self, A: np.ndarray) -> float:
        """dP/dz at the current field (paper Eq. (14): −∫α π dω)."""
        if self.alpha == 0.0:
            return 0.0
        spec = np.abs(self.fft(A)) ** 2
        if self.invariant_kind == "photon":
            return float(-self.alpha * np.sum(spec * self._inv_omega) * self.grid.dw)
        return float(-self.alpha * np.sum(spec) * self.grid.dw)


# --------------------------------------------------------------------------
# integrators (one full step, no adaptivity)
# --------------------------------------------------------------------------


class SSFIntegrator:
    """Symmetric split-step Fourier, paper η = 3 (local error order)."""

    eta = 3

    def __init__(self, op: GNLSEOperator) -> None:
        self.op = op

    def step(self, A: np.ndarray, dz: float) -> np.ndarray:
        A = self.op.D(A, 0.5 * dz)
        A = self.op.N(A, dz)
        return self.op.D(A, 0.5 * dz)


class RK4IPIntegrator:
    """Runge–Kutta in the interaction picture (Hult 2007, Eq. (12)), paper η = 5.

    ``N̂`` enters as a *vector field* (not as a propagator), which is what
    makes the scheme fourth-order.  With the step midpoint as the
    interaction/normal separation distance (Hult 2007, Eq. (12);
    Hochbruck & Ostrmann, *Acta Numerica* **19**, 209 (2010)):

        A_I = e^{+hD̂/2}A
        k₁ = e^{+hD̂/2}[h N̂(A)]
        k₂ = h N̂(A_I + k₁/2)
        k₃ = h N̂(A_I + k₂/2)
        k₄ = h N̂(e^{+hD̂/2}(A_I + k₃))          (kept in the normal picture)
        A⁺ = e^{+hD̂/2}[A_I + k₁/6 + k₂/3 + k₃/3] + k₄/6

    i.e. four nonlinear evaluations and four half-step dispersions per step
    (the "8 FFTs" of the source papers; the Raman convolution and the shock
    derivative each add one further transform pair per evaluation, which the
    FFT counter below records).  With a commuting linear ``N̂`` the scheme
    integrates the linear flow exactly, as classical RK4 must — that identity
    is regression-tested in ``reproduce.py``.
    """

    eta = 5

    def __init__(self, op: GNLSEOperator) -> None:
        self.op = op

    def step(self, A: np.ndarray, dz: float) -> np.ndarray:
        D, nf = self.op.D, self.op.nl_field
        half = 0.5 * dz
        A_I = D(A, half)
        k1 = D(dz * nf(A), half)
        k2 = dz * nf(A_I + 0.5 * k1)
        k3 = dz * nf(A_I + 0.5 * k2)
        k4 = dz * nf(D(A_I + k3, half))
        return D(A_I + (k1 + 2 * k2 + 2 * k3) / 6.0, half) + k4 / 6.0


INTEGRATORS = {"ssf": SSFIntegrator, "rk4ip": RK4IPIntegrator}


# --------------------------------------------------------------------------
# step-size controllers
# --------------------------------------------------------------------------


@dataclass
class PropagationResult:
    A: np.ndarray
    z_steps: np.ndarray
    errors: np.ndarray
    dz_steps: np.ndarray
    n_fft: int
    n_steps: int
    history: dict = field(default_factory=dict)


class _Stepper:
    """Common driver: fixed or adaptive h, records the paper's diagnostics."""

    scheme = "rk4ip"
    method = "constant"

    def __init__(
        self,
        op: GNLSEOperator,
        scheme: str = "rk4ip",
        dz: float | None = None,
        dz_max: float | None = None,
        dz_min: float | None = None,
        goal_error: float = 1e-6,
        dz_init: float | None = None,
        z_start: float = 0.0,
        max_steps: int = 300_000,
    ) -> None:
        self.op = op
        self.integrator = INTEGRATORS[scheme](op)
        self.eta = self.integrator.eta
        self.dz0 = dz
        self.dz_max = dz_max if dz_max is not None else np.inf
        self.dz_min = dz_min if dz_min is not None else 0.0
        self.goal_error = goal_error
        self.dz_init = dz_init
        self.z_start = z_start
        self.max_steps = max_steps

    # -- hooks ---------------------------------------------------------------

    def _advance(self, A: np.ndarray, dz: float) -> tuple[np.ndarray, float]:
        return self.integrator.step(A, dz), 0.0

    def _control(self, A: np.ndarray, dz: float, err: float) -> float:
        return dz

    # -- driver --------------------------------------------------------------

    def run(self, A0: np.ndarray, length: float) -> PropagationResult:
        op = self.op
        A = np.array(A0, dtype=complex)
        z = self.z_start
        z_end = z + length
        dz = self.dz_init if self.dz_init is not None else self.dz0
        z_steps, errs, dzs = [z], [], []
        n_accepted = 0
        max_reject = 0
        reject_streak = 0
        op.reset_count()
        # safety net: a goal error below the attainable error floor (round-off,
        # or a model whose invariant is not exactly conserved) would make the
        # controller shrink the step forever, so floor the step at
        # length/max_steps_min and fail loudly when the goal is unreachable.
        dz_floor = max(self.dz_min, length / 1.0e5)
        max_steps = self.max_steps
        while z < z_end - 1e-15:
            dz = min(dz, z_end - z)
            A_try, err = self._advance(A, dz)
            new_dz = self._control(A, dz, err)
            if new_dz < dz:  # step rejected: discard and retry with a smaller h
                max_reject += 1
                reject_streak += 1
                if new_dz <= dz_floor:
                    raise RuntimeError(
                        f"{type(self).__name__}: goal error "
                        f"{self.goal_error:.3e} is unreachable — at z = {z:g} m "
                        f"the error estimate {err:.3e} stays above "
                        f"2*goal and the step size has reached its floor "
                        f"({dz_floor:g} m)."
                    )
                if reject_streak > 200:
                    raise RuntimeError(
                        f"{type(self).__name__}: step rejected "
                        f"{reject_streak}x in a row at z = {z:g} m "
                        f"(dz = {dz:g} m, error estimate {err:.3e}, goal "
                        f"{self.goal_error:.3e}) — the goal error is below the "
                        "attainable error floor of this deck/step."
                    )
                dz = new_dz
                continue
            reject_streak = 0
            if dz <= dz_floor and err > 2 * self.goal_error:
                raise RuntimeError(
                    f"{type(self).__name__}: step size hit the floor "
                    f"{dz_floor:g} m at z = {z:g} m with error estimate "
                    f"{err:.3e} > 2*goal ({2 * self.goal_error:.3e})."
                )
            A = A_try
            z += dz
            n_accepted += 1
            z_steps.append(z)
            errs.append(err)
            dzs.append(dz)
            dz = new_dz
            if n_accepted > max_steps:
                raise RuntimeError(
                    f"{type(self).__name__}: exceeded {max_steps} accepted steps "
                    f"at z = {z:g} m of {z_end:g} m — the step controller is not "
                    "making progress."
                )
        return PropagationResult(
            A=A,
            z_steps=np.array(z_steps),
            errors=np.array(errs),
            dz_steps=np.array(dzs),
            n_fft=op.n_fft,
            n_steps=n_accepted,
            history={"n_reject": max_reject},
        )


class ConstantStepper(_Stepper):
    method = "constant"


class DiagnosticStepper(_Stepper):
    """Fixed step that only *records* an error estimate — paper Fig. 1(c,d).

    The paper plots the two error estimates along a run with a constant,
    "relatively large" step of 40 µm, i.e. without any step control.  The
    estimator is switched by ``estimator``: ``"local"`` (step doubling, Eq. (6))
    or ``"cqe"`` (photon-number change between consecutive steps, Eq. (13)).
    The accepted field is the plain integrator step in both cases, so the two
    tracks of Fig. 1(c) and 1(d) come from *the same* underlying propagation.
    """

    method = "diagnostic"

    def __init__(self, *args, estimator: str = "local", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if estimator not in ("local", "cqe"):
            raise ValueError(f"estimator must be 'local' or 'cqe', got {estimator!r}")
        self.estimator = estimator

    def _advance(self, A: np.ndarray, dz: float) -> tuple[np.ndarray, float]:
        integ = self.integrator
        if self.estimator == "local":
            A_coarse = integ.step(A, dz)
            A_fine = integ.step(integ.step(A, 0.5 * dz), 0.5 * dz)
            diff = A_fine - A_coarse
            err = float(np.linalg.norm(diff) / max(np.linalg.norm(A_fine), 1e-300))
            return A_coarse, err
        op = self.op
        P_prev, slope = op.invariant(A), op.invariant_rate(A)
        A_next = integ.step(A, dz)
        err = abs(op.invariant(A_next) - (P_prev + slope * dz)) / max(
            abs(P_prev + slope * dz), 1e-300
        )
        return A_next, err


class LocalErrorStepper(_Stepper):
    """Sinkin local error method with local extrapolation, paper Eq. (6)/(7).

    The returned error is δ = ‖A_fine − A_coarse‖ / ‖A_fine‖ and the accepted
    field is the extrapolated Eq. (7) solution.
    """

    method = "local"

    def _advance(self, A: np.ndarray, dz: float) -> tuple[np.ndarray, float]:
        integ = self.integrator
        A_coarse = integ.step(A, dz)
        A_fine = integ.step(integ.step(A, 0.5 * dz), 0.5 * dz)
        diff = A_fine - A_coarse
        err = float(np.linalg.norm(diff) / max(np.linalg.norm(A_fine), 1e-300))
        p = 2**self.eta
        A_next = (p / (p - 1.0)) * A_fine - A_coarse / (p - 1.0)
        return A_next, err

    def _control(self, A, dz, err) -> float:
        g = self.goal_error
        if err > 2.0 * g:
            return 0.5 * dz  # paper: "half the step size"
        if err < 0.5 * g:
            return min(dz * 2 ** (1.0 / self.eta), self.dz_max)
        return dz


class CQEStepper(_Stepper):
    """Heidt's conservation-quantity-error controller, paper Eq. (13).

    No step doubling: the relative change of the conserved quantity over the
    accepted step is the error estimate, compared against the *true* value
    including the exact linear-loss correction of Eq. (14)/(15). The paper's
    0.16·δ_G threshold on step-size increase suppresses the modulus-zero
    "spiking" discussed in Sec. III-B.
    """

    method = "cqe"
    increase_threshold = 0.16

    def _advance(self, A: np.ndarray, dz: float) -> tuple[np.ndarray, float]:
        op = self.op
        P_prev = op.invariant(A)
        slope = op.invariant_rate(A)
        A_next = self.integrator.step(A, dz)
        P_next = op.invariant(A_next)
        P_true = P_prev + slope * dz  # Eq. (15)
        err = abs(P_next - P_true) / max(abs(P_true), 1e-300)
        return A_next, err

    def _control(self, A, dz, err) -> float:
        g = self.goal_error
        if err > 2.0 * g:
            return 0.5 * dz
        if err < self.increase_threshold * g:
            return min(dz * 2 ** (1.0 / self.eta), self.dz_max)
        return dz


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def global_average_error(
    op: GNLSEOperator, A_calc: np.ndarray, A_ref: np.ndarray
) -> float:
    """Paper Eq. (17): ε = ∫‖I_calc − I_ref‖dω / ∫I_ref dω with I = |Ã|²."""
    I_calc = np.abs(op.fft(A_calc)) ** 2
    I_ref = np.abs(op.fft(A_ref)) ** 2
    return float(np.sum(np.abs(I_calc - I_ref)) / max(np.sum(I_ref), 1e-300))
