"""Multimode (few-mode fiber) coupled GNLSE.

Split-step Fourier solver for *N* simultaneously-guided spatial modes:
``structured.py`` gives static LG/OAM portraits; this module propagates
their modal envelopes **nonlinearly**.

Model (all channels coupled through one shared scalar γ):

- linear: per-mode Taylor dispersion ``β_k^(m)``, shared scalar loss
  ``e^{−α dz/2}``, and modal group delay (walk-off in the retarded frame
  of channel 0);
- nonlinear SPM/XPM with a selectable coefficient set — degenerate
  linearly-polarized (LP) spatial modes: SPM ``1``, XPM ``2/3``
  (Agrawal §6.4 LP limit) — or the isotropic all-ones model
  (Manakov-like, for randomly-coupled bases);
- opt-in inter-modal four-wave mixing ``iγ·f·A_n A_p A_q*`` (RK4IP
  frequency-domain substep, Strang-split around the diagonal phase),
  restricted by angular momentum conservation
  ``ℓ_m = ℓ_n + ℓ_p − ℓ_q`` when OAM indices are supplied;
- optional **pump depletion**: the full Manley–Rowe-consistent exchange
  (creation arms ``+iγ f A_n²A_q*`` / ``+iγ f A_n²A_m*`` and the
  back-conversion pump arm ``+2iγ f* A_m A_q A_n*``), which conserves
  ``Σ|A|²`` to RK4 round-off (Mumtaz Eq. 6 three-recoupling structure);
- optional **mode-specific nonlinear overlap weights** (Mumtaz Eq. 8 /
  Poletti & Horak Eq. 7): ``xpm_weights`` (N×N, SPM/XPM slot) and
  ``fwm_weights`` (N×N×N×N, FWM slot) override the uniform
  ``coef_model`` factors mode-pair-wise.

The degenerate two-channel strip-down of this engine is the variety
implemented in :mod:`photonics_helper.vector_gnlse` (the two polarization
axes of one mode); the multimode engine generalizes the same machinery to
arbitrary mode counts.

References
----------
G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed. §6.4 (scalar XPM 2/3, FWM);
M. Poletti & P. Horak, "Description of ultrashort pulse propagation in
multimode optical fibers," J. Opt. Soc. Am. B **25**, 1645 (2008),
doi:10.1364/JOSAB.25.001645 (vector modal equations, overlap tensors,
photon-number conservation Eq. 15); S. Mumtaz, R.-J. Essiambre & G. P.
Agrawal, "Nonlinear propagation in multimode and multicore fibers:
generalization of the Manakov equations," J. Lightwave Technol. **31**, 398
(2013), doi:10.1109/JLT.2012.2235414 (Eq. 6/8: ``f_lmnp`` overlap tensor,
``γ/3`` coherent mixing + ``2γ/3`` SPM/XPM structure; arXiv:1207.6645);
L. G. Wright et al., *Nat. Commun.* **6**, 6682 (2015) (resonant inter-modal
FWM).
"""

from __future__ import annotations

import warnings
from math import factorial
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import Length
from photonics_helper.gnlse import BetasUnit, FiberProfile, _normalize_betas

if TYPE_CHECKING:  # pragma: no cover
    from photonics_helper.pulse import TemporalGrid, Wave

__all__ = [
    "CoeffModel",
    "MultimodeSplitStepEngine",
]

CoeffModel = Literal["lp_degenerate", "isotropic"]

#: XPM anisotropy of the degenerate linearly-polarized LP basis.
_LP_XPM = 2.0 / 3.0
#: Inter-modal FWM factor of the same degenerate LP model.
_LP_FWM = 2.0 / 3.0


class MultimodeSplitStepEngine:
    """Split-step engine for N coupled modal channels (few-mode GNLSE).

    Parameters
    ----------
    waves : sequence of Wave
        One input envelope per guided mode. All channels must share one
        :class:`TemporalGrid` (each wave's ``grid`` is the *same* object)
        and equal-length fields; each may carry its own central wavelength
        but the engine treats them all as co-polarized modal envelopes.
    fiber : FiberProfile
        Shared scalar parameters (``n2``, ``alpha``, ``A_eff``,
        ``confinement_factor``, ``length``). One scalar γ is applied in
        every channel; mode-specific area/overlap corrections are to be
        handled at the caller by scaling ``fiber.A_eff``.
    betas : array_like or sequence of array_like
        Taylor coefficients ``[β₂, β₃, …]`` per mode (``ps^k/m`` with Ω in
        rad/ps), or one shared array for every channel.
    betas_unit : BetasUnit
        Unit of the dispersion coefficients. Default ``"ps^k/m"``.
    group_delays : sequence of float, optional
        Modal group delay ``β₁⁽ᵐ⁾ − β₁⁽⁰⁾`` (SI, s/m) in the retarded
        frame of channel 0. ``None`` (default) = co-riding modes; forced
        ``group_delays[0] == 0``.
    coef_model : {"lp_degenerate", "isotropic"}
        Nonlinear coefficients:

        * ``"lp_degenerate"`` (default) — degenerate-LP spatial-mode
          model: SPM 1, XPM 2/3, inter-modal FWM 2/3.
        * ``"isotropic"`` — all-ones coefficients (Manakov-like spatially
          averaged modes; use for a non-degenerate basis).
    include_fwm : bool
        Opt-in inter-modal four-wave mixing. Default False — the standard
        *averaged* few-mode model keeps only SPM/XPM (the heterodyne FWM
        beats wash out over the group-delay walk-off in real fiber).
    oam_l : sequence of int, optional
        OAM azimuthal order ℓ of each channel. When supplied (and FWM is
        on) FWM triples are restricted to ``ℓ_m = ℓ_n + ℓ_p − ℓ_q``;
        otherwise every triplet is allowed.
    xpm_weights : array_like, optional
        Mode-specific SPM/XPM overlap weights (Mumtaz Eq. 8 style), an
        ``N×N`` real array. ``w[i, j]`` multiplies ``|A_j|²`` entering
        channel ``i`` (including ``i == j``, i.e. the SPM slot). When
        given, this overrides the uniform ``coef_model`` factors.
    fwm_weights : array_like, optional
        Mode-specific FWM overlap weights, an ``N×N×N×N`` real array;
        ``w[m, n, p, q]`` weights the ``→ m`` transition pumped by
        ``(n, p)`` consuming ``q``. When given, it multiplies every
        allowed triple (OAM gating, if any, still applies first).
    fwm_pump_depletion : bool
        Include the Manley–Rowe-consistent back-conversion pump arm
        ``+2iγ f* A_m A_q A_n*`` in the FWM substep (requires
        ``include_fwm=True``). With this on, ``Σ|A|²`` is conserved to
        RK4 round-off and strong pumps show the parametric
        back-conversion oscillation. Default False (pump-driven
        approximation: the pump evolves only through SPM/XPM).
    step_size : Length | None
        Fixed step size (m); ``None`` uses ``length/num_steps``.
    """

    _COEF_MODELS = ("lp_degenerate", "isotropic")

    def __init__(
        self,
        waves,
        fiber: FiberProfile,
        betas,
        *,
        betas_unit: BetasUnit = "ps^k/m",
        group_delays: list[float] | None = None,
        coef_model: CoeffModel = "lp_degenerate",
        include_fwm: bool = False,
        oam_l: list[int] | None = None,
        xpm_weights=None,
        fwm_weights=None,
        fwm_pump_depletion: bool = False,
        step_size: Length | None = None,
    ):
        waves = list(waves)
        if len(waves) < 1:
            raise ValueError("at least one channel is required")
        self._n = len(waves)
        if any(w.grid is not waves[0].grid for w in waves):
            raise ValueError(
                "all channels must share one TemporalGrid object "
                "(wave.grid is waves[0].grid)"
            )
        if coef_model not in self._COEF_MODELS:
            raise ValueError(
                f"Unknown coef_model {coef_model!r}; accepted values are "
                f"{', '.join(repr(c) for c in self._COEF_MODELS)}."
            )
        if include_fwm and coef_model == "lp_degenerate" and oam_l is None:
            warnings.warn(
                "include_fwm=True with coef_model='lp_degenerate' but no "
                "oam_l: every FWM triplet is treated as allowed (isotropic "
                "selection rule). Pass oam_l for angular-momentum gating.",
                UserWarning,
                stacklevel=2,
            )
        if oam_l is not None and len(oam_l) != self._n:
            raise ValueError("oam_l must match the channel count exactly.")
        if group_delays is not None:
            if len(group_delays) != self._n:
                raise ValueError(
                    "group_delays must match the channel count exactly."
                )
            if abs(group_delays[0]) > 0:
                raise ValueError(
                    "group_delays[0] must be 0 (the reference frame)."
                )
        if fiber.length.as_m <= 0:
            raise ValueError(f"fiber.length must be positive, got {fiber.length!r}")
        if step_size is not None and step_size.as_m <= 0:
            raise ValueError(f"step_size must be positive, got {step_size!r}")
        if fwm_pump_depletion and not include_fwm:
            raise ValueError(
                "fwm_pump_depletion=True requires include_fwm=True (the "
                "back-conversion pump arm lives in the FWM substep)."
            )
        if xpm_weights is not None:
            w = np.asarray(xpm_weights, dtype=float)
            if w.shape != (self._n, self._n):
                raise ValueError(
                    f"xpm_weights must be ({self._n}, {self._n}), got "
                    f"{w.shape}."
                )
            if not np.all(np.isfinite(w)):
                raise ValueError("xpm_weights must be finite.")
        if fwm_weights is not None:
            fw = np.asarray(fwm_weights, dtype=float)
            if fw.shape != (self._n,) * 4:
                raise ValueError(
                    f"fwm_weights must be ({self._n},)*4 = "
                    f"{(self._n,) * 4}, got {fw.shape}."
                )
            if not np.all(np.isfinite(fw)):
                raise ValueError("fwm_weights must be finite.")
        if betas is None:
            raise ValueError("betas is required")

        # betas: one shared array broadcast to every channel, or a list
        # with one array per channel.
        if isinstance(betas, list) and betas and isinstance(betas[0], (list, tuple, np.ndarray)):
            if len(betas) != self._n:
                raise ValueError(
                    "betas list must match the channel count exactly, or "
                    "pass a single array to share across modes."
                )
            per_mode = list(betas)
        else:
            per_mode = [betas] * self._n
        self.betas = [_normalize_betas(b, betas_unit) for b in per_mode]

        self.waves = waves
        self.fiber = fiber
        self.group_delays = None if group_delays is None else list(group_delays)
        self.coef_model: CoeffModel = coef_model
        self.include_fwm = include_fwm
        self.oam_l = None if oam_l is None else list(oam_l)
        self.xpm_weights = None if xpm_weights is None else np.asarray(xpm_weights, dtype=float).copy()
        self.fwm_weights = None if fwm_weights is None else np.asarray(fwm_weights, dtype=float).copy()
        self.fwm_pump_depletion = fwm_pump_depletion
        self.step_size = step_size

        self.grid: TemporalGrid = waves[0].grid
        self.omega0 = waves[0].central_frequency
        self.A: list[NDArray] = []
        for w in waves:
            f = np.asarray(w.envelope_field, dtype=complex)
            if self.A and f.shape != self.A[0].shape:
                raise ValueError(
                    f"channel field shapes differ: {self.A[0].shape} vs "
                    f"{f.shape}."
                )
            self.A.append(f)

        self.evolution: list[list[Wave]] = []
        self._z_positions: list[float] = [0.0]
        self._current_z: float = 0.0
        self._energy_vs_z: list[float] | None = None
        self._spectra: tuple[NDArray, NDArray] | None = None

    # ------------------------------------------------------------------
    # step helpers
    # ------------------------------------------------------------------

    def _gamma_v(self) -> float:
        """Scalar γ of the shared profile, at the carrier frequency."""
        from photonics_helper.gnlse import _gamma

        return _gamma(
            self.fiber.n2,
            self.omega0,
            self.fiber.A_eff,
            self.fiber.confinement_factor,
        )

    def _linear_step(self, field: NDArray, dz: float, m: int) -> NDArray:
        """Linear step (dispersion + group delay + shared loss) for mode m."""
        f_w = self.grid.fft(field)
        omega_ps = self.grid.w * 1e-12  # rad/s → rad/ps

        phi = np.zeros_like(omega_ps, dtype=float)
        for k, beta_k in enumerate(self.betas[m], start=2):
            phi += beta_k * omega_ps**k / factorial(k)
        phi *= dz
        if self.group_delays is not None:
            gd = self.group_delays[m]
            if gd:
                # retardation in the frame of channel 0: enters the φ
                # stack as a first-order Taylor term with a minus sign
                # (matching the vector engine's walkoff convention), so a
                # slower mode (Δβ₁ > 0) drifts to later times.
                phi -= gd * self.grid.w * dz
        f_w = f_w * np.exp(1j * phi)
        alpha = self.fiber.alpha
        if alpha > 0:
            f_w = f_w * np.exp(-alpha * dz / 2)
        return np.asarray(self.grid.ifft(f_w), dtype=complex)

    def _xpm_factor(self, i: int, j: int) -> float:
        """SPM/XPM coefficient between channels i and j.

        ``xpm_weights`` (Mumtaz Eq. 8, when supplied) overrides the uniform
        ``coef_model`` factors mode-pair-wise, including the SPM slot.
        """
        if self.xpm_weights is not None:
            return float(self.xpm_weights[i, j])
        if i == j:
            return 1.0
        return 1.0 if self.coef_model == "isotropic" else _LP_XPM

    def _fwm_allowed(self, m: int, n: int, p: int, q: int) -> bool:
        """Selection mask for the → m FWM term ``A_n A_p A_q*``.

        With OAM indices assigned: allowed iff ``ℓ_m = ℓ_n + ℓ_p − ℓ_q``
        (angular momentum conservation of the Kerr interaction). Without
        OAM indices every triplet is allowed (isotropic rule).
        """
        if self.oam_l is None:
            return True
        return self.oam_l[m] == self.oam_l[n] + self.oam_l[p] - self.oam_l[q]

    def _fwm_factor(self, m: int, n: int, p: int, q: int) -> float:
        """Numeric FWM coefficient of the (m n p q) transition.

        ``fwm_weights`` (Mumtaz Eq. 8, when supplied) multiplies every
        allowed triple mode-pair-wise; OAM gating (if any) is applied
        first in :meth:`_fwm_allowed`.
        """
        if self.fwm_weights is not None:
            return float(self.fwm_weights[m, n, p, q])
        if self.coef_model == "isotropic":
            return 1.0
        return _LP_FWM

    def _diagonal_phase(self, A: list[NDArray]) -> list[NDArray]:
        """Instantaneous diagonal SPM/XPM phase rate ``γ·P_m`` per channel.

        With mode-specific ``xpm_weights`` (Mumtaz Eq. 8), the weights are
        per-pair floats so the returned rate is an intensity array per
        channel; the docstring's scalar-model interpretation is unchanged
        for the default coefficient models.
        """
        gamma = self._gamma_v()
        out: list[NDArray] = []
        for i in range(self._n):
            tot = self._xpm_factor(i, i) * (np.abs(A[i]) ** 2)
            for j in range(self._n):
                if j != i:
                    tot = tot + self._xpm_factor(i, j) * np.abs(A[j]) ** 2
            out.append(gamma * tot)
        return out

    def _fwm_rhs(self, A: list[NDArray]) -> list[NDArray]:
        """Pump-driven FWM right-hand side for every channel.

        Each pump channel ``n`` drives the exchange pair ``(m, q)`` with
        ``m != q``, ``m != n``, ``q != n`` through the Mumtaz Eq. (6)
        creation arms (γ/3-weighted coherent terms of the vector modal
        equation, here written with the engine's uniform-Gamma
        convention absorbed into ``f``):

            dA_m/dz|FWM = i Σ_n γ f(mnnq)  · A_n²  A_q*
            dA_q/dz|FWM = i Σ_n γ f(nnmq)  · A_n²  A_m*

        With ``fwm_pump_depletion`` the Manley–Rowe-consistent
        back-conversion arm of the same exchange is added on the pump:

            dA_n/dz|FWM += 2iγ conj(f(nnmq)) · A_m A_q A_n*

        (the factor 2 = two pump photons per exchange event; the doubled
        index-ordering count of the fully symmetric overlap tensor).
        The depleted set is exactly photon-conserving — verified against
        an RK4 probe to round-off — and reproduces the parametric
        back-conversion oscillation of strong pumps. Without the pump
        arm the substep is the standard pump-driven approximation.
        """
        gamma = self._gamma_v()
        N = self._n
        rhs: list[NDArray] = [np.zeros_like(A[0], dtype=complex) for _ in range(N)]
        if not self.include_fwm:
            return rhs
        for n in range(N):
            pump_sq = A[n] * A[n]
            for m in range(N):
                if m == n:
                    continue
                for q in range(m + 1, N):
                    if q == n:
                        continue
                    if not self._fwm_allowed(m, n, n, q):
                        continue
                    f_m = self._fwm_factor(m, n, n, q)
                    f_q = self._fwm_factor(q, n, n, m)
                    rhs[m] = rhs[m] + 1j * gamma * f_m * pump_sq * np.conj(A[q])
                    rhs[q] = rhs[q] + 1j * gamma * f_q * pump_sq * np.conj(A[m])
                    if self.fwm_pump_depletion:
                        rhs[n] = (
                            rhs[n]
                            + 2j * gamma * np.conj(f_m) * A[m] * A[q] * np.conj(A[n])
                        )
        return rhs

    def _fwm_substep_count(self, A: list[NDArray], dz: float) -> int:
        """Frequency-domain RK4 substeps keeping Δφ_FWM ≲ 0.05 rad/step."""
        rhs = self._fwm_rhs(A)
        rate = max((float(np.max(np.abs(r))) for r in rhs), default=0.0)
        rate = max(rate, 1e-30)
        return int(min(200, max(1, int(np.ceil(rate * dz / 0.05)))))

    def _coupled_nonlinear_step(
        self, A: list[NDArray], dz: float
    ) -> list[NDArray]:
        """Coupled nonlinear step for every channel, at ``self._current_z``.

        SPM/XPM advance exactly (unitary phase rotation per channel). With
        ``include_fwm``, the heterodyne inter-modal FWM term is advanced by
        frequency-domain RK4 substeps, Strang-split around the exact
        diagonal phase (the FWM term is ``O(γP·f)`` while the diagonal
        phase is ``O(γP)``, so no stage integrates the stiff part).
        """
        if not self.include_fwm:
            diag = self._diagonal_phase(A)
            return [Ai * np.exp(1j * d * dz) for Ai, d in zip(A, diag)]

        # --- FWM mode ----------------------------------------------------
        # Strang split: half of the diagonal phase, FWM RK4, trailing half.
        diag_in = self._diagonal_phase(A)
        n_sub = self._fwm_substep_count(A, dz)
        h_sub = dz / n_sub
        states = [Ai * np.exp(0.5j * d * dz) for Ai, d in zip(A, diag_in)]
        for _ in range(n_sub):
            rhs = self._fwm_rhs(states)
            states = [Ai + h_sub * k for Ai, k in zip(states, rhs)]
        diag_out = self._diagonal_phase(states)
        return [Ai * np.exp(0.5j * d * dz) for Ai, d in zip(states, diag_out)]

    # ------------------------------------------------------------------
    # propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        num_steps: int,
        *,
        nsaves: int | None = None,
        show_progress: bool = False,
    ) -> None:
        """Run the split-step propagation for ``num_steps`` steps.

        Snapshot semantics match the scalar engine: ``nsaves`` evenly
        spaced snapshots (including ``z=0`` and ``z=L``), every step when
        ``nsaves=None``; a total-photon-number monitor warns on >5% drift
        in lossless runs.
        """
        length = self.fiber.length.as_m
        if self.step_size is not None:
            dz = self.step_size.as_m
            n_steps = int(round(length / dz))
            dz = length / n_steps  # fit the fiber end exactly
        else:
            if num_steps < 1:
                raise ValueError(f"num_steps must be >= 1, got {num_steps}")
            n_steps = num_steps
            dz = length / num_steps

        from photonics_helper.pulse import Wave

        save_z = np.linspace(0.0, length, nsaves) if nsaves is not None else None
        next_save_idx = 1 if nsaves is not None else None

        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm. Install with: pip "
                    "install tqdm)"
                ) from exc
            step_iter = tqdm(range(n_steps), desc="multimode GNLSE", unit="step")
        else:
            step_iter = range(n_steps)

        self.evolution = []
        self._z_positions = [0.0]
        self._energy_vs_z = []

        def _snapshot() -> None:
            snaps = []
            for m, wave in enumerate(self.waves):
                wm = Wave(
                    grid=self.grid,
                    envelope=wave.envelope,
                    central_wavelength=wave.central_wavelength,
                )
                wm._pulse_train_field = self.A[m].copy()
                snaps.append(wm)
            self.evolution.append(snaps)
            total = sum(
                float(np.sum(np.abs(Ai) ** 2)) * self.grid.dt for Ai in self.A
            )
            assert self._energy_vs_z is not None
            self._energy_vs_z.append(total)

        _snapshot()  # z = 0

        for step in step_iter:
            self._current_z = step * dz
            # Strang split: half linear → coupled nonlinear → half linear
            self.A = [self._linear_step(self.A[m], dz / 2, m) for m in range(self._n)]
            self.A = self._coupled_nonlinear_step(self.A, dz)
            self.A = [self._linear_step(self.A[m], dz / 2, m) for m in range(self._n)]
            z = step * dz + dz
            self._z_positions.append(z)

            if save_z is None:
                _snapshot()
            else:
                while (
                    next_save_idx is not None
                    and next_save_idx < len(save_z)
                    and z >= save_z[next_save_idx] - 1e-15
                ):
                    _snapshot()
                    next_save_idx += 1

        if save_z is not None:
            assert next_save_idx is not None and nsaves is not None
            while next_save_idx < nsaves:  # final snapshot at z = L
                _snapshot()
                next_save_idx += 1
            self._z_positions = save_z[: len(self.evolution)].tolist()

        self._emit_energy_drift_warning()

        spectra = np.zeros((len(self.evolution), self.grid.N))
        for i, snaps in enumerate(self.evolution):
            acc = np.zeros(self.grid.N)
            for wave in snaps:
                acc = acc + np.abs(self.grid.fft(wave.envelope_field)) ** 2
            spectra[i] = acc
        self._spectra = (self.grid.w, spectra)

    def _emit_energy_drift_warning(self) -> None:
        """Warn if the total photon number drifted >5% with loss disabled."""
        assert self._energy_vs_z is not None
        if not self._energy_vs_z:
            return
        e0, e1 = self._energy_vs_z[0], self._energy_vs_z[-1]
        if e0 <= 0 or self.fiber.alpha != 0:
            return
        drift = abs(e1 / e0 - 1.0)
        if drift > 0.05:
            warnings.warn(
                f"multimode GNLSE total-energy drift {drift * 100:.2f}% "
                "exceeds 5% with loss disabled; increase num_steps or "
                "reduce step_size.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # readouts
    # ------------------------------------------------------------------

    @property
    def z_array(self) -> NDArray:
        """Propagation distances (m) for each saved snapshot."""
        return np.asarray(self._z_positions, dtype=float)

    def fields_vs_z(self) -> list[NDArray]:
        """Complex field histories: one ``(n_saves, N)`` array per channel."""
        return [
            np.array([snaps[m].envelope_field for snaps in self.evolution], dtype=complex)
            for m in range(self._n)
        ]

    @property
    def energy_vs_z(self) -> NDArray:
        """Total photon number ``Σᵢ∫|A_i|²dt`` at every snapshot."""
        if self._energy_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return np.asarray(self._energy_vs_z, dtype=float)

    @property
    def spectra_vs_z(self) -> tuple[NDArray, NDArray]:
        """``(ω [rad/s], total summed spectrum over all channels)`` per snapshot."""
        if self._spectra is None:
            raise RuntimeError("Call propagate() first.")
        return self._spectra

    def nonlinear_phase_measure(self, m: int) -> float:
        """Nonlinear phase of channel m measured against its own input CW.

        Returns ``arg(A_m(t₀, z) / A_m(t₀, 0))`` at the last snapshot for
        CW or constant-amplitude inputs.
        """
        if not self.evolution:
            raise RuntimeError("Call propagate() first.")
        ref = self.A[m]
        ratio = ref / np.asarray(self.waves[m].envelope_field, dtype=complex)
        return float(np.angle(ratio[0]))
