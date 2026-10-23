"""Multimode (few-mode fiber) coupled GNLSE.

Split-step Fourier solver for *N* simultaneously-guided spatial modes:
``structured.py`` gives static LG/OAM portraits; this module propagates
their modal envelopes **nonlinearly**.

Model (all channels coupled through one shared scalar γ):

- linear: per-mode Taylor dispersion ``β_k^(m)``, shared scalar loss
  ``e^{−α dz/2}``, and modal group delay (walk-off in the retarded frame
  of channel 0);
- nonlinear SPM/XPM with a selectable coefficient set:
  ``"co_polarized"`` — co-polarized scalar spatial modes: SPM ``1``,
  XPM ``2`` (Mumtaz Eq. 6 scalar limit; Agrawal co-polarized Kerr limit);
  ``"lp_degenerate"`` — the two-polarization degenerate LP pair (the
  vector-GNLSE strip-down): SPM ``1``, XPM ``2/3``; or
  ``"isotropic"`` — all-ones (Manakov-like, for randomly-coupled bases).
  The two-polarization model is *not* the XPM of two co-polarized
  spatial modes; ``coef_model="co_polarized"`` is the spatial-mode
  default when no overlap tensor is supplied;
- opt-in inter-modal four-wave mixing ``iγ·f·A_n A_p A_q*`` (RK4IP
  frequency-domain substep, Strang-split around the diagonal phase),
  restricted by angular momentum conservation
  ``ℓ_m = ℓ_n + ℓ_p − ℓ_q`` when OAM indices are supplied;
- per-channel (intramodal) Raman ``iγ·A_m·[(1−f_R)|A_m|² +
  f_R·(h_R ⊗ |A_m|²)]`` with the scalar-engine convolution (step8), plus an
  opt-in intermodal delayed-coupling tensor ``raman_tensors``
  (``T_R^(m,n,p,q)``, Raman-weighted) evaluated as an FFT memory kernel on
  the RK4 path — Wright JSTQE Eq. (1) Raman term with ``S^R = S^K``
  (``ph_pdfs/pages_wright/page-03.png``, image-read); step7 ``S⁽²⁾`` tensors
  are admissible as ``T_R`` (hand-supplied fallback documented);
- optional **pump depletion**: the full Manley–Rowe-consistent exchange
  (creation arms ``+iγ f A_n²A_q*`` / ``+iγ f A_n²A_m*`` and the
  back-conversion pump arm ``+2iγ f* A_m A_q A_n*``), which conserves
  ``Σ|A|²`` to RK4 round-off (Mumtaz Eq. 6 three-recoupling structure);
- optional **mode-specific nonlinear overlap weights** (Mumtaz Eq. 8 /
  Poletti & Horak Eq. 7): ``xpm_weights`` (N×N, SPM/XPM slot) and
  ``fwm_weights`` (N×N×N×N, FWM slot) override the uniform
  ``coef_model`` factors mode-pair-wise.
- per-mode full-β(ω) spectral dispersion (``full_beta`` — scalar
  ``TaperedGNLSESolver`` precedent; decision 1.1: full operator, not a
  Taylor-order bump) and full-``S⁽²⁾`` FWM contraction
  (``fwm_full_tensor=True``; Wright Eq. 1 tensor term, Poletti Eq. 18/19 +
  ``oam_l`` selection preserved; diagonal arms excluded — no double count).

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
    "CavityResult",
    "CoeffModel",
    "MultimodeSplitStepEngine",
]

CoeffModel = Literal["lp_degenerate", "isotropic", "co_polarized"]

#: XPM anisotropy of the degenerate linearly-polarized (two-polarization) basis.
_LP_XPM = 2.0 / 3.0
#: Inter-modal FWM factor of the same degenerate two-polarization model.
#: Mumtaz Eq. 6 gives 1 for a co-polarized pump/signal pair and 1/3 for an
#: orthogonal one; 2/3 is their equal mix (the polarization-averaged value
#: the engine documents). A single co-polarized basis has 1 — use
#: ``coef_model="co_polarized"`` for that case.
_LP_FWM = 2.0 / 3.0
#: Co-polarized scalar-spatial-mode XPM factor: two equal index permutations
#: in the MM-GNLSE contraction (Mumtaz Eq. 6, Agrawal co-polarized limit).
_CO_POL_XPM = 2.0
#: Pump-driven FWM factor of the co-polarized scalar-spatial-mode model.
_CO_POL_FWM = 1.0

#: Accepted ``coef_model`` values (also aliased as ``MultimodeSplitStepEngine._COEF_MODELS``).
_MM_COEF_MODELS = ("lp_degenerate", "isotropic", "co_polarized")


def _mm_check_channels(waves: list) -> int:
    """Validate the channel list; return the channel count."""
    if len(waves) < 1:
        raise ValueError("at least one channel is required")
    if any(w.grid is not waves[0].grid for w in waves):
        raise ValueError(
            "all channels must share one TemporalGrid object "
            "(wave.grid is waves[0].grid)"
        )
    return len(waves)


def _mm_check_topology(
    n: int,
    coef_model: CoeffModel,
    include_fwm: bool,
    oam_l: list[int] | None,
    group_delays: list[float] | None,
    phase_offsets: list[float] | None,
) -> None:
    """Validate mode-model selection and per-channel frame arguments."""
    if coef_model not in _MM_COEF_MODELS:
        raise ValueError(
            f"Unknown coef_model {coef_model!r}; accepted values are "
            f"{', '.join(repr(c) for c in _MM_COEF_MODELS)}."
        )
    if include_fwm and coef_model == "lp_degenerate" and oam_l is None:
        warnings.warn(
            "include_fwm=True with coef_model='lp_degenerate' but no "
            "oam_l: every FWM triplet is treated as allowed (isotropic "
            "selection rule). Pass oam_l for angular-momentum gating.",
            UserWarning,
            stacklevel=3,
        )
    if oam_l is not None and len(oam_l) != n:
        raise ValueError("oam_l must match the channel count exactly.")
    if group_delays is not None:
        if len(group_delays) != n:
            raise ValueError("group_delays must match the channel count exactly.")
        if abs(group_delays[0]) > 0:
            raise ValueError("group_delays[0] must be 0 (the reference frame).")
    if phase_offsets is not None:
        if len(phase_offsets) != n:
            raise ValueError("phase_offsets must match the channel count exactly.")


def _mm_check_lengths(
    fiber: FiberProfile,
    step_size: Length | None,
    fwm_pump_depletion: bool,
    include_fwm: bool,
) -> None:
    """Validate shared length/step sizes and FWM option combinations."""
    if fiber.length.as_m <= 0:
        raise ValueError(f"fiber.length must be positive, got {fiber.length!r}")
    if step_size is not None and step_size.as_m <= 0:
        raise ValueError(f"step_size must be positive, got {step_size!r}")
    if fwm_pump_depletion and not include_fwm:
        raise ValueError(
            "fwm_pump_depletion=True requires include_fwm=True (the "
            "back-conversion pump arm lives in the FWM substep)."
        )


def _mm_coerce_xpm_weights(xpm_weights, n: int):
    """Validate and copy XPM weights (``None`` stays ``None``)."""
    if xpm_weights is None:
        return None
    w = np.asarray(xpm_weights, dtype=float)
    if w.shape != (n, n):
        raise ValueError(f"xpm_weights must be ({n}, {n}), got {w.shape}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("xpm_weights must be finite.")
    return np.asarray(xpm_weights, dtype=float).copy()


def _mm_coerce_fwm_weights(fwm_weights, n: int):
    """Validate and copy FWM weights, preserving a complex overlap."""
    if fwm_weights is None:
        return None
    # Complex-capable: full-vectorial S⁽²⁾ overlaps (step7) can carry
    # an imaginary part; it is physically meaningful and must not be
    # silently discarded.
    fw = np.asarray(fwm_weights)
    if not np.iscomplexobj(fw):
        fw = fw.astype(float)
    if fw.shape != (n,) * 4:
        raise ValueError(f"fwm_weights must be ({n},)*4 = {(n,) * 4}, got {fw.shape}.")
    if not np.all(np.isfinite(fw)):
        raise ValueError("fwm_weights must be finite.")
    return (
        np.asarray(fwm_weights).astype(float, copy=True)
        if not np.iscomplexobj(fwm_weights)
        else np.asarray(fwm_weights, dtype=complex).copy()
    )


def _mm_coerce_coupling_tensor(coupling_tensor, n: int):
    """Validate and copy the absolute coupling tensor."""
    if coupling_tensor is None:
        return None
    ct = np.asarray(coupling_tensor, dtype=complex)
    if ct.shape != (n, n):
        raise ValueError(f"coupling_tensor must be ({n}, {n}), got {ct.shape}.")
    if not np.all(np.isfinite(ct)):
        raise ValueError("coupling_tensor must be finite.")
    return np.asarray(coupling_tensor, dtype=complex).copy()


def _mm_coerce_fwm_coupling_tensor(fwm_coupling_tensor, include_fwm: bool, n: int):
    """Validate and copy the absolute FWM tensor."""
    if fwm_coupling_tensor is None:
        return None
    if not include_fwm:
        raise ValueError(
            "fwm_coupling_tensor requires include_fwm=True (it fills "
            "the FWM arm, which is only integrated when FWM is on)."
        )
    fc = np.asarray(fwm_coupling_tensor, dtype=complex)
    if fc.shape != (n,) * 4:
        raise ValueError(
            f"fwm_coupling_tensor must be ({n},)*4 = {(n,) * 4}, got {fc.shape}."
        )
    if not np.all(np.isfinite(fc)):
        raise ValueError("fwm_coupling_tensor must be finite.")
    return np.asarray(fwm_coupling_tensor, dtype=complex).copy()


def _mm_betas_per_mode(betas, betas_unit: BetasUnit, n: int) -> list:
    """Broadcast one shared betas array, or check a per-mode list."""
    if betas is None:
        raise ValueError("betas is required")
    if (
        isinstance(betas, list)
        and betas
        and isinstance(betas[0], (list, tuple, np.ndarray))
    ):
        if len(betas) != n:
            raise ValueError(
                "betas list must match the channel count exactly, or "
                "pass a single array to share across modes."
            )
        per_mode = list(betas)
    else:
        per_mode = [betas] * n
    # betas: one shared array broadcast to every channel, or a list
    # with one array per channel.
    return [_normalize_betas(b, betas_unit) for b in per_mode]


def _mm_coerce_raman_tensors(
    raman_tensors, include_raman: bool, n: int
) -> tuple[NDArray | None, NDArray | None]:
    """Validate Raman tensors; return ``(tensors, sparsity mask)``."""
    if raman_tensors is None:
        return None, None
    rt = np.asarray(raman_tensors, dtype=complex)
    if rt.shape != (n,) * 4:
        raise ValueError(
            f"raman_tensors must be ({n},)*4 = {(n,) * 4}, got {rt.shape}."
        )
    if not np.all(np.isfinite(rt)):
        raise ValueError("raman_tensors must be finite.")
    if not include_raman:
        raise ValueError(
            "raman_tensors rides on include_raman=True (they share the "
            "same h_R memory kernel and f_R weight); pass "
            "include_raman=True with it, or use coupling_tensor for "
            "instantaneous-only coupling."
        )
    # Poletti-type sparsity pruning: mask of significant entries.
    scale = float(np.abs(rt).max())
    return rt.copy(), np.abs(rt) > 1e-12 * max(scale, 1e-300)


def _mm_coerce_channel_alpha(channel_alpha, n: int):
    """Validate per-channel loss (``None`` stays ``None``)."""
    if channel_alpha is None:
        return None
    ca = np.asarray(channel_alpha, dtype=float)
    if ca.shape != (n,):
        raise ValueError(f"channel_alpha must be ({n},), got {ca.shape}.")
    if not np.all(np.isfinite(ca)) or np.any(ca < 0):
        raise ValueError("channel_alpha must be finite and non-negative.")
    return np.asarray(channel_alpha, dtype=float)


def _mm_coerce_cavity(
    cavity_reflectivity_input, cavity_reflectivity_output, n: int
) -> tuple[NDArray | None, NDArray | None]:
    """Validate the cavity mirror pair (both given or both omitted)."""
    if cavity_reflectivity_input is not None or cavity_reflectivity_output is not None:
        if cavity_reflectivity_input is None or cavity_reflectivity_output is None:
            raise ValueError(
                "cavity_reflectivity_input and cavity_reflectivity_output must "
                "be supplied together"
            )
        r_in = np.asarray(cavity_reflectivity_input, dtype=float)
        r_out = np.asarray(cavity_reflectivity_output, dtype=float)
        if r_in.shape != (n,) or r_out.shape != (n,):
            raise ValueError(
                f"cavity reflectivity must have shape ({n},), got "
                f"input {r_in.shape} and output {r_out.shape}"
            )
        if (
            np.any(r_in < 0)
            or np.any(r_in > 1)
            or np.any(r_out < 0)
            or np.any(r_out > 1)
        ):
            raise ValueError("cavity reflectivity must be in [0, 1]")
        return r_in, r_out
    return None, None


def _mm_coerce_full_beta(
    full_beta, n: int, group_delays, phase_offsets, grid, omega0: float
) -> tuple[list | None, list | None]:
    """Validate full-β(ω) curves and build the retarded-frame phases."""
    if full_beta is None:
        return None, None
    if len(full_beta) != n:
        raise ValueError(
            f"full_beta must carry one β(ω) per mode ({n}), got {len(full_beta)}."
        )
    if group_delays is not None or phase_offsets is not None:
        raise ValueError(
            "full_beta carries group delay and β₀ phase itself; pass "
            "group_delays=None and phase_offsets=None with it."
        )
    w = np.asarray(grid.w, dtype=float)
    w0 = float(omega0)
    curves = []
    for b in full_beta:
        curves.append(
            np.asarray(b(w + w0), dtype=float)
            if callable(b)
            else np.asarray(b, dtype=float)
        )
    for m, B in enumerate(curves):
        if B.shape != w.shape:
            raise ValueError(
                f"full_beta[{m}] must match the grid ({w.shape}), got {B.shape}."
            )
        if not np.all(np.isfinite(B)):
            raise ValueError(f"full_beta[{m}] must be finite.")
    # Reference-frame subtraction (retarded frame of channel 0):
    # φ_m(Ω) = [B_m(Ω) − B_0(0)] − B1_0·Ω. Narrowband this is exactly
    # (β1,m − β1,0)·Ω + Σ_{k≥2} βk,m·Ω^k/k! (decision 1.1: full-β
    # operator, not Taylor-order bump — octave MIR is where truncated
    # Taylor fails; scalar TaperedGNLSESolver precedent).
    B0 = curves[0]
    i0 = int(np.argmin(np.abs(w)))
    dw = float(w[1] - w[0])
    # 4th-order centre slope + cubic centre value: nearest-sample
    # errors (O(dw²)) would leave O(0.1) rad/m phases at the band
    # edge — fatal for the narrowband-agreement contract (task 1.2).
    lo = max(i0 - 2, 0)
    seg_w = w[lo : lo + 5]
    seg_b = B0[lo : lo + 5]
    # Cubic Lagrange value at Ω = 0.
    b_at_0 = 0.0
    for j in range(5):
        lj = 1.0
        for k in range(5):
            if k != j:
                lj *= (0.0 - seg_w[k]) / (seg_w[j] - seg_w[k])
        b_at_0 += seg_b[j] * lj
    # 4th-order central slope at the centre sample.
    ci = i0 - lo
    dB0 = float(
        (-seg_b[ci + 2] + 8.0 * seg_b[ci + 1] - 8.0 * seg_b[ci - 1] + seg_b[ci - 2])
        / (12.0 * dw)
    )
    return curves, [(B - float(b_at_0) - dB0 * w) for B in curves]


class CavityResult:
    """Result of a cavity iteration.

    Attributes
    ----------
    fields
        Converged complex fields, one per channel.
    iterations
        Number of cavity iterations performed.
    converged
        Whether the iteration converged.
    history
        Relative change per iteration.
    beat_period_hz
        Longitudinal mode beat period of the cavity in Hz.
    """

    def __init__(
        self,
        fields: list[NDArray],
        iterations: int,
        converged: bool,
        history: list[float],
        beat_period_hz: float,
    ) -> None:
        self.fields = fields
        self.iterations = iterations
        self.converged = converged
        self.history = history
        self.beat_period_hz = beat_period_hz


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
    group_delays: sequence of float, optional
        Modal group delay ``β₁⁽ᵐ⁾ − β₁⁽⁰⁾`` (SI, s/m) in the retarded
        frame of channel 0. ``None`` (default) = co-riding modes; forced
        ``group_delays[0] == 0``.
    phase_offsets : sequence of float, optional
        Modal propagation-constant offset ``Δβ₀⁽ᵐ⁾ = β₀⁽ᵐ⁾ − β₀⁽⁰⁾``
        (rad/m) in the retarded frame of channel 0. Frequency-independent
        phase accumulated as ``e^{i Δβ₀ z}`` in each channel's linear
        step; this is the absolute modal phase the retarded-frame Taylor
        expansion omits, and it is what provides discrete intermodal
        quasi-phase matching (e.g. the GRIN geometric-parametric-
        instability ladder, where ``Δβ₀⁽ᵖ⁾ = −2πp/ξ`` from self-imaging).
    gamma_reference : float, optional
        Absolute nonlinear scale ``γ₀`` (W⁻¹m⁻¹) used by every
        tensor-normalised term, overriding ``_fiber_gamma(fiber, ω₀)``.
        Needed when ``fwm_weights`` / ``raman_tensors`` were divided by
        ``S_0000`` (as :func:`photonics_helper.mode_analysis
        .overlap_tensors_from_fields` does): their absolute coefficient is
        ``γ₀·weight``, independent of the ``A_eff`` placed on ``fiber``.
        With ``coupling_tensor``/``xpm_weights`` from the same deck it also
        makes the default SPM/XPM path per-mode correct
        (``γ₀·xpm_ii = γ_i``). Omitted (default) keeps the fiber-derived
        scalar γ, byte-identically.
    coef_model : {"lp_degenerate", "isotropic", "co_polarized"}
        Nonlinear coefficients:

        * ``"lp_degenerate"`` (default) — the two-polarization degenerate
          LP pair (the vector-GNLSE strip-down): SPM 1, XPM 2/3 (the
          orthogonal-linear-polarization value), inter-modal FWM 2/3 (the
          equal mix of the co-polarized 1 and orthogonal 1/3 values).
          **Not** the co-polarized spatial-mode model — for two
          co-polarized spatial modes the Kerr XPM limit is 2, not 2/3.
        * ``"isotropic"`` — all-ones coefficients (Manakov-like spatially
          averaged modes; use for a non-degenerate basis).
        * ``"co_polarized"`` — co-polarized scalar spatial modes (Mumtaz
          Eq. 6 scalar limit / Agrawal co-polarized Kerr limit): SPM 1,
          XPM 2, inter-modal FWM 1. Use this for a few-mode LP basis that
          shares one linear polarization, or for a mode-analysis deck
          with ``xpm_model="co_polarized"``.
    include_fwm : bool
        Opt-in inter-modal four-wave mixing. Default False — the standard
        *averaged* few-mode model keeps only SPM/XPM (the heterodyne FWM
        beats wash out over the group-delay walk-off in real fiber).
    oam_l : sequence of int, optional
        OAM azimuthal order ℓ of each channel. When supplied (and FWM is
        on) FWM triples are restricted to ``ℓ_m = ℓ_n + ℓ_p − ℓ_q``
        (transverse-photon-momentum balance for the ``A_n A_p A_q*`` term
        feeding channel ``m``); otherwise every triplet is allowed.

        This gate **is** the Poletti & Horak Eq. (18) type-2 spatial
        selection rule, not merely an approximation of it: their ``Q^(2)``
        term in Eq. (6) is ``Q^(2)_plmn A_l* A_m A_n`` feeding channel ``p``
        with ``l`` the *conjugated* field, so their balance
        ``m_m + m_n = m_p + m_l`` is the printed rule
        ``−m_p − m_l + m_m + m_n = 0``; under the index map
        ``(p, l, m, n) = (m, q, n, p)`` it becomes ``ℓ_m + ℓ_q = ℓ_n + ℓ_p``,
        which is this gate (verified element-wise in
        ``reproductions/poletti_2008_multimode``).

        Two caveats:

        * The gate carries the **spatial** rule only. The polarisation rule
          Eq. (19) and the exact magnitudes are in the overlap tensor — pass
          them via ``fwm_weights`` / ``xpm_weights`` when a real fibre's mode
          functions are known.
        * Uniform entries make the gate inert: if all labels are equal (e.g.
          ``[0, 0, 0]``, the common "these channels have no distinguished
          azimuthal order" case) the condition holds for every triplet and the
          gate is exactly the isotropic fallback of ``oam_l=None``. It is a
          selection filter, not a cost control — it never reduces the engine's
          work, only the number of permitted exchanges.
    xpm_weights : array_like, optional
        Mode-specific SPM/XPM overlap weights (Mumtaz Eq. 8 style), an
        ``N×N`` real array. ``w[i, j]`` multiplies ``|A_j|²`` entering
        channel ``i`` (including ``i == j``, i.e. the SPM slot). When
        given, this overrides the uniform ``coef_model`` factors.
    fwm_weights : array_like, optional
        Mode-specific FWM overlap weights, an ``N×N×N×N`` array (real, or
        complex when the full-vectorial ``S⁽²⁾`` overlap carries an
        imaginary part — never silently clipped);
        ``w[m, n, p, q]`` weights the ``→ m`` transition pumped by
        ``(n, p)`` consuming ``q``. When given, it multiplies every
        allowed triple (OAM gating, if any, still applies first).
    coupling_tensor : array_like, optional
        Absolute SPM/XPM coupling coefficients, an ``N×N`` **complex**
        array in W⁻¹ m⁻¹; ``C[i, j]`` is the coefficient multiplying
        ``|A_j|²`` in the rate of channel ``i``. Supplied when given in
        preference to ``coef_model``/``xpm_weights``, which carry
        dimensionless multipliers of a single scalar γ.

        Two properties make this the right slot for a Raman cascade
        rather than a widened ``xpm_weights``:

        * An **imaginary antisymmetric** off-diagonal pair,
          ``C[i, j] = −conj(C[j, i])``, is a photon-number-conserving
          energy exchange — channel ``i`` gains what ``j`` loses — which
          is the structure a Raman Stokes cascade needs and which a real
          symmetric overlap weight cannot express.
        * The coefficients are **absolute**, so pump and Stokes orders
          carry their own nonlinear scales without the engine holding a
          per-order γ.

        The instantaneous arm is integrated exactly while ``C`` is real
        (the closed-form phase rotation) and by RK4 substeps otherwise,
        so a real tensor reproduces the default path exactly.
    fwm_coupling_tensor : array_like, optional
        Absolute FWM coefficients, an ``N×N×N×N`` **complex** array in
        W⁻¹ m⁻¹; ``C[m, n, p, q]`` is the coefficient of the ``→ m``
        transition pumped by ``(n, p)`` consuming ``q``. Supplied when
        given in preference to ``fwm_weights`` (OAM gating still applies
        first). Requires ``include_fwm=True``.
    channel_alpha : array_like, optional
        Per-channel loss in ``1/m``, length ``N``, overriding the shared
        ``fiber.alpha``. Raman cascade orders sit at different wavelengths and have
        genuinely different losses (Mermelstein 2003 Table I: 0.279 / 0.220 / 0.179
        dB/km at 1375 / 1465 / 1560 nm), so a single scalar cannot represent them.

        This is deliberately a *constructor* argument and not part of
        ``coupling_tensor``: the coupling tensor multiplies ``|A_j|**2``, so folding
        loss into it would make the loss amplitude-dependent. Omitted (the default)
        leaves the shared ``fiber.alpha`` in charge, byte-identically.
    fwm_pump_depletion : bool
        Include the Manley–Rowe-consistent back-conversion pump arm
        ``+2iγ f* A_m A_q A_n*`` in the FWM substep (requires
        ``include_fwm=True``). With this on, ``Σ|A|²`` is conserved to
        RK4 round-off and strong pumps show the parametric
        back-conversion oscillation. Default False (pump-driven
        approximation: the pump evolves only through SPM/XPM).
    step_size : Length | None
        Fixed step size (m); ``None`` uses ``length/num_steps``.
    cavity_reflectivity_input : sequence of float, optional
        Per-channel power reflectivity at the input end (z=0) of a
        cavity, length ``N``. ``None`` (default) = no cavity, amplifier
        mode. When supplied, :meth:`cavity_iterate` closes the
        propagation into a resonator.
    cavity_reflectivity_output : sequence of float, optional
        Per-channel power reflectivity at the output end (z=L) of a
        cavity, length ``N``. Must be supplied together with
        ``cavity_reflectivity_input``.
    cavity_round_trip_length : float, optional
        Round-trip length of the cavity in metres. Used for the
        beat-period calculation and the round-trip phase. Defaults
        to ``2 * fiber.length`` (the physical cavity length).
    cavity_group_index : float, optional
        Group index ``n_g`` of the fiber, used both for the round-trip
        phase and for :meth:`cavity_beat_period`. Defaults to 1.466
        (silica near 1.5 um), which is the Babin 2005 value. Must be
        positive.
    """

    _COEF_MODELS = _MM_COEF_MODELS

    def __init__(
        self,
        waves,
        fiber: FiberProfile,
        betas,
        *,
        betas_unit: BetasUnit = "ps^k/m",
        gamma_reference: float | None = None,
        group_delays: list[float] | None = None,
        phase_offsets: list[float] | None = None,
        coef_model: CoeffModel = "lp_degenerate",
        include_fwm: bool = False,
        oam_l: list[int] | None = None,
        xpm_weights=None,
        fwm_weights=None,
        coupling_tensor=None,
        fwm_coupling_tensor=None,
        channel_alpha=None,
        fwm_pump_depletion: bool = False,
        include_raman: bool = False,
        raman_response=None,
        raman_tensors=None,
        include_shock: bool = False,
        tau_shock: float | None = None,
        conserving_shock: bool = False,
        include_tpa: bool = False,
        include_free_carriers: bool = False,
        beta_tpa=None,
        sigma_fca=None,
        sigma_tpa=None,
        full_beta=None,
        fwm_full_tensor: bool = False,
        step_size: Length | None = None,
        cavity_reflectivity_input: list[float] | None = None,
        cavity_reflectivity_output: list[float] | None = None,
        cavity_round_trip_length: float | None = None,
        cavity_group_index: float = 1.466,
        rayleigh_backscatter: float | None = None,
    ):
        waves = list(waves)
        for _w in waves:
            _conv = getattr(
                getattr(_w, "envelope", None), "amplitude_convention", "power"
            )
            if _conv == "field":
                raise ValueError(
                    "a channel envelope is in field units (V/m), but the "
                    "multimode engine uses |A|² in watts. Convert each wave "
                    "with Wave.to_engine_units(A_eff) (or build it with "
                    "Envelope.from_energy)."
                )
        self._n = _mm_check_channels(waves)
        _mm_check_topology(
            self._n, coef_model, include_fwm, oam_l, group_delays, phase_offsets
        )
        _mm_check_lengths(fiber, step_size, fwm_pump_depletion, include_fwm)
        _xpm = _mm_coerce_xpm_weights(xpm_weights, self._n)
        _fwm_w = _mm_coerce_fwm_weights(fwm_weights, self._n)
        _ct = _mm_coerce_coupling_tensor(coupling_tensor, self._n)
        _fct = _mm_coerce_fwm_coupling_tensor(fwm_coupling_tensor, include_fwm, self._n)
        self.betas = _mm_betas_per_mode(betas, betas_unit, self._n)

        self.waves = waves
        self.fiber = fiber
        if gamma_reference is not None:
            gamma_reference = float(gamma_reference)
            if not np.isfinite(gamma_reference):
                raise ValueError(
                    f"gamma_reference must be finite, got {gamma_reference!r}"
                )
        self.gamma_reference = gamma_reference
        self.group_delays = None if group_delays is None else list(group_delays)
        self.phase_offsets = None if phase_offsets is None else list(phase_offsets)
        self.coef_model: CoeffModel = coef_model
        self.include_fwm = include_fwm
        self.oam_l = None if oam_l is None else list(oam_l)
        self.xpm_weights = _xpm
        self.fwm_weights = _fwm_w
        self.coupling_tensor = _ct
        self.fwm_coupling_tensor = _fct
        self.fwm_pump_depletion = fwm_pump_depletion
        self.include_raman = bool(include_raman)
        self.raman_response = raman_response
        self.raman_tensors: NDArray | None
        self.raman_tensors, self._raman_mask = _mm_coerce_raman_tensors(
            raman_tensors, include_raman, self._n
        )
        if self.include_raman and raman_response is None:
            raise ValueError(
                "include_raman=True but raman_response is None. "
                "Provide a RamanResponse (e.g. RamanSpec.response()) "
                "or set include_raman=False."
            )
        self._h_R_fft: NDArray | None = None
        self._raman_fR: float = 0.0
        self.channel_alpha = _mm_coerce_channel_alpha(channel_alpha, self._n)
        self.step_size = step_size

        # Cavity configuration
        r_in, r_out = _mm_coerce_cavity(
            cavity_reflectivity_input, cavity_reflectivity_output, self._n
        )
        self.cavity_reflectivity_input = r_in
        self.cavity_reflectivity_output = r_out
        self.cavity_round_trip_length = (
            cavity_round_trip_length
            if cavity_round_trip_length is not None
            else 2.0 * self.fiber.length.as_m
        )
        if not cavity_group_index > 0:
            raise ValueError(
                f"cavity_group_index must be positive, got {cavity_group_index!r}"
            )
        self.cavity_group_index = float(cavity_group_index)

        # Rayleigh backscatter coefficient (1/m)
        if rayleigh_backscatter is not None:
            if rayleigh_backscatter < 0:
                raise ValueError(
                    f"rayleigh_backscatter must be non-negative, got {rayleigh_backscatter!r}"
                )
        self.rayleigh_backscatter = rayleigh_backscatter

        self.grid: TemporalGrid = waves[0].grid
        self.omega0 = waves[0].central_frequency
        self.fwm_full_tensor = bool(fwm_full_tensor)
        if self.fwm_full_tensor and fwm_pump_depletion:
            raise ValueError(
                "fwm_full_tensor=True already contains every exchange arm; "
                "fwm_pump_depletion=True would double-count the back-conversion."
            )
        self.full_beta: list[NDArray] | None = None
        _full_curves, _full_phi = _mm_coerce_full_beta(
            full_beta,
            self._n,
            group_delays,
            phase_offsets,
            self.grid,
            self.omega0,
        )
        if _full_curves is not None:
            self.full_beta = _full_curves
            self._full_phi = _full_phi
        self.include_shock = bool(include_shock)
        if tau_shock is not None and not tau_shock > 0.0:
            raise ValueError(
                f"tau_shock must be positive (seconds, SI), got {tau_shock!r}"
            )
        self._tau_shock_override = float(tau_shock) if tau_shock is not None else None
        self.conserving_shock = bool(conserving_shock)
        if self.conserving_shock and not self.include_shock:
            raise ValueError(
                "conserving_shock=True requires include_shock=True "
                "(the pcGNLSE modification is on the shock arms)."
            )
        if self.conserving_shock and (
            self.coupling_tensor is not None
            or self.raman_tensors is not None
            or self.include_fwm
        ):
            raise ValueError(
                "conserving_shock=True (pcGNLSE) is only defined on the "
                "per-channel default path: it requires coupling_tensor=None, "
                "raman_tensors=None and include_fwm=False."
            )
        self.include_tpa = bool(include_tpa)
        self.include_free_carriers = bool(include_free_carriers)
        self.beta_tpa = self._broadcast_per_mode(beta_tpa, "beta_tpa")
        self.sigma_fca = self._broadcast_per_mode(sigma_fca, "sigma_fca")
        self.sigma_tpa = self._broadcast_per_mode(sigma_tpa, "sigma_tpa")
        self._U_list: list[float] | None = None
        self._N_list: list[NDArray | None] | None = None
        self._photon_vs_z: list[float] | None = None
        self._power_l_vs_z: list[NDArray] | None = None
        self.power_l_labels: list[int] | None = (
            sorted(set(self.oam_l)) if self.oam_l is not None else None
        )
        if self.include_shock:
            self._validate_shock_grid()
        if self.include_raman:
            from photonics_helper.gnlse import _delayed_h_R, _response_fR

            self._raman_fR = _response_fR(self.raman_response)
            h_R = _delayed_h_R(self.raman_response, self.grid.t)
            self._h_R_fft = np.asarray(self.grid.fft(h_R), dtype=complex)
        self.A: list[NDArray] = []
        for w in waves:
            f = np.asarray(w.envelope_field, dtype=complex)
            if self.A and f.shape != self.A[0].shape:
                raise ValueError(
                    f"channel field shapes differ: {self.A[0].shape} vs {f.shape}."
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
        """Nonlinear scale γ used by the model, at the carrier frequency.

        ``gamma_reference`` (a mode-analysis deck's fundamental γ₀) wins
        when supplied; otherwise the shared fiber's scalar γ. Every
        tensor-normalised term multiplies this scale, so the two routes
        cannot drift apart.
        """
        if self.gamma_reference is not None:
            return self.gamma_reference
        from photonics_helper.gnlse import _fiber_gamma

        return _fiber_gamma(self.fiber, self.omega0)

    def _linear_step(self, field: NDArray, dz: float, m: int) -> NDArray:
        """Linear step (dispersion + group delay + phase offset + shared loss)
        for mode m.

        The dispersion phase `phi(dz) = dz * Σ_k (β_k/k!) ω^k` is linear in
        `dz`, so the ω-polynomial (Taylor terms + group delay) is precomputed
        per (mode, grid) once — only the `dz` scaling and the phase-offset
        `dz` term are applied per step. dtype fast path avoided: real `phi`
        arrays are small (1 float array) and the multiply kernels let numpy
        pick the pairing.
        """
        if getattr(self, "full_beta", None) is not None:
            assert self._full_phi is not None
            phi_base = self._full_phi[m]
        else:
            phi_base = self._phi_base(m)  # Σ β_k ω^k / k! (+ group delay)
        f_w = self.grid.fft(field)
        phi = phi_base * dz
        if self.phase_offsets is not None:
            db0 = self.phase_offsets[m]
            if db0:
                phi = phi + db0 * dz
        f_w = f_w * np.exp(1j * phi)
        alpha = (
            self.fiber.alpha if self.channel_alpha is None else self.channel_alpha[m]
        )
        if alpha > 0:
            f_w = f_w * np.exp(-alpha * dz / 2)
        return np.asarray(self.grid.ifft(f_w), dtype=complex)

    def _phi_base(self, m: int) -> NDArray:
        """Cached per-(mode, grid) dispersion phase polynomial (1/length units).

        Σ_k (β_k/k!) ω^k over the Taylor orders, plus the group-delay
        Taylor-1 term `Δβ₁·ω` (retardation in the frame of channel 0, with
        a plus sign — post-#0 synthesis kernel ``e^{−iΩt}``; a slower mode,
        Δβ₁ > 0, drifts to later times — sign flipped from the pre-#0
        mirrored-kernel convention, see ISSUES.md #0). The caller multiplies
        by `dz` per step.
        """
        if not hasattr(self, "_phi_base_cache"):
            self._phi_base_cache: dict[int, np.ndarray] = {}
        cache = self._phi_base_cache
        phi = cache.get(m)
        if phi is None:
            omega_ps = self.grid.w * 1e-12  # rad/s -> rad/ps
            phi = np.zeros_like(omega_ps, dtype=float)
            for k, beta_k in enumerate(self.betas[m], start=2):
                phi += beta_k * omega_ps**k / factorial(k)
            if self.group_delays is not None:
                gd = self.group_delays[m]
                if gd:
                    phi += gd * self.grid.w
            cache[m] = phi
        return phi

    def _xpm_factor(self, i: int, j: int) -> float:
        """SPM/XPM coefficient between channels i and j.

        ``xpm_weights`` (Mumtaz Eq. 8, when supplied) overrides the uniform
        ``coef_model`` factors mode-pair-wise, including the SPM slot.
        """
        if self.xpm_weights is not None:
            return float(self.xpm_weights[i, j])
        if i == j:
            return 1.0
        if self.coef_model == "isotropic":
            return 1.0
        if self.coef_model == "co_polarized":
            return _CO_POL_XPM
        return _LP_XPM

    def _fwm_allowed(self, m: int, n: int, p: int, q: int) -> bool:
        """Selection mask for the → m FWM term ``A_n A_p A_q*``.

        With OAM indices assigned: allowed iff ``ℓ_m = ℓ_n + ℓ_p − ℓ_q``
        (angular momentum conservation of the Kerr interaction). Without
        OAM indices every triplet is allowed (isotropic rule).
        """
        if self.oam_l is None:
            return True
        return self.oam_l[m] == self.oam_l[n] + self.oam_l[p] - self.oam_l[q]

    def _fwm_factor(self, m: int, n: int, p: int, q: int) -> complex:
        """Numeric FWM coefficient of the (m n p q) transition.

        ``fwm_weights`` (Mumtaz Eq. 8, when supplied) multiplies every
        allowed triple mode-pair-wise; OAM gating (if any) is applied
        first in :meth:`_fwm_allowed`. Complex-capable: a full-vectorial
        ``S⁽²⁾`` overlap (step7 ``overlap_tensors_from_fields``) may carry
        an imaginary part, which the engine keeps (never silently drops).
        """
        if self.fwm_weights is not None:
            return complex(self.fwm_weights[m, n, p, q])
        if self.coef_model == "isotropic":
            return 1.0
        if self.coef_model == "co_polarized":
            return _CO_POL_FWM
        return _LP_FWM

    def _diagonal_phase(self, A: list[NDArray]) -> list[NDArray]:
        """Instantaneous diagonal SPM/XPM phase rate ``γ·P_m`` per channel.

        With mode-specific ``xpm_weights`` (Mumtaz Eq. 8), the weights are
        per-pair floats so the returned rate is an intensity array per
        channel; the docstring's scalar-model interpretation is unchanged
        for the default coefficient models.

        This is the **default** (no ``coupling_tensor``) arm and is
        deliberately left as written: the tensor path in
        :meth:`_coupling_rate` multiplies γ per pair, which rounds
        differently, and task 1.5 of the stage 1 change requires the
        default path to be bit-identical to the pre-tensor implementation.
        """
        gamma = self._gamma_v()
        out: list[NDArray] = []
        for i in range(self._n):
            tot = self._xpm_factor(i, i) * self._raman_polarization(A[i])
            for j in range(self._n):
                if j != i:
                    tot = tot + self._xpm_factor(i, j) * np.abs(A[j]) ** 2
            out.append(gamma * tot)
        return out

    def _broadcast_per_mode(self, value, name: str) -> NDArray | None:
        """Scalar-or-sequence TPA/FCA coefficient → per-mode float array."""
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self._n, float(arr))
        if arr.shape != (self._n,):
            raise ValueError(f"{name} must be scalar or ({self._n},), got {arr.shape}.")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name} must be finite and non-negative.")
        return arr

    @property
    def tau_shock(self) -> float:
        """Shock time: override or ``1/ω₀`` (Blow & Wood default)."""
        if self._tau_shock_override is not None:
            return self._tau_shock_override
        return 1.0 / float(self.omega0)

    def _validate_shock_grid(self) -> None:
        """Port of the scalar shock-grid guard (task 1.3).

        Hard: ``Ω_max < ω₀`` (positive absolute frequencies only) — raises.
        Soft: ``τ_shock·Ω_max ≤ 0.2`` (Blow & Wood GNLS bandwidth) — warns.
        """
        omega_max = float(self.grid.omega_max)
        omega0 = float(self.omega0)
        if omega_max >= omega0:
            raise ValueError(
                "self-steepening requires Ω_max < ω₀, but got "
                f"Ω_max = {omega_max:.4e} >= ω₀ = {omega0:.4e}. Increase "
                "Tmax (or reduce N), or disable self-steepening."
            )
        if float(self.tau_shock) * omega_max > 0.2:
            warnings.warn(
                "self-steepening outside the GNLS derivation regime "
                f"(τ_shock·Ω_max = {float(self.tau_shock) * omega_max:.3f} > 0.2); "
                "the ω/ω₀ symbol stays exact, only the equation's bandwidth "
                "assumption degrades.",
                UserWarning,
                stacklevel=3,
            )

    def _photon_number(self, A: list[NDArray]) -> float:
        """Blow & Wood App. Eq. (A5) photon number summed over channels.

        The spectral sum carries the grid measure ``dω`` exactly as the
        scalar engine's ``_photon_number`` does; omitting it left the
        recorded number off by a constant ``dω`` factor (relative drift
        was unaffected, absolute values were not).
        """
        omega0 = float(self.omega0)
        dw = float(self.grid.dw)
        tot = 0.0
        for Ai in A:
            Aw = self.grid.fft(Ai)
            denom = omega0 + self.grid.w
            tot += float(np.sum(np.abs(Aw) ** 2 / denom) * dw)
        return tot

    def power_by_l(
        self, fields: list[NDArray] | tuple[NDArray, ...] | None = None
    ) -> dict[int, float]:
        """Power per OAM class, ``Σ_m ∫|A_m|²dt`` grouped by ``oam_l`` label (W).

        The OAM-survival invariant (GAP A4): with the FWM gate off this must
        be flat versus z, so any change is either an allowed exchange or a
        symmetry break. Raises when the engine was built without ``oam_l``
        labels. There is no azimuthal class to group by.
        """
        if self.oam_l is None or self.power_l_labels is None:
            raise ValueError(
                "power_by_l() needs oam_l labels; construct the engine with"
                "oam_l=[...] to group channels by azimuthal order"
            )
        A = self.A if fields is None else list(fields)
        if len(A) != self._n:
            raise ValueError(f"expected {self._n} channel fields, got {len(A)}")
        out = {int(label): 0.0 for label in self.power_l_labels}
        for label, Ai in zip(self.oam_l, A):
            out[int(label)] += float(np.sum(np.abs(Ai) ** 2)) * self.grid.dt
        return out

    @property
    def power_by_l_vs_z(self) -> tuple[NDArray, list[int], NDArray] | None:
        """``(z, labels, powers)`` per saved step of the last ``propagate``.

        ``powers`` has shape ``(n_saves, len(labels))``; ``None`` until a
        propagation with ``oam_l`` has run.
        """
        if self._power_l_vs_z is None or self.power_l_labels is None:
            return None
        z = np.asarray(self._z_positions[: len(self._power_l_vs_z)], dtype=float)
        return z, list(self.power_l_labels), np.asarray(self._power_l_vs_z)

    def _raman_polarization(self, Ai: NDArray) -> NDArray:
        """Per-channel Raman polarization (step8 layer 1, intramodal).

        ``(1−f_R)|A_i|² + f_R·(h_R ⊗ |A_i|²)`` with the scalar-engine FFT
        convolution (``grid.fft`` / ``grid.ifft`` pair, plain ``h_R_fft`` —
        post-#0 Stokes-correct, no ``conj``). With Raman off this is plain
        ``|A_i|²`` and the default path is bit-identical.
        """
        intensity = np.asarray(np.abs(Ai) ** 2)
        if not self.include_raman or self._h_R_fft is None:
            return intensity
        I_fft = self.grid.fft(intensity)
        delayed = self._raman_fR * np.real(self.grid.ifft(self._h_R_fft * I_fft))
        return np.asarray((1.0 - self._raman_fR) * intensity + delayed)

    def _coupling_rate(self, A: list[NDArray]) -> list[NDArray]:
        """Instantaneous SPM/XPM rate per channel from ``coupling_tensor``.

        ``d_i = Σ_j C_ij |A_j|²`` with ``C`` in W⁻¹ m⁻¹, i.e. the rate that
        the default path exponentiates as ``exp(1j·d_i·dz)``. The result is
        complex whenever the tensor carries an imaginary (gain-transfer)
        part, in which case the update is no longer a phase rotation and is
        integrated by RK4 instead.
        """
        if self.coupling_tensor is None:
            raise RuntimeError("_coupling_rate requires coupling_tensor.")
        C = self.coupling_tensor
        out: list[NDArray] = []
        for i in range(self._n):
            tot = C[i, i] * (np.abs(A[i]) ** 2)
            for j in range(self._n):
                if j != i:
                    tot = tot + C[i, j] * np.abs(A[j]) ** 2
            out.append(tot)
        return out

    def _fwm_coupling(self, m: int, n: int, p: int, q: int) -> complex:
        """Absolute FWM coefficient of the ``(m n p q)`` transition.

        ``fwm_coupling_tensor`` (absolute, W⁻¹ m⁻¹) when supplied, else
        ``fwm_weights``/``coef_model`` multiplied by the scalar γ. A real
        result reproduces the pre-tensor expression exactly.
        """
        if self.fwm_coupling_tensor is not None:
            return complex(self.fwm_coupling_tensor[m, n, p, q])
        return complex(self._gamma_v() * self._fwm_factor(m, n, p, q))

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
        N = self._n
        rhs: list[NDArray] = [np.zeros_like(A[0], dtype=complex) for _ in range(N)]
        if not self.include_fwm:
            return rhs
        if self.fwm_full_tensor:
            # Full S⁽²⁾_mnpq contraction (Wright JSTQE Eq. 1 tensor term,
            # Poletti Eq. 6 arms): dA_m = i·Σ_{npq} C(mnpq)·A_n·A_p·A_q*.
            # Index sets already carried by the diagonal SPM/XPM arm
            # ((n==m,q==p) or (p==m,q==n)) are excluded — no double count.
            # Selection rules via _fwm_allowed (oam_l gate + weights).
            # NOTE: the shock operator (include_shock) is NOT applied here —
            # _fwm_rhs feeds the arm into _nonlinear_rhs_tensor, which applies
            # it once to the assembled sum. Applying it in both places would
            # give (1 + iτΩ)² on the FWM arm (regression guard in
            # tests/test_multimode_full_beta_fwm.py).
            for m in range(N):
                acc = rhs[m]
                for n in range(N):
                    for p in range(N):
                        for q in range(N):
                            if (n == m and q == p) or (p == m and q == n):
                                continue
                            if not self._fwm_allowed(m, n, p, q):
                                continue
                            c = self._fwm_coupling(m, n, p, q)
                            if c == 0:
                                continue
                            acc = acc + 1j * c * A[n] * A[p] * np.conj(A[q])
                rhs[m] = acc
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
                    c_m = self._fwm_coupling(m, n, n, q)
                    c_q = self._fwm_coupling(q, n, n, m)
                    rhs[m] = rhs[m] + 1j * c_m * pump_sq * np.conj(A[q])
                    rhs[q] = rhs[q] + 1j * c_q * pump_sq * np.conj(A[m])
                    if self.fwm_pump_depletion:
                        rhs[n] = rhs[n] + 2j * np.conj(c_m) * A[m] * A[q] * np.conj(
                            A[n]
                        )
        # NOTE: as in the full-tensor arm above, shock is applied once by the
        # caller on the assembled nonlinear sum — never here.
        return rhs

    def _raman_tensor_rhs(self, A: list[NDArray]) -> list[NDArray]:
        """Intermodal delayed-coupling Raman RHS (step8 layer 2).

        Wright JSTQE Eq. (1) Raman term (``ph_pdfs/pages_wright/page-03.png``
        image-read): ``∂A_m/∂z ⊃ iγf_R·Σ T_R^(m,n,p,q)·A_n·∫h_R·A_p·A_q*``
        with ``S^R = S^K`` per Eq. (2) — one FFT convolution per (p, q) pair
        per RK4 stage, Poletti-type sparsity pruning via ``_raman_mask``.
        """
        assert (
            self.raman_tensors is not None
            and self._h_R_fft is not None
            and self._raman_mask is not None
        )
        N = self._n
        gamma = self._gamma_v()
        fR = self._raman_fR
        T = self.raman_tensors
        mask = self._raman_mask
        # Pairwise memory kernels C_pq = h_R ⊗ (A_p · A_q*).
        C: dict[tuple[int, int], NDArray] = {}
        rhs: list[NDArray] = [np.zeros_like(A[0], dtype=complex) for _ in range(N)]
        for m in range(N):
            acc = np.zeros_like(A[0], dtype=complex)
            for n in range(N):
                for p in range(N):
                    for q in range(N):
                        if not mask[m, n, p, q]:
                            continue
                        key = (p, q)
                        c = C.get(key)
                        if c is None:
                            prod = A[p] * np.conj(A[q])
                            c = np.asarray(
                                self.grid.ifft(self._h_R_fft * self.grid.fft(prod))
                            )
                            C[key] = c
                        acc = acc + T[m, n, p, q] * A[n] * c
            rhs[m] = 1j * gamma * fR * acc
        return rhs

    def _apply_shock_rhs(self, R: list[NDArray]) -> list[NDArray]:
        """Wright §III spectral shock: ``R → R + τ·∂_t R`` via ``iΩ``.

        The ``∂/∂t`` acts on the nonlinear polarization with the mode
        fields/couplings evaluated at the local frequency — here realized
        spectrally on the assembled RHS (exact ``ω/ω₀`` symbol per bin, no
        Taylor truncation). Only used on the coupled (tensor/FWM) path; the
        per-channel default path mirrors the scalar Strang scheme instead.
        """
        tau = float(self.tau_shock)
        return [
            np.asarray(Rm + tau * self.grid.ifft(1j * self.grid.w * self.grid.fft(Rm)))
            for Rm in R
        ]

    def _shock_step_default(
        self, Ai: NDArray, P_NL: NDArray, gamma: float, dz: float
    ) -> NDArray:
        """Scalar-identical Strang shock step for one channel (task 1.2).

        Mirrors ``SplitStepEngine._nonlinear_step`` op-for-op (half exact
        Kerr/Raman phase, frequency-domain RK4 on ``iγτ·∂_t(A·P)``, trailing
        half phase, same substep policy) so N=1 matches the scalar shock
        deck (``examples/39``) at machine precision.
        """
        tau = float(self.tau_shock)
        shock_kernel = 1j * gamma * tau * self.grid.w
        use_pc = bool(self.conserving_shock)
        if use_pc and not (self.include_raman and self._h_R_fft is not None):
            raise ValueError(
                "conserving_shock=True requires include_raman=True and a "
                "Raman response (the pcGNLSE modification is on the Raman arms)."
            )
        fR0 = float(self._raman_fR)
        g_abs = abs(gamma)
        shock_kernel_pc = 1j * g_abs * tau * self.grid.w

        def _delayed(i_state: NDArray) -> NDArray:
            assert self._h_R_fft is not None
            return np.asarray(
                fR0 * np.real(self.grid.ifft(self._h_R_fft * self.grid.fft(i_state))),
                dtype=float,
            )

        def shock_rhs(A_state: NDArray) -> NDArray:
            i2 = np.abs(A_state) ** 2
            if use_pc:
                i_state = (1.0 - fR0) * i2
                d_state = _delayed(i2)
                rhs = shock_kernel * self.grid.fft(
                    i_state * A_state
                ) + shock_kernel_pc * self.grid.fft(d_state * A_state)
                return np.asarray(self.grid.ifft(rhs), dtype=complex)
            p_nl = (1.0 - fR0) * i2
            if self._h_R_fft is not None and self.include_raman:
                p_nl = p_nl + _delayed(i2)
            src_w = self.grid.fft(p_nl * A_state)
            return np.asarray(self.grid.ifft(shock_kernel * src_w), dtype=complex)

        def _rk4_shock(A_state: NDArray, h: float) -> NDArray:
            k1 = shock_rhs(A_state)
            k2 = shock_rhs(A_state + 0.5 * h * k1)
            k3 = shock_rhs(A_state + 0.5 * h * k2)
            k4 = shock_rhs(A_state + h * k3)
            return np.asarray(A_state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        half_phase = 0.5j * gamma * P_NL * dz
        if use_pc:
            P_inst = (1.0 - fR0) * np.abs(Ai) ** 2
            P_del = _delayed(np.abs(Ai) ** 2)
            half_phase = (0.5j * gamma * P_inst + 0.5j * g_abs * P_del) * dz
        A_state: NDArray = np.asarray(Ai * np.exp(half_phase), dtype=complex)
        shock_rate = float(np.max(np.abs(shock_kernel))) * float(np.max(P_NL))
        n_sub = int(min(200, max(1, np.ceil(shock_rate * dz / 0.05))))
        h_sub = dz / n_sub
        for _ in range(n_sub):
            A_state = _rk4_shock(A_state, h_sub)
        P_final_int = np.abs(A_state) ** 2
        if use_pc:
            P_inst_f = (1.0 - fR0) * P_final_int
            P_del_f = _delayed(P_final_int)
            return np.asarray(
                A_state
                * np.exp(0.5j * gamma * P_inst_f * dz + 0.5j * g_abs * P_del_f * dz)
            )
        P_final = (1.0 - fR0) * P_final_int
        if self._h_R_fft is not None and self.include_raman:
            P_final = P_final + _delayed(P_final_int)
        return np.asarray(A_state * np.exp(0.5j * gamma * P_final * dz))

    def _tpa_fca_step(self, A: list[NDArray], dz: float) -> list[NDArray]:
        """Opt-in time-resolved TPA/FCA per channel (task 2.1).

        Mirrors the scalar ``tpa_step`` / ``free_carrier_step`` operator
        split via per-mode fiber copies. Per-mode ``Im(χ⁽³⁾)`` scaling is
        the caller's job (the step7 deck's ``Γ_m`` is unity — the export
        contract carries no core mask): pass pre-scaled
        ``beta_tpa[m]`` / ``sigma_fca[m]`` arrays built from each mode's
        confinement/A_eff.
        """
        import dataclasses

        if self._U_list is None:
            self._U_list = [0.0] * self._n
        if self._N_list is None:
            self._N_list = [None] * self._n
        assert self._U_list is not None and self._N_list is not None
        from photonics_helper.gnlse import free_carrier_step, tpa_step

        out: list[NDArray] = []
        for m, Ai in enumerate(A):
            fb = dataclasses.replace(
                self.fiber,
                beta_tpa=float(self.beta_tpa[m]) if self.beta_tpa is not None else 0.0,
                sigma_fca=float(self.sigma_fca[m])
                if self.sigma_fca is not None
                else 0.0,
                sigma_tpa=float(self.sigma_tpa[m])
                if self.sigma_tpa is not None
                else 0.0,
            )
            Bi, U_new = tpa_step(
                Ai, fb, self.grid, dz, self.include_tpa, self._U_list[m], self.omega0
            )
            self._U_list[m] = float(U_new)
            if self.include_free_carriers:
                Bi, N_new = free_carrier_step(
                    Bi, fb, self.grid, dz, self._N_list[m], self.omega0
                )
                self._N_list[m] = N_new
            out.append(np.asarray(Bi))
        return out

    def _nonlinear_rhs_tensor(self, A: list[NDArray]) -> list[NDArray]:
        """Full coupled nonlinear RHS under the supplied tensors.

        ``dA_i/dz = 1j * ( sum_j C_ij |A_j|**2 * A_i + FWM arms )``.

        The ``A_i`` factor of the instantaneous arm is implicit in the closed-form
        ``A_i * exp(1j * d * dz)`` of the default path, and must be made explicit here
        where the same term is integrated by RK4. Omitting it adds a pure imaginary
        increment to the envelope, which looks like amplitude growth rather than phase
        and drives the pump up instead of depleting it.
        """
        if self.coupling_tensor is not None:
            rates = self._coupling_rate(A)
            rhs = [1j * r * Ai for r, Ai in zip(rates, A)]
        else:
            # Default instantaneous arm (diagonal phase rates, Raman-aware).
            rhs = [1j * d * Ai for d, Ai in zip(self._diagonal_phase(A), A)]
        if (
            self.include_raman
            and self._h_R_fft is not None
            and self.coupling_tensor is not None
        ):
            # Per-channel delayed correction on the tensor path: replace
            # |A_i|² with P_i in the self term (C_ii is absolute, W⁻¹m⁻¹).
            C = self.coupling_tensor
            for i in range(self._n):
                Pi = self._raman_polarization(A[i])
                rhs[i] = rhs[i] + 1j * C[i, i] * (Pi - np.abs(A[i]) ** 2) * A[i]
        if self.raman_tensors is not None:
            ram = self._raman_tensor_rhs(A)
            rhs = [r + g for r, g in zip(rhs, ram)]
        if not self.include_fwm:
            if self.include_shock:
                return self._apply_shock_rhs(rhs)
            return rhs
        fwm = self._fwm_rhs(A)
        out = [r + f for r, f in zip(rhs, fwm)]
        if self.include_shock:
            return self._apply_shock_rhs(out)
        return out

    _FWM_SUBSTEP_CAP = 200
    #: Max-FWM rate·η within one explicit RK4 substep for stability margin.
    #: Explicit RK4's absolute stability region on the imaginary axis ends at
    #: η·h ≈ 2.83; keep ≤ 2.5 (ISSUES.md #11: the old hard cap of 200 left
    #: h·λ unbounded beyond it and produced the |A| ~ 1e240 algebraic blow-up).
    _FWM_SUBSTEP_RATE = 2.5
    #: Coupling-based estimator for the exchange-pair growth rate: the tightest
    #: stability limiting eigenvalue of the 2×2 pair is ±2c with
    #: c = γ f P (frequency-domain amplitude coupling per unit length).
    _FWM_SUBSTEP_MAX = 8192

    def _fwm_rate_max(self, A: list[NDArray], dz: float) -> tuple[float, int]:
        """Max exchange-pair exponential rate η and its offending channel.

        The FWM driver amplitude for channel m is |C·A²_n·A_q*|, i.e. the
        creation rate per unit length; feasibility of the explicit RK4 update
        is governed by η·h ≤ 2.5 on the strongest pair (ISSUES.md #11).
        ``_fwm_coupling`` is used rather than ``γ·factor`` so an absolute
        ``fwm_coupling_tensor`` is not ignored by the stability estimate.
        """
        N = self._n
        rate_max, arg_ch = 0.0, 0
        if self.fwm_full_tensor:
            # Conservative full-contraction bound: per-output sum of |C|·Pmax.
            pmax = max(float(np.max(np.abs(Ai) ** 2)) for Ai in A)
            for m in range(N):
                tot = 0.0
                for n in range(N):
                    for p in range(N):
                        for q in range(N):
                            if (n == m and q == p) or (p == m and q == n):
                                continue
                            if not self._fwm_allowed(m, n, p, q):
                                continue
                            tot += abs(complex(self._fwm_coupling(m, n, p, q)))
                rate = tot * pmax
                if rate > rate_max:
                    rate_max, arg_ch = rate, m
            return rate_max, arg_ch
        for n in range(N):
            pump = float(np.max(np.abs(A[n]) ** 2))
            for m in range(N):
                if m == n:
                    continue
                for q in range(m + 1, N):
                    if q == n or not self._fwm_allowed(m, n, n, q):
                        continue
                    # Both exchange arms are integrated; an absolute
                    # fwm_coupling_tensor may be asymmetric, so the
                    # stability estimate must see the larger one. Using
                    # gamma*factor here ignored fwm_coupling_tensor
                    # entirely (step10 audit, 2026-10).
                    c_m = self._fwm_coupling(m, n, n, q) * pump
                    c_q = self._fwm_coupling(q, n, n, m) * pump
                    c = max(abs(c_m), abs(c_q))
                    if c > rate_max:
                        rate_max, arg_ch = c, n
        return rate_max, arg_ch

    def _coupling_rate_max(self, A: list[NDArray]) -> tuple[float, int]:
        """Max instantaneous coupling rate ``|d_i * A_i|`` and its channel.

        Drives the RK4 substep count of the tensor arm under the same
        stability criterion as the FWM arm (:meth:`_fwm_rate_max`). The
        ``A_i`` factor matters here: the substep is stable when
        ``|dA_i/dz| * h`` is bounded, not when ``|d_i| * h`` is.
        """
        if self.coupling_tensor is None:
            return 0.0, 0
        rate_max, arg_ch = 0.0, 0
        for i, (r, Ai) in enumerate(zip(self._coupling_rate(A), A)):
            m = float(np.max(np.abs(r * Ai)))
            if m > rate_max:
                rate_max, arg_ch = m, i
        return rate_max, arg_ch

    def _fwm_substep_count(self, A: list[NDArray], dz: float) -> int:
        """Explicit-RK4 substeps: stability-driven with an accuracy floor.

        Policy (ISSUES.md #11):
        - stability requires η·h_sub ≤ 2.5 for the strongest exchange pair;
        - accuracy targets η·h_sub ≲ 0.05 rad but at most ``_FWM_SUBSTEP_CAP``
          = 200 inner steps (the historical cap);
        - the number of inner steps is the *stability* requirement; if that
          exceeds ``_FWM_SUBSTEP_MAX`` the configuration is not integrable
          with this explicit scheme at this outer dz and a loud error names
          the channel and rate (never silently NaNs).
        """
        rate, ch = self._fwm_rate_max(A, dz)
        rate = max(rate, 1e-30)
        n_stab = int(np.ceil(rate * dz / self._FWM_SUBSTEP_RATE))
        if n_stab > self._FWM_SUBSTEP_MAX:
            raise ValueError(
                f"FWM exchange pair rate |γ f P| = {rate:.3e} /m over outer "
                f"dz = {dz:.3e} m exceeds the explicit RK4 stability region "
                f"even with {self._FWM_SUBSTEP_MAX} substeps (η·h ≈ "
                f"{rate * dz / self._FWM_SUBSTEP_MAX:.2f} > 2.5); reduce the "
                "outer step size or disable FWM on channel "
                f"{ch} (ISSUES.md #11)."
            )
        n_acc = min(self._FWM_SUBSTEP_CAP, int(np.ceil(rate * dz / 0.05)))
        return int(max(1, max(n_stab, n_acc)))

    def _nonlinear_is_rotation(self) -> bool:
        """True when the nonlinear arm is an exact per-channel rotation.

        Holds when no *complex* tensor was supplied: a real ``C`` makes the
        instantaneous arm a pure phase, which the closed-form exponential
        integrates exactly, so the real-tensor case still takes the Strang
        path. A complex tensor (a Raman gain-transfer arm) is not a rotation
        and is integrated by RK4 instead.
        """
        if self.coupling_tensor is not None and np.any(
            np.abs(self.coupling_tensor.imag) > 0.0
        ):
            return False
        if self.include_shock:
            # Shock correction is not a rotation (spectral prefactor).
            return False
        if self.raman_tensors is not None:
            # Delayed memory kernel: not a rotation even when real-valued.
            return False
        if self.fwm_coupling_tensor is not None and np.any(
            np.abs(self.fwm_coupling_tensor.imag) > 0.0
        ):
            return False
        return True

    def _coupled_nonlinear_step(self, A: list[NDArray], dz: float) -> list[NDArray]:
        """Coupled nonlinear step for every channel, at ``self._current_z``.

        SPM/XPM advance exactly (unitary phase rotation per channel). With
        ``include_fwm``, the heterodyne inter-modal FWM term is advanced by
        frequency-domain RK4 substeps, Strang-split around the exact
        diagonal phase (the FWM term is ``O(γP·f)`` while the diagonal
        phase is ``O(γP)``, so no stage integrates the stiff part).

        With a ``coupling_tensor`` the arm is integrated instead of
        exponentiated whenever it is not a pure rotation, because a complex
        tensor carries a gain-transfer (exchange) flow for which the
        exponential identity does not hold.
        """
        if self._nonlinear_is_rotation():
            if not self.include_fwm:
                diag = self._diagonal_phase(A)
                return [Ai * np.exp(1j * d * dz) for Ai, d in zip(A, diag)]
            # --- FWM mode ------------------------------------------------
            # Strang split: half of the diagonal phase, FWM RK4, trailing half.
            diag_in = self._diagonal_phase(A)
            n_sub = self._fwm_substep_count(A, dz)
            h_sub = dz / n_sub
            states = [Ai * np.exp(0.5j * d * dz) for Ai, d in zip(A, diag_in)]
            for _ in range(n_sub):
                states = self._rk4_fwm_substep(states, h_sub)
            diag_out = self._diagonal_phase(states)
            return [Ai * np.exp(0.5j * d * dz) for Ai, d in zip(states, diag_out)]

        # --- tensor mode --------------------------------------------------
        # No pure-phase diagonal exists to split against, so the whole
        # coupled term (instantaneous + FWM arms) advances by RK4 substeps.
        # --- per-channel shock (no tensors/FWM) --------------------------
        # Scalar-identical Strang shock scheme per channel (task 1.2): the
        # stiff Kerr/Raman phase stays analytic, only the shock correction
        # sees RK4 — N=1 matches SplitStepEngine at machine precision.
        if (
            self.include_shock
            and self.coupling_tensor is None
            and self.raman_tensors is None
            and not self.include_fwm
        ):
            gamma = self._gamma_v()
            if gamma == 0.0:
                return list(A)
            diag = self._diagonal_phase(A)
            return [
                self._shock_step_default(Ai, np.asarray(d) / gamma, gamma, dz)
                for Ai, d in zip(A, diag)
            ]
        n_sub = self._tensor_substep_count(A, dz)
        h_sub = dz / n_sub
        states = list(A)
        for _ in range(n_sub):
            states = self._rk4_tensor_substep(states, h_sub)
        return states

    def _rk4_fwm_substep(self, states: list[NDArray], h_sub: float) -> list[NDArray]:
        """One classical RK4 substep of the FWM flow (default coefficients)."""
        # Frozen-coefficient FWM flow (anti-Hermitian when the weight tensor
        # is exchange-symmetric -> photon-number conserving to O(h_sub^5)).
        # ISSUES.md #12: an explicit Euler step here pumps |A|^2 at O(dz).
        k1 = self._fwm_rhs(states)
        k2 = self._fwm_rhs([Ai + 0.5 * h_sub * ki for Ai, ki in zip(states, k1)])
        k3 = self._fwm_rhs([Ai + 0.5 * h_sub * ki for Ai, ki in zip(states, k2)])
        k4 = self._fwm_rhs([Ai + h_sub * ki for Ai, ki in zip(states, k3)])
        out = [
            Ai + (h_sub / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
            for Ai, k1i, k2i, k3i, k4i in zip(states, k1, k2, k3, k4)
        ]
        for m, s in enumerate(out):
            if not np.all(np.isfinite(s.view(float) if s.dtype == complex else s)):
                raise FloatingPointError(
                    f"FWM substep produced non-finite amplitudes in "
                    f"channel {m} at z = {self._current_z:.4e} m; the "
                    "substep integrator diverged — reduce the outer "
                    "step size or the seed amplitude (ISSUES.md #11)."
                )
        return out

    def _rk4_tensor_substep(self, states: list[NDArray], h_sub: float) -> list[NDArray]:
        """One classical RK4 substep of the full tensor-driven flow."""
        k1 = self._nonlinear_rhs_tensor(states)
        k2 = self._nonlinear_rhs_tensor(
            [Ai + 0.5 * h_sub * ki for Ai, ki in zip(states, k1)]
        )
        k3 = self._nonlinear_rhs_tensor(
            [Ai + 0.5 * h_sub * ki for Ai, ki in zip(states, k2)]
        )
        k4 = self._nonlinear_rhs_tensor([Ai + h_sub * ki for Ai, ki in zip(states, k3)])
        out = [
            Ai + (h_sub / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
            for Ai, k1i, k2i, k3i, k4i in zip(states, k1, k2, k3, k4)
        ]
        for m, s in enumerate(out):
            if not np.all(np.isfinite(s.view(float) if s.dtype == complex else s)):
                raise FloatingPointError(
                    f"coupling-tensor substep produced non-finite amplitudes "
                    f"in channel {m} at z = {self._current_z:.4e} m; the "
                    "substep integrator diverged — reduce the outer step "
                    "size or the coupling magnitude (ISSUES.md #11)."
                )
        return out

    def _tensor_substep_count(self, A: list[NDArray], dz: float) -> int:
        """RK4 substeps covering both the instantaneous and FWM tensor arms.

        Same policy as :meth:`_fwm_substep_count` — stability needs
        ``η·h_sub ≤ 2.5``, accuracy targets ``η·h_sub ≲ 0.05`` rad capped at
        ``_FWM_SUBSTEP_CAP``, and a configuration that cannot be integrated
        at this ``dz`` raises naming the channel.
        """
        rate, ch = self._coupling_rate_max(A)
        if self.include_fwm:
            fwm_rate, fwm_ch = self._fwm_rate_max(A, dz)
            if fwm_rate > rate:
                rate, ch = fwm_rate, fwm_ch
        rate = max(rate, 1e-30)
        n_stab = int(np.ceil(rate * dz / self._FWM_SUBSTEP_RATE))
        if n_stab > self._FWM_SUBSTEP_MAX:
            raise ValueError(
                f"coupling rate = {rate:.3e} /m over outer dz = {dz:.3e} m "
                f"exceeds the explicit RK4 stability region even with "
                f"{self._FWM_SUBSTEP_MAX} substeps (η·h ≈ "
                f"{rate * dz / self._FWM_SUBSTEP_MAX:.2f} > 2.5); reduce the "
                f"outer step size or shorten the coupling on channel {ch} "
                "(ISSUES.md #11)."
            )
        n_acc = min(self._FWM_SUBSTEP_CAP, int(np.ceil(rate * dz / 0.05)))
        return int(max(1, max(n_stab, n_acc)))

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
                    "show_progress=True requires tqdm. Install with: pip install tqdm)"
                ) from exc
            step_iter = tqdm(range(n_steps), desc="multimode GNLSE", unit="step")
        else:
            step_iter = range(n_steps)

        self.evolution = []
        self._z_positions = [0.0]
        self._energy_vs_z = []
        self._photon_vs_z = [] if self.include_shock else None
        self._power_l_vs_z = [] if self.oam_l is not None else None

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
            total = sum(float(np.sum(np.abs(Ai) ** 2)) * self.grid.dt for Ai in self.A)
            assert self._energy_vs_z is not None
            self._energy_vs_z.append(total)
            if self._photon_vs_z is not None:
                self._photon_vs_z.append(self._photon_number(self.A))
            if self._power_l_vs_z is not None and self.power_l_labels is not None:
                p = self.power_by_l()
                self._power_l_vs_z.append(
                    np.array([p[label] for label in self.power_l_labels], dtype=float)
                )

        _snapshot()  # z = 0

        for step in step_iter:
            self._current_z = step * dz
            # Strang split: half linear → coupled nonlinear → half linear
            self.A = [self._linear_step(self.A[m], dz / 2, m) for m in range(self._n)]
            self.A = self._coupled_nonlinear_step(self.A, dz)
            if self.include_tpa or self.include_free_carriers:
                self.A = self._tpa_fca_step(self.A, dz)
            # Apply Rayleigh backscatter if enabled
            if self.rayleigh_backscatter is not None:
                self.A = self._apply_rayleigh_backscatter(self.A, dz)
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

    def _apply_rayleigh_backscatter(self, A: list[NDArray], dz: float) -> list[NDArray]:
        """Apply Rayleigh backscatter coupling to the fields.

        The Rayleigh backscatter is a distributed feedback mechanism that couples
        forward and backward propagating waves. In the coupled power equations:

            dP+/dz = ... + eps * P-
            dP-/dz = ... + eps * P+

        where eps is the Rayleigh backscatter coefficient in 1/m.

        For the multimode GNLSE engine, this is implemented as a perturbation to
        the field. The Rayleigh backscatter coefficient eps has units of 1/m, and
        the coupling is applied as:

            A -> A + eps * A * dz

        This is a simplified treatment that captures the essential physics of the
        random distributed feedback mechanism.

        Parameters
        ----------
        A
            List of complex field arrays, one per channel.
        dz
            Step size in metres.

        Returns
        -------
        list[NDArray]
            The fields after applying the Rayleigh backscatter coupling.
        """
        if self.rayleigh_backscatter is None or self.rayleigh_backscatter == 0.0:
            return A
        # Simplified Rayleigh backscatter: A -> A + eps * A * dz
        # This captures the essential physics of the random distributed feedback
        eps = self.rayleigh_backscatter
        return [a + eps * a * dz for a in A]

    def _emit_energy_drift_warning(self) -> None:
        """Warn if the total photon number drifted >5% with loss disabled."""
        assert self._energy_vs_z is not None
        if not self._energy_vs_z:
            return
        e0, e1 = self._energy_vs_z[0], self._energy_vs_z[-1]
        if e0 <= 0:
            return
        if self.fiber.alpha != 0 or self.channel_alpha is not None:
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
    # cavity iteration
    # ------------------------------------------------------------------

    @property
    def has_cavity(self) -> bool:
        """Whether cavity reflectivity is configured."""
        return self.cavity_reflectivity_input is not None

    def cavity_beat_period(self, group_index: float | None = None) -> float:
        """Longitudinal mode beat period of the cavity, in Hz.

        The beat period is set by the cavity round-trip time:
        ``f_beat = c / (2 * n_g * L_rt)`` where ``L_rt`` is the round-trip
        length and ``n_g`` is the group index. For the Babin 2005
        configuration (L = 16 m, n_g = 1.466) this gives approximately
        6 MHz.

        Parameters
        ----------
        group_index
            Group index of the fiber. ``None`` (default) uses the engine's
            ``cavity_group_index``, itself defaulting to 1.466 (silica at 1.5 um).

        Returns
        -------
        float
            Beat period in Hz.
        """
        from photonics_helper.base import C_MS

        n_g = self.cavity_group_index if group_index is None else float(group_index)
        if not n_g > 0:
            raise ValueError(f"group_index must be positive, got {n_g!r}")
        return C_MS / (2.0 * n_g * self.cavity_round_trip_length)

    def _apply_cavity_boundary(self, relaxation: float = 1.0) -> None:
        """Apply round-trip boundary conditions to the circulating field.

        After propagation, the Stokes output at z=L is reflected by the
        output mirror, propagates back (round-trip phase), and is reflected
        by the input mirror. The result is the new input field at z=0.

        The pump (channel 0) is not reflected; it is re-injected by the
        caller before the next propagation.

        Parameters
        ----------
        relaxation
            Under-relaxation factor ``w`` in ``A <- (1 - w) A_old + w A_boundary``,
            in ``(0, 1]``. ``1.0`` is the plain map. See
            :meth:`cavity_iterate` for why an above-threshold cavity needs ``w < 1``.
        """
        assert self.cavity_reflectivity_input is not None
        assert self.cavity_reflectivity_output is not None
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(f"relaxation must be in (0, 1], got {relaxation!r}")
        r_in = self.cavity_reflectivity_input
        r_out = self.cavity_reflectivity_output
        n = self._n

        # Save the output fields at z=L
        output_fields = [self.A[k].copy() for k in range(n)]

        # Apply output mirror reflectivity and compute round-trip phase
        for k in range(1, n):  # Skip pump (channel 0)
            # Output mirror
            output_fields[k] = output_fields[k] * np.sqrt(r_out[k])
            # Round-trip phase: the field travels back through the cavity.
            # beta_k is taken at the group index, since the round trip is set by the
            # group velocity. Using a hard-coded index here would make the phase, and
            # therefore the converged circulating field, silently depend on a constant
            # that the caller cannot see.
            lam_m = self.waves[k].central_wavelength.as_m
            if lam_m > 0:
                beta_k = 2.0 * np.pi * self.cavity_group_index / lam_m
                phase = beta_k * self.cavity_round_trip_length
                output_fields[k] = output_fields[k] * np.exp(1j * phase)
            # Input mirror
            output_fields[k] = output_fields[k] * np.sqrt(r_in[k])

        # Set the new input fields (Stokes only; pump is re-injected)
        for k in range(1, n):
            if relaxation < 1.0:
                self.A[k] = (1.0 - relaxation) * self.A[k] + relaxation * output_fields[
                    k
                ]
            else:
                self.A[k] = output_fields[k]

    def cavity_iterate(
        self,
        num_steps: int,
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        pump_field: NDArray | None = None,
        relaxation: float = 1.0,
        callback=None,
    ) -> "CavityResult":
        """Iterate propagation and boundary application to convergence.

        The cavity iteration propagates all channels through the fiber,
        applies the round-trip boundary conditions, re-injects the pump,
        and repeats until the circulating field converges.

        Parameters
        ----------
        num_steps
            Number of split-step propagation steps per iteration.
        tolerance
            Convergence tolerance on the relative change of the circulating
            *intensity* ``|A|**2`` between successive round trips, i.e.
            ``|| |F(A)|**2 - |A|**2 || / || |A|**2 ||``. Intensity, not the complex
            field, because the round-trip map carries an arbitrary absolute phase;
            see the comment at the metric below.
        max_iterations
            Maximum number of cavity iterations. Raises on non-convergence.
        relaxation
            Under-relaxation factor ``w`` in ``A <- (1 - w) A_old + w A_boundary``,
            in ``(0, 1]``. Default ``1.0`` is the plain round-trip map.

            **An above-threshold cavity needs ``w < 1``.** Its fixed point exists —
            pump depletion and output coupling balance it — but plain iteration
            diverges: the small-signal loop gain at the origin is much greater than
            one, and the gain saturation that would arrest the growth only sets in far
            from the origin, so every iterate runs away instead of climbing toward the
            fixed point. This was confirmed directly: a seeded first-Stokes cavity with
            mirror reflectivities of 0.99 raised ``RuntimeError`` at every launch power
            and seed tried. Damping moves each iterate inside the basin of attraction
            instead of over it. ``w = 1`` remains the default so that a below-threshold
            cavity — the only case that converges unconditionally — is untouched.
        pump_field
            Pump field to inject at z=0 at each iteration. If None, the
            original pump field is used.
        callback
            Optional callback function called after each iteration with
            the iteration index and current relative change.

        Returns
        -------
        CavityResult
            The converged field, iteration count, and convergence history.

        Raises
        ------
        RuntimeError
            If the cavity does not converge within ``max_iterations``. A cavity above
            its Stokes threshold genuinely has no fixed point under plain
            fixed-point iteration, and this is the one case where raising is the
            correct answer rather than a bug.
        ValueError
            If no cavity is configured.
        """
        if not self.has_cavity:
            raise ValueError(
                "cavity_iterate requires cavity_reflectivity_input and "
                "cavity_reflectivity_output to be configured"
            )

        # Save the original pump field
        original_pump = self.A[0].copy()
        if pump_field is not None:
            self.A[0] = np.asarray(pump_field, dtype=complex).copy()

        history: list[float] = []
        converged = False

        for iteration in range(max_iterations):
            # The state fed into this round trip. It is the previous iterate's
            # post-boundary field, i.e. the fixed-point variable.
            incoming = [self.A[k].copy() for k in range(1, self._n)]

            # Propagate through the fiber
            self.propagate(num_steps)

            # Apply cavity boundary conditions
            self._apply_cavity_boundary(relaxation)

            # Re-inject pump
            self.A[0] = original_pump.copy()

            # Convergence is judged on the circulating intensity, not on the complex
            # field. The round-trip map carries an absolute phase exp(i beta_k L_rt),
            # and beta_k L_rt is O(1e8) rad, so its residue mod 2pi is an arbitrary
            # constant that alternates with the optical frequency rather than
            # converging. A complex-field residual ||F(x) - x|| / ||x|| therefore
            # settles at |1 - m e^{i phi}| — for a converged, contracting cavity that is
            # a nonzero constant, not zero — and a solver written on it would raise on
            # exactly the configurations it is supposed to accept. The circulating
            # intensity envelope is the phase-insensitive observable every consumer of
            # this result (threshold, spectra, RIN) actually reads, and it does
            # converge. The complex fields themselves are returned untouched.
            max_change = 0.0
            for j, k in enumerate(range(1, self._n)):
                new_i = np.abs(self.A[k]) ** 2
                old_i = np.abs(incoming[j]) ** 2
                old_norm = float(np.sqrt(np.sum(old_i)))
                new_norm = float(np.sqrt(np.sum(new_i)))
                if old_norm > 0:
                    change = float(np.sqrt(np.sum((new_i - old_i) ** 2))) / old_norm
                elif new_norm > 0:
                    change = float("inf")
                else:
                    change = 0.0
                max_change = max(max_change, change)

            history.append(max_change)

            if callback is not None:
                callback(iteration, max_change)

            if max_change < tolerance:
                converged = True
                break

        if not converged:
            raise RuntimeError(
                f"Cavity iteration did not converge within {max_iterations} "
                f"iterations (last relative change: {max_change:.3e}, "
                f"tolerance: {tolerance:.3e}). The cavity may be "
                f"over-pumped or the reflectivity too high for convergence."
            )

        return CavityResult(
            fields=[self.A[k].copy() for k in range(self._n)],
            iterations=iteration + 1,
            converged=converged,
            history=history,
            beat_period_hz=self.cavity_beat_period(),
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
            np.array(
                [snaps[m].envelope_field for snaps in self.evolution], dtype=complex
            )
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
