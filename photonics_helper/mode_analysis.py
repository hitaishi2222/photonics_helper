"""Mode-solver to multimode-engine bridge (step7).

Loads an N-mode tidy3d/femwell export into a permutation-invariant,
fundamental-first ordering (so the deck's channel 0 is the engine's
``group_delays[0] == 0`` fundamental), derives the per-mode dispersion deck,
computes overlap tensors and per-mode ``gamma`` from arbitrary vector fields,
and builds :class:`~photonics_helper.multimode_gnlse.MultimodeSplitStepEngine`
kwargs in one call. No circular-symmetry assumption anywhere.

Equation grounding (read as page images, never ``pdftotext``)
-------------------------------------------------------------
* Wright et al., JSTQE **24**, 5100516 (2018) — ``ph_pdfs/wright_2017.pdf`` p3
  (``ph_pdfs/pages_wright/page-03.png``, verified by image read): the GMMNLSE
  Eq. (1) with shock ``(1 + i/ω₀∂_t)``, Raman convolution over ``h_R``, and
  the ``S^R_plmn = S^K_plmn`` overlap-tensor definition Eq. (2). The tensors
  :func:`overlap_tensors_from_fields` returns are that ``S^K`` object in the
  real-LP basis (where Poletti Sec. 3 gives ``Q^(1) = Q^(2)``).
* Afshar & Monro, Opt. Express **17**, 2298 (2009) — ``ph_pdfs/afshar_2009.pdf``
  (page images in ``ph_pdfs/pages_afshar/``): full-vectorial Kerr overlaps for
  high-contrast guides — ``γ`` / ``A_eff`` / orthogonality definitions.
* Poletti & Horak, JOSA B **25**, 1645 (2008) — in-repo
  ``reproductions/poletti_2008_multimode/JOSAB.25.001645.pdf``: MM-GNLSE +
  Eq. 7 overlaps (engine spec), Eq. 18/19 selection rules. The fibre-specific
  solver stays in ``reproductions/poletti_2008_multimode/overlap.py``; only
  the shape-agnostic quadrature idea is reused here.

Per-mode gamma routing decision (task 3.3)
------------------------------------------
High-index-contrast rib modes differ in ``γ_m`` by ≳2×, but the engine takes
one shared ``_fiber_gamma``. We route mode-resolved couplings in absolute
units through the existing ``coupling_tensor`` / ``fwm_coupling_tensor``
slots — **no engine signature change**. Rationale: the slots already multiply
``|A_j|²`` / FWM arms in absolute units, so stuffing ``γ_mn`` there works
today and keeps steps 8–10 diffs additive-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, PI, Wavelength, WavelengthArray
from photonics_helper.fiber import WaveguideMode

__all__ = [
    "DispersionSet",
    "LoadedMode",
    "ModePairParameters",
    "build_multimode_deck",
    "ensure_continuation",
    "load_modes",
    "mode_dispersion",
    "overlap_tensors_from_fields",
]


@dataclass
class LoadedMode:
    """One vector mode: ``n_eff(λ)`` table plus transverse fields.

    ``Ex`` / ``Ey`` have shape ``(n_wl, Ny, Nx)`` (complex), sampled on the
    Cartesian grid ``x_um`` / ``y_um`` at the wavelengths ``wavelengths_um``.
    ``label`` is the persistent physical label (OAM-like integer); eigen-index
    ordering is NOT trusted across λ.
    """

    label: int
    wavelengths_um: NDArray
    neff: NDArray
    ng: NDArray | None
    Ex: NDArray
    Ey: NDArray
    x_um: NDArray
    y_um: NDArray

    def to_waveguide_mode(self) -> WaveguideMode:
        """The ``n_eff(λ)`` half as a library :class:`WaveguideMode`."""
        return WaveguideMode(
            neff=np.asarray(self.neff, dtype=float),
            wavelengths=WavelengthArray(
                np.asarray(self.wavelengths_um, dtype=float), "um"
            ),
            ng=None if self.ng is None else np.asarray(self.ng, dtype=float),
        )


@dataclass
class DispersionSet:
    """Per-mode Taylor deck for the multimode engine.

    ``betas[m]`` is the ``[β₂, β₃, …]`` array (ps^k/m) for mode ``m``;
    ``group_delays[m] = β₁⁽ᵐ⁾ − β₁⁽⁰⁾`` (s/m) with ``group_delays[0] == 0``;
    ``phase_offsets[m]`` is the ``β₀`` relative phase (rad/m, GRIN-like only).
    """

    betas: list[NDArray]
    group_delays: NDArray
    phase_offsets: NDArray


@dataclass
class ModePairParameters:
    """Overlap / nonlinear parameters from real fields.

    ``gamma`` is per-mode ``γ_m`` (1/W/m); ``A_eff`` per-mode effective area
    (m²); ``xpm_weights`` (N×N) and ``fwm_weights`` (N×N×N×N) are the
    ``S⁽¹⁾``/``S⁽²⁾`` tensors normalised to ``S_0000 = 1`` (``fwm_weights``
    real when the fields are real, complex when not — never clipped);
    ``oam_l`` the azimuthal labels from the export; ``coupling_tensor`` the
    absolute-unit ``γ_mn`` matrix routed through the engine slot.
    ``confinement`` is currently ``Γ_m = 1``: the loader contract carries
    no core mask (see :func:`overlap_tensors_from_fields`).
    """

    gamma: NDArray
    A_eff: NDArray
    xpm_weights: NDArray
    fwm_weights: NDArray
    oam_l: list[int]
    coupling_tensor: NDArray
    confinement: NDArray = field(default_factory=lambda: np.array([]))


def _mode_files(export_dir: Path) -> list[Path]:
    files = sorted(export_dir.glob("mode-*.npz"))
    if not files:
        raise ValueError(
            f"no mode-*.npz files in {export_dir}; expected one per mode "
            "(see examples/generate_multimode_mode_data.py)"
        )
    return files


def load_modes(export_dir: str | Path) -> list[LoadedMode]:
    """Load an N-mode export in a permutation-invariant ordering.

    Each ``mode-*.npz`` carries ``wavelength_um``, ``neff``, optional ``ng``,
    ``Ex``/``Ey`` ``(n_wl, Ny, Nx)``, ``x_um``/``y_um``, and ``label``.
    Within one file the λ-walk of one physical mode is the solver's contract;
    across files the ordering is canonicalised by :func:`ensure_continuation`,
    so shuffled inputs yield identical decks.
    """
    export_dir = Path(export_dir)
    modes: list[LoadedMode] = []
    for f in _mode_files(export_dir):
        d = dict(np.load(f, allow_pickle=False))
        modes.append(
            LoadedMode(
                label=int(d.get("label", len(modes))),
                wavelengths_um=np.asarray(d["wavelength_um"], dtype=float),
                neff=np.asarray(d["neff"], dtype=float),
                ng=None if "ng" not in d else np.asarray(d["ng"], dtype=float),
                Ex=np.asarray(d["Ex"], dtype=complex),
                Ey=np.asarray(d["Ey"], dtype=complex),
                x_um=np.asarray(d["x_um"], dtype=float),
                y_um=np.asarray(d["y_um"], dtype=float),
            )
        )
    return ensure_continuation(modes)


def ensure_continuation(modes: list[LoadedMode]) -> list[LoadedMode]:
    """Canonicalise the file ordering: fundamental first, stable under
    input permutation.

    Decks are sorted by ``n_eff`` at the mid-grid wavelength (descending) —
    a pure function of the physical tables, not of file order — which puts
    the fundamental mode, the engine's ``group_delays[0] == 0`` anchor, at
    index 0. Cross-λ labelling / continuation of the fields themselves is
    the solver-export contract (one file carries one physical mode's λ-walk);
    the azimuthal ``label`` travels untouched from the export.
    """
    if len(modes) <= 1:
        return sorted(modes, key=lambda m: m.label)
    # Fundamental first (highest n_eff at mid-grid): physical ordering that is
    # still a pure function of the fields/tables, hence permutation-invariant.
    mid_idx = [len(m.neff) // 2 for m in modes]
    order = sorted(
        range(len(modes)),
        key=lambda i: -float(modes[i].neff[mid_idx[i]]),
    )
    return [modes[i] for i in order]


def mode_dispersion(
    modes: list[LoadedMode], order: int, lambda0: float
) -> DispersionSet:
    """Per-mode Taylor ``betas``, ``group_delays``, ``phase_offsets``.

    ``betas[m]`` comes from :meth:`Dispersion.get_betas` on each mode's
    ``n_eff(λ)`` (ps^k/m, Ω in rad/ps). ``group_delays[m] = β₁⁽ᵐ⁾ − β₁⁽⁰⁾``
    with ``β₁ = n_g/c`` (``n_g`` from the table when present, else ``n_eff −
    λ dn_eff/dλ``); ``group_delays[0] == 0`` enforced. ``phase_offsets``
    from ``β₀⁽ᵐ⁾`` differences — meaningful for GRIN-like profiles only
    (step-index / high-contrast ribs accumulate differential phase that the
    engine's common-carrier picture cannot use; see task 2.2 note in the
    API docs).
    """
    lam0 = Wavelength(float(lambda0), "um")
    betas: list[NDArray] = []
    beta1: list[float] = []
    beta0: list[float] = []
    for m in modes:
        wm = m.to_waveguide_mode()
        disp = wm.to_dispersion()
        b = np.asarray(disp.get_betas(order, wavelength=lam0), dtype=float).ravel()
        betas.append(b)
        wl = np.asarray(m.wavelengths_um)
        ne = np.asarray(m.neff)
        if m.ng is not None:
            ng0 = float(np.interp(float(lambda0), wl, np.asarray(m.ng)))
        else:
            dne = float(np.gradient(ne, wl)[np.argmin(np.abs(wl - lambda0))])
            ng0 = float(np.interp(float(lambda0), wl, ne)) - float(lambda0) * dne
        beta1.append(ng0 / C_MS)
        ne0 = float(np.interp(float(lambda0), wl, ne))
        omega0 = 2.0 * PI * C_MS / (float(lambda0) * 1e-6)
        beta0.append(ne0 * omega0 / C_MS)
    beta1_arr = np.asarray(beta1)
    gd = beta1_arr - beta1_arr[0]
    gd[0] = 0.0
    po = np.asarray(beta0) - beta0[0]
    return DispersionSet(betas=betas, group_delays=gd, phase_offsets=po)


def overlap_tensors_from_fields(
    modes: list[LoadedMode],
    n2: float,
    lambda0: float,
) -> ModePairParameters:
    """Grid quadrature overlaps with no circular-symmetry assumption.

    Fields are the transverse vector ``F = (Ex, Ey)`` evaluated at the **pump**
    wavelength; normalised profiles ``∬|F_m|² = 1`` on the Cartesian grid; then
    ``A_eff,m = (∬|F_m|²)²/∬|F_m|⁴``, ``γ_m = n₂ω₀/(c·A_eff,m)`` (Afshar
    Part I Kerr), ``S⁽¹⁾_mn = ∬|F_m|²|F_n|² → xpm_weights``,
    ``S⁽²⁾_mnpq = ∬(F_m*·F_n)(F_p·F_q*) → fwm_weights`` (Poletti–Horak Eq. 7;
    in the real basis ``Q⁽¹⁾ = Q⁽²⁾``; the vector form reduces to
    ``∬F_m*F_nF_pF_q*`` for single-component / LP fields), ``oam_l`` the
    azimuthal labels from the export (real LP-like fields carry no winding
    phase to recover numerically), ``coupling_tensor`` the absolute ``γ_mn``
    matrix for the engine slot.

    Confinement ``Γ_m`` is unity: the loader contract carries fields and
    ``n_eff(λ)`` only, no core mask, so a power-in-core fraction cannot be
    derived here. A deck that wants confinement-scaled ``n₂`` per mode
    (multi-cladding guides) must pre-scale ``n2·Γ_m`` per mode and route
    the result through the engine's per-mode coupling slots.
    """
    n = len(modes)
    dx = float(np.mean(np.diff(modes[0].x_um))) * 1e-6
    dy = float(np.mean(np.diff(modes[0].y_um))) * 1e-6
    dA = dx * dy
    omega0 = 2.0 * PI * C_MS / (float(lambda0) * 1e-6)

    # Fields are the two TRANSVERSE components (Ex, Ey) of the vector mode.
    # Two corrections over the first cut (STATUS finding #10):
    #  * evaluate at the PUMP wavelength, not the grid-median index — a
    #    nonlinear deck is defined at the pump, and the median of a broad
    #    export can sit microns away from it;
    #  * contract the FULL transverse vector (|Ex|²+|Ey|² and the vector dot
    #    products below), not |Ex|² alone — for a mode whose plane is normal
    #    to x, Ex is the *longitudinal* component and |Ex|² is not the
    #    intensity.
    Fx_list: list[NDArray] = []
    Fy_list: list[NDArray] = []
    for ld in modes:
        wl = np.asarray(ld.wavelengths_um, dtype=float)
        iw = int(np.argmin(np.abs(wl - float(lambda0))))
        fx = ld.Ex[iw].astype(complex)
        fy = ld.Ey[iw].astype(complex)
        norm = float(np.sum(np.abs(fx) ** 2 + np.abs(fy) ** 2) * dA)
        Fx_list.append(fx / np.sqrt(norm))
        Fy_list.append(fy / np.sqrt(norm))
    Fx = np.asarray(Fx_list)  # (N, Ny, Nx)
    Fy = np.asarray(Fy_list)
    I2 = np.abs(Fx) ** 2 + np.abs(Fy) ** 2  # (N, Ny, Nx)

    Aeff = np.empty(n)
    Gamma = np.ones(n)
    for ii in range(n):
        denom = float(np.sum(I2[ii] ** 2) * dA)
        Aeff[ii] = 1.0 / denom
    gamma = n2 * omega0 * Gamma / (C_MS * Aeff)

    S1 = np.einsum("mij,nij->mn", I2, I2) * dA
    # S2 with the full vector conjugation structure (F_m*·F_n)(F_p·F_q*),
    # which reduces exactly to F_m* F_n F_p F_q* when the field lives in a
    # single transverse component (the scalar / real-LP case).
    D = np.einsum("mij,nij->mnij", np.conj(Fx), Fx) + np.einsum(
        "mij,nij->mnij", np.conj(Fy), Fy
    )
    S2 = np.einsum("mnij,pqij->mnpq", D, np.conj(D)) * dA
    # Keep the dtype honest: S2 is real for real (LP-like) mode sets but
    # genuinely complex for propagating-mode field exports — never clip
    # the imaginary part silently (the engine now accepts complex
    # fwm_weights), and never emit fake complex128 when values are real.
    S2 = np.real_if_close(S2)
    scale = float(abs(S2[0, 0, 0, 0]))
    xpm = (S1 / scale).astype(float)
    fwm = S2 / scale
    # Absolute-unit coupling matrix for the engine slot
    cpl = np.empty((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            cpl[a, b] = 0.5 * (gamma[a] + gamma[b]) * xpm[a, b]
    return ModePairParameters(
        gamma=gamma,
        A_eff=Aeff,
        xpm_weights=xpm,
        fwm_weights=fwm,
        oam_l=[int(modes[i].label) for i in range(n)],
        coupling_tensor=cpl,
        confinement=Gamma,
    )


def build_multimode_deck(
    export_dir: str | Path,
    lambda0: float,
    dispersion_order: int,
    n2: float,
    *,
    pump: dict | None = None,
) -> dict:
    """One call from geometry + pump to engine kwargs (task 4.1).

    Returns the ``MultimodeSplitStepEngine`` kwargs: per-mode ``betas`` list,
    ``group_delays``, ``phase_offsets``, ``xpm_weights`` / ``fwm_weights``
    from ``S⁽¹⁾``/``S⁽²⁾``, ``oam_l``, and absolute-unit ``coupling_tensor``.
    ``pump`` is accepted for forward-compat (peak power / duration shape the
    grid elsewhere) and recorded, not consumed.

    Note for step-index / high-contrast decks: ``phase_offsets`` are computed
    here for completeness, but the differential ``β₀`` phase has no meaning
    under the engine's common-carrier picture — pass
    ``phase_offsets=None`` to the engine for rib/step-index propagations.
    """
    modes = load_modes(export_dir)
    disp = mode_dispersion(modes, dispersion_order, float(lambda0))
    pars = overlap_tensors_from_fields(modes, float(n2), float(lambda0))
    deck = {
        "betas": [np.asarray(b) for b in disp.betas],
        "group_delays": np.asarray(disp.group_delays, dtype=float),
        "phase_offsets": np.asarray(disp.phase_offsets, dtype=float),
        "xpm_weights": np.asarray(pars.xpm_weights, dtype=float),
        "fwm_weights": np.asarray(pars.fwm_weights),
        "oam_l": list(pars.oam_l),
        "coupling_tensor": np.asarray(pars.coupling_tensor),
        "n_modes": len(modes),
        "lambda0_um": float(lambda0),
        "pump": dict(pump or {}),
    }
    return deck
