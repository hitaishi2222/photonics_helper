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

Per-mode gamma routing decision (task 3.3, corrected 2026-10)
----------------------------------------------------------
High-index-contrast rib modes differ in ``γ_m`` by ≳2×, but the engine's
uniform models take one shared γ. The bridge routes mode-resolved couplings
through the absolute ``coupling_tensor`` slot as
``C_ij = (n₂ω₀/c)·Q⁽¹⁾_iijj = γ₀·xpm_ij``. Its diagonal is exactly ``γ_i``;
the off-diagonal is the literature XPM coefficient, **not** a
``½(γ_i+γ_j)`` average (the old form underestimated SPM on a larger mode
by ``A_eff,0/A_eff,i`` and had no source). ``gamma_reference = γ₀`` is
also returned so the S₀₀₀₀-normalised FWM/Raman tensors keep their absolute
scale independent of the caller's ``Fiber.A_eff``. The engine grows a
``gamma_reference`` override for that purpose; no per-mode γ array enters
the public signature.
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
    ``S⁽¹⁾``/``S⁽²⁾`` tensors normalised to the fundamental SPM coefficient
    ``S⁽¹⁾_0000 = 1`` (``fwm_weights`` real when the fields are real, complex
    when not — never clipped); ``oam_l`` the azimuthal labels from the
    export; ``coupling_tensor`` the **absolute** ``C_ij`` matrix in W⁻¹m⁻¹,
    ``C_ij = γ_0·xpm_weights[i, j] = (n₂ω₀/c)·S⁽¹⁾_iijj`` — its diagonal is
    exactly the per-mode ``γ_i``. ``confinement`` is currently ``Γ_m = 1``:
    the loader contract carries no core mask (see
    :func:`overlap_tensors_from_fields`).
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
    *,
    xpm_model: str = "co_polarized",
) -> ModePairParameters:
    """Grid quadrature overlaps with no circular-symmetry assumption.

    Fields are the transverse vector ``F = (Ex, Ey)`` evaluated at the **pump**
    wavelength; normalised profiles ``∬|F_m|² = 1`` on the Cartesian grid; then
    ``A_eff,m = (∬|F_m|²)²/∬|F_m|⁴``, ``γ_m = n₂ω₀/(Γ_m c·A_eff,m)`` (Afshar
    Part I Kerr), ``Q⁽¹⁾_mn = ∬|F_m|²|F_n|²`` (xpm), ``Q⁽²⁾_mnpq =
    ∬(F_m*·F_n)(F_p·F_q*)`` (fwm) — Poletti–Horak Eq. 7.

    The engine's ``xpm_weights[i, j]`` multiplies ``|A_j|²`` **once** in
    channel ``i``, whereas the MM-GNLSE (Mumtaz Eq. 6) carries two XPM
    index permutations. For a **co-polarized scalar LP basis** those
    permutations are equal, so the physical XPM coefficient is
    ``2·Q⁽¹⁾_iijj`` while SPM stays ``Q⁽¹⁾_iiii`` (SPM 1, XPM 2 —
    Agrawal's co-polarized Kerr limit). ``xpm_model="co_polarized"``
    (default) applies that factor to the off-diagonal entries only.
    ``xpm_model="vector"`` keeps the bare ``Q⁽¹⁾`` ratio for a
    polarization-resolved / two-copy basis; a genuine vector basis needs
    separate polarization channels because the orthogonal-polarization
    XPM factor (2/3 linear, 2 circular) cannot be represented by one
    scalar weight (see the ``coef_model`` discussion in
    :mod:`photonics_helper.multimode_gnlse`).

    ``coupling_tensor`` is the **absolute** matrix for the engine slot,
    ``C_ij = γ₀·xpm_weights[i, j] = (n₂ω₀/c)·Q⁽¹⁾_iijj``. Its diagonal is
    exactly the per-mode ``γ_i``: ``γ₀·Q⁽¹⁾_iiii/Q⁽¹⁾_0000 =
    n₂ω₀/(cA_eff,i)``. This is the step7 scaling correction — averaging
    ``½(γ_i+γ_j)`` underestimates SPM on a larger mode by the factor
    ``A_eff,0/A_eff,i`` (0.75× in the committed rib deck) and has no
    literature basis.

    ``oam_l`` are the azimuthal labels from the export (real LP-like fields
    carry no winding phase to recover numerically). Confinement ``Γ_m`` is
    unity: the loader contract carries fields and ``n_eff(λ)`` only, no core
    mask, so a power-in-core fraction cannot be derived here. A deck that
    wants confinement-scaled ``n₂`` per mode (multi-cladding guides) must
    pre-scale ``n2·Γ_m`` per mode.
    """
    if xpm_model not in ("co_polarized", "vector"):
        raise ValueError(
            f"xpm_model must be 'co_polarized' or 'vector', got {xpm_model!r}"
        )
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
    # the imaginary part silently (the engine accepts complex
    # fwm_weights), and never emit fake complex128 when values are real.
    S2 = np.real_if_close(S2)
    # S⁽¹⁾ and S⁽²⁾ share the same self-normalisation for unit-power modes:
    # S⁽²⁾_0000 = ∬|F_0|⁴ = S⁽¹⁾_0000 = 1/A_eff,0.
    scale = float(abs(S2[0, 0, 0, 0]))
    xpm = S1 / scale
    if xpm_model == "co_polarized":
        # Two equal XPM index permutations (Mumtaz Eq. 6); diagonal SPM
        # stays at one. See the docstring derivation.
        off = ~np.eye(n, dtype=bool)
        xpm = np.where(off, 2.0 * xpm, xpm)
    xpm = np.real_if_close(xpm).astype(float)
    fwm = S2 / scale
    # Absolute-unit coupling matrix for the engine slot. C_ii = γ_i exactly;
    # C_ij = (n₂ω₀/c)·S⁽¹⁾_iijj. Never average ½(γ_i+γ_j) — that destroys
    # the per-mode SPM scale (step7 audit, 2026-10).
    gamma0 = float(gamma[0])
    cpl = (gamma0 * xpm).astype(complex)
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
    xpm_model: str = "co_polarized",
) -> dict:
    """One call from geometry + pump to engine kwargs (task 4.1).

    Returns the ``MultimodeSplitStepEngine`` kwargs: per-mode ``betas`` list,
    ``group_delays``, ``phase_offsets``, ``xpm_weights`` / ``fwm_weights``
    from ``S⁽¹⁾``/``S⁽²⁾``, ``oam_l``, absolute-unit ``coupling_tensor``, and
    ``gamma_reference = γ₀`` (the fundamental-mode ``γ``).

    ``gamma_reference`` is the load-bearing scale for every tensor-normalised
    term: ``fwm_weights`` and ``raman_tensors`` are divided by ``S_0000``,
    so their absolute coefficient is ``γ₀·weight``. Passing it makes the
    deck independent of whichever ``A_eff`` the caller puts on the
    :class:`~photonics_helper.gnlse.Fiber`; without it, a fiber ``A_eff``
    other than ``A_eff,0`` silently mis-scales FWM/Raman. ``A_eff`` and
    ``gamma`` are returned for inspection and for per-mode loss/pre-scaling
    calculations.

    ``pump`` is accepted for forward-compat (peak power / duration shape the
    grid elsewhere) and recorded, not consumed.

    Note for step-index / high-contrast decks: ``phase_offsets`` are computed
    here for completeness, but the differential ``β₀`` phase has no meaning
    under the engine's common-carrier picture — pass
    ``phase_offsets=None`` to the engine for rib/step-index propagations.
    """
    modes = load_modes(export_dir)
    disp = mode_dispersion(modes, dispersion_order, float(lambda0))
    pars = overlap_tensors_from_fields(
        modes, float(n2), float(lambda0), xpm_model=xpm_model
    )
    deck = {
        "betas": [np.asarray(b) for b in disp.betas],
        "group_delays": np.asarray(disp.group_delays, dtype=float),
        "phase_offsets": np.asarray(disp.phase_offsets, dtype=float),
        "xpm_weights": np.asarray(pars.xpm_weights, dtype=float),
        "fwm_weights": np.asarray(pars.fwm_weights),
        "oam_l": list(pars.oam_l),
        "coupling_tensor": np.asarray(pars.coupling_tensor),
        "gamma_reference": float(pars.gamma[0]),
        "gamma": np.asarray(pars.gamma, dtype=float),
        "A_eff": np.asarray(pars.A_eff, dtype=float),
        "n_modes": len(modes),
        "lambda0_um": float(lambda0),
        "pump": dict(pump or {}),
    }
    return deck


# ---------------------------------------------------------------------------
# Azimuthal / spin decomposition: the OAM measurement (GAP A3, PLAN §5 A2)
# ---------------------------------------------------------------------------
# ``oam_l`` in every deck above is a *label*: it is copied out of the export
# metadata (``LoadedMode.label``) and nothing verifies it. The functions below
# are the measurement that turns it into a checked number, and the purity
# metric ``P_l`` that any OAM / vortex claim has to quote.
#
# Sign convention, stated because it is not obvious and a mirror flips it:
# the azimuth is ``phi = arctan2(z, y)`` on the transverse pair ``(y, z)`` in
# the order the arrays arrive. Swapping that pair (or any axis reflection)
# maps ``l -> -l``. Export paths that rename transverse axes must therefore be
# audited before their ``l`` is trusted (oam_scg GAP B1).


def azimuthal_spectrum(
    field: NDArray,
    y: NDArray,
    z: NDArray,
    *,
    rmax: float | None = None,
    nr: int = 160,
    nphi: int = 256,
    lmax: int = 6,
) -> tuple[NDArray, NDArray]:
    """Azimuthal decomposition ``A_l = ∬ f(r,φ) e^{-ilφ} r dr dφ`` of one component.

    Parameters
    ----------
    field
        Complex transverse field on the regular grid ``(y, z)`` (e.g. ``E_x``).
    y, z
        Monotonic transverse coordinate axes in µm.
    rmax
        Integration radius in µm. The default is ``0.3·min(ptp(y), ptp(z))``,
        inside the simulation window but wide enough to hold the guided core.
        Power outside ``rmax`` is **not** counted, so report the metric at two
        radii when the tail is large (see oam_scg/STATUS.md).
    nr, nphi
        Polar quadrature sizes; ``nphi`` must exceed ``2·lmax`` (Nyquist in φ).
    lmax
        Maximum ``|l|`` returned.

    Returns
    -------
    ``(amps, ls)`` with ``amps[i]`` the complex amplitude of ``ls[i]``.
    Unnormalised. Use :func:`spin_azimuthal_spectrum` for a normalised basis.

    Notes
    -----
    The quadrature is a plain midpoint rule on a polar grid with weights
    ``r dr dφ``; it is exact for fields well resolved on both grids. The
    interpolation is bilinear (:class:`scipy.interpolate.RegularGridInterpolator`),
    zero outside the grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    field = np.asarray(field)
    if field.shape != (y.size, z.size):
        raise ValueError(
            f"field shape {field.shape} does not match (y, z) = ({y.size}, {z.size})"
        )
    if nphi <= 2 * lmax:
        raise ValueError(f"nphi={nphi} cannot resolve lmax={lmax} (need nphi > 2·lmax)")
    if rmax is None:
        rmax = 0.3 * min(np.ptp(y), np.ptp(z))

    rr = np.linspace(0.0, rmax, nr)
    pp = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    weights = (
        np.broadcast_to(rr[:, None], (nr, nphi))
        * (rr[1] - rr[0])
        * (2.0 * np.pi / nphi)
    )

    fi = RegularGridInterpolator((y, z), field, bounds_error=False, fill_value=0.0)
    yp = rr[:, None] * np.cos(pp)[None, :]
    zp = rr[:, None] * np.sin(pp)[None, :]
    fp = fi(np.stack([yp.ravel(), zp.ravel()], axis=-1)).reshape(nr, nphi)

    ls = np.arange(-lmax, lmax + 1)
    amps = np.array(
        [np.sum(fp * np.exp(-1j * order * pp)[None, :] * weights) for order in ls]
    )
    return amps, ls


def spin_azimuthal_spectrum(
    ey: NDArray, ez: NDArray, y: NDArray, z: NDArray, **kwargs
) -> tuple[NDArray, NDArray]:
    """Spin-resolved azimuthal spectrum in the circular-polarisation basis.

    ``E_σ = (E_y ∓ i·E_z)/√2`` with ``σ = ±1``: row 0 of the result carries
    ``σ = +1``, row 1 ``σ = −1``. This is the basis in which a pure vortex has
    a single nonzero entry, and the basis arXiv:2402.16492 writes its Kerr
    coupling blocks in (``J = l ± σ`` conserved for circular symmetry).

    Returns
    -------
    ``(amps, ls)`` with ``amps.shape == (2, 2·lmax+1)``, normalised so
    ``sum(|amps|²) == 1``. Take ``purity = |amps|².max()``, or use
    :func:`azimuthal_purity`.

    Reference values (both are asserted in ``tests/test_mode_azimuthal.py``):
    a circularly polarised analytic ``LG(0, ℓ)`` gives purity ``1.000`` at
    ``(σ, ℓ)``; the same field **linearly** polarised (``E_z = 0``) gives
    exactly ``0.500`` for any ``ℓ``, because a real field populates both spins
    equally. A measured purity below 0.5 is therefore not"mostly one vortex",
    it is noise.
    """
    ep, ls = azimuthal_spectrum((ey + 1j * ez) / np.sqrt(2.0), y, z, **kwargs)
    em, _ = azimuthal_spectrum((ey - 1j * ez) / np.sqrt(2.0), y, z, **kwargs)
    amps = np.stack([ep, em])
    tot = float(np.sum(np.abs(amps) ** 2))
    if not np.isfinite(tot) or tot == 0.0:
        raise ValueError("field is zero or non-finite on the polar grid")
    return amps / np.sqrt(tot), ls


def azimuthal_purity(amps: NDArray, ls: NDArray) -> tuple[float, int, int]:
    """``(P, sigma, l)`` of the dominant channel in a ``(2, nl)`` amplitude array.

    ``P = |A_{σ,l}|² / Σ|A|²``. For a launched ``OAM±ℓ`` this is the number a
    purity spectrum plots against ``λ``.
    """
    p = np.abs(np.asarray(amps)) ** 2
    idx = np.unravel_index(int(np.argmax(p)), p.shape)
    return float(p[idx]), int(idx[0]), int(ls[idx[1]])


def azimuthal_charge(
    field_xy: NDArray,
    x: NDArray,
    y: NDArray,
    *,
    center: tuple[float, float] | None = None,
) -> int:
    """Integer azimuthal charge ``l`` of a single complex field component.

    Wraps the phase around a closed square-ish loop of radius
    ``R = 0.25·min(ptp(x), ptp(y))`` and divides the total unwrapped phase by
    ``2π``. Cheaper and more robust than :func:`azimuthal_spectrum` when the
    field is clean; it gives no purity, only the winding number.

    This is the value the engine's ``oam_l`` gate consumes (Poletti and Horak
    Eq. 18/19), so it is the natural cross-check on an exported label:
    ``azimuthal_charge(m.ey[:, :, k], m.y, m.z) == m.label``.

    Sign convention: ``φ = arctan2(y, x)`` with ``field_xy[i, j]`` at ``(x[i], y[j])``.
    This is the same rule as :func:`azimuthal_spectrum`, so the two measurements agree
    on one field (they must; a sign flip between them would make the ``oam_l``
    gate disagree with the exported label). Any axis swap or reflection flips
    the sign, so an export path that renames transverse axes has to be audited
    before its ``l`` is trusted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    field_xy = np.asarray(field_xy)
    if field_xy.shape != (x.size, y.size):
        raise ValueError(
            f"field shape {field_xy.shape} does not match (x, y) = ({x.size}, {y.size})"
        )
    R = 0.25 * min(np.ptp(x), np.ptp(y))
    cx = 0.0 if center is None else center[0]
    cy = 0.0 if center is None else center[1]
    m = max(16, 4 * int(max(abs(R) / (x[1] - x[0]), 1)))
    th = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    xs = cx + R * np.cos(th)  # first transverse axis
    ys = cy + R * np.sin(th)  # second transverse axis

    def _sample(xx, yy):
        ix = np.clip(np.searchsorted(x, xx) - 1, 0, x.size - 2)
        iy = np.clip(np.searchsorted(y, yy) - 1, 0, y.size - 2)
        tx = (xx - x[ix]) / (x[ix + 1] - x[ix])
        ty = (yy - y[iy]) / (y[iy + 1] - y[iy])
        f00 = field_xy[ix, iy]
        f10 = field_xy[ix + 1, iy]
        f01 = field_xy[ix, iy + 1]
        f11 = field_xy[ix + 1, iy + 1]
        return (
            f00 * (1 - tx) * (1 - ty)
            + f10 * tx * (1 - ty)
            + f01 * (1 - tx) * ty
            + f11 * tx * ty
        )

    vals = _sample(xs, ys)
    phase = np.unwrap(np.angle(vals))
    return int(np.round((phase[-1] - phase[0]) / (2.0 * np.pi)))


# ---------------------------------------------------------------------------
# Launch-field projection (GAP A7)
# ---------------------------------------------------------------------------


@dataclass
class ModalProjection:
    """Modal amplitudes of a launched transverse field.

    ``amplitudes[i]`` is the complex overlap with mode ``labels[i]`` (in the
    square root of the launched field's power units); ``coupling_efficiency``
    is ``Σ|a_m|² / ∬|E|²``, the fraction of launched power captured by the
    mode set. ``residual_power`` is the part of the launch that the set does
    not describe.
    """

    amplitudes: NDArray
    labels: list[int]
    coupling_efficiency: float
    launched_power: float
    residual_power: float


def project_field_onto_modes(
    field_xy: NDArray,
    x_um: NDArray,
    y_um: NDArray,
    modes: list[LoadedMode],
    lambda0: float,
    *,
    component: str = "y",
) -> ModalProjection:
    """Project a transverse launch field onto a solved mode set.

    ``a_m = ∬ conj(F_m) E dA`` with each ``F_m`` normalised so
    ``∬|F_m|² dA = 1`` on the same grid, which is the field-level launch model
    ``E = Σ_m a_m F_m`` (PLAN §2.4). ``component`` selects the transverse
    component to project (``"x"`` or ``"y"``); modes are evaluated at the
    export wavelength nearest ``lambda0``.

    The launch grid must be the export grid: shape mismatch raises instead of
    silently broadcasting, because a silently resampled launch is a silently
    wrong coupling efficiency.

    Parameters
    ----------
    field_xy
        Complex launch field, shape ``(len(y_um), len(x_um))``.
    x_um, y_um
        Transverse axes in µm, matching the modes' export grid.
    modes
        Solved modes from :func:`load_modes`.
    lambda0
        Launch wavelength in µm (nearest export sample is used).
    component
        ``"y"`` (default, the usual transverse component) or ``"x"``.

    Returns
    -------
    :class:`ModalProjection`
    """
    if component not in ("x", "y"):
        raise ValueError(f"component must be 'x' or 'y', got {component!r}")
    if not modes:
        raise ValueError("project_field_onto_modes needs at least one mode")
    x_um = np.asarray(x_um, dtype=float)
    y_um = np.asarray(y_um, dtype=float)
    field_xy = np.asarray(field_xy, dtype=complex)
    if field_xy.shape != (y_um.size, x_um.size):
        raise ValueError(
            f"field shape {field_xy.shape} does not match (y, x) = "
            f"({y_um.size}, {x_um.size})"
        )
    grid_x = np.asarray(modes[0].x_um, dtype=float)
    grid_y = np.asarray(modes[0].y_um, dtype=float)
    if (
        grid_x.shape != x_um.shape
        or grid_y.shape != y_um.shape
        or not (np.allclose(grid_x, x_um) and np.allclose(grid_y, y_um))
    ):
        raise ValueError(
            "launch grid does not match the mode export grid; interpolate the"
            "launch field before projecting"
        )
    dx = abs(float(np.mean(np.diff(x_um)))) * 1e-6
    dy = abs(float(np.mean(np.diff(y_um)))) * 1e-6
    dA = dx * dy

    launched = float(np.sum(np.abs(field_xy) ** 2) * dA)
    if not np.isfinite(launched) or launched <= 0.0:
        raise ValueError("launch field has zero or non-finite power")

    amps = np.empty(len(modes), dtype=complex)
    labels: list[int] = []
    for i, mode in enumerate(modes):
        wl = np.asarray(mode.wavelengths_um, dtype=float)
        iw = int(np.argmin(np.abs(wl - float(lambda0))))
        raw = mode.Ex[iw] if component == "x" else mode.Ey[iw]
        raw = np.asarray(raw, dtype=complex)
        if raw.shape != field_xy.shape:
            raise ValueError(
                f"mode {mode.label} field shape {raw.shape} does not match"
                f"the launch grid {field_xy.shape}"
            )
        norm = float(np.sum(np.abs(raw) ** 2) * dA)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"mode {mode.label} has zero {component}-component power")
        amps[i] = np.sum(np.conj(raw) * field_xy) * dA / np.sqrt(norm)
        labels.append(int(mode.label))
    captured = float(np.sum(np.abs(amps) ** 2))
    return ModalProjection(
        amplitudes=amps,
        labels=labels,
        coupling_efficiency=captured / launched,
        launched_power=launched,
        residual_power=launched - captured,
    )
