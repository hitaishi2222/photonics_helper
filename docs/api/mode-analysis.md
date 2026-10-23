# mode-analysis API

Step7 bridge: mode-solver export → multimode-engine deck.

## Export format

`examples/generate_multimode_mode_data.py` writes one `mode-*.npz` per mode
into `examples/data/multimode_rib/` (`wavelength_um`, `neff`, `ng`, `Ex`/`Ey`
`(n_wl, Ny, Nx)`, `x_um`/`y_um`, `label`) plus `meta.json`. Replace the
analytic Hermite–Gaussian fields with femwell/tidy3d dumps — the file format
is the contract. The single-mode `generate_waveguide_mode_data.py` path is
untouched.

## Functions

- `load_modes(export_dir) -> list[LoadedMode]` — N-mode loader,
  permutation-invariant ordering, fundamental first (highest `n_eff` at the
  mid-grid wavelength). Cross-λ continuation is the solver-export contract:
  one `mode-*.npz` file carries one physical mode's λ-walk.
- `mode_dispersion(modes, order, lambda0) -> DispersionSet` — per-mode Taylor
  `betas`, `group_delays[m] = β₁⁽ᵐ⁾ − β₁⁽⁰⁾` (`group_delays[0] == 0`),
  `phase_offsets` from `β₀` differences.
- `overlap_tensors_from_fields(modes, n2, lambda0, *, xpm_model="co_polarized")`
  — grid quadrature (no circular symmetry): `A_eff,m`, `γ_m`,
  `S⁽¹⁾ → xpm_weights`, `S⁽²⁾ → fwm_weights` (real when the fields are,
  complex when not — the engine accepts complex `fwm_weights`),
  `oam_l` (the export's azimuthal labels), absolute-unit
  `coupling_tensor`.
  - **Absolute scale:** `C_ij = γ₀·xpm_ij = (n₂ω₀/c)·Q⁽¹⁾_iijj`. Its diagonal
    is exactly the per-mode `γ_i` (`C_ii = γ_i`); it is not a
    `½(γ_i+γ_j)` average. Per-mode SPM on a larger mode is therefore
    `γ₀·A_eff,0/A_eff,i`, matching `γ_i`.
  - **`xpm_model`:** `"co_polarized"` (default) doubles the **off-diagonal**
    `S⁽¹⁾` ratio (two equal XPM permutations in Mumtaz Eq. 6; SPM 1, XPM 2
    for co-polarized spatial modes). `"vector"` keeps the bare `S⁽¹⁾`
    ratio for a polarization-resolved basis; a genuine vector basis needs
    separate polarization channels because the orthogonal-polarization XPM
    factor depends on the polarization basis (2/3 linear, 2 circular).
  - Confinement `Γ_m` is unity: the export contract carries no core mask,
    so confinement-scaled `n₂` must be pre-scaled by the deck author.
- `build_multimode_deck(export_dir, lambda0, order, n2, *, pump=None,
  xpm_model="co_polarized")` — one call to engine kwargs
  (`phase_offsets` are emitted but should be dropped — pass `None` — for
  step-index / high-contrast rib decks). It also returns
  `gamma_reference = γ₀`, `gamma` (per mode) and `A_eff` (per mode). Pass
  `gamma_reference` to the engine: it is the absolute scale of the
  `S_0000`-normalised `fwm_weights`/`raman_tensors`, so the run does not
  silently inherit the `Fiber.A_eff` scale.

## OAM measurement (GAP A3)

- `azimuthal_spectrum(field, y, z, *, rmax=None, lmax=6)` computes the
  azimuthal DFT `A_ℓ = ∬ f e^{−iℓφ} r dr dφ` of one component.
- `spin_azimuthal_spectrum(ey, ez, y, z, ...)` does the same in the circular
  basis, normalised to unit power. Control values: a circularly polarised
  `LG(0, ℓ)` gives purity 1.000; the same field linearly polarised gives
  exactly 0.500.
- `azimuthal_purity(amps, ls)` returns `(P, σ, ℓ)` for the dominant channel.
- `azimuthal_charge(field, x, y, *, center=None)` returns the winding number
  using the same azimuth convention. Swapping the transverse axes flips the sign, so an
  export that renames axes (`multimode_scg` GAP B1) is auditable.

## Launch projection (GAP A7)

```python
from photonics_helper.mode_analysis import load_modes, project_field_onto_modes

modes = load_modes("exports/rib")           # solved modes
res = project_field_onto_modes(E_launch, x_um, y_um, modes, lambda0=2.0)
res.amplitudes, res.coupling_efficiency, res.residual_power
```

`a_m = ∬ conj(F_m) E dA` with `∬|F_m|² dA = 1` on the export grid, so
`E = Σ a_m F_m` and `|a_m|²` is captured power. The launch grid must be the
export grid. A shape or grid mismatch raises rather than silently resampling.

## Notes

- **Per-mode γ routing (task 3.3):** absolute-unit `γ_mn` travels in the
  existing `coupling_tensor` / `fwm_coupling_tensor` slots — no engine change.
- **`phase_offsets` meaningful for GRIN-like profiles only** (task 2.2):
  step-index / high-contrast ribs accumulate differential `β₀` phase the
  engine's common-carrier picture cannot use; pass them only for GRIN decks.
- **Honesty note (tasks 5.2/8–10):** all four GAP steps have landed (step7 bridge, step8 Raman, step9 shock/TPA, step10 full-β/FWM) — MM-SCG runs no longer carry a switched-off-physics disclaimer. Retired 2026-10-16.
- **Grounding:** Wright JSTQE p3 Eq. 1/2 (`ph_pdfs/pages_wright/page-03.png`,
  image-read); Afshar OE 17 γ/A_eff (Part I Kerr); Poletti–Horak Eq. 7/18/19
  (in-repo PDF; fibre solver stays in `reproductions/`).
