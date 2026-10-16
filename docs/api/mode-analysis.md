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
- `overlap_tensors_from_fields(modes, n2, lambda0) -> ModePairParameters` —
  grid quadrature (no circular symmetry): `A_eff,m`, `γ_m`,
  `S⁽¹⁾ → xpm_weights`, `S⁽²⁾ → fwm_weights` (real when the fields are,
  complex when not — the engine accepts complex `fwm_weights`),
  `oam_l` (the export's azimuthal labels), absolute-unit
  `coupling_tensor`. Confinement `Γ_m` is unity: the export contract
  carries no core mask, so confinement-scaled `n₂` must be pre-scaled by
  the deck author.
- `build_multimode_deck(export_dir, lambda0, order, n2, pump=None)` — one call
  to engine kwargs (`phase_offsets` are emitted but should be dropped —
  pass `None` — for step-index / high-contrast rib decks).

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
