# SHG coupling in x-cut LNOI — Wang et al. (2017)

**Reference (as cited throughout this repo):** Wang et al., *Opt. Express*
**25**(6), 6963 (2017) — the χ⁽²⁾ mode-overlap coupling paper behind the
`chi2` overlap API. See `openspec/specs/chi2-mode-overlap/spec.md`,
`chi2.py:753` and `tests/test_materials.py:173`, which all cite it the same way.

> ⚠️ **Citation not verified.** The repo records no DOI for this paper, and
> *Opt. Express* **25**, 6963–6973 (2017) is also the page range of Allison et
> al. on PPLN waveguides for supercontinuum generation. The author should
> confirm which paper the `chi2` overlap API was implemented against before
> this deck is cited externally; until then the parameter anchors below are
> "as-implemented", not "as-published".

**What this deck reproduces.** The *overlap-based* coupling constants the
article builds its PGLN design on, computed with the `photonics_helper.chi2`
mode-overlap APIs on the article's waveguide geometry:

- `g` — uniform-waveguide coupling, `shg_coupling_overlap`
- `g'` — PGLN effective coupling including the grating first order `d^(1)`,
  `Δε₁` and the residual mismatch `Δk`, `pgln_overlap`
- `σ_plane` — the plane-wave/Boyd-normalised scalar reference, `shg_coupling`

Run it with `python reproductions/shg_lnoi_shg/reproduce.py`; results land in
`results.json`. `validate()` checks math sanity only (finite, positive, non-zero
grating phase) — the quantitative FEM anchors live in the standalone `shg_solve`
study, not here.

## Status: ⚠️ partial — the overlap machinery is exercised, not the paper's FEM fields

| Quantity | This deck | Article | Ratio |
|---|---|---|---|
| `σ_plane` | 1331.3 m⁻¹·W^−1/2 | — (normalisation reference) | — |
| `g` uniform | 203.4 | 77.4 | 2.63 |
| `g'` PGLN effective | 9.93 | 34.5 | 0.288 |

**Why the ratios are not ≈1.** The article's `g` and `g'` come from *FEM-solved*
mode fields of the real rib waveguide. This deck substitutes **scalar toy modes**
(a Gaussian pump and a `cos(3πx/Λ)` third-order mode) on the article's etch
geometry, which fixes the normalization conventions and the order of magnitude
but not the field overlap integral. The ratios above are therefore a record of
the convention gap, not a failed physics claim — the plane-wave reference
`σ_plane` matches the Boyd-normalised article anchor (1330.8) to 0.03 %,
confirming the normalization is the article's.

**Index data.** `n_e(λ)` for both 1550 nm and 775 nm now comes from the
birefringent Sellmeier curve in `materials.db`
(`RefractiveIndex.from_material_database("LiNbO3", axis="extraordinary")`),
seeded by the openspec change `fix-shg-replication-gaps` (Group 4). The
Edwards & Lawrence anchor interpolation is retained only as a fallback for a
database without the LiNbO₃ row. Switching sources moved every result by
<0.1 % (e.g. `g` 203.412 → 203.386, −0.013 %).

## Recorded caveats

1. **Toy fields, not FEM fields** — the dominant limitation (see above). A real
   reproduction of the article's `g`/`g'` needs the actual mode profiles.
2. **`Δk` is article-anchored**, not computed from the groove period and poling
   period of this specific geometry.
3. **The citation itself is unverified** (see the warning at the top).
4. Not registered in `tests/test_reproductions.py` (fast, but the quantitative
   anchors live outside the repo).