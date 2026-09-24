# Planned reproductions (parallel work queue)

One folder per paper; each `README.md` is a standalone implementation plan
(reference, DOIs, parameters, engine calls, asserts, checklist) so multiple
agent instances can pick up folders and work independently. Follow the house
contract of `../README.md`: `parameters.json` + `reproduce.py` + README with
the measured metric, then a row in the main `reproductions/README.md` table.

**Source PDFs and rendered `pages/p-*.png` are LOCAL-ONLY (gitignored) —
do not commit them.**

| Folder | Reference (DOI) | Module stack | PDF | Priority |
|---|---|---|---|---|
| ~~`renninger_wise_2013_grin_solitons/`~~ → **moved to [`../renninger_wise_2013_grin_solitons/`](../renninger_wise_2013_grin_solitons/)** | Renninger & Wise, Nat. Commun. 4, 1719 (2013), 10.1038/ncomms2739 | `multimode_gnlse` + paraxial GRIN modes | ✅ done | was **P0** — reproduced (locking, Eq. 6 fixed point, self-imaging via `phase_offsets`); higher-mode blue-shift sign open → **ISSUES.md #0** |
| ~~`raissi_2019_pinn_nlse/`~~ → **moved to [`../raissi_2019_pinn_nlse/`](../raissi_2019_pinn_nlse/)** | Raissi et al., JCP 378, 686 (2019), 10.1016/j.jcp.2018.10.045 | new PINN layer (`[pinns]` torch) with GNLSE-generated data | ✅ done | was **P0** — reproduced 2026-09-22 (§I Schrödinger example, PINN rel-L2 6.1e-3 accepted ≤ 1e-2 worst-case, loss 1.24e-6; details in the folder README; deviation analysis in `ISSUES.md` #6) |
| ~~`mumtaz_2013_multimode_jlt/`~~ → **moved to [`../mumtaz_2013_multimode_jlt/`](../mumtaz_2013_multimode_jlt/)** | Mumtaz et al., JLT 31, 398 (2013), 10.1109/JLT.2012.2235414 | `multimode_gnlse` v2 (spec-source paper) | ✅ done | was **P0** — reproduced 2026-09-21 (walk-offs exact, SPM 8/9 to ~1e-3, generalized Manakov 4/3 confirmed at M=2; details in the folder README) |
| `wright_2015_self_organized_instability/` | Wright et al., Nat. Photon. (2016), 10.1038/nphoton.2015.60 (arXiv:1603.07414) | `multimode_gnlse` + FWM Jacobian | ✅ | P1 |
| ~~`krupa_2019_multimode/`~~ → **moved to [`../krupa_2019_multimode/`](../krupa_2019_multimode/)** | Krupa et al., APL Photonics 4, 110901 (2019), 10.1063/1.5119434 | `multimode_gnlse` + `structured.py` | ✅ done | P1 — GPI sidebands (Fig. 14 left / PRL 116, 183901); heavy ~35 min, NOT in test suite |
| `eftekhar_2019_parametric_cascades/` | Nat. Commun. 10, 1638 (2019), 10.1038/s41467-019-09687-9 | `multimode_gnlse` + chunked taper + Raman + DW phase matching | ✅ | P2 |
| `eftekhar_2019_parametric_cascades/` | Nat. Commun. 10, 1638 (2019), 10.1038/s41467-019-09687-9 | `multimode_gnlse` + chunked taper + Raman + DW phase matching | ✅ | P2 |
| ~~`wai_menyuk_1991_random_birefringence_solitons/`~~ → **moved to [`../wai_menyuk_1991_random_birefringence_solitons/`](../wai_menyuk_1991_random_birefringence_solitons/)** | Wai, Menyuk & Chen, Opt. Lett. 16, 1231 (1991), 10.1364/OL.16.001231 | `RandomBirefringenceEngine` | ✅ done | was **P2** — reproduced (Fig. 1 shadow vs Eq. (5) peak ratio 1.10; Fig. 2 delay = δ·∫cos2θ dξ corr 0.978; Fig. 3 bounded widths, no splitting; Fig. 4 polarization budget; energy ≤6e-12; details in the folder README; δ=7.5 width caveat noted) |
| ~~`conforti... → guasoni_2015_generalized_mi_multimode/`~~ → **moved to [`../guasoni_2015_generalized_mi_multimode/`](../guasoni_2015_generalized_mi_multimode/)** | Guasoni, Phys. Rev. A 92, 033849 (2015), 10.1103/PhysRevA.92.033849 | multimode linear-stability eigenvalues | ✅ done | was **P1** — reproduced 2026-09-24 (Fig. 3 anchors to 0.7 %: g₁ = 0.9071 / g₂ = 0.7070 vs 0.90 / 0.71; Fig. 3 inset eigenvectors to ~0.1 in ln (−0.349/−3.30/−3.37 vs −0.35/−3.22/−3.35); single-mode MI closed form < 1e-9. Split-step layer recorded-outstanding — satisfies `ISSUES.md` #7; ~90 s) |
| `oe_2001_nn_nlse/` | Opt. Express 9, 72 (2001), 10.1364/OE.9.000072 | PINN baseline | ✅ | P3 |

## PDFs

All 11 PDFs present locally (gitignored). OE 2001 is the only one without a URL-fetchable copy — supplied by author.

## Mode-analysis / FDTD cross-checks (as instructed)

- GRIN/step-index **mode analysis** → **femwell** (radial LG modes; compare
  β_p(z) vs the paraxial Eq. used in the reproductions).
- **tidy3d** optional for the same purpose (API-key needed online; prefer femwell
  for offline rigs).
- **meep** not applicable to any of the planned papers so far (no full-FDTD
  figures); revisit for a spatiotemporal panel if we add a waveguide
  scattering-relation reproduction.

## Workflow per folder

1. `pdftoppm -png -r 130 <pdf> pages/p` (already done in folders with ✅).
2. Read the PNGs (vision) → extract every simulation parameter with page refs.
3. `parameters.json` (strict house format), `reproduce.py` with asserts vs
   analytic/published target, plain figures, and a README describing the
   ground truth + outcome.
4. Run `python -m pytest tests/test_reproductions.py` (add the new script to
   the test list only when it runs < ~2 min, or mark heavy).
5. Move the folder out of `planned/` once green, i.e. to `reproductions/<name>/`
   and update the main README row.
