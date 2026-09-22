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
| `raissi_2019_pinn_nlse/` | Raissi et al., JCP 378, 686 (2019), 10.1016/j.jcp.2018.10.045 | new PINN layer (`[pinns]` torch) with GNLSE-generated data | ✅ | **P0** — first reproduction of the PINN direction |
| `mumtaz_2013_multimode_jlt/` | Mumtaz et al., JLT 31, 398 (2013), 10.1109/JLT.2012.2235414 | `multimode_gnlse` v2 (spec-source paper) | ✅ | **P0** — conformance test for FWM/XPM tensors |
| `wright_2015_self_organized_instability/` | Wright et al., Nat. Photon. (2016), 10.1038/nphoton.2015.60 (arXiv:1603.07414) | `multimode_gnlse` + FWM Jacobian | ✅ | P1 |
| ~~`krupa_2019_multimode/`~~ → **moved to [`../krupa_2019_multimode/`](../krupa_2019_multimode/)** | Krupa et al., APL Photonics 4, 110901 (2019), 10.1063/1.5119434 | `multimode_gnlse` + `structured.py` | ✅ done | P1 — GPI sidebands (Fig. 14 left / PRL 116, 183901); heavy ~35 min, NOT in test suite |
| `eftekhar_2019_parametric_cascades/` | Nat. Commun. 10, 1638 (2019), 10.1038/s41467-019-09687-9 | `multimode_gnlse` + chunked taper + Raman + DW phase matching | ✅ | P2 |
| `wai_menyuk_1991_random_birefringence_solitons/` | Opt. Lett. 16, 1231 (1991), 10.1364/OL.16.001231 | `RandomBirefringenceEngine` | ✅ | P2 |
| `conforti... → guasoni_2015_generalized_mi_multimode/` | Guasoni, Phys. Rev. A 92, 033849 (2015), 10.1103/PhysRevA.92.033849 | multimode linear-stability eigenvalues | ✅ | P1 — analytic gain curves |
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
