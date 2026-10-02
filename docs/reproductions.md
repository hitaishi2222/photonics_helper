# Reproductions

Self-contained, validated reproductions of results from the literature. Each
folder under `reproductions/` contains a `parameters.json`, a `reproduce.py`
that asserts its result against an analytic/closed-form reference, and a
`README.md` with the ground truth and outcome. Run them all with:

```bash
python -m pytest tests/test_reproductions.py
```

| Reproduction | Reference | Stack | Headline outcome |
|---|---|---|---|
| [`eftekhar_2019_parametric_cascades`](../reproductions/eftekhar_2019_parametric_cascades/) | Eftekhar et al., *Nat. Commun.* **10**, 1638 (2019) | `multimode_gnlse` (3 GRIN modes) + chunked taper | ⚠️ MFD-oscillation periods match the window-avg L_si to 0.1 % / 2.4 %; acceleration ratio x3.33 vs window model x3.25…. |
| [`stolen_lin_1978_spm`](../reproductions/stolen_lin_1978_spm/) | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) | GNLSE (Kerr) + pulse | ✅ spectrum matches closed form to ~1e-13; peak-count rule exact. |
| [`macleod_quarter_wave_dbr`](../reproductions/macleod_quarter_wave_dbr/) | Macleod, *Thin-Film Optical Filters*; Born & Wolf (textbook) | `dbr` TMM | ✅ peak reflectance matches exact closed form to <1e-6; stopband width within 7%. |
| [`shg_textbook`](../reproductions/shg_textbook/) | Boyd, *Nonlinear Optics* (3rd ed.), Ch. 2; Fejer et al., *IEEE JQE* **28**, 2631 (1992) | `chi2` (RK4IP) | ✅ η matches `tanh²(κL)` to ≤8e-15; first-order QPM recovers `tanh²((2/π)κL)` to 0.12%. |
| [`gordon_1986_ssfs`](../reproductions/gordon_1986_ssfs/) | Gordon, *Opt. Lett.* **11**, 662 (1986) | GNLSE + Raman + soliton | ✅ measured +1.08 nm / 20 m vs Gordon +0.91 nm (ratio 1.19). |
| [`dudley_2006_cherenkov_dw`](../reproductions/dudley_2006_cherenkov_dw/) | Akhmediev & Karlsson, *Phys. Rev. A* **51**, 2602 (1995) | phase_matching + GNLSE | ✅ `dispersive_wave_roots` = analytic 699.3 nm. |
| [`dudley_2006_scg`](../reproductions/dudley_2006_scg/) | Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006) | GNLSE + Raman + phase_matching + soliton | ✅ **9 figure scripts** (Figs. 3, 4, 5, 6, 7, 8, 9, 10, 23). |
| [`kuznetsov_ma_2012_breather`](../reproductions/kuznetsov_ma_2012_breather/) | Kibler et al., *Sci. Rep.* **2**, 463 (2012) | GNLSE (NLSE) + phase_matching | ✅ exact Kuznetsov–Ma soliton reproduced over one period: peak 7.6129 W vs 7.6130 W, intensity L2 9e-4, `T₀`/`z_p` exact. |
| [`narhi_2016_mi_breathers`](../reproductions/narhi_2016_mi_breathers/) | Närhi et al., *Nat. Commun.* **7**, 13675 (2016) | GNLSE (NLSE) + phase_matching + SplitStepEngine | ✅ MI sideband 46.41 GHz = paper 46.4; exact Peregrine ratio 8.9996 (theory 9). |
| [`tomlinson_1985_wave_breaking`](../reproductions/tomlinson_1985_wave_breaking/) | Tomlinson, Stolen & Johnson, *Opt. Lett.* **10**, 457 (1985) | GNLSE (NLSE) | ✅ steep edges → flat top → oscillations; steepening onset at 0.57 `z_WB`; scaling `z ∝ P₀^(−0.54)` (theory −0.5) with…. |
| [`marcuse_menyuk_wai_1997_manakov_pmd`](../reproductions/marcuse_menyuk_wai_1997_manakov_pmd/) | Marcuse, Menyuk & Wai, *JLT* **15**, 1735 (1997) | `vector_gnlse` (`manakov` coupling + `RandomBirefringenceEngine`) | ✅ Eq. (30) 8/9 law exact (ratio 9/8, shapes invariant to 7×10⁻⁴ over 30 z₀). |
| [`renninger_wise_2013_grin_solitons`](../reproductions/renninger_wise_2013_grin_solitons/) | Renninger & Wise, *Nat. Commun.* **4**, 1719 (2013) | `multimode_gnlse` (3 GRIN modes, isotropic tensors) | ✅ temporal locking 0.06 of linear walk-off. |
| [`menyuk_1987_birefringent_pulses`](../reproductions/menyuk_1987_birefringent_pulses/) | Menyuk, *IEEE JQE* **QE-23**, 174 (1987) | `vector_gnlse` (incoherent 2/3 XPM + coherent FWM branch) | ⚠️ Eq. (9)/(10) soliton filaments at machine precision (shape L2 7×10⁻⁶); coherent-FWM filament deviation O(1/Rδ) as…. |
| [`krupa_2019_multimode`](../reproductions/krupa_2019_multimode/) | Krupa et al., *APL Photonics* **4**, 110901 (2019) (GPI: Krupa, *PRL* **116**, 183901 (2016)) | `multimode_gnlse` (modal GNLSE + `phase_offsets` grating) | ✅ ξ = 0.6157 mm / f_m = 124.98 THz (PRL 0.615 / 125.0); √h·f_m ladder to 0.5 % for h ≤ 3 (h = 1..5 measured); f₁ ×4-power…. |
| [`mumtaz_2013_multimode_jlt`](../reproductions/mumtaz_2013_multimode_jlt/) | Mumtaz, Essiambre & Agrawal, *JLT* **31**, 398 (2013) (spec source of v2 multimode Eq. 12/29) | `multimode_gnlse` (Manakov `xpm_weights`) + Eq. 6 stochastic harness | ✅ Table-II walk-offs exact (≤3e-6). |
| [`wai_menyuk_1991_random_birefringence_solitons`](../reproductions/wai_menyuk_1991_random_birefringence_solitons/) | Wai, Menyuk & Chen, *Opt. Lett.* **16**, 1231 (1991) | `vector_gnlse` (`RandomBirefringenceEngine`, Wai-1991 Eq.-(2) rotation law) | ⚠️ Fig. 1 shadow peak ratio 1.10 vs Eq. (5). |
| [`wright_2015_self_organized_instability`](../reproductions/wright_2015_self_organized_instability/) | Wright et al., *Nat. Photon.* **10**, 471 (2016 online 2015) | `multimode_gnlse` (FWM Jacobian, `oam_l` gating) + `phase_offsets` | ✅ check A: analytic STMI ladder vs digitized Fig. 3d circles to −5.2/−0.9/−2.8 % (tol 15 %); check-B' probe: engine…. |
| [`guasoni_2015_generalized_mi_multimode`](../reproductions/guasoni_2015_generalized_mi_multimode/) | Guasoni, *Phys. Rev. A* **92**, 033849 (2015) | `multimode_gnlse` (XPM-coupled deck) + Eq.-(8)/(9) eigen solver | ⚠️ Fig. 3 anchor to **0.7 %** (g₁ = 0.9071 / g₂ = 0.7070 vs paper 0.90 / 0.71). |
| [`raissi_2019_pinn_nlse`](../reproductions/raissi_2019_pinn_nlse/) | Raissi, Perdikaris & Karniadakis, *J. Comput. Phys.* **378**, 686 (2019) | reproduction-local PINN (torch float64, Adam + L-BFGS) + `SplitStepEngine` data | ✅ §I Schrödinger example: PINN rel-L2 **6.1e-3** vs paper 1.97e-3 (accepted ≤ 1e-2 per the folder plan's worst-case…. |
| [`dw_timing_gas_hollowcore`](../reproductions/dw_timing_gas_hollowcore/) | Brahms & Travers, arXiv:2101.04014 (2021) | `TaperedGNLSESolver` gas β(ω,z) (Marcatili–Schmeltzer + Boerzsoenyi He) + RDW arrival-time statistics | ⚠️ REPRODUCED 2026-09-30 (plasma-free subset): transmission 86.8 % vs paper ~87 %. |
| [`kibler_2010_peregrine`](../reproductions/kibler_2010_peregrine/) | Kibler et al., *Nat. Phys.* **6**, 790 (2010) | GNLSE (β₂-only) + `breathers` (`general_sfb` AB, `peregrine_soliton`) | ✅ REPRODUCED 2026-09-30: engine tracks the analytic AB pointwise over the growth leg (peak ≤ 2 %); a = 0.42…. |
| [`hult_2007_rk4ip`](../reproductions/hult_2007_rk4ip/) | Hult, *JLT* **25**, 3770 (2007) | `RK4IPIntegrator` + `GNLSEOperator` (from `heidt_adaptive.py`) | ⚠️ REPRODUCED 2026-10-01, both decks. Deck B keeps the house two-exponential silica Raman model rather than the paper's Hollenbeck–Cantrell modal sum (recorded deviation, shown non-limiting). |
| [`shg_lnoi_shg`](../reproductions/shg_lnoi_shg/) | Wang et al., *Opt. Express* **25**(6), 6963 (2017) — *citation unverified* | `chi2` mode-overlap (`shg_coupling_overlap`, `pgln_overlap`) + `materials.db` LiNbO₃ extraordinary Sellmeier | ⚠️ Partial: overlap machinery on the article geometry with scalar toy modes — `g` = 203.4 vs 77.4 and `g'` = 9.93 vs 34.5 are convention gaps; Boyd-normalised `σ` = 1331.3 vs 1330.8 (0.03 %). |
| [`heidt_2009_adaptive_step`](../reproductions/heidt_2009_adaptive_step/) | Heidt, *JLT* **27**, 3984 (2009) | reproduction-local adaptive-step layer (`heidt_adaptive.py`: `SSFIntegrator`/`RK4IPIntegrator`, `LocalErrorStepper`/`CQEStepper`, counted-FFT cost) + `GNLSEOperator` | ⚠️ REPRODUCED 2026-10-01, all 16 checks green in 290 s. |
| [`huang_202x_pcgnlse_attractors`](../reproductions/huang_202x_pcgnlse_attractors/) | Huang et al., arXiv:2607.05244 (2026) | `SplitStepEngine(conserving_shock=True)` + signed γ + folder-local dimensionless pcGNLSE/moment-ODE layer | ⚠️ REPRODUCED partly vs recorded caveats (2026-09-30): ENGINE SIGN BENCH green — pcGNLSE γ>0 redshift path identical to…. |

✅ = the paper's claim is reproduced within its stated tolerance. ⚠️ = reproduced, with a deviation or an out-of-scope arm recorded in the folder `README.md` and in the repository `ISSUES.md` register.


Reproducing the papers also surfaced real bugs (inverted GNLSE dispersion sign,
Raman response direction); see the repository `REPORT.md` for the full list.
