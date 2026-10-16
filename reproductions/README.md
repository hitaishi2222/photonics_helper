# Paper reproductions

Self-contained, validated reproductions of results from the literature. Each
folder contains:

- `parameters.json` — every physical input (documented, reproducible).
- `reproduce.py` — the script (`python .../reproduce.py`), which asserts its
  result against an analytic/closed-form reference and writes a figure PNG.
- `README.md` — what was reproduced, the ground truth, and the outcome.

Run all reproduction tests with:

```bash
python -m pytest tests/test_reproductions.py
```

> **`tests/test_reproductions.py` is LOCAL-ONLY (gitignored, see `.gitignore`)**
> — it is not published with the repo. Each reproduced paper is registered
> there as a pytest wrapper around its folder's `validate()`; keep new
> reproductions in the suite locally (mark heavy ones `pytest.mark.slow`),
> just never commit the file.

## Available

| Reproduction | Reference (DOI) | Module stack | Status |
|---|---|---|---|
| [`eftekhar_2019_parametric_cascades`](eftekhar_2019_parametric_cascades/) | Eftekhar et al., *Nat. Commun.* **10**, 1638 (2019) · [10.1038/s41467-019-09687-9](https://doi.org/10.1038/s41467-019-09687-9) | `multimode_gnlse` (3 GRIN modes) + chunked taper | ✅ MFD-oscillation periods match the window-avg L_si to 0.1 % / 2.4 %; acceleration ratio x3.33 vs window model x3.25 (pointwise claim x4); walk-off integral to 0.06 % (linear) / 0.05 % (N≈1 nonlinear), energy exact. DW-cascade/Fig. 2c out of scope (3-mode, no Raman) |
| [`stolen_lin_1978_spm`](stolen_lin_1978_spm/) | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) · [10.1103/PhysRevA.17.1448](https://doi.org/10.1103/PhysRevA.17.1448) | GNLSE (Kerr) + pulse | ✅ spectrum matches closed form to ~1e-13; peak-count rule exact |
| [`macleod_quarter_wave_dbr`](macleod_quarter_wave_dbr/) | Macleod, *Thin-Film Optical Filters*; Born & Wolf (textbook) | `dbr` TMM | ✅ peak reflectance matches exact closed form to <1e-6; stopband width within 7% |
| [`shg_textbook`](shg_textbook/) | Boyd, *Nonlinear Optics* (3rd ed.), Ch. 2; Fejer et al., *IEEE JQE* **28**, 2631 (1992) · [10.1109/3.159513](https://doi.org/10.1109/3.159513) | `chi2` (RK4IP) | ✅ η matches `tanh²(κL)` to ≤8e-15; first-order QPM recovers `tanh²((2/π)κL)` to 0.12% |
| [`gordon_1986_ssfs`](gordon_1986_ssfs/) | Gordon, *Opt. Lett.* **11**, 662 (1986) · [10.1364/OL.11.000662](https://doi.org/10.1364/OL.11.000662) | GNLSE + Raman + soliton | ✅ measured +1.08 nm / 20 m vs Gordon +0.91 nm (ratio 1.19); Stokes gain, red-shift |
| [`shg_lnoi_shg`](shg_lnoi_shg/) | Wang et al., *Opt. Express* **25**(6), 6963 (2017) — *citation unverified, see folder README* | `chi2` mode-overlap (`shg_coupling_overlap`, `pgln_overlap`) + `materials.db` LiNbO₃ extraordinary Sellmeier | ⚠️ PARTIAL — overlap machinery exercised on the article geometry with **scalar toy modes**, so `g` = 203.4 vs article 77.4 and `g'` = 9.93 vs 34.5 are convention gaps, not physics failures; the Boyd-normalised plane-wave reference `σ` = 1331.3 matches the article anchor 1330.8 to 0.03 %. Source PDF not in the repo and the *Opt. Express* 25, 6963 page range is claimed by two papers — **author must confirm the citation**. Not in the test suite |
| [`dudley_2006_cherenkov_dw`](dudley_2006_cherenkov_dw/) | Akhmediev & Karlsson, *Phys. Rev. A* **51**, 2602 (1995) · [10.1103/PhysRevA.51.2602](https://doi.org/10.1103/PhysRevA.51.2602) | phase_matching + GNLSE | ✅ `dispersive_wave_roots` = analytic 699.3 nm; GNLSE DW peak 702.3 nm (0.44%). *Linear two-term limit.* |
| [`dudley_2006_scg`](dudley_2006_scg/) | Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006) · [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135) | GNLSE + Raman + phase_matching + soliton | ✅ **9 figure scripts** (Figs. 3, 4, 5, 6, 7, 8, 9, 10, 23); Kodama–Hasegawa fission to 4%, DW phase matching to 2.6%, `z_sol` periodicity 0.998 |
| [`kuznetsov_ma_2012_breather`](kuznetsov_ma_2012_breather/) | Kibler et al., *Sci. Rep.* **2**, 463 (2012) · [10.1038/srep00463](https://doi.org/10.1038/srep00463) | GNLSE (NLSE) + phase_matching | ✅ exact Kuznetsov–Ma soliton reproduced over one period: peak 7.6129 W vs 7.6130 W, intensity L2 9e-4, `T₀`/`z_p` exact |
| [`narhi_2016_mi_breathers`](narhi_2016_mi_breathers/) | Närhi et al., *Nat. Commun.* **7**, 13675 (2016) · [10.1038/ncomms13675](https://doi.org/10.1038/ncomms13675) | GNLSE (NLSE) + phase_matching + SplitStepEngine | ✅ MI sideband 46.41 GHz = paper 46.4; exact Peregrine ratio 8.9996 (theory 9); Akhmediev ratio 5.8284; noise-seeded MI reproduces the triangular wings and ratio-9 events |
| [`tomlinson_1985_wave_breaking`](tomlinson_1985_wave_breaking/) | Tomlinson, Stolen & Johnson, *Opt. Lett.* **10**, 457 (1985) · [10.1364/OL.10.000457](https://doi.org/10.1364/OL.10.000457) | GNLSE (NLSE) | ✅ steep edges → flat top → oscillations; steepening onset at 0.57 `z_WB`; scaling `z ∝ P₀^(−0.54)` (theory −0.5) with `z/√(L_D L_NL)` constant to 9 % |
| [`marcuse_menyuk_wai_1997_manakov_pmd`](marcuse_menyuk_wai_1997_manakov_pmd/) | Marcuse, Menyuk & Wai, *JLT* **15**, 1735 (1997) · [10.1109/50.622902](https://doi.org/10.1109/50.622902) | `vector_gnlse` (`manakov` coupling + `RandomBirefringenceEngine`) | ✅ Eq. (30) 8/9 law exact (ratio 9/8, shapes invariant to 7×10⁻⁴ over 30 z₀); Fig. 5 peak fluctuation 2.2 % (paper ≈ 1 %, tolerance 3 %); Fig. 4 NRZ birefringent-vs-averaged currents to 2×10⁻⁴ L2 (paper: "exactly the same") |
| [`renninger_wise_2013_grin_solitons`](renninger_wise_2013_grin_solitons/) | Renninger & Wise, *Nat. Commun.* **4**, 1719 (2013) · [10.1038/ncomms2739](https://doi.org/10.1038/ncomms2739) | `multimode_gnlse` (3 GRIN modes, isotropic tensors) | ✅ temporal locking 0.06 of linear walk-off; FWHM on Eq. (6) fixed point to 3.3 %; self-imaging 400.5 µm direct via `phase_offsets` (1.7 %); higher-mode blue-shift −1.30/−4.58 nm vs required −1.53/−4.61 (post-#0 audit + FWM Euler fix 2026-09-30 — **ISSUES.md #0 closed, #12 resolved**) |
| [`menyuk_1987_birefringent_pulses`](menyuk_1987_birefringent_pulses/) | Menyuk, *IEEE JQE* **QE-23**, 174 (1987) · [10.1109/JQE.1987.1073308](https://doi.org/10.1109/JQE.1987.1073308) | `vector_gnlse` (incoherent 2/3 XPM + coherent FWM branch) | ✅ Eq. (9)/(10) soliton filaments at machine precision (shape L2 7×10⁻⁶); coherent-FWM filament deviation O(1/Rδ) as predicted (0.4–1.6 %, Rδ = 70); FWM decisively active at Rδ = 0.7 (83 % field change); linear-split criterion δ* = 0.04 reproduced to 0.4 % (10.04 ps = 2 × FWHM over 20 km); nonlinear lock holds to δ ≈ 0.56 (paper's δ ≤ 1 border) |
| [`krupa_2019_multimode`](krupa_2019_multimode/) | Krupa et al., *APL Photonics* **4**, 110901 (2019) · [10.1063/1.5119434](https://doi.org/10.1063/1.5119434) (GPI: Krupa, *PRL* **116**, 183901 (2016)) | `multimode_gnlse` (modal GNLSE + `phase_offsets` grating) | ✅ ξ = 0.6157 mm / f_m = 124.98 THz (PRL 0.615 / 125.0); √h·f_m ladder to 0.5 % for h ≤ 3 (h = 1..5 measured); f₁ ×4-power shift 0.04 THz pre-swap / −0.11 THz fresh post-#0 (re-recorded); photon drift 6.5e-3; no-grating control = 0 peaks. **Registered `slow`-marked in the (local) test suite — GREEN 2026-09-30 (23.5 min cold, warm-cache cases seconds; post-#0 re-record −0.11 THz)** |
| [`mumtaz_2013_multimode_jlt`](mumtaz_2013_multimode_jlt/) | Mumtaz, Essiambre & Agrawal, *JLT* **31**, 398 (2013) · [10.1109/JLT.2012.2235414](https://doi.org/10.1109/JLT.2012.2235414) (spec source of v2 multimode Eq. 12/29) | `multimode_gnlse` (Manakov `xpm_weights`) + Eq. 6 stochastic harness | ✅ Table-II walk-offs exact (≤3e-6); SPM 1→8/9 at M=1 to 8.6e-4; M=2 ensemble-vs-generalized-Manakov: inter-mode XPM **2→4/3** confirmed (best-fit weight exactly 4/3, L2 2.9e-2 @ 32 seeds, 1/√N scaling); both sides energy-conserving to 1e-12; in the test suite (~9 s) |
| [`wai_menyuk_1991_random_birefringence_solitons`](wai_menyuk_1991_random_birefringence_solitons/) | Wai, Menyuk & Chen, *Opt. Lett.* **16**, 1231 (1991) · [10.1364/OL.16.001231](https://doi.org/10.1364/OL.16.001231) | `vector_gnlse` (`RandomBirefringenceEngine`, Wai-1991 Eq.-(2) rotation law) | ✅ Fig. 1 shadow peak ratio 1.10 vs Eq. (5); Fig. 2 delay = δ·∫cos2θ dξ (corr 0.978, paper's "agreed quite well"); Fig. 3 widths bounded, no splitting at δ ≤ 5 to ~5 % (δ=7.5 caveat in folder README); Fig. 4 polarization budget 0.966/0.919/0.800/0.845/0.630; energy conserved to ≤6e-12; in the (local-only) test suite (~30 s) |
| [`wright_2015_self_organized_instability`](wright_2015_self_organized_instability/) | Wright et al., *Nat. Photon.* **10**, 471 (2016 online 2015) · [10.1038/nphoton.2015.60](https://doi.org/10.1038/nphoton.2015.60) | `multimode_gnlse` (FWM Jacobian, `oam_l` gating) + `phase_offsets` | ✅ check A: analytic STMI ladder vs digitized Fig. 3d circles to −5.2/−0.9/−2.8 % (tol 15 %); check-B' probe: engine growth band = the corrected 2x2 band |dbar| < c for both orders, out-of-band |b2(z)| turnover at z ≈ π/xi_eff (Fig. `wright_2015_gain_ztraj.png`); full-parameter deck: MI peaks 0.024/0.021 % of the roots, gains 29x/263x above the floor. Corrected Kerr condition dbar = 0.5·sym − N·κ − γP0/3 (see `../ISSUES.md` #10). ~35 min slow path / ~15 min check-B' |
| [`guasoni_2015_generalized_mi_multimode`](guasoni_2015_generalized_mi_multimode/)   | Guasoni, *Phys. Rev. A* **92**, 033849 (2015) · [10.1103/PhysRevA.92.033849](https://doi.org/10.1103/PhysRevA.92.033849) | `multimode_gnlse` (XPM-coupled deck) + Eq.-(8)/(9) eigen solver | ✅ Fig. 3 anchor to **0.7 %** (g₁ = 0.9071 / g₂ = 0.7070 vs paper 0.90 / 0.71); Fig. 3 inset eigenvectors to ~0.1 in ln (−0.349/−3.30/−3.37 vs −0.35/−3.22/−3.35); single-mode MI closed form to 1e-9. **Band structure asserted 2026-10-02**: the flat end-to-end Eq.-12 readout (contrast 1.013) is a saturation artifact — the *local* unsaturated gain gives band/edge **4.13** at z = 0.10 m (`../ISSUES.md` #8); ~96 s |
| [`raissi_2019_pinn_nlse`](raissi_2019_pinn_nlse/) | Raissi, Perdikaris & Karniadakis, *J. Comput. Phys.* **378**, 686 (2019) · [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) | reproduction-local PINN (torch float64, Adam + L-BFGS) + `SplitStepEngine` data | ✅ §I Schrödinger example: PINN rel-L2 **3.62e-3 on \|h\|** (the magnitude the paper's Fig. 2 plots) and **1.52e-3 on \|h\|²**, vs the paper's 1.97e-3; the previously reported 6.1e-3 is the *complex-field* convention, still recorded as the conservative number — the "3× gap" was a metric mismatch (`ISSUES.md` #6); loss 1.24e-6; engine ground truth verified to 1e-12 energy drift; cut profiles at t = 0.59/0.79/0.98 overlay the exact solution. ~55 min CPU (Adam + L-BFGS, `--nf-chunk 5000`). Not in the test suite (heavy; data-validation + smoke-train test only) |
| [`dw_timing_gas_hollowcore`](dw_timing_gas_hollowcore/) | Brahms & Travers, arXiv:2101.04014 (2021) | `TaperedGNLSESolver` gas β(ω,z) (Marcatili–Schmeltzer + Boerzsoenyi He) + RDW arrival-time statistics | ✅ REPRODUCED 2026-09-30 (plasma-free subset): transmission 86.8 % vs paper ~87 %; ZDW 477.9 nm pressure-insensitive; τ(E) rising in all 10 decks (ρ ≥ 0.98, Fig. 1c mechanism); Fig. 5 jitter < 300 as in 9/10 decks (medians 99–129 as; 0.8-bar exception 371 as recorded); jitter ∝ pump noise (1 %/2 % median ratio 0.51–0.84); Δv_g(RDW) −0.85..−1.0 km/s vs Fig. 5c circles. Arrival time reconstructed via the analytic β₁ propagation leg (Eq. 11/12 with simulated RDW λ) — post-#0 time-direction validation; ionisation arm out of scope v1 |
| [`kibler_2010_peregrine`](kibler_2010_peregrine/) | Kibler et al., *Nat. Phys.* **6**, 790 (2010) · [10.1038/nphys1740](https://doi.org/10.1038/nphys1740) | GNLSE (β₂-only) + `breathers` (`general_sfb` AB, `peregrine_soliton`) | ✅ REPRODUCED 2026-09-30: engine tracks the analytic AB pointwise over the growth leg (peak ≤ 2 %); a = 0.42 max-compression peak/background 8.166 vs analytic 8.026 (+1.7 %); compressed-train period rel err 1.6e-4; compressed FWHM 0.964 ps vs analytic 0.965 ps; Peregrine-limit (a = 0.495) ratio 8.944 (theory 9) with |ψ_PS|² shape overlay L2 0.0025. Real-cos eigenmode second-recurrence cycle recorded; paper-f_mod discrepancy (241 vs 74 GHz) recorded, not asserted. ~3 min |
| [`hult_2007_rk4ip`](hult_2007_rk4ip/) | Hult, *JLT* **25**, 3770 (2007) · [10.1109/JLT.2007.909373](https://doi.org/10.1109/JLT.2007.909373) | `RK4IPIntegrator` + `GNLSEOperator` (from `heidt_adaptive.py`) | ✅ REPRODUCED 2026-10-01, both decks. Deck A (N=2 soliton): 4th-order slope **−3.97**, best ε = 5.0e-6. Deck B (SCG, Table-I PCF, 10 cm): fission into 24 temporal peaks, Raman red-shift **199.6 nm** (876 → 1076 nm), dispersive wave at **576 nm** (16 % of peak), −20 dB span 551–1179 nm; convergence ladder slope **−3.86** to ε = 2.9e-12. Found two load-bearing bugs in `heidt_adaptive.py` (Raman spectrum stored time-domain; β arrays converted from ps^0 not ps^2) — see folder README |
| [`heidt_2009_adaptive_step`](heidt_2009_adaptive_step/) | Heidt, *JLT* **27**, 3984 (2009) · [10.1109/JLT.2009.2021538](https://doi.org/10.1109/JLT.2009.2021538) | reproduction-local adaptive-step layer (`heidt_adaptive.py`: `SSFIntegrator`/`RK4IPIntegrator`, `LocalErrorStepper`/`CQEStepper`, counted-FFT cost) + `GNLSEOperator` | ✅ REPRODUCED 2026-10-01, all 16 checks green in 290 s. Deck A (SCG, 10 cm): Fig. 1(c)/(d) error tracks peak 3.4e-4 / 2.5e-6 (ratio 7.4e-3) at z = 1.20 cm inside the paper's 0.7–1.5 cm fission window, log-log correlation 0.79; Fig. 2 efficiency: RK4IP-CQE **0.34×** constant / **0.70×** local at ε = 1e-3 (paper ~0.30 / 0.60–0.75), most efficient of six, slopes RK4IP −3.91/−4.40, SSF-constant −2.02. Deck B (collision, 400 km): step collapse to 0.082 at z = 197 km with full recovery (1.00), CQE dz higher than local outside (1795 vs 1031 m), CQE/local matched-ε cost 0.56 @ 1e-4 → 1.00 @ 1e-6 (paper "up to 45 % faster" — mid band yes, tight end no). Findings: CQE needs an exactly conserved quantity (first-order shock breaks it → both decks shock-free, paper Eq. (16)); SSF-CQE blind on an exactly-conserving split-step; chaotic ε floor ~7e-6 (deck A). Deviations in folder README; `ISSUES.md` #14 closed |
| [`huang_202x_pcgnlse_attractors`](huang_202x_pcgnlse_attractors/) | Huang et al., arXiv:2607.05244 (2026) | `SplitStepEngine(conserving_shock=True)` + signed γ + folder-local dimensionless pcGNLSE/moment-ODE layer | ✅ REPRODUCED partly vs recorded caveats (2026-09-30): ENGINE SIGN BENCH green — pcGNLSE γ>0 redshift path identical to the standard engine, γ<0 pcGNLSE keeps the SSFS redshift while the standard GNLSE blueshifts (the paper's headline pathology), validating `conserving_shock=True` for `ISSUES.md` #1; bright Cases I/II (Table I) anchors to ≤ 15 %; dark-soliton structural identities (S47/S48/S102/S104/S105 incl. M = M_core + ΩE) closed to 1e-14 after fixing a double-Γ bug in `dark_moments`; **dark ODE machinery fully transcribed + implemented** (`diagnostics/dark_ode.py`, printed-vs-derived `variant` switch; overlay run blocked by a now-diagnosed seam-discontinuity artifact of the non-periodic dark ansatz — folder README caveat 3 follow-up); bright Case III E-anchor reproduces at ξ≈105 (E→0.517 vs paper 0.5), Ω magnitude is the recorded deviation (caveat 2); ~30 s
| [`poletti_2008_multimode`](poletti_2008_multimode/) | Poletti & Horak, *JOSA B* **25**, 1645 (2008) · [10.1364/JOSAB.25.001645](https://doi.org/10.1364/JOSAB.25.001645) | `multimode_gnlse` (Eq. 7 `fwm_weights`/`xpm_weights` slots) + folder-local LP mode solver / overlap-integral layer | ✅ Sec. 4.A symmetry analysis of the Eq. (7) coefficients, all rules **emerging from quadrature** (nothing hard-coded), Fig. 1 fibre a = 6 µm / NA 0.17 / 1.5 µm → V = 4.2726, 10 modes = 2×10⁴ coefficients. Eq. (18): 1360 survivors per type = the combinatorial prediction exactly, forbidden entries at 7.8e-17; Eq. (19): 1360 → 340, the 1020 discarded quadruples carry \|Q\| up to 1.0; Eq. (16): all 11 identities to 2.2e-16; Sec. 3 `Q⁽¹⁾=Q⁽²⁾` exact in the real LP basis and *false* in the helical Eq.-17 basis (0.672); Sec. 5: survivors fit M^3.65, a **31–47×** cut of the O(M⁴N) term; engine: opposite-σ power **0.0** (Eq. 14 closure, ten channels, five launched empty) vs 1.09 for an isotropic control, energy 4.8e-8. Recorded: the paper's 17872/936 Fig. 1 counts need a vectorial mode solver (folder README caveat 1). The engine's `oam_l` gate is **exactly** Eq. (18) type 2 (0 mismatches over 10⁴ quadruples) — an earlier claim here that it admitted 64 extra quadruples was an index-mapping error, corrected in `ISSUES.md` #15; ~10 s |
| [`xing_2022_multimode_chg`](xing_2022_multimode_chg/) | Xing et al., *Laser Phys.* **32**, 115401 (2022) · [10.1088/1555-6611/ac979a](https://doi.org/10.1088/1555-6611/ac979a) | `materials.db` As₂Se₃/As₂S₃ Sellmeier + folder-local scalar-LP / exact-vector (Okamoto) mode solver + Eq.-(7) phase matching (dispersion layer only; no `multimode_gnlse` run) | ⚠️ **PARTIAL / NEGATIVE** — the paper's 10 µm-core As₂Se₃/As₂S₃ MMF. **Material reproduces exactly**: Eq. (5) As₂S₃ to 3e-9; Eq. (6) As₂Se₃ **as printed is broken** (the middle `B` lost its square, `λ²−19` → `λ²−19²`; corrected it matches the repo row to 4e-6, n(4 µm)=2.78304, material ZDW 7.225 µm). **Dispersion does not reproduce**: exact-vector HE₁₁ ZDW = **4.849 µm** vs the paper's 6.104 µm (scalar LP 4.962), and the higher-order modes are **anomalous** at the pump (ZDWs 3.79–3.90 µm) where the paper says they are normal. The **10.8 µm intermodal DW does not exist**: no pair has an Eq. (7) root in 8–13 µm; the paper's own TE₀₁↔HE₂₁ pair instead phase-matches a **blue** DW at ~1.94 µm. Found an upstream gap — `phase_matching.dispersive_wave_roots` is **intramodal-only** (it reads β_sol, β(ω) and β₁ from one `beta_fn`), returning the soliton wavelength as a spurious root for an intermodal DW. ~29 s |

## Findings surfaced while reproducing (see `../ISSUES.md`)

Reproducing papers is also a bug hunt. Landmark bugs found and fixed:

- **Inverted dispersion sign** — `_linear_step` formed a bright soliton for
  β₂ > 0 (and inverted the group delay). Fixed to the standard convention
  (β₂ < 0 → soliton).
- **Raman response direction** — the delayed convolution amplified the
  anti-Stokes sideband; corrected to give Stokes gain and a red-shifting soliton.
- **Coherent-coupling FWM integrator** (Menyuk 1987) — missing second Strang
  half of the diagonal + substep rule; resolved, exact vector solitons at
  shape-L2 7×10⁻⁶.
- **FROG chirped-pulse retrieval** — PCA projection with seeded restarts.
- **MI gain convention** — exact linear-stability result 4γP/|β₂|.

**CLOSED 2026-09-30** (see `ISSUES.md` #0): the original failing symptom
(the Renninger & Wise higher-mode blue-shift) re-measured **blue** under
the post-swap convention (−1.06/−3.80 nm vs required −1.53/−4.61), and the
arrival-time channel was independently validated by the Brahms & Travers
reproduction the same day. The multicomponent-soliton results (locking,
compression, self-imaging) were unaffected throughout.

## Findings across reproductions

- **Analytic ground truth is decisive.** Where a closed form exists (SPM
  Fourier integral, quarter-wave DBR, Gordon SSFS, Cherenkov root,
  Kodama–Hasegawa solitons, MI gain) the reproductions match it to
  `1e-13`–few %. Solver bugs that survived the unit tests (inverted dispersion
  sign, Raman sideband direction, DBR matrix order, `_estimate_beta2`) were all
  exposed only by the literature comparisons.
- **Grid/step convergence matters more than expected.** Raman fission and full
  SCG drift by tens of percent if `dt > ~3 fs` or if the self-steepening step is
  under-resolved. The scripts that depend on this (Dudley Figs. 3/6) pin the
  grid explicitly.
- **The Taylor reconstruction reproduces the paper's ZDW.** From the Table I
  coefficients alone, β₂(λ) = 0 at 779.9 nm, matching Dudley's Fig. 2 and
  enabling the MI-gain figure without digitising the dispersion curve.

## ISSUES (cross-cutting)

1. **Self-steepening residual energy drift (≈5–6 % with Raman).** Mitigated
   twice (see the dedicated write-up below): the `τ_shock` override and the
   interaction-picture (RK4IP) shock integrator have both landed, cutting the
   drift from ≈12 % to ≈5–6 %. The remainder is **not an integrator bug** —
   it is intrinsic to the first-order Blow–Wood shock model, which does not
   conserve photon number for a time-dependent nonlinear drive. Cause,
   evidence, and solution options: [**Issue detail — shock energy drift**](#issue-detail-—-shock-energy-drift).
2. **No quantum/shot noise.** The Dudley coherence figures (18, 20–22) are not
   reproducible without a stochastic Raman source; the deterministic
   reproductions are single-shot only. *(Partially closed: the stochastic
   Raman source landed in `step2-raman-noise-source`; **Fig. 19 coherence
   reproduced 2026-09-30** (`dudley_2006_scg/fig19_coherence.py`: |g₁₂| 0.91
   in the pump/soliton band vs 0.03–0.13 in the far wings — the noise-driven
   decoherence structure of the paper); Figs. 18/20–22/28 subsequently reproduced
   (`fig18/fig20/fig21/fig22/fig28_*.py` ensemble scripts + suite entries).)*
3. **Full GVD curves are not digitised.** Where a paper's figure needs β(ω)
   across wavelengths (Dudley Fig. 23), this repo uses a Taylor
   reconstruction; far-from-835-nm results are qualitative.
4. **Reproductions validate physics, not pixels.** No figure is digitised
   curve-for-curve; tolerances and caveats are stated in each folder's README
   rather than hidden.

## ISSUE detail — shock energy drift

*(Full write-up 2025-09-21, after the Krupa GPI reproduction and a
literature check; supersedes the one-line issue and the corresponding
`REPORT.md` entry, both of which were stale.)*

### Symmetry of the problem

With `include_self_steepening=True` (+ Raman), long SCG-class runs show a
monotonic ≈5–6 % drift of the pulse *energy* (was ≈12 % before the mitigations
below). **The photon number does not drift** (see the resolution below): pure-
Raman runs are exactly conserving of both; pure SPM with shock on a frozen
drive is conservative to machine precision; only *shock + Raman together* moves
the spectrum enough for the energy↔photon-number distinction to become visible.

### What was already fixed (do not re-file)

| Mitigation | Where | Effect |
|---|---|---|
| `τ_shock` override (Dudley's effective-area-corrected 0.56 fs vs the 1/ω₀ = 0.443 fs default) | `SplitStepEngine(tau_shock=…)`, `photonics_helper/gnlse.py` | removes the 26 % timescale bias; **this part of the old issue text is closed** |
| Interaction-picture shock integrator (Strang-split exact Kerr/Raman phase + frequency-domain RK4 of the shock correction, adaptive substepping, no bin clamping) | `SplitStepEngine._nonlinear_step` | drift ≈12 % → ≈5–6 %; integrator error < 1e-4 vs a 4000-substep classical-RK4 reference |
| `energy_vs_z` monitor + >5 % drift warning | engine + `tests/test_gnlse_shock_energy.py` | makes residual drift visible instead of silent |
| `photon_number_vs_z` — the Blow & Wood (A5) invariant | engine, recorded when self-steepening is on | the *conserved* quantity for Raman+shock runs; see the resolution below |

Verified by `tests/test_gnlse_unitarity.py` and
`tests/test_gnlse_shock_energy.py` (13 tests, all green).

### Resolution (2026-10-27) — the wrong invariant, not a wrong model

**The drift is the difference between energy and photon number, and the
conserved one is not the one that was being watched.** Blow & Wood (1989),
Appendix Eq. (A5) (read from `formal/refs/blow.pdf`, p. 2672): the GNLS with
self-steepening conserves the *frequency-weighted* photon number

```
Q = ∫ |Û(ω)|² / (ω₀ + ω) dω
```

— with `τ_shock = 1/ω₀` the `n_eff/N` prefactor may be taken outside the
integral and "(A5) is then the obvious classical analog of the total photon
number in the optical wave". Their abstract is explicit that the *unweighted*
energy is not invariant: "Optical energy is not conserved by the equation, as
indeed it is not conserved in stimulated Raman scattering because photons are
preferentially emitted at a lower frequency."

Measured on the fissioning deck below (same parameters as every previous
measurement of this issue):

| `τ_shock·Ω_max` | Δ unweighted energy | Δ Q (photon number, A5) |
|---|---|---|
| 0.073 | −0.3 % | ≈ −0.01 % |
| 0.145 | −3.1 % | ≈ −0.01 % |
| 0.290 | −5.2 % | ≈ −0.01 % |
| 0.88 | −14.4 % | ≈ −0.001 % |

So the "drift scales with grid validity" table measured the *energy* — which
legitimately changes because `Σ|A|²dω = ⟨ω⟩·Q` and the spectrum moves — while
attributing it to a first-order truncation of `ω/ω₀` that does not exist: the
operator's symbol `1 + Ω·τ_shock` **is** `(ω₀ + Ω)/ω₀` bin-by-bin
(`formal/Photonics/Nonlinear/Shock.lean::code_symbol_is_exact`). Both earlier
readings were wrong: the 2025-09-21 "the model does not conserve it" reading
was aimed at the wrong invariant, and the 2026-10-02 "grid artifact" reading
invented a truncation to explain the same numbers. The integrator was
never at fault (both were agreed on that).

What is true and kept from the earlier analyses:

- pure-Raman runs are **exactly** energy-conserving (no shock ⇒ energy is the
  invariant again); pure SPM with shock on a frozen drive is
  machine-precision conservative; only *shock + Raman together* moves the
  spectrum enough to show the energy↔photon distinction.
- `conserving_shock=True` is a **no-op for γ > 0** and is a different model
  (the |γ| pcGNLSE substitutions), not a repair of a truncation — its
  validation tests (γ < 0 sign bench) are unaffected.
- Kim, Park & Shin, *Phys. Rev. E* **58**, 6746 (1998): higher-order terms
  violate the *energy* conservation laws in general — consistent with (A5)
  being about photon number, and still the right citation for energy.
- New: `photon_number_vs_z` records (A5) at every saved step when
  self-steepening is on, and `tests/test_gnlse_photon_number.py` pins the
  invariant; the >5 % drift warning now monitors the conserved quantity.

**Shipped:** `_validate_shock_grid` warns when `τ_shock·Ω_max` exceeds
`_SHOCK_TAYLOR_LIMIT = 0.2`, now justified as Blow & Wood's `ω₀/3`
derivation regime for the GNLS (the code's window is `2ω₀/5`; strict
adherence needs `1/6` — recorded in `CONFORMANCE.md` F11, not changed), and
raises on the hard positivity violation `Ω_max ≥ ω₀`. Pinned by
`tests/test_gnlse_shock_energy.py::test_fissioning_shock_photon_number_is_grid_independent`
and `::test_shock_regime_warns_only_outside_derivation_window`.

### Solutions (ranked)

1. **Read the right monitor (no code change; do this today).** With
   self-steepening on, judge conservation by `photon_number_vs_z` (Blow &
   Wood's (A5) invariant); `energy_vs_z` will drift by the spectrum's mean
   frequency change and that is *correct physics*, not a defect. Keep
   `Ω_max < ω₀` (hard guard), stay near the `ω₀/3` derivation regime the
   guard warns about, and quote the *photon-number* drift as the model
   tolerance in reproduction notes (house style, cf. the Renninger & Wise
   "energy drift 1.0 %" row — which, note, is an energy row and only makes
   sense with shock off).
2. **`conserving_shock=True` remains the γ < 0 model it always was.** The
   pcGNLSE |γ| substitutions are a different model (Hernández, *J. Opt.* 2020,
   [10.1080/17455030.2020.1856970](https://doi.org/10.1080/17455030.2020.1856970);
   arXiv:2607.05244), validation-tested on the sign bench. They are *not* a
   repair of a truncation (there is none) and are a no-op for γ > 0.
3. **No expansion of the shock factor is needed.** The code's `1 + Ω·τ_shock`
   already *is* the exact `ω/ω₀` symbol; there is no `O((τΩ_max)²)` band-edge
   term to reduce. (The one approximation left is dropping Blow & Wood's
   material-dispersion term `(∂/∂ω)log(N/n_eff)` from `τ_shock`, which they
   call "extremely weak" in a low-loss guide — a stated approximation, not a
   truncation of `ω/ω₀`.)
4. **Accepted-as-documented (default).** The ≈5–6 % energy figure is retained
   per reproduction as the *energy* tolerance for shock+Raman runs, with the
   photon-number monitor as the conservation check. The Krupa GPI
   reproduction is unaffected (shock off, as in the PRL); the Dudley
   high-order SCG bandwidths remain the main affected case — the
   `τ_shock` fix there is the dominant correction anyway.

> This issue is tracked in the repo-root **`ISSUES.md`** (now issue #1 there;
> the new top entry **#0** is the dispersion time-direction convention issue
> found by this reproduction). Solution status and the resolved-issues
> register live in `ISSUES.md`.

