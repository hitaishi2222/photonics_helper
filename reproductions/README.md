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

## Available

| Reproduction | Reference (DOI) | Module stack | Status |
|---|---|---|---|
| [`stolen_lin_1978_spm`](stolen_lin_1978_spm/) | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) · [10.1103/PhysRevA.17.1448](https://doi.org/10.1103/PhysRevA.17.1448) | GNLSE (Kerr) + pulse | ✅ spectrum matches closed form to ~1e-13; peak-count rule exact |
| [`macleod_quarter_wave_dbr`](macleod_quarter_wave_dbr/) | Macleod, *Thin-Film Optical Filters*; Born & Wolf (textbook) | `dbr` TMM | ✅ peak reflectance matches exact closed form to <1e-6; stopband width within 7% |
| [`shg_textbook`](shg_textbook/) | Boyd, *Nonlinear Optics* (3rd ed.), Ch. 2; Fejer et al., *IEEE JQE* **28**, 2631 (1992) · [10.1109/3.159513](https://doi.org/10.1109/3.159513) | `chi2` (RK4IP) | ✅ η matches `tanh²(κL)` to ≤8e-15; first-order QPM recovers `tanh²((2/π)κL)` to 0.12% |
| [`gordon_1986_ssfs`](gordon_1986_ssfs/) | Gordon, *Opt. Lett.* **11**, 662 (1986) · [10.1364/OL.11.000662](https://doi.org/10.1364/OL.11.000662) | GNLSE + Raman + soliton | ✅ measured +1.08 nm / 20 m vs Gordon +0.91 nm (ratio 1.19); Stokes gain, red-shift |
| [`dudley_2006_cherenkov_dw`](dudley_2006_cherenkov_dw/) | Akhmediev & Karlsson, *Phys. Rev. A* **51**, 2602 (1995) · [10.1103/PhysRevA.51.2602](https://doi.org/10.1103/PhysRevA.51.2602) | phase_matching + GNLSE | ✅ `dispersive_wave_roots` = analytic 699.3 nm; GNLSE DW peak 702.3 nm (0.44%). *Linear two-term limit.* |
| [`dudley_2006_scg`](dudley_2006_scg/) | Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006) · [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135) | GNLSE + Raman + phase_matching + soliton | ✅ **9 figure scripts** (Figs. 3, 4, 5, 6, 7, 8, 9, 10, 23); Kodama–Hasegawa fission to 4%, DW phase matching to 2.6%, `z_sol` periodicity 0.998 |
| [`kuznetsov_ma_2012_breather`](kuznetsov_ma_2012_breather/) | Kibler et al., *Sci. Rep.* **2**, 463 (2012) · [10.1038/srep00463](https://doi.org/10.1038/srep00463) | GNLSE (NLSE) + phase_matching | ✅ exact Kuznetsov–Ma soliton reproduced over one period: peak 7.6129 W vs 7.6130 W, intensity L2 9e-4, `T₀`/`z_p` exact |
| [`narhi_2016_mi_breathers`](narhi_2016_mi_breathers/) | Närhi et al., *Nat. Commun.* **7**, 13675 (2016) · [10.1038/ncomms13675](https://doi.org/10.1038/ncomms13675) | GNLSE (NLSE) + phase_matching + SplitStepEngine | ✅ MI sideband 46.41 GHz = paper 46.4; exact Peregrine ratio 8.9996 (theory 9); Akhmediev ratio 5.8284; noise-seeded MI reproduces the triangular wings and ratio-9 events |
| [`tomlinson_1985_wave_breaking`](tomlinson_1985_wave_breaking/) | Tomlinson, Stolen & Johnson, *Opt. Lett.* **10**, 457 (1985) · [10.1364/OL.10.000457](https://doi.org/10.1364/OL.10.000457) | GNLSE (NLSE) | ✅ steep edges → flat top → oscillations; steepening onset at 0.57 `z_WB`; scaling `z ∝ P₀^(−0.54)` (theory −0.5) with `z/√(L_D L_NL)` constant to 9 % |
| [`marcuse_menyuk_wai_1997_manakov_pmd`](marcuse_menyuk_wai_1997_manakov_pmd/) | Marcuse, Menyuk & Wai, *JLT* **15**, 1735 (1997) · [10.1109/50.622902](https://doi.org/10.1109/50.622902) | `vector_gnlse` (`manakov` coupling + `RandomBirefringenceEngine`) | ✅ Eq. (30) 8/9 law exact (ratio 9/8, shapes invariant to 7×10⁻⁴ over 30 z₀); Fig. 5 peak fluctuation 2.2 % (paper ≈ 1 %, tolerance 3 %); Fig. 4 NRZ birefringent-vs-averaged currents to 2×10⁻⁴ L2 (paper: "exactly the same") |
| [`renninger_wise_2013_grin_solitons`](renninger_wise_2013_grin_solitons/) | Renninger & Wise, *Nat. Commun.* **4**, 1719 (2013) · [10.1038/ncomms2739](https://doi.org/10.1038/ncomms2739) | `multimode_gnlse` (3 GRIN modes, isotropic tensors) | ✅ temporal locking 0.06 of linear walk-off; FWHM on Eq. (6) fixed point to 1.0 %; self-imaging 400.5 µm direct via `phase_offsets` (1.7 %). ⚠ higher-mode blue-shift sign pending **ISSUES.md #0** (convention audit) |
| [`menyuk_1987_birefringent_pulses`](menyuk_1987_birefringent_pulses/) | Menyuk, *IEEE JQE* **QE-23**, 174 (1987) · [10.1109/JQE.1987.1073308](https://doi.org/10.1109/JQE.1987.1073308) | `vector_gnlse` (incoherent 2/3 XPM + coherent FWM branch) | ✅ Eq. (9)/(10) soliton filaments at machine precision (shape L2 7×10⁻⁶); coherent-FWM filament deviation O(1/Rδ) as predicted (0.4–1.6 %, Rδ = 70); FWM decisively active at Rδ = 0.7 (83 % field change); linear-split criterion δ* = 0.04 reproduced to 0.4 % (10.04 ps = 2 × FWHM over 20 km); nonlinear lock holds to δ ≈ 0.56 (paper's δ ≤ 1 border) |
| [`krupa_2019_multimode`](krupa_2019_multimode/) | Krupa et al., *APL Photonics* **4**, 110901 (2019) · [10.1063/1.5119434](https://doi.org/10.1063/1.5119434) (GPI: Krupa, *PRL* **116**, 183901 (2016)) | `multimode_gnlse` (modal GNLSE + `phase_offsets` grating) | ✅ ξ = 0.6157 mm / f_m = 124.98 THz (PRL 0.615 / 125.0); √h·f_m ladder to 0.5 % for h ≤ 3 (h = 1..5 measured); f₁ power shift 0.04 THz; photon drift 6.5e-3; no-grating control = 0 peaks. **Not in the test suite — ~35 min runtime** |

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

**Currently open and blocking one reproduction:** the engine's dispersion
operator runs with a time direction opposite to its own Raman and
group-delay operators (dense-DFT arbiter evidence at 1e-13) — see
**`../ISSUES.md` #0**.  First surfaced by the Renninger & Wise 2013
higher-mode blue-shift caveat; the multicomponent-soliton results
(locking, compression, self-imaging) are unaffected.

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
   Raman source landed in `step2-raman-noise-source`; the coherence
   figures still await a reproduction.)*
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
monotonic ≈5–6 % drift of the pulse energy / photon number (was ≈12 % before
the mitigations below). Pure-Raman runs are **exactly** conserving
(E ratio = 1.00000); pure SPM with shock on a frozen drive is conserving to
machine precision; only the combination *shock + evolving drive* drifts.

### What was already fixed (do not re-file)

| Mitigation | Where | Effect |
|---|---|---|
| `τ_shock` override (Dudley's effective-area-corrected 0.56 fs vs the 1/ω₀ = 0.443 fs default) | `SplitStepEngine(tau_shock=…)`, `photonics_helper/gnlse.py` | removes the 26 % timescale bias; **this part of the old issue text is closed** |
| Interaction-picture shock integrator (Strang-split exact Kerr/Raman phase + frequency-domain RK4 of the shock correction, adaptive substepping, no bin clamping) | `SplitStepEngine._nonlinear_step` | drift ≈12 % → ≈5–6 %; integrator error < 1e-4 vs a 4000-substep classical-RK4 reference |
| `energy_vs_z` monitor + >5 % drift warning | engine + `tests/test_gnlse_shock_energy.py` | makes residual drift visible instead of silent |

Verified by `tests/test_gnlse_unitarity.py` and
`tests/test_gnlse_shock_energy.py` (13 tests, all green).

### Cause (established, with evidence)

The drift is a **property of the first-order shock model itself, not of the
integrator**:

1. **The model's own conservation law.** For
   `iγ(1 + (i/ω₀)∂_t)(A·P_NL)` (Blow & Wood 1989),
   `d/dz ∫|A|² dt = −2γτ ∫ P_NL·Im(A* ∂_t A) dt`,
   which vanishes only for a *time-independent* `P_NL`. Any state where the
   drive keeps changing (fissioning soliton, SCG) violates it at order
   `O(τ_shock Ω_max)`. The docstring of `_nonlinear_step` derives this
   in-line.
2. **The integrator is exonerated.**
   `test_gnlse_unitarity.py::test_integrator_fidelity` advances the same
   state with (a) the RK4IP shock step and (b) 4000 classical-RK4 substeps
   of the *exact* same flow: fields agree to < 1e-4 and **both drift in
   photon number identically**. No integrator can conserve what the model
   does not.
3. **Literature confirms.** Kim, Park & Shin, *Phys. Rev. E* **58**, 6746
   (1998) (arXiv:solv-int/9904008), "Conservation Laws in Higher-Order
   Nonlinear Optical Effects": NLSE conservation laws are "violated in
   general" once self-steepening is added; only special integrable
   higher-order combinations (Hirota, Sasa–Satsuma) have closed sets. And
   the photon-conserving GNLSE line (pcGNLSE; see e.g. Huang et al.,
   arXiv:2607.05244, and references therein) exists precisely because the
   standard GNLSE with a shock term is not photon-conserving — restoring
   conservation requires modifying the *equation*, not the solver.

### Solutions (ranked)

1. **Grid hygiene at the call site (no code change; do this today).**
   Keep `Ω_max·τ_shock ≪ 1` (the construction-time validity guard
   `_validate_shock_grid` already rejects `Ω_max ≥ ω₀`), resolve the shock
   front (`dt` small enough that the shock correction per substep stays
   small — the adaptive `n_sub` logic handles this), and read the
   `energy_vs_z` monitor. For reproductions that need drift-free books,
   quote the drift as a stated model tolerance (house style, cf. the
   Renninger & Wise "energy drift 1.0 %" row).
2. **Implement a photon-conserving shock term (the real fix, medium
   effort).** Reformulate the shock+Raman coupling in the pcGNLSE form:
   frequency-domain pcGNLSE, or the time-domain form derived in
   arXiv:2607.05244 (two sign/|γ| modifications relative to the standard
   GNLSE: absolute-value Kerr coefficient in the Raman-shift and
   shock–Raman dissipation terms). This changes model physics, so it must
   ship as an opt-in flag (`conserving_shock=True`) with its own
   validation tests (conservation to integrator error on fissioning
   states, red-shift direction preserved, blue-skew sign preserved), not
   as a silent replacement.
3. **Raise the shock expansion order (smaller gain, same caveat).** Using
   the exact `(1 + Ω/ω₀)` factor instead of the first-order Taylor factor
   reduces the band-edge error from `O(τΩ_max)` to `O((τΩ_max)²)` but does
   not restore an exact conservation law; only worth doing together with
   (2) if pcGNLSE validation demands it.
4. **Accepted-as-documented (default).** Until (2) lands: keep the
   ≈5–6 % figure stated per reproduction, with the `>5 %` warning as the
   tripwire. The Krupa GPI reproduction is unaffected (shock off, as in
   the PRL); the Dudley high-order SCG bandwidths remain the main
   affected case — the `τ_shock` fix there is the dominant correction
   anyway.

> This issue is tracked in the repo-root **`ISSUES.md`** (now issue #1 there;
> the new top entry **#0** is the dispersion time-direction convention issue
> found by this reproduction). Solution status and the resolved-issues
> register live in `ISSUES.md`.

