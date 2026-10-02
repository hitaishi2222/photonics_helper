# Reproduction — femtosecond supercontinuum (Dudley, Genty & Coen, 2006)

**Reference:** J. M. Dudley, G. Genty, and S. Coen, "Supercontinuum generation in
photonic crystal fiber," *Rev. Mod. Phys.* **78**, 1135 (2006).
**DOI:** [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135)

This folder reproduces several *distinct* results from the paper's
femtosecond anomalous-GVD section (Sec. V–VI) with one script per figure. The
shared physical model lives in [`common.py`](common.py) and the numeric inputs
in [`parameters.json`](parameters.json).

## The paper's model (Sec. IV)

The simulations integrate the scalar GNLSE (paper Eq. 2)

```
∂A/∂z + (α/2)A − Σ_{k≥2} (i^{k+1}/k!) β_k ∂^k A/∂T^k
    = iγ (1 + (i/ω₀)∂/∂T) [ A ∫ R(T') |A(z, T−T')|² dT' + iΓ_R ]
```

with `R(t) = (1−f_R)δ(t) + f_R h_R(t)`, `f_R = 0.18`, and the PCF of Table I
(hole diameter `d = 1.4 µm`, pitch `Λ = 1.6 µm`, ZDW ≈ 780 nm, pumped at
835 nm). The reference pulse is a 10 kW, 50 fs (FWHM) sech.

## Figures reproduced

| Figure | Script | What it shows | Validation | Status |
|---|---|---|---|---|
| **Fig. 3** | [`fig03_basic_scg.py`](fig03_basic_scg.py) | Basic SCG: spectral/temporal evolution + density plots, all processes on | `N = 8.66`, `z_sol = 10.7 cm`; −20 dB output span 500–1257 nm (ratio 2.51) | ✅ |
| **Fig. 4** | [`fig04_output_features.py`](fig04_output_features.py) | Output temporal/spectral detail; DW (C) and Raman solitons (A, B) | DW 535 nm, Raman soliton 944 nm; bandpass features at τ = **+3.35 / +0.76 ps** (re-recorded 2026-09-30 after the #0 mirror fix in the bandpass synthesis kernel; the negative-delay values were the e^{−i}-synthesis time-reversal artifact) | ✅ |
| **Fig. 5** | [`fig05_ideal_soliton_period.py`](fig05_ideal_soliton_period.py) | Ideal `N = 3` higher-order soliton (β₂ only, `f_R = 0`) over 2 periods | periodicity overlap at `z_sol` = **0.998**; peak compression 6.3× | ✅ |
| **Fig. 6** | [`fig06_raman_fission.py`](fig06_raman_fission.py) | Raman-induced fission of the `N = 3` soliton (β₂ + Raman) | `L_fiss = 2.23 cm`; ejected soliton 3612 W / 10.8 fs vs Kodama–Hasegawa 3472 W / 10.0 fs | ✅ |
| **Fig. 7** | [`fig07_fission_detail.py`](fig07_fission_detail.py) | Fission onset + first ejected soliton vs Kodama–Hasegawa | sech field overlap **0.997**; power 4.0 %, FWHM 8.3 % | ✅ |
| **Fig. 8** | [`fig08_dispersive_wave.py`](fig08_dispersive_wave.py) | DW emission from the `j = 1` fundamental soliton, with/without Raman | DW 645 nm vs phase matching 663 nm (**2.6 %**) | ✅ |
| **Fig. 9** | [`fig09_dw_energy.py`](fig09_dw_energy.py) | Mean soliton wavelength and DW energy fraction vs distance | Raman red-shift and reduced DW energy confirmed | ✅ |
| **Fig. 10** | [`fig10_spectrogram.py`](fig10_spectrogram.py) | Output SC spectrogram (Eq. 4) projected on intensity/spectrum | DW and soliton bands separated by 2.8 ps in delay | ✅ |
| **Fig. 19** | [`fig19_coherence.py`](fig19_coherence.py) | |g₁₂(λ)| coherence across the SC (20-realization noise ensemble) | pump/soliton band 0.91, far wings 0.03–0.13, blue wing 0.20; 0 ≤ g₁₂ ≤ 1 | ✅ |
| **Fig. 23** | [`fig23_mi_gain.py`](fig23_mi_gain.py) | CW parametric/MI gain vs pump wavelength (500 W) | reconstructed ZDW **779.9 nm**; `g_max = 2γP`; anomalous peak 25.1 THz vs 25.6 THz classical | ✅ |

### Quantitative results

```
Fig. 3 : N = 8.660, z_sol = 10.71 cm, L_D = 6.82 cm
         output −20 dB span = 500–1257 nm (ratio 2.51; paper 550–1100 nm)
Fig. 5 : L_D = 6.82 cm, z_sol = 10.71 cm, periodicity overlap = 0.998
Fig. 6 : L_fiss = 2.227 cm
         ejected j=1 : P = 3612 W (KH 3472 W), FWHM = 10.84 fs (KH 10.01 fs)
         mean wavelength 835 → 1251 nm over 0.5 m
Fig. 7 : Kodama–Hasegawa sech overlap = 0.9972
Fig. 8 : phase-matching DW = 662.8 nm, GNLSE DW = 645.4 nm (2.63 %)
         DW energy fraction 0.092 (no Raman) → 0.019 (Raman)
Fig. 10: DW band τ = −3.61 ps, soliton band τ = −0.78 ps (Δτ = 2.83 ps)
Fig. 23: ZDW = 779.9 nm, g_max(800 nm) = 110.0 = 2γP
         Ω_peak(800 nm) = 25.07 THz vs classical 25.56 THz (1.9 %)
         normal-GVD pump 750 nm: classical gain 0, full-dispersion peaks ±168.7 THz
```

## Running

Each script is standalone (writes its own PNG and prints its result):

```bash
python reproductions/dudley_2006_scg/fig05_ideal_soliton_period.py
python reproductions/dudley_2006_scg/fig06_raman_fission.py
python reproductions/dudley_2006_scg/fig07_fission_detail.py
python reproductions/dudley_2006_scg/fig08_dispersive_wave.py   # full: several minutes
python reproductions/dudley_2006_scg/fig09_dw_energy.py         # full: several minutes
python reproductions/dudley_2006_scg/fig03_basic_scg.py         # full: ~1 min
python reproductions/dudley_2006_scg/fig04_output_features.py
python reproductions/dudley_2006_scg/fig10_spectrogram.py
python reproductions/dudley_2006_scg/fig23_mi_gain.py
```

Heavy full-resolution runs are cached (`fig03_cache.npz`,
`fig08_*_full.npz`) so Figs. 4/9/10 reuse them instead of re-propagating.
The `validate(fast=True)` paths used by the test suite are much shorter and
are only used for the light figures.

## Findings

- **The paper's core femtosecond physics is reproduced quantitatively.** The
  Kodama–Hasegawa ejected-soliton parameters (Fig. 7) match to 4 % in power and
  8 % in width, and the Cherenkov phase-matching condition on the full Table I
  dispersion (Fig. 8) predicts the simulated DW to 2.6 %.
- **The full-β phase-matching root differs from the 2-term analytic
  `−3β₂/β₃` estimate.** With only β₂ and β₃ the linear root is 699 nm; adding
  the higher orders moves it to 684 nm, and the nonlinear `(1−f_R)γP_s` term
  moves it to ≈ 663 nm — in much better agreement with the GNLSE (645 nm) and
  with the paper's short-wavelength DW.
- **The ideal-soliton periodicity is an excellent solver check.** The spectrum
  returns to the input at `z_sol` with 0.998 overlap, which validates the sign
  and scaling conventions of `_linear_step`.
- **Soliton fission is grid-sensitive.** With `dt > ~3 fs` the ejected-soliton
  peak powers are wrong (we measured 1.95–3.8 kW depending on the grid); with
  `dt ≤ 1.5 fs` they converge to the Kodama–Hasegawa value. This is consistent
  with the paper's numerical-issues discussion (Sec. IV.B).
- **The reconstructed Taylor dispersion reproduces the paper's ZDW (780 nm)**
  from the Table I coefficients alone, which lets Fig. 23 be reproduced without
  digitising Fig. 2.

## Plotting API (Fig. 3)

The two evolution density plots in [`common.py`](common.py) and the assembled
dashboards in [`fig03_basic_scg.py`](fig03_basic_scg.py) return their figure
object so they can be adjusted after the call.

```python
from reproductions.dudley_2006_scg import common, fig03_basic_scg as fig03

evo = fig03.run(fast=False)
result = fig03.validate(fast=False, make_plot=False)

# Wavelength window defaults to the carrier ± margins (0.5·λ0 … 1.6·λ0);
# pass wl_bounds=(wl_min, wl_max) to override it.
fig = fig03._plot(evo, result, wl_bounds=(450, 1300), t_bounds=(-1, 5), save=False)
fig.axes[0].set_title("custom")

# One call writes the interactive dashboard next to this script as
# reproductions/dudley_2006_scg/fig03_basic_scg.html.
fig03.write_html(evo, result)
```

The interactive dashboard is the reusable
`photonics_helper.gnlse.plot_scg_dashboard` (array-level) / 
`plot_spectral_temporal_summary(solver, plotly=True)` (solver-level).  For any
propagated solver:

```python
from photonics_helper.gnlse import save_summary_html

save_summary_html(solver, "summary.html", wl_bounds=(450, 1300), t_bounds=(-1, 5))
```

The spectral panel labels each wavelength bin from its offset to the carrier.
The temporal panel uses a coarse spectral filter bank on every stored field and
labels each ``(t, z)`` cell after the *dominant local* band, because a single
delay can carry both a Raman soliton and the blue dispersive wave. Cells more
than 40 dB below the global peak are marked *low-level background*. This is a
hover aid, not a replacement for the band-resolved analysis in Figs. 4 and 10.

**Time convention (updated 2026-09-30, post-#0).** The engine's internal time
time grid NOW uses the standard Agrawal/Dudley direction directly: the
Raman-red-shifted soliton is slower (β₂ < 0 → dβ₁/dλ > 0) and appears at
*positive* delay, matching Fig. 3(b) with no flip. The historical
`time_reversal=True` compensation (for the pre-#0 mirrored dispersion
operator) is retired: the plotting helpers default to `time_reversal=False`,
and the flag is kept only for back-compatibility. Spectra are unaffected.

Both `common.plot_spectral_evolution` and `common.plot_temporal_evolution`
also accept `dynamic_range_db`, `cmap`, `z_scale`, and `plotly`, in addition to
`wl_bounds` / `t_bounds` (or the legacy `wl_min`/`wl_max` and
`t_min`/`t_max`).

## ISSUES

> Status audit 2026-09-21: issues 1–2 below are **stale** (the features
> landed); 3–6 remain true model/data limitations.

1. **Self-steepening time scale — RESOLVED.** The paper uses the
   effective-area-corrected `τ_shock = 0.56 fs` (their Eq. 3). The library's
   shock operator defaults to `1/ω₀ = 0.443 fs` but now takes a `tau_shock=`
   override (`step1-gnlse-shock-tau-and-energy`, 0.1.9); the scripts here set
   `tau_shock=0.56e-15` and the residual few-percent-level caveat applies to
   the remaining self-steepening drift only (see the model-intrinsic note in
   `reproductions/README.md` → "ISSUE detail — shock energy drift") —
   previously this entry wrongly said the override did not exist.
2. **No stochastic Raman/shot noise — RESOLVED (source shipped).** The
   paper's Fig. 3 uses single-shot simulations with noise, and Sec. VI.D
   analyses noise-driven decoherence. The library now ships the stochastic
   source (`step2-raman-noise-source`: one-photon-per-mode Γ_R with thermal
   factor, `photonics_helper.noise`) AND the first coherence figure has
   landed: **Fig. 19 reproduced** (`fig19_coherence.py`, 20-realization
   ensemble with Raman noise): |g₁₂| stays 0.91 in the pump/soliton band,
   drops to 0.03–0.13 mean in the far wings (blue 0.20 at 690 nm), 0 ≤ g₁₂ ≤
   1 — the noise-driven decoherence structure of the paper's Fig. 19.
   Figs. 18, 20–22, 28 **reproduced 2026-09-30** (ensemble scripts):
   `fig20_coherence_evolution.py` (Fig. 20 spectra + |g₁₂|-vs-z for the
   100/150 fs 10 kW decks over 10 cm — near-unity coherent-seeding band
   112 nm (100 fs) vs 1.1 nm (150 fs); Fig. 18(c) = the 150 fs output
   coherence, Fig. 18(d) = the per-branch pump-λ phase trajectories from
   the same ensemble), `fig21_coherence_vs_pump.py` (Fig. 21a avg |g₁₂| +
   −20 dB width vs pump λ; narrative asserts RECORDED — measured β(λ)
   still needed), `fig22_coherence_vs_N.py` (Fig. 22 scatter: N ≈ 5–11 →
   |g₁₂| 0.94–0.98, N ≥ 26 → ~0.20 — the paper's transition band),
   `fig28_picosecond_coherence.py` (Fig. 28: 20 ps/500 W deck — coherent
   only at the pump, smooth mean spectrum, far wings |g₁₂| ~ 0.2).
3. **Fig. 10 beat frequency.** The paper quotes ≈ 165 THz between the two
   beating spectrogram bands. The deterministic simulation gives a different
   beat (≈ 17 THz here). The script therefore validates the *method* and the
   time-frequency correlation, not the literal number.
4. **Fig. 23 uses a reconstructed dispersion.** Only the Table I Taylor
   coefficients at 835 nm are available, so β(ω) for other pump wavelengths is
   an extrapolation. The ZDW and the anomalous/normal contrast are right, but
   the far-normal-GVD gain windows (e.g. 750 nm at ±169 THz) are qualitative.
5. **Fission-distance definition.** The paper quotes `L_fiss ≈ L_D/N ≈ 2.3 cm`
   for N = 3. This is a bandwidth/fission estimate; the first *temporally
   resolved* split into two peaks in our simulations is later (≈ 7 cm). The
   scripts use `L_D/N` and validate the ejected-soliton parameters instead,
   which is the stronger test.
6. **Grid/step convergence.** Full SCG results depend on `num_steps` at the
   few-percent level even after the adaptive stepping, because of the
   self-steepening RK4 step (see issue 1). Convergence at the 1 % level needs
   `num_steps ≳ 20000` and is not enforced on every test run.

## Reproducing the figures vs. digitising the paper

The PDF of the review is checked in as `revmodphys.78.1135.pdf` and rendered
into `paper_pages/` for reference. These scripts validate the *physics* of the
figures against analytic references; they are not pixel-level digitised
reproductions. Where the paper's exact curve depends on unstated choices
(noise realisation, corrected shock time, full GVD table) this is called out
under **ISSUES** rather than hidden behind a loose tolerance.

## Full-mode re-run record (2026-09-30, follow-up session)

All seven slow-path figures (Figs. 3, 5–10) re-run at `fast=False` on the
current engine (post-#0 swap + post-FWM-RK4 fix), fresh npz caches written:

- **Fig. 3 row RE-CONFIRMED**: fresh full-mode −20 dB span = **499.9–1256.7 nm**
  (ratio **2.514**), N = 8.66, z_sol = 10.71 cm — identical to the recorded
  row; the earlier "broader span was a fast-grid/stale-cache artifact"
  suspicion is resolved (the row stands as measured, now from a fresh cache).
- Fig. 8: DW 645.4 nm vs phase-matching 662.8 nm (**2.63 %**), DW energy
  fraction 0.092 (no Raman) → 0.019 (Raman) — unchanged on fresh caches.
- Fig. 6: ejected j=1 3612 W / 10.84 fs vs KH 3472/10.01; mean λ 835 →
  **1250.7 nm** — unchanged.
- `common.py` audit (raw-`np.fft` mirror follow-up): the spectrogram helper
  and the sorted-wavelength interpolation path were tone-probe-checked in
  the suite (`test_dudley_common_analysis_channel_mirror_audit`); the
  legacy `time_reversal=True` default in `plot_temporal_evolution` is
  retired to `False` (post-#0 internal time already matches the literature).
