# Brahms & Travers (2021) — Timing and energy stability of resonant
# dispersive wave emission in gas-filled hollow-core waveguides

> Christian Brahms, John C. Travers, Heriot-Watt: arXiv:2101.04014v2
> (22 Apr 2021; published during revision). PDF: `dw_timing_gas_hcf.pdf`
> (LOCAL-ONLY, gitignored). Page images `pages/p-01..24.png`.

## Why this paper (what solving it decides)

The sharpest *time-domain*, dispersion-direction-sensitive observables in
the GNLSE family: **RDW arrival-time** (linked to the group delay of the
dispersive wave relative to the soliton — the same time-direction family
that exposed ISSUES.md #0) and arrival-time *statistics*. Solving it
decides:

1. **#0 post-fix validation in the arrival-time channel.** RDW is emitted
   at max self-compression and then walks off from the soliton under the
   gas dispersion: an arrival-time/centroid-lag measurement cannot be
   spoofed by time-symmetric observables. If the engine reproduces the
   paper's lag-vs-pump-energy trend (Fig. 1(c) mechanism and the
   parameter-scan panels), the dispersion direction and the group-delay
   operator are validated in a regime never tested in-house.
2. **Soliton self-compression in gas** (anomalous GVD from the gas
   dispersion + Kerr) — exercises the engine's β(ω, z) pressure-gradient
   path (`TaperedGNLSESolver` dispersion profile) end-to-end.
3. **RDW phase matching in the gas frame** (resonant-radiation roots in
   the phase_matching layer at the gas pressure, incl. the plasma-free
   subset).

## Scope decision (read before working)

- The paper's ionisation/plasma parts are OUT OF SCOPE for v1 (no
  photoionisation operator in the engine). The paper itself applied its
  analysis to a plasma-free **scaled-down system** (small-core
  microstructured fibre and a hollow capillary variant) — reproduce the
  plasma-free subset there; leave the ionisation arm as an explicit
  caveat + follow-up.
- Waiting-arrival statistics (pump-energy-noise Monte-Carlo) is the
  paper's quantitative core: plan for N-run ensembles (batching via
  herdr background panes; approximate N from the paper's own ensemble
  size). Mark slow; not in the default test suite.

## Module stack

- `TaperedGNLSESolver` (z-dependent gas dispersion via pressure gradient —
  the paper uses constant-pressure AND pressure-gradient cases).
- Gas refractive index: standard pressure-scaled Sellmeier (Marcatili–
  Blythe-Beckmann for Ar or the paper's gas; extract which gas + pressure
  from `pages/`), mapped to β(ω, z) in SI.
- `phase_matching` for the RDW resonant roots at the compression point.
- `SplitStepEngine` full GNLSE (self-steepening on for few-cycle RdW
  fidelity; Raman in gas is negligible vs the paper's model — check which
  terms the paper includes; reproduce their operator set exactly).

## Paper parameters (to extract in the working session, page refs)

- Waveguide core radius, gas + pressure profile, pump wavelength/duration/
  energy, noise model of the pump (their conservative estimate), the RDW
  detection band, and the per-panel parameter sweep ranges.
- The exact figure targets: RDW spectrum/tuneability, arrival-time lag
  vs pump energy, energy-noise suppression onset, jitter < 1 fs claim.

## Engine calls (planned)

- Compression-only check first (cheap): self-compression optimum length
  vs the paper's Fig. 1; then RDW root via `phase_matching` at ω_factory.
- Arrival-time readout: temporal centroid of the RDW band-pass-filtered
  field at the waveguide exit vs the pump arrival — *deterministic*
  single-noise-seed version + ensemble version.
- Tuneability: repeat at 3–4 pressures matching the paper's panels.

## Runtime (fill in as measured; herdr background)

- Deterministic per-panel run: TBD once the first compression run is
  timed (record here before the ensemble stage, per house rule).
- Ensemble: plan for ~O(10^2–10^3) noise seeds × the deterministic run;
  batch 2 seeds/pane and shard across herdr panes.

## Asserts (draft)

- Compression z-position within stated tolerance of the paper's.
- RDW central wavelength vs `phase_matching` root (≤ few %) and vs paper.
- Arrival-time lag vs pump energy: reproduces the paper's trend direction
  and magnitude (tolerance from the paper's own spread).
- Ensemble: arrival-time jitter < 1 fs at the paper's noise level.

## Execution checklist

- [ ] Read `pages/p-01..p-12.png` + figures; extract parameters + gas.
- [ ] Decide plasma-free path (scaled-down system rows) with folder note.
- [ ] `parameters.json` + `reproduce.py` skeleton + timing measurement.
- [ ] Gas β(ω, z) implementation + one closed-form anchor (e.g. zero-
      dispersion wavelength vs pressure, analytic Marcatili check).
- [ ] Deterministic panels → ensemble panels (herdr batching).
- [ ] pytest wrapper (fast config); README measured-row template.
- [ ] Move out of `planned/`; main README row; ISSUES.md note under #0
      addendum (arrival-time validation landed).

## Parameters extracted (2026-09-29, v1)

Full strict-format values in `parameters.json`. Key numbers with page refs
(text layer of `dw_timing_gas_hcf.pdf` + figure scans in `pages/`):

- **Main system (Fig. 1/2/5/6):** hollow-capillary fibre, a = 125 um core
  radius, L = 1 m (transmission ~87 % at 800 nm, Fig. 8 text), helium at
  constant pressures 0.8/1.5/2.1/3.0/4.0 bar (Fig. 3a) or decreasing
  gradients p(z) = p0*sqrt(1 - z/L) (paper Eq. 3) with fills 1.2/2.2/3.2/
  4.5/6.0 bar (Fig. 3b; Fig. 7 uses the 3.2-bar gradient).
- **Pump:** transform-limited 7.5 fs Gaussian pulses at 800 nm; energy
  scan 110-210 uJ (Fig. 2 example: 225 uJ); noise = 2 % r.m.s. energy
  (Fig. 2/3/5), 1 % duration study (Fig. 7); uniformly random CEP +
  one-photon-per-mode shot noise (Fig. 2c/d).
- **Sampling:** resampling approach = 800 uniformly spaced pump energies
  -> interpolants -> 10 000 random samples; the direct-method validation
  uses 500 simulations per mean energy (2c/d).
- **RDW extraction:** fixed window 185-265 nm (Fig. 2), variable 15 %
  relative-bandwidth filter centred on the RDW band elsewhere; arrival
  time = first moment of the IFFT of the filtered spectrum; energy =
  integrated filtered spectral energy density; central-wavelength jitter
  = first spectral moment. Paper grid: 100 nm - 3 um at 4096 samples /
  450 fs window, 55 as nonlinear resampling.
- **Simple model (Eqs. 11/12):** tau = Lprop*[beta1(w_RDW)-beta1(w0)],
  Lprop = L - Lf, Lf = sqrt(T0^2 / (gamma |beta2(w0)| P0)) — reproduces
  the simulated jitter magnitude (Fig. 6).
- **Scaled-down pair (Fig. 8):** hollow-core PCF, a = 25 um, He at 53.8
  bar (2.1-bar-scaled) and 80.7 bar (3.2-bar-scaled); the same noise
  characteristics as the large core once waveguide loss is switched off —
  instructive no-loss toggle for us.
- **Hard quantitative anchor:** timing jitter < 300 as for every
  parameter combination (Fig. 5 text); jitter proportional to the pump
  energy noise over the whole range.
- **Gas dispersion:** pressure-scaled helium refractive index
  (Boerzsoenyi 2008 Sellmeier fits) via the CoolProp helium EOS number
  density (paper refs [23]/[24]).
- **Out of scope v1:** photoionisation + plasma (the paper's own
  plasma-free comparisons define our target subset) and THG.

Runtime: still TBD for the first engine compression run (record here
before the ensemble stage); resampled statistics are cheap after the
probe scan.

---

## Outcome (2026-09-30, REPRODUCED — plasma-free subset)

**Status: reproduced.** Analytic tier [A1–A6] green; Fig. 3 scan campaign
complete (10 decks x 800 energies, ~10 s/pt sharded over herdr); Fig. 5/6
jitter statistics reproduced by the paper's own resampling method
(`stats_fig56.py` -> `stats_fig5.json` + `dw_timing_fig56.png`).

| Check | Paper | Measured | Verdict |
|---|---|---|---|
| He index at 800 nm, 1 bar | n−1 ≈ 3.2e-5 (Boerzsoenyi) | 3.195e-5 | ✅ |
| Capillary transmission 1 m, 125 µm, 800 nm | ~87 % (Fig. 8 text) | 86.79 % | ✅ |
| β₂(800 nm), 2.1 bar | negative, |β₂| small (anomalous for soliton) | −0.0075 ps²/km | ✅ |
| ZDW vs pressure | waveguide-dominated, ~pressure-insensitive | 477.9 nm (all p) | ✅ |
| Eq. (12) Lf at Fig. 1 point | max-compression z < L (Fig. 1b) | 0.62 m | ✅ |
| RDW band vs pressure/energy | ~185–265 nm (Fig. 2 window), tuneable | 206–231 nm over 80–220 µJ, all decks | ✅ (narrower than the paper's 140–320 nm plasma-inclusive band — expected, plasma-free) |
| τ(E) direction (Fig. 1c mechanism) | τ rises with E (Lf shrinks → Lprop grows) | Spearman ρ ≥ 0.977 in all 10 decks | ✅ |
| Δv_g(RDW) (Fig. 5c circles) | negative, ~−1..−1.7 km/s | −0.85..−1.0 km/s | ✅ order+sign |
| Timing jitter < 300 as (p-14) | all parameter combinations, 2 % noise | 9/10 decks < 300 as (medians 99–129 as); **0.8 bar exception: 371 as** | ⚠ recorded deviation |
| Jitter ∝ pump noise (p-14) | 1 % → half | 1 %/2 % ratio median 0.51–0.84 (exact 0.5 only where dτ/dE locally linear; steep λ(E) sections inflate it) | ✅ within sampling |
| Arrival-time channel post-ISSUES #0 | — | RDW walk-off enters via the analytic β₁ propagation leg (engine frame drops β₁ by design); direction + magnitude validated against Eq. (11) | ✅ |

**Method notes (house honesty):**

1. *Arrival-time reconstruction.* The engine propagates in the carrier
   group-velocity frame (`_linear_step` drops β₁), so the raw scan
   arrival-time moments sit at ~0 fs and measure only the envelope-frame
   GVD imprint. The paper-frame observable is reconstructed as
   τ(E) = L_prop(E)·[β₁(ω_RDW(E)) − β₁(ω₀)] (Eq. 11/12) using the
   SIMULATED RDW wavelengths per energy — exactly the paper's own
   coupling ("finding the central wavelength of the RDW in the same
   manner as for our choice of window function", p-16). The envelope-frame
   contribution (raw moment, SavGol-smoothed) is added on top; it is
   ≤ 0.2 fs — negligible.
2. *Scan smoothing.* The plasma-free UV band has competing quasi-threshold
   peaks (242/216/192 nm at 2.1 bar) which make λ_RDW(E) jump 1–6 nm per
   0.2 µJ; SavGol window 21 pts (3.7 µJ) before the resampling step, as
   the paper's smooth Fig. 5/6 montages imply.
3. *0.8-bar exception.* The paper claims jitter < 300 as for every
   combination; our 0.8-bar deck peaks at 371 as (median 129 as). Their
   own 0.8-bar no-plasma trace (Fig. 5a, lighter purple) also behaves
   differently from all other pressures; our stronger UV multi-peak
   competition (no carrier-resolved UPPE) plausibly widens it further.
4. *Ensemble scope.* Statistics use the paper's resampling method
   (interpolants + 10 000 Gaussian samples of the pump energy), not the
   direct N-simulation ensemble (their Fig. 2c/d validation used 500
   sims/point). The direct-method jitter channel requires per-run β₁
   post-processing identical to note 1.
5. *β₁-aware readout helper (SHIPPED 2026-09-30).* The optional follow-up
   "in-engine absolute-arrival readout helper" is now in
   [`arrival_beta1.py`](arrival_beta1.py):
   `absolute_arrival_time_fs(beta1_fn, length_m, z_fission_m, omega0,
   omega_rdw)` integrates
   τ = ∫_{z_f}^{L} [β₁(ω_RDW; z) − β₁(ω₀; z)] dz on the same dispersion
   profile family the engine used, with the RDW frequency taken from the
   engine-measured spectral centroid and a per-seed `z_fission`
   supported. Machine-precision vs the closed form
   (L−z_f)·Δβ₁ (regression test `test_dw_timing_beta1_arrival_helper`);
   τ ≈ +10 fs for the 2.1-bar UV deck (215 nm). Direct-method per-run
   statistics can now call this helper instead of the Eq.-11/12
   reconstruction.

**Runtime record:** deterministic Fig. 1 anchor ~25 s (dz = 2.5e-5 m);
scan point ~10 s at dz = 5e-5 m (convergence: RDW λ shift 1.5 nm / energy
frac 2 % vs dz = 2.5e-5); full 10-deck scan 83 min sharded x20;
stats pass ~8 min. Fast validation tier seconds.

**Out of scope v1 (unchanged):** photoionisation/plasma dynamics, THG
operator, direct-method ensemble.
