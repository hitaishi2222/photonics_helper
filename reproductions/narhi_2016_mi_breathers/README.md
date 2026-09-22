# Reproduction — spontaneous MI breathers and rogue waves (Närhi et al., 2016)

**Reference:** M. Närhi, B. Wetzel, C. Billet, S. Toenger, T. Sylvestre,
J.-M. Merolla, R. Morandotti, F. Dias, G. Genty, J. M. Dudley, "Real-time
measurements of spontaneous breathers and rogue wave events in optical fibre
modulation instability," *Nature Communications* **7**, 13675 (2016).
**DOI:** [10.1038/ncomms13675](https://doi.org/10.1038/ncomms13675)

## Result reproduced

The paper measures the real-time breakup of a noisy continuous wave in SMF-28
and shows that the emerging localized structures are **nonlinear Schrödinger
breathers**, verifying the analytic soliton-on-finite-background (SFB) theory.
Five things are reproduced here, all against the paper's parameters
(λ = 1550.3 nm, β₂ = −21.4×10⁻²⁷ s²/m, γ = 1.3×10⁻³ W⁻¹m⁻¹, loss
0.18 dB/km, P₀ = 0.7 W):

**A. MI gain.** For `i A_z = (β₂/2)A_TT − γ|A|²A` with β₂ < 0 the maximum-gain
modulation frequency is `Ω = √(2γP₀/|β₂|)` and `g_max = 2γP₀` (Agrawal
Eq. 5.1.9). The paper quotes **46.4 GHz** for these parameters.

**B. Deterministic MI growth.** Two weak sidebands are seeded at Ω and the
exponential intensity growth `g_meas = d ln(P_sideband)/dz` is compared with the
analytic `g(Ω)`.

**C. Peregrine soliton (PS).** Paper Fig. 4b uses the exact profile
(normalised to the background)

```
I_PS(t) = ( 1 − 4 / (1 + 4 γ P₀ t² / |β₂|) )²,
```

whose peak-to-background ratio is exactly **9**. The exact PS field

```
ψ_P(ξ,τ) = e^{iξ} ( 1 − 4(1 + 2iξ)/(1 + 4τ² + 4ξ²) )
```

is propagated from ξ = −3 to ξ = +3 with the GNLSE engine (β₂ only) and
compared with the analytic solution.

**D. Akhmediev breather (AB).** The `a < 1/2` member of the SFB family (same
form as the Kuznetsov–Ma reproduction, paper's Ref. 31) grows and decays once.
For `a = 0.25` the peak ratio is `|ψ(0,0)|² = 5.8284`.

**E. Spontaneous MI.** A CW field with 1.4 % intensity contrast broadband noise
is propagated 11.7 km (the paper's first measurement distance). The
ensemble-averaged spectrum develops the characteristic **triangular MI wings**
around ±Ω_peak, and individual realizations contain breather peaks whose
peak-to-background ratio reaches the ratio-9 rogue-wave limit and beyond
(breather collisions).

## Ground truth

- MI gain `g(Ω) = |β₂Ω|√(Ω_c²−Ω²)`, `Ω_c² = 4γP₀/|β₂|`, `g_max = 2γP₀`.
- Exact PS and AB solutions of the NLSE (verified by finite differences during
  implementation; see the Kuznetsov–Ma reproduction for the general Eq. 2).
- Paper's quoted values: 46.4 GHz sideband, PS ratio 9 + collision threshold.

## Outcome

```
MI:  Ω_peak = 46.41 GHz  (paper 46.4),  g_max = 1.820e-3 /m,
     cutoff = 65.6 GHz
deterministic growth:  measured 1.514e-3 vs theory 1.820e-3 /m  (ratio 0.83)
Peregrine:  peak/background 8.9996 (theory 9),  profile L2 6.85e-03
Akhmediev (a=0.25):  peak/background 5.8284 (theory 5.8284),  profile L2 1.14e-06
spontaneous MI (12 realisations, 11.7 km):
     ensemble sideband peak 39.5 GHz,  max peak/mean 9.49, mean 7.53
```

Figure: `mi_breathers.png` (MI gain, PS propagation + profile, AB propagation +
profile, ensemble spectrum). Paper page images in `paper_pages/`.

## Findings

- The library's analytic MI convention (`Ω_c² = 4γP₀/|β₂|`, `g_max = 2γP₀`)
  matches the paper's quoted 46.4 GHz sideband **exactly** and the engine's
  dynamical MI growth to within ~17 % (the seed is a real cosine, so it excites
  both the growing and decaying eigenmodes; the finite 2 km window and the
  discrete grid account for the rest).
- Both exact breather solutions are reproduced to ~10⁻⁶–10⁻³ relative; the PS
  reaches `P/⟨P⟩ = 9` to 4×10⁻⁴ and its profile matches the paper's Fig. 4b
  analytic curve.
- The spontaneous-MI ensemble reproduces the paper's central qualitative claim:
  noise on a CW breaks up into breathers, the average spectrum grows the
  triangular MI wings around ±Ω_peak, and the peak-to-background distribution
  reaches and exceeds the single-breather PS limit of 9 (→ collisions).

## ISSUES

> Status audit 2026-09-21: the "library has no stochastic noise source"
> sentence below is **stale** — `step2-raman-noise-source` shipped
> `photonics_helper.noise` (ASE background, one-photon-per-mode spontaneous
> Raman/shot seed). This reproduction still seeds its own white Gaussian by
> construction; re-wiring it onto the shipped source (and an ASE-shaped −50 dB
> background) is a possible upgrade.

- The library's stochastic noise source (`photonics_helper.noise`) landed
  **after** this reproduction was written; the noise is generated inside this
  reproduction (white complex Gaussian, 1.4 % intensity contrast — the paper
  states <5 %). The paper's ASE model (a −50 dB spectral background with
  random phase) is not reproduced exactly, and the ensemble size here (4–12)
  is far smaller than the paper's ~50 000 peaks, so no histogram-statistics or
  coherence claim is made.
- The paper's **Figs. 1, 2, 3** (single-shot temporal/spectral evolution,
  experimental traces, and the long-tailed peak histograms) are **not**
  reproduced: they need the experiment or a large Monte-Carlo ensemble, neither
  of which is digitised here.
- Part E drives `SplitStepEngine` directly with a fixed 20 m step for determinism;
  `GNLSESolver` has no `step_size` passthrough (its adaptive `dz ≈ 11 m` works fine
  for this CW field). Note the `TemporalGrid.ifft` `1/dt` scaling: the input noise
  is built in the **time domain** because mapping a small spectral perturbation back
  with `grid.ifft` inflates it by `~1/dt ≈ 3×10¹³`. See `PLAN.md` §3.1.
- The deterministic MI growth is a secondary check: it is only approximately
  quantitative (ratio 0.83), and the measured peak in the ensemble spectrum
  (39.5 GHz) sits below the analytic 46.4 GHz because of the finite ensemble,
  the discrete grid and pump self-phase-modulation, all of which the paper also
  discusses.
