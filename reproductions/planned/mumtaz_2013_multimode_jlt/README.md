# Planned — Nonlinear propagation in multimode and multicore fibers (Mumtaz, Essiambre & Agrawal 2013)

**Reference.** S. Mumtaz, R.-J. Essiambre, G. P. Agrawal, *J. Lightwave Technol.*
**31**, 398 (2013), **doi:10.1109/JLT.2012.2235414**. PDF:
`1207.6645v1.pdf` (arXiv author v1); 10 pages under `pages/`.

**Status:** `[~]` parameters fully extracted (below), `reproduce.py` to be written.
**Target:** `photonics_helper.multimode_gnlse.MultimodeSplitStepEngine` —
this paper is *the spec source* of the v2 engine (Eq. 12 stochastic coupled NLSE
→ Eq. 29 generalized Manakov). This reproduction is the conformance test against
the paper itself.

## Parameters (paper §IV, "Multimode fibers")

Step-index MMF, core Ø 12.3 µm, NA 0.2 (Δ = 0.01), V = 5 at 1550 nm; supports
LP01, LP02, LP11(a/b), LP21(a/b) → **6 spatial modes**. No birefringent linear
coupling (q̄mp = 0), δβ̄1p = 0. Fundamental γ = 1.4 W⁻¹km⁻¹. Split-step step size
100 m. Signal: 114 Gb/s PDM-QPSK (we do *not* reproduce BER — see below).

Per-mode dispersion (paper Table II):

| mode | DMGD [ns/km] | D [ps/(nm·km)] |
|---|---|---|
| LP01 | 0 | 25 |
| LP11 | 6.5 | 27.3 |
| LP02 | 9.9 | −2.3 |
| LP21 | 12 | 20.8 |

Spans: 1000 km (we may shrink to 100–200 km for runtime; document).

## What to reproduce (nonlinear-physics level, not BER)

1. **Per-mode linear channel** (Table II sanity): launch a BW/TDL pulse per
   mode; walk-off between LP01/LP21 over 100 km = 12 ns·100 = 1.2 µs →
   engine `group_delays` must separate pulses exactly.- numeric delta vs table.
2. **Manakov reduction claim (paper's core result):** with random birefringence
   (paper Eq. 12), the *ensemble-averaged* modal powers/spectra of
   `MultimodeSplitStepEngine` with randomized SU(2) mixing frames converge to
   the deterministic Manakov-equivalent run (Eq. 29) — for the M = 2 and
   M = 3 co-propagating mode cases. Metric: L2 < 5 % over ≥ 6 seeds, matching
   the paper's Fig. 1–3 finding ("Manakov Eq. (29) agree very well with the
   full stochastic equation (12)").
3. **XPM-averaging factor for GRIN degenerate modes** (paper §V discussion):
   rapid birefringence reduces the intermodal XPM coefficient from 2 → 4/3
   (degenerate-group modes). Verify numerically: CW XPM phase-shift measurement
   in the 2-mode degenerate case — deterministic 2 vs averaged Manakov 4/3
   pairwise ratio, ≤ 2 % agreement.
4. **Time-step/step-size convergence** mirroring paper Table III duration-ordering
   logic (Manakov needs far larger steps than Eq. 12): document wall-clock ratio
   (expect ≈ 10× for M=2 at 1000 km with dt 2^17 grid — reproduce qualitatively).

## Engine mapping

- β2 per mode from D (β₂ = −λ²D/(2πc) with D in SI → ps²/m), β₃ from materials
  DB or zero per paper's simplification; `group_delays` from DMGD [s/m].
- Nonlinear coefficients: paper uses full overlap-integral equations with
  γ_pmpm...; our engine maps via `xpm_weights` (N×N) with the uniform
  LP-degenerate factors when `coef_model="lp_degenerate"`, or explicit tensors:
  compute overlap integrals γ_mn = ω0 n2/c · ∫|F_m|²|F_n|² dA/(A_m A_n)
  numerically from paper Table I mode profiles (they are Bessel-Gauss-like),
  pass as `xpm_weights` normalized to mode 0 (γ = 1.4 W⁻¹km⁻¹).
- Communication-level check (Fig. 1 BER curves) is out of scope — the DSP/BER
  layer is not part of the library; we validate the *propagation field-level*
  Manakov-vs-stochastic agreement that the BER curves were generated from.

## Execution checklist

- [x] PDF rendered (`pages/p-01..10.png`).
- [ ] `parameters.json` (values above with page refs).
- [ ] `reproduce.py`: (i) Table-II linear checks; (ii) Manakov-vs-stochastic
  ensemble convergence at M=2,3; (iii) XPM 2→4/3 GRIN check.
- [ ] Figures: random-frame ensemble vs Manakov output spectra/pulse.
- [ ] Move folder to `reproductions/mumtaz_2013_multimode_jlt/` when green,
  update main index row.
