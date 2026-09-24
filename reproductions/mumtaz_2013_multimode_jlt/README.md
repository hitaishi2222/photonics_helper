# Reproduction — Nonlinear propagation in multimode and multicore fibers
(Mumtaz, Essiambre & Agrawal 2013)

**Reference.** S. Mumtaz, R.-J. Essiambre, G. P. Agrawal, *J. Lightwave Technol.*
**31**, 398 (2013), **doi:10.1109/JLT.2012.2235414** (arXiv:1207.6645). PDF:
`1207.6645v1.pdf`; rendered pages `pages/p-01..10.png`.

**Status: ✅ reproduced (2026-09-21).** `reproduce.py` runs green in ~9 s and is
in the pytest suite (`tests/test_reproductions.py`, 24/24). The folder was moved
out of `planned/`.

This is the **spec-source paper of the v2 multimode engine** (stochastic Eq. 12 →
generalized Manakov Eq. 29), so the reproduction doubles as the engine's
conformance test.

## What is reproduced

1. **Table II linear conformance.** Step-index MMF (core Ø 12.3 µm, NA 0.2,
   Δ = 0.01, V = 5 at 1550 nm, γ = 1.4 W⁻¹km⁻¹): the engine's `group_delays`
   map produces the DMGD walk-offs of Table II over a 150 m window-limited
   span (the coefficient table itself, not the span, is the paper quantity):

   | pair       | measured   | expected   | rel. err |
   |---|---|---|---|
   | LP01–LP11  | 975.00 ps  | 975.00 ps  | ≤ 1e-6   |
   | LP01–LP21  | 1799.99 ps | 1800.00 ps | 2.9e-6   |

   β₂ conversion sanity: β₂ = −λ²D/(2πc) with D = 25 ps/(nm·km) gives
   −31.9 ps²/km (asserted analytically to 2⁻¹⁰⁰ rel.).

2. **SPM 1 → 8/9 (M = 1).** One spatial mode (two polarizations), independent
   Haar SU(2) frames per 1 km segment over 100 km, 32 seeds versus the engine
   run with `xpm_weights` = 8/9**: ensemble-averaged pol-summed
   intensities/spectra match the engine's Manakov output to
   **rel-L2 8.6e-4 (time), 6.5e-4 (freq)**.

3. **Generalized Manakov, M = 2 degenerate modes (the paper's core claim).**
   Stochastic Eq. 12 ensemble (independent per-mode Haar frames per segment)
   vs the engine's deterministic Eq. 29 run with inter-mode XPM **4/3**:
   rel-L2 **2.9e-2 (time), 4.1e-2 (freq) at 32 seeds**. A sweep of the
   engine's inter-mode weight shows the **best fit at exactly 4/3**
   (L2 0.0525 at 8 seeds; next best, 1.0, at 0.14) with clean 1/√N
   convergence: 5.3 % → 2.9 % → 1.0 % at 8 / 32 / 128 seeds. Both sides
   conserve total energy to ~1e-12. The no-birefringence fixed-frame
   reference deviates strongly (0.44), reproducing the paper's point that
   the un-averaged Eq. 6 physics differs fundamentally from the ensemble.

4. **Wall-clock ratio** (paper Table III logic): harness/engine = 4.7× (M=1),
   2.8× (M=2) — the deterministic Manakov path is cheaper, as claimed
   (qualitative; our ensemble is Python-level RK4 vs the vectorised engine).

## Bugs found and fixed while landing this reproduction

These were the "some are worked on but have issues" symptoms — the folder had a
half-written `reproduce.py` that failed at check 1:

1. **Dispersion applied in the WRONG space** (`stochastic_run`):
   `A = A * phi_lin` multiplied the *time-domain* array by the spectral
   phase. With 1 km segments that applies a spurious time-domain quadratic
   mask every segment, cumulatively destroying the spectrum (spectral peak
   0.945 → 0.0105 over 20 segments). It was hidden by the single-segment
   case, where the mask is nearly flat → the fixed linear step is now
   `A = ifft(fft(A)·phi)` per segment.
2. **Truncated Eq. 6 cubic** (`mumtaz_cubic`): only the `n = p` arms of the
   (l, m, n) triple sum were retained. That truncated cubic (a) leaks ~0.4 %
   of the total energy per 10 segments once random frames mix polarizations
   (the FULL cubic is a closed Kerr system and conserves Σ|A|² to 1e-12) and
   (b) makes the ensemble's inter-mode XPM wash out toward ~0.2 instead of
   the paper's 4/3 (M=2 rel-L2 was 0.32 with it, 0.0525 with the full cubic).
   With the complete sum, everything snaps into place and the 4/3 claim is
   quantitatively confirmed.
3. **`einsum` subscripts** in the cubic dropped the third index.
4. **`engine_manakov_run` snapshot indexing**: `eng.evolution[-1]` is a flat
   list of Waves (one per channel), not nested; the reshape was wrong.
5. **Comparison level**: the ensemble must average *intensities/spectra per
   pol-sum*, not complex amplitudes — the SU(2) frames decorrelate the
   polarization phases seed-to-seed while |A|² is frame-invariant, so
   amplitude-averaging collapses the mode energy and saturates the L2 at ~1.
6. **Unit slips**: β's are ps^k/m in the engine (with Ω in rad/ps) —
   passing SI s^k/m silently scales dispersion by 1e-24 (`betas_unit` is
   decorative); window semantics (`TemporalGrid(Tmax)` is the TOTAL window);
   the sanity β₂ value −21.7 ps²/km belongs to D = 17, not 25 (fixed to
   −31.9); a ns/µs print-label slip.

## Scope notes

- Paper's BER/DSP layer (Fig. 1) is out of scope; we validate the field-level
  ensemble-vs-Manakov agreement the BER curves were generated from.
- Signal stand-in: unmodulated gaussians (100 ps FWHM, 7 dBm per mode); the
  statistical depletion/walk-off physics of the QPSK stream is not modelled.
- 100 km spans (paper: 1000 km = 10×100 km, same coefficients per km).
- Overlap coefficients idealized to f = 1 (isotropic degenerate pair); the
  paper's step-index values (f_11ab = 0.3) are quoted in `parameters.json`.
- Check 1 span is 150 m because a 2 ns window cannot hold the paper's
  650 ns walk-off (grid-wrap makes the centroid meaningless); the Table II
  coefficients themselves are what the engine is validated against.
- Original planned checks are all in: (i) Table-II linear ✓, (ii)
  Manakov-vs-stochastic ensemble convergence at M=1,2 ✓ (iii) the §V XPM
  2→4/3 GRIN-style check = the M=2 degenerate case with f=1 ✓, and (iv) the
  wall-clock ratio, documented qualitatively.

## Files

- `parameters.json` — all physical inputs with page references.
- `reproduce.py` — four checks; asserts vs Table II and the Manakov factors;
  writes `mumtaz_2013_manakov.png`.
- `mumtaz_2013_manakov.png` — (a) M=1 SPM ensemble vs Manakov spectra,
  (b) M=2 per-mode spectra, (c) temporal panel incl. the fixed-frame ref.
