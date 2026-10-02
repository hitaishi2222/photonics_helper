# Reproduction — The Peregrine soliton in nonlinear fibre optics (Kibler et al., 2010)

**Reference:** B. Kibler, J. Fatome, C. Finot, G. Millot, F. Dias, G. Genty,
N. Akhmediev, J. M. Dudley, "The Peregrine soliton in nonlinear fibre
optics," *Nature Physics* **6**, 790 (2010).
DOI: [10.1038/nphys1740](https://doi.org/10.1038/nphys1740)

Paper PDF: `nphys1740.pdf` (LOCAL-ONLY, gitignored), pages in `pages/`.

## What is reproduced

The paper designs the experiment through the breather formalism: the
self-focusing NLSE (paper Eq. 1) has the Akhmediev-breather solution (paper
Eq. 2), whose a -> 1/2 limit is the Peregrine soliton (Eq. 3). The
experiment drives an a = 0.42 AB in a 900 m HNLF (beta_2 = -8.85e-27 s^2/m,
gamma = 0.01 W^-1 m^-1, lambda_0 = 1550 nm, loss 1 dB/km) and reads
Peregrine characteristics at the compression point (P_0 = 0.30 W,
xi = 2.5; z = 833 m sits exactly at Leff/L_NL = 2.54 of the fibre).

**A. Engine-vs-analytic AB evolution (paper Eqs. 1-2).** The validated
`photonics_helper.breathers.general_sfb` Akhmediev breather, seeded at
xi = -2.5 through the paper's SI scales (L_NL = 333.33 m, T_0 = 1.7176 ps),
is propagated 1667 m (2 x 2.5 L_NL, loss-free per the paper's NLSE
correspondence) by the GNLSE engine (beta_2 only; explicit split step
dz = 0.25 m per the CW-deck step guidance). The engine tracks the analytic
AB pointwise over the whole growth leg at snapshot peaks within 1e-2
(worst full-profile rel-L2 6% near the steep compression gradients, at
EQUAL peaks — engine lag, not physics).

**B. Max-compression peak ratio.** a = 0.42 deck: simulated
peak/background 8.166 vs the analytic `sfb_peak_ratio(0.42)` = 8.026
(+1.7%). Peregrine-limit deck (a = 0.495): 8.944 vs the ideal-PS limit 9
(asymptotic, per the paper's Fig. 2b).

**C. Compressed-train structure (Fig. 2a analogue).** Train period matches
the AB modulation period 2 pi/nu with nu = 2 sqrt(1-2a) to 1.6e-4; the
compressed FWHM matches the analytic |psi(0,tau)|^2 FWHM to <0.2% (0.964
vs 0.965 ps).

**D. Peregrine-limit profile (paper Eq. 3).** The a = 0.495 max-compression
profile overlays |psi_PS(0,tau)|^2 in shape to rel-L2 0.0025 (peak
normalized).

## Ground truth

- AB anchors: a = 0.25 -> peak/background 5.8284, a -> 1/2 -> 9 (the exact
  `general_sfb` family, previously validated through the Narhi and
  Kuznetsov-Ma reproductions).
- L_eff(1 dB/km, 900 m) = 846 m -> L_eff/L_NL = 2.54 ~= the paper's
  xi = 2.5 deck: the experimental window ends exactly at max compression.

## Outcome

```
growth-leg max rel-L2 = 0.061        (peak ratios within 2% everywhere)
peak ratio (a=0.42):  8.166 vs 8.026 theory (+1.7%, tol 15%)
train period rel err = 1.6e-4        (tol 5%)
compressed FWHM:      0.964 ps vs analytic 0.965 ps (< 0.2%)
Peregrine-limit ratio = 8.944        (theory 9)
PS-profile shape L2   = 0.00246      (tol 10%)
```

Figure: `peregrine_soliton.png` (deck evolution vs analytic AB, snapshot at
xi = +2.5; max-compression train; Peregrine-limit profile overlay).

## Findings

1. **Real-cos eigenmode mismatch.** The paper's input field
   A(0,T) = sqrt(P0)(1 + a_mod cos(w_mod T)) is not the exact AB eigenmode;
   the mismatch grows into a SECOND breather recurrence cycle at
   xi ~ +1.5–2.3 (recorded, not asserted — the paper prints only one AB
   period).
2. **Eq. (2) typesetting.** The printed mid-term ("i sqrt(2a) cos(2 Om
   tau)") is ambiguous; the standard Akhmediev–Korneev numerator
   (2(1-2a)cosh(b xi) + i b sinh(b xi) over sqrt(2a)cos(nu tau) - cosh(b
   xi)) is the only candidate realizing the anchors 5.8284 / 9, and it is
   what the library's `general_sfb` implements.

## ISSUES

- **241 GHz vs 74 GHz.** The Methods' stated omega_mod = 241 GHz for a =
  0.42 is not reconcilable with the paper's own omega_c^2 = 4 gamma P0 /
  |beta_2| and nu = 2 sqrt(1-2a) (which gives 74.1 GHz for the AB temporal
  period). The reproduction asserts against the analytic breather, so the
  mismatch does not affect the asserts; flag for re-check against any
  supplementary sheet.
- **Adaptive-step trap (engine guidance).** The adaptive split-step
  heuristic is meaningless on this finite-background CW deck (>2 h without
  making progress before the explicit dz = 0.25 m switch); supply
  `step_size=` explicitly for CW/MI decks (SplitStepEngine docstring rule).
- Full validate ~3 min on this box.
