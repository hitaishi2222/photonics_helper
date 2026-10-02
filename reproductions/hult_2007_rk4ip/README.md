# Hult 2007 (JLT 25, 3770) — RK4IP validation

> **Status — both decks ✅ PASS.**
> Deck A (soliton): slope −3.97, best ε = 5.0 × 10⁻⁶ (∼10 s).
> Deck B (SCG): Raman red-shift 199.6 nm (soliton at 1076 nm), dispersive
> wave at 576 nm (16 % of the peak), 24 temporal peaks, −20 dB span
> 551–1179 nm; convergence ladder slope −3.86 to ε = 2.9 × 10⁻¹² (∼2 min).
> Run with `--fast` to skip deck B's ladder.

**Reference.** J. Hult, "A Fourth-Order Runge-Kutta in the Interaction Picture
Method for Simulating Supercontinuum Generation in Optical Fibers", *J.
Lightwave Technol.* **25**(12), 3770 (2007), doi:10.1109/JLT.2007.909373.
Local PDF and `pages/p-*.png` are gitignored.

## What is reproduced

| Claim | Deck | Assertion | Measured |
|-------|------|-----------|----------|
| 4th-order RK4IP convergence for the N=2 soliton (Fig. 1) | A | slope −4 ± 0.35; ε < 2 × 10⁻⁵ | **−3.97**; 5.0 × 10⁻⁶ |
| SCG fission (Fig. 2) | B | ≥ 2 temporal peaks | **24** |
| Raman self-frequency shift of the ejected soliton | B | red-shift > 50 nm | **199.6 nm** (876 → 1076 nm) |
| Blue-side dispersive wave (Fig. 2a/b) | B | DW < 0.95 λ₀, > 1 % of peak | **576 nm**, 16.4 % |
| Supercontinuum span | B | ratio > 1.2 | **2.14** (551–1179 nm, −20 dB) |
| 4th-order convergence on an SCG deck (Fig. 3) | B | slope −4 ± 0.35; ε < 10⁻¹⁰ | **−3.86**; 2.9 × 10⁻¹² |

Deck B's soliton order is N = 5.33 from the paper's own β₂, γ and P₀ (the
paper says "around five"), z_sol = 9.93 cm — the fission onset is inside the
paper's 10 cm, as intended.

## Bugs this reproduction found (both were load-bearing)

1. **Delayed Raman switched off** — `heidt_adaptive.GNLSEOperator.__post_init__`
   stored `h * dt`, the *time-domain* array, in the field `_h_R` that
   `nl_field`/`_P_of` multiply against a spectrum
   (`ifft(fft(I) * _h_R)`). The convolution therefore ran against a smooth
   broadband kernel instead of a causal `h_R` spectrum, reducing the delayed
   Raman term to a near-instantaneous `(1−f_R)+f_R` drive: the
   self-frequency shift was **3 nm instead of 200 nm**, and the result was
   bit-identical with the Raman response swapped for anything else. Fixed to
   `self._h_R = self.grid.fft(h)`, matching `h_R_fft = grid.fft(h_R)` in
   `photonics_helper/gnlse.py::_raman_polarization`.

2. **β coefficients left in ps^k/m** — the Taylor arrays were converted with
   `10.0 ** (-12 * np.arange(len(betas)))`, whose first entry is 10⁰, so
   β₂ entered the operator as −1.276 × 10⁻² s²/m instead of
   −1.276 × 10⁻²⁶ s²/m — **10²⁴ too large**. The "SCG" was then a purely
   dispersive field that smeared into a low pedestal (the same failure mode
   as `ISSUES.md` #14 symptom 2 in the Heidt deck). Fixed to
   `arange(2, 2 + len(betas))`.

Both bugs are shared with `reproductions/heidt_2009_adaptive_step/`,
which imports this module.

## Recorded deviations

- **Self-steepening off on deck B.** Forced by the operator, not by
  convenience. The first-order shock term `iγτ ∂_t(A·P)` is an *unbounded*
  frequency-domain multiplier, and inside the explicit interaction-picture RK
  its round-off seed at the extreme FFT bin grows geometrically. Measured on
  this deck (8192 bins, 0.1 m, 256 → 4096 steps): `τΩ_max = 0.36` and `0.18`
  both NaN, `0.09` stable — the blow-up is step-size independent, and the
  bound is unsatisfiable, because `Ω_max = π/dt` means a stable grid needs
  `dt ≳ 14 fs` (≈ 2 samples per pulse FWHM). The library engine survives the
  same deck only because it never puts the shock inside the RK: it applies
  the exact nonlinear phase in Strang half-steps and advances just the shock
  correction with a substepped frequency-domain RK4
  (`SplitStepEngine._nonlinear_step`). Since self-steepening's role here is
  to *oppose* the Raman red-shift, leaving it off makes the 199.6 nm shift an
  upper bound on the paper's value, not a weaker claim. Cf. `ISSUES.md` #1.
- **Deck A window 20 ps** (paper: 2 ps) to reduce spectral leakage: the N = 2
  soliton's exact recurrence loses intensity at the grid edges. The floor
  settles at ε ≈ 5 × 10⁻⁶; the pre-floor ladder measures −3.97.
- **Deck A invariant** is `energy`, `photon` being unusable on the deck-B grid.
- **Raman response** is the house two-exponential silica model
  (τ₁ = 12.2 fs, τ₂ = 32 fs, f_R = 0.18) rather than the paper's
  Hollenbeck–Cantrell modal sum. With the fix above this is demonstrably
  *not* the limiting factor — the shift is already 13× the assertion.
- **Deck B convergence ladder runs on a 2 cm section**, not the full 10 cm.
  The five-soliton cascade is chaotically sensitive on that scale: per
  doubling at 0.1 m the error goes 1.4e-2, 1.0e-2, 2.3e-3, 1.5e-3, 2.6e-4,
  2.8e-5, 9.1e-7, 3.5e-8 — no single fitted order. At 2 cm (past fission
  onset) it is clean 4th order, 4.7e-5 → 2.9e-12 over 160 → 20480 steps.
  Convergence order is a statement about the scheme, so this measures the
  paper's claim without the chaos; the physics deck still runs the paper's
  full 10 cm.

## Run

```bash
python reproductions/hult_2007_rk4ip/reproduce.py         # full  (~2 min)
python reproductions/hult_2007_rk4ip/reproduce.py --fast  # deck A (~10 s)
```
