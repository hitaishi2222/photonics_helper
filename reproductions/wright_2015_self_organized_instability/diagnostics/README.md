# Session diagnostics (2026-09, bisecting the missing engine-side STMI gain)

Run with `python diagnostics/<script>.py` from the repo root (`photonics_helper/`).
All numbers below are as measured in this session.

## What is solid

1. **Check A (analytic phase matching) stands.** `stmi_shift_thz(N)` vs the
digitized Fig. 3d circles: N=2 -5.2 %, N=4 -0.9 %, N=5 -2.8 % (<15 % tol).
2. **The previous check B was invalid.** The asserted `peak_bin_gain`
(113/297x) is an argmax over a chi2-noise floor: the bin drawn at the exact
analytic root of a full 1 m run gives out/in power ratio 0.896 and a
coherent-tone probe at the analytic root over 0.3 m gives amp-gain 0.998.
The engine spectrum is a noise floor; no resolved MI bump on this grid.

## State of the bisect (as of this session - open)

- Offsets sign scan (minus/plus/split/none, isotropic): NO parametric gain
anywhere, all band ratios ~1.0.
- Coherent conjugate-pair probe at the analytic root (rel phase scan):
amp out/in 0.97..1.03 over 0.3 m - NO growth (theory at resonance:
cosh(gamma_f P0 L) = 1.47).
- Measured z-oscillation of the pair (dense z-sampling): delta_eff ~ 10.4
rad/m at the analytic root - i.e. the ENGINE effectively sits FAR off its
own (claimed) resonance there.
- Hand derivation from the engine's operator algebra (`_linear_step`,
`_diagonal_phase`, `_fwm_rhs`, Strang halves): the pump channel self-phases
at the FULL gamma P0 (4.7 /m), while the sideband XPM diagonals are only
(2/3) gamma P0 (3.13 /m). The effective pair mismatch is therefore
    Xi = (2/3) gamma P0 - (beta2 Om^2 + 2*offset)      [rad/m]
(z-osc probe measured delta_eff ~ 10.4 at the old root, vs Xi(old) = 9.4 -
the gamma P0/3-vs-gamma_f P0 bookkeeping is not yet pinned down to the last
factor; the next step is a clean 2x2 eigen-map, see probe_map.py).
- Consequence: the reproduce's analytic condition
`0.5*sym - N*kappa + gamma_f*P0 = 0` is inconsistent with the engine's
effective mismatch; the corrected root is ~0.02 THz higher, still within the
Fig. 3d tolerance, but the engine-side gain check must be rebuilt on the
corrected condition - and the botched factor must first be confirmed.
- Also observed: with ch2 = 0 (spontaneous idler), the run EXPLODES
(|A2| ~ 1e240) - the FWM substep's adaptive-substep logic
(`_fwm_substep_count`, cap 200) is not safety-limited against RK4 blow-up
in strongly driven configurations.

## Session-2 result (2026-09-27, probe_eigengain.py, 12-min run, complete)

- [A] betas_taylor "sym error 1.6e5 rad/m" is a PROBE unit bug: the probe
  multiplies SI s^k/m betas by 1e12**k as if they were ps^k/m. The engine
  receives them with betas_unit='s^k/m' and normalizes internally. Not
  evidence against the Taylor fit.
- Corrected root confirmed analytically: f_new = 115.692 THz
  (0.5*sym = N*kappa + gamma*P0/3; +0.016 THz vs the old root),
  Xi(corrected root) = 0 by construction, g_model(root) = 3.133 /m.
- [C] BUT the engine still shows NO pair gain at the corrected root over
  L = 1 m: 2x2-map eigenvalues ~ 1e-15 (noise floor), mu ~ -33 /m, while
  the reduced model demands sinh(cL) growth, cosh(cL) = 11.5
  (c = (2/3) gamma P0 = 3.13 /m). [D] control at the old root behaves
  identically - on/off-resonance do not separate.
- CONSEQUENCE: the missing-gain bisect is NOT closed by the mismatch fix.
  Suspects narrow to the FWM creation arm in _fwm_rhs for the fully
  degenerate pair (factor/sign), or the Strang-split interaction with it.
  Next probe: run the same pair through _fwm_rhs in isolation (engine
  substep vs direct RK4 of the reduced pair ODE from an identical state).
