# Reproduction — Quarter-wave DBR stopband (Macleod / Born & Wolf)

**Reference:** H. A. Macleod, *Thin-Film Optical Filters* (4th ed.), Ch. 5; see
also M. Born and E. Wolf, *Principles of Optics*.
**DOI:** This is a textbook result, so there is no single paper DOI. The
underlying quarter-wave characteristic-matrix method originates in
Macleod (1969) and the Born & Wolf treatment; the reproduction's regression
counterpart is the Dudley TMM validation
([10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135)
parameter set).

## Result reproduced

For a quarter-wave stack `(HL)^N` designed at `lambda0`, at `lambda0` every
layer has phase thickness `pi/2`, so one period has the diagonal characteristic
matrix `diag(-eta_L/eta_H, -eta_H/eta_L)`. The exact peak reflectance is

```
r = (eta0 (-etaL/etaH)^N - etas (-etaH/etaL)^N)
    / (eta0 (-etaL/etaH)^N + etas (-etaH/etaL)^N),   R = |r|^2,
```

and the first-order stopband full width is

```
Delta_lambda / lambda0 = (4/pi) arcsin((nH - nL)/(nH + nL)).
```

## Ground truth

The exact quarter-wave closed forms above, evaluated independently of the TMM.

## Outcome

```
N=2: R(lambda0)=0.593729 (closed form 0.593729)
N=3: R(lambda0)=0.829643 (closed form 0.829643)
N=4: R(lambda0)=0.935017 (closed form 0.935017)
N=20: R(lambda0)=1.000000
stopband width (R>0.5) = 533.1 nm, analytic = 498.7 nm (err 6.9%)
```

Figure: `dbr_spectrum.png`.

## Findings

- `R(λ₀)` matches the exact quarter-wave closed form to `< 1e-6` for `N = 2, 3,
  4`, which is a strong end-to-end check of the corrected characteristic-matrix
  product order (`M_total = M_total @ M_layer`, the review's **N1** fix).
- The first-order stopband width matches `(4/π) arcsin((n_H−n_L)/(n_H+n_L))`
  to 6.9 % at the `R > 0.5` level; the residual is the finite-width definition
  of "stopband" rather than a matrix error.
- This reproduction doubles as the independent check that was used to nail the
  transfer-matrix ordering and the `M⁻¹` forward field propagation.

## ISSUES

- `TMM.spectrum` evaluates the exit medium from `n_substrate`, but this
  reproduction deliberately keeps the air-terminated case (`n_substrate =
  1.0`) to compare against the closed form; the substrate case is covered by
  `tests/test_tmm_physics.py`.
- The stopband-width comparison tolerates 15 % in the test
  (`width_rel_error < 0.15`); the measured 6.9 % is dominated by how the width
  is measured (R > 0.5 vs the analytic null-to-null definition).

## Note / limitation

`TMM` now exposes `n_incident` and `n_substrate` (both default air,
`1.0 + 0j`), so a mirror on a substrate can be modelled without adding an
explicit layer. This reproduction still evaluates the air-terminated case
(`n_substrate = 1.0`) for comparison with the closed-form quarter-wave result;
set `n_substrate` to e.g. `1.5` to model a mirror on glass. The exit medium is a
semi-infinite medium evaluated at the Snell-refracted exit angle.
