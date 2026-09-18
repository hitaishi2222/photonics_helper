# Reproduction — Textbook SHG efficiency and QPM recovery

**Reference:** R. W. Boyd, *Nonlinear Optics* (3rd ed.), Ch. 2 (coupled-wave
SHG); M. M. Fejer, G. A. Magel, D. H. Jundt & R. L. Byer, *IEEE J. Quantum
Electron.* **28**, 2631 (1992) · [10.1109/3.159513](https://doi.org/10.1109/3.159513)
(quasi-phase-matching).
**Module stack:** `photonics_helper.chi2` (RK4IP coupled-wave solver).

## Result reproduced

For second-harmonic generation with pump depletion, at perfect phase matching
(`Δk = 0`) the exact conversion efficiency is

```
eta(L) = |A_sh(L)|² / |A_f(0)|² = tanh²(kappa L),   kappa = sigma sqrt(P0),
```

where `sigma` is the effective coupling, `P0` the input fundamental power, and
`L` the interaction length. For a deliberately mismatched interaction
(`Δk ≠ 0`), first-order quasi-phase-matching with poling period
`Lambda = 2 pi / |Δk|` recovers the same curve with the square-wave grating's
fundamental harmonic, i.e. `eta = tanh²((2/pi) kappa L)`.

## Ground truth

The closed forms `tanh²(kappa L)` and `tanh²((2/pi) kappa L)` evaluated
independently of the solver, with
`sigma = (2 omega d_eff / (n c)) sqrt(2 / (n c eps0 A_eff))`.

## Outcome

```
kappa L=0.25: eta=0.059985 (tanh^2 0.059985, err 5.55e-15)
kappa L=0.50: eta=0.213552 (tanh^2 0.213552, err 1.95e-15)
kappa L=1.00: eta=0.580026 (tanh^2 0.580026, err 1.15e-15)
kappa L=1.50: eta=0.819293 (tanh^2 0.819293, err 4.47e-15)
kappa L=2.00: eta=0.929349 (tanh^2 0.929349, err 7.88e-15)
QPM: eta_off=2.059e-05, eta_on=0.316132 (tanh^2((2/pi)kL) 0.316512, err 1.20e-03)
```

Figure: `shg_efficiency.png` (left: `Δk = 0` solver vs `tanh²(κL)`; right:
QPM recovery vs `tanh²((2/π)κL)` and the off-QPM baseline).

Parameters: λ = 1550 nm, `d_eff` = 10 pm/V, n = 2.0, `A_eff` = 1 µm²,
`P0` = 100 mW (see `parameters.json`).

## Findings

- The solver reproduces `tanh²(κL)` to **machine precision** (≤ 8e-15) at every
  sampled `κL`, confirming the coupled-wave bookkeeping, the RK4IP
  interaction-picture substitution, and the `|A|² = power` normalization.
- The QPM recovery matches `tanh²((2/π)κL)` to **0.12 %**. The residual is the
  square-wave grating's third harmonic (a physical effect, not a numerical
  one): increasing the step count from 4 000 to 32 000 changes `eta_on` by
  < 2e-7, while the offset from the first-harmonic formula stays at 1.2e-3.
- Off-QPM efficiency over the same length is `2e-5`, i.e. the poling improves
  the conversion by more than four orders of magnitude — the expected QPM
  signature.

## ISSUES / limitations

- Scalar envelopes only: no group-velocity mismatch, dispersion, walk-off or
  loss. The `2/π` QPM factor assumes a 50 % duty-cycle square grating and
  low-to-moderate conversion; at `κL ≳ 2` higher grating harmonics shift the
  curve by several percent.
- The coupling assumes a single refractive index for fundamental and
  second harmonic (`n₁ = n₂`); the non-dispersive approximation is stated in
  `shg_coupling`.
