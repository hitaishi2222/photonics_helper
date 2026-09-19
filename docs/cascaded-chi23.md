# Cascaded χ⁽²⁾–χ⁽³⁾ coupled-wave solver

`photonics_helper.chi2.solve_cascaded_shg` extends the degenerate SHG
integrator with the bulk χ⁽³⁾ Kerr terms:

```
dA_f/dz  = i σ A_SH A_f* + iγ_f |A_f|² A_f + iγ_cross |A_SH|² A_f − (α_f/2) A_f
dA_SH/dz = i σ A_f² − iΔk A_SH + iγ_sh |A_SH|² A_SH + iγ_cross |A_f|² A_SH
```

Limit contracts (all regression-tested in `tests/test_cascaded_chi23.py`):

- pure quadratic (`gamma_f=0`) equals `solve_shg` **exactly** — all existing
  χ⁽²⁾ reproductions unaffected;
- pure Kerr (`sigma=0`): the fundamental solves the scalar SPM equation —
  analytic `exp(iγ_f P₀ L)` phase exactly (the SH stays empty);
- **cascaded-Kerr limit:** at large phase mismatch (|Δk| ≫ σ√P₀) the
  quadratic coupling acts like an effective Kerr coefficient
  `γ_φ = σ²P₀/Δk` (Epstein; Saltiel et al.; Agrawal §10.5) — the solver
  recovers the combined `(γ_f + σ²P₀/Δk)` fundamental phase <5%.
- XPM defaults to the `2/3` degenerate linearly-polarized mode-pair factor
  used by the vector GNLSE (:mod:`photonics_helper.vector_gnlse`);
  non-degenerate waveguide geometries pass the mode overlap explicitly.
- QPM poling and loss work exactly as in :func:`solve_shg`; the χ³ terms
  are unaffected by the poling.

# Inverse-design layer (Phase 4 item 5, v1)

`photonics_helper.inverse_design` wraps deterministic least-squares
identification and design over the exact forward solvers:

- `fit_two_wave(z_samples, ratios)` — identifies the **well-posed**
  invariants of a χ⁽²⁾ SHG run from measured `η(z)` data:
  `κ = σ√P₀` and the phase mismatch `Δk` (validated to recover κ and
  |Δk| exactly on synthetic data). The individual `(σ, P₀)` pair inside
  κ is a **degenerate direction** of η(z) — the reported `sigma_P0`
  drifts with the optimizer basin while η(z) stays flat.
  Multi-start least squares with parameter scaling; failure reports
  loudly instead of returning a basin value silently.
- `design_efficiency(target, P0, sigma, …)` — bounded bisection of
  `solve_shg` for the device *length* achieving a target efficiency;
  exact against the analytic `tanh²(κL)` design formula
  (`L = atanh√η / κ`) and fails loudly when the target saturates
  beyond reach in the bracket.

Scope: the PINN/differentiable direction shares these forward calls;
training-time autodiff is declared future work.
