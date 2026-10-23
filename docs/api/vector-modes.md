---
quadrant: reference
---

# Exact vector modes of a circular fibre

Closed-form vector eigenmodes of a circular step-index guide: `HE`, `EH`,
`TE`, and `TM`, any order. The solver uses no mesh, no weakly-guiding
approximation, and no materials. It is the symmetry-preserving reference for
OAM and polarisation splitting (gap A1).

```python
from photonics_helper.vector_modes import vector_modes

modes = vector_modes(
    n_core=2.65, n_clad=1.0, radius_um=0.4, wavelength_um=2.0,
    kinds=["HE11", "TE01", "TM01", "HE21", "EH11", "HE31", "EH21"],
)
for m in modes:
    print(m.kind, m.ell, m.u, m.n_eff, m.beta)   # beta in rad/m
```

## Contract

`kind` is `<family><order><m>`, for example `HE21`, `EH11`, or `TE01`. For
`HE_ν` and `EH_ν` the transverse azimuthal order is `ℓ = ν - 1` and
`ℓ = ν + 1`. `TE` and `TM` carry `ℓ = 1` and equation order 0.

`kinds=None` enumerates guided `HE` and `EH` (orders 1 to 4, radial 1 to 3)
plus `TE` and `TM` (radial 1 to 3).

An explicitly requested cut-off mode is returned with NaN `u`, `n_eff`, and
`β`. The default enumeration omits cut-off candidates.

The index relation is `n_eff² = n_core² - (n_core² - n_clad²)(u/V)²`, and
`β = (2π/λ)n_eff` in rad/m. Geometry and wavelength are in µm, the convention
of the reproduction this code comes from. Dispersion is the caller's problem.

## Method

The solver evaluates the Okamoto hybrid characteristic equation (Eqs.
3.97 to 3.99) with a full, pole-aware sign-change scan. It selects the root
nearest the scalar LP root of the mode's own group. That fixes the failure
mode of the ±0.5-around-the-LP-guess bracket, which returns NaN for `HE21`
and `TM01` on a strongly guiding guide. In the weakly-guiding limit the roots
reduce to `lp_roots`: tested at Δn = 1e-4 to better than 1e-4.

See the [mode-analysis bridge](mode-analysis.md) for projecting a launched
field onto solved mode exports.

::: photonics_helper.vector_modes
