# GNLSE physics: shock invariant, multi-mode Raman, free carriers, validation

This page documents four pieces of solver physics added in 0.1.9, each stated
against the literature and validated by an independent analytic or invariant
check. Docstrings remain the source of truth; this page gives the derivations
and *honest caveats* in one place.

## 1. Interaction-picture self-steepening (RK4IP)

The nonlinear GNLSE term (Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135
(2006), Eq. (3); Blow & Wood, *IEEE J. Quantum Electron.* **25**, 2665 (1989))
is

```text
∂A/∂z |_NL = iγ (1 + (i/ω₀)∂_t) [ A · P_NL ] ,   P_NL = (1−f_R)|A|² + f_R (h_R ⊛ |A|²)
```

The step integrates the exactly-integrable phase `exp(iγP_NL Δz)` in two
symmetric half-steps (Strang splitting) and advances only the much smaller
shock correction `iγτ_shock ∂_t(A·P_NL)` with frequency-domain RK4 — the
interaction-picture (RK4IP) idea of Hult, *J. Lightwave Technol.* **25**, 3770
(2007) and Hochbruck & Ostermann, *Acta Numerica* **19**, 209 (2010). No
Runge–Kutta stage ever integrates the stiff nonlinear phase, and **no
spectral bin is clamped**.

**Photon-number balance — an honest caveat.** A draft of this change claimed
the shock term is a total time derivative, making the photon number
`∫|A|²dt` an exact invariant. That identity is **false in general**: with a
real response,

```text
d/dz ∫|A|² dt = −2γτ_shock ∫ P_NL · Im(A* ∂_t A) dt ,
```

which vanishes only when the field evolves under a *time-independent* `P_NL`
(pure SPM with a frozen drive — then the step is exactly conservative,
verified to machine precision). In general the residual is a **physical**
error of the first-order `ω/ω₀` expansion itself, `O(τ_shock·Ω_max)` at the
band edge: a fully-resolved (heavily substepped) reference of the *same flow*
drifts identically on fissioned states, while the RK4IP step matches that
reference to ≈1e-5 in the field. No integrator can conserve what the model
does not conserve; the historical `max(1 + Ω·τ_shock, 0)` clamp, by contrast,
was a *numerical* corruption of valid bins and is removed.

### Spectral validity guard

The factor `1 + Ω·τ_shock` is the first-order form of `ω/ω₀` and is only
meaningful for channels with positive absolute frequency, `ω = ω₀ + Ω > 0`.
When self-steepening is enabled the engine therefore requires
**`Ω_max < ω₀`** and raises a `ValueError` naming the grid values (`N`,
`Tmax`), the ratio `Ω_max/ω₀`, and the remedy (increase `Tmax`, reduce `N`,
or disable self-steepening). The grid must resolve only physical frequencies;
nothing is silently discarded.

References — Blow & Wood (1989); Agrawal, *Nonlinear Fiber Optics*, 5th ed.,
§2.3; Hult (2007).

## 2. Multi-phonon Raman response

`PhononResponse.h_R(t)` builds a causal, unit-integral delayed response from
the material's phonon modes (a compound-glass / multi-vibrational-mode model,
Hollenbeck & Cantrell, *JOSA B* **19**, 2886 (2002)):

```text
h_R(t) = Z⁻¹ Σ_i w_i e^{−t/τ_i} sin(ω_i t) θ(t),   τ_i = 2/γ_i,   ω_i = 2πc·ν̃_i
```

with `γ_i` the angular-frequency FWHM of mode *i* (Lorentzian ↔ damped
oscillator relation `Δω = 2/τ`, Agrawal §2.3.2) and `Z` fixed by
`∫₀^∞ h_R dt = 1`. The solver's Raman dispatch reads `fR` and `h_R(t)` from
whatever response object the `FiberProfile` carries — the single-mode silica
Blow–Wood `RamanResponse` or a multi-mode `PhononResponse` — so crystalline
materials (LiNbO₃, YAG, KTP, …) can be propagated with their real multi-mode
response. Single-mode behaviour is unchanged.

## 3. Time-resolved TPA / free carriers (opt-in)

The legacy TPA model spatially averages the carrier density (one scalar `U`
per step). SplitStepEngine additionally offers
`include_free_carriers=True`, which resolves the carrier population
**per retarded-time sample** (Soref & Bennett, *IEEE J. Quantum Electron.*
**23**, 123 (1987); Cowan, Rieger & Young, *Appl. Phys. Lett.* **82**, 1745
(2003); Yin & Agrawal, *Opt. Lett.* **32**, 2951 (2007)):

```text
|A|²(z+dz) = |A|² / (1 + β_TPA |A|² dz)      (exact TPA attenuation)
∂N/∂z = β_TPA |A|⁴ / (2 ħω₀) − N / (τ_c v)   (generation − recombination)
∂A/∂z|_FCA = −(σ_FCA/2) N A                  (free-carrier absorption)
```

each integrated exactly per sample within a step. The legacy
`include_tpa` path is untouched; both models are documented side by side.
Out of scope by design: free-carrier **refraction** (the `μ` index term),
carrier diffusion and drift — all listed as follow-ups.

## 4. Convergence and validation harness

`photonics_helper.gnlse_validation` (exported from the package root) is the
"demonstrate grid independence on *your* configuration" tool the original
code review called out as missing.

**Grid/step convergence** — `convergence_study(build_solver, refinements,
observables, tolerance, shared)` runs a caller-supplied GNLSE factory at
successively refined resolutions and returns, per caller-selected observable,
the value at each resolution, the relative change between successive
refinements, and a `converged` verdict at a caller tolerance (Sinkin,
Holzlöhner, Zweck & Menyuk, *J. Lightwave Technol.* **21**, 61 (2003);
Agrawal §2.4). Built-in observables: `peak_intensity`, `pulse_energy`,
`rms_bandwidth`, `rms_width`; pass a `{name: callable}` mapping for anything
else.

**Cited analytical checks** raise `ValidationFailure` when a closed form is
violated, so they can be wired into an application's own test suite:

| Check | Reference | Closed form |
|---|---|---|
| `check_spm` | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) | Fourier-integral spectrum + `N_peaks = ⌊φ_max/π⌋+1` |
| `check_mi` | Agrawal §5.1, Eq. 5.1.9 | `g(Ω)=\|β₂Ω\|√(Ω_c²−Ω²)`, `Ω_c²=4γP/\|β₂\|`, `g_max=2γP` (power-gain convention, propagated sideband intensity `∝ e^{gz}`) |
| `check_soliton` | Agrawal §5.2 | fundamental soliton returns after `z_sol = (π/2)L_D` |
| `check_gordon_ssfs` | Gordon, *Opt. Lett.* **11**, 662 (1986) | `dΩ/dz = −8\|β₂\|T_R/(15T₀⁴)`, `T_R = f_R ∫t·h_R dt` |

Example:

```python
from photonics_helper import convergence_study

report = convergence_study(
    build,                      # your factory → propagated GNLSESolver
    refinements=[
        {"N": 8192,  "Tmax_s": 8e-12, "num_steps": 1000},
        {"N": 16384, "Tmax_s": 8e-12, "num_steps": 2000},
        {"N": 32768, "Tmax_s": 8e-12, "num_steps": 2000},
    ],
    observables=["peak_intensity", "rms_bandwidth"],
    tolerance=2e-3,
)
assert report.converged, report.summary()
```

Regression coverage: `tests/test_gnlse_convergence.py`,
`tests/test_gnlse_unitarity.py` (shock integrator fidelity + the
constant-drive conservation limit), `tests/test_gnlse_phonon_raman.py`
(multi-mode response), `tests/test_gnlse_free_carrier.py` (analytic TPA
attenuation `I₀/(1+βI₀z)` and the legacy-path contract).
