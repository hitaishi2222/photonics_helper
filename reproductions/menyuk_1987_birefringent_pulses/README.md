# Reproduced — Nonlinear pulse propagation in birefringent optical fibers
# (Menyuk 1987)

**Reference.** C. R. Menyuk, *IEEE J. Quantum Electron.* **QE-23**, 174 (1987),
**doi:10.1109/JQE.1987.1073308**. PDF + rendered pages: local-only.

## About the article — the physics in a nutshell

A "single-mode" optical fiber still carries the one transverse spatial mode
in **two polarization axes**. In a perfectly round fiber the two would behave
identically; in a real fiber the axes are **birefringent** — they see
slightly different refractive indices, so the two polarization components of
a pulse differ in two fundamental ways:

- **Differential group delay** (β₁ₓ ≠ β₁ᵧ): one polarization component
  drifts through the other along the fiber ("walk-off"). Left alone, a
  pulse injected into both axes arrives as *two* time-shifted pulses —
  "linear pulse splitting".
- **Phase birefringence** (β₀ₓ ≠ β₀ᵧ): the axes are statically detuned in
  phase, so the *nonlinear* polarization of one axis beats coherently
  against the other at the mismatch frequency along the fiber.

The paper opens with this observation and derives, from the full Kerr
nonlinearity E = ((χ/3)(E·E)E) vector polarization, the closed-form coupled
envelope equations. Three coupled effects appear:

1. **Linear birefringence** — different global group velocities and a
   constant phase mismatch Δβ between the axes.
2. **SPM + XPM** — each envelope sees self-phase modulation with
   coefficient 1 *and* cross-phase modulation from the other axis with the
   coefficient **⅔** of the degenerate linearly-polarized mode pair. This
   `2/3` factor is the hallmark of the model.
3. **Coherent polarization four-wave mixing** — the "oscillating"
   terms `½χ V²U*·e^{−2iΔβz}` (paper's `⅓v²u*e^{−iRδξ}` in normalized
   form): energy that can transfer coherently between the axes, oscillating
   at 2Δβ along the fiber. Menyuk points out a symmetry: without these terms
   the equations would not be invariant to a rotation of the polarization
   state in the k = l limit, so they are necessary *in general* — but
   because `Rδξ` accumulates far faster than the soliton evolution, they
   **average out** for all practical fibers.

The Kerr coupling law is simple: each axis's amplitude sees the *sum* of
SPM (coefficient 1) and XPM from the other axis with the **⅔**
coefficient of the degenerate linearly-polarized mode pair, plus a coherent
mixing term ±⅓ that exchanges energy between the axes at the birefringence
beat. Because the beat phase `2Δβ·z` accumulates far faster than the
soliton evolves in any practical fiber (R·δ ≫ 1), Menyuk drops it
throughout the soliton analysis but is explicit that this is a
limiting case, not a generality — in very-low-birefringence fiber the
coherent term *must* be kept, and it is the part of the model that cannot
come from a scalar NLSE argument.

The paper then derives three families of results that this reproduction
targets:

### The normalized model (Eqs. (7)–(8)), what the letters mean

Paper's scaling with the Mollenauer–Stolen soliton normalization:

- Retarded time `s = (t − z/ū_g)/t₀` in the **mean group frame of the two
  axes**, with `ū_g⁻¹ = (k′ + l′)/2`; soliton width `t₀ = 0.568 τ`
  (sech convention, τ = intensity FWHM);
- Distance in soliton periods `ξ = πz/(2z₀)`,
  **z₀ = π²cτ₀²/(D(λ)λ₀)** from the dispersion D(λ) = dβ₁/dλ;
- Envelope `u = √(χ/2)·U` — folds the Kerr coefficient so the normalized
  cubic *diagonal* coefficient is exactly 1;
- **δ = (k′ − l′)·t₀/(2|k″|)** — normalized group-delay walk-off between
  the axes (the two pulses drift −δ and +δ per unit ξ in the mean frame);
- **R = 8πc·t₀/(D(λ)·λ₀)** — large (~10⁵–10⁶), setting the phase scale of
  the FWM mismatch: the coherent terms carry e^{∓iRδξ}.

The equations (8a)/(8b) exactly as printed:

```text
i(u_ξ + δu_s) + ½u_ss + (|u|² + ⅔|v|²)u   + ⅓v²u*e^{−iRδξ} = 0   (8a)
i(v_ξ − δv_s) + ½v_ss + (⅔|u|² + |v|²)v   + ⅓u²v*e^{+iRδξ} = 0   (8b)
```

The `±δ·∂/∂s` terms are the ± walk-off of the two axes about the mean
frame; the ⅓-coherent terms couple the two axes at the birefringence phase.

### The paper's exact results

1. **Eq. (9), single-polarization soliton** (v = 0): `u =
   exp[i½(1+δ²)ξ − iδs]·sech(s)`. A scalar soliton of peak P₁ = |β₂|/(γ t₀²)
   boosted by chirp −iδs — the chirp translates the pulse by
   `δ|β₂|z/t₀` (a Galilean boost; comoving phase ½(1+δ²)·ξ),
   stationary in the mean frame.
2. **Eq. (10), the locked two-polarization soliton**:
   `u = v = √(3/5)·sech(s)·exp[i½(1+δ²)ξ ∓ iδs]`.
   * Both axes carry 3/5 of the scalar peak: total intensity 6/5 of a
     scalar soliton of the same width — because each axis, summing
     SPM+XPM, sees 1 + ⅔ = 5/3 of its own nonlinearity, the pair must
     carry 6/5 aggregate power for one "soliton wave" total;
   * The ±chirps are the walk-off-compensating boost: the pair is
     stationary and *coincident in time* in the mean frame while the walk
     of the two axes is ±δ·dispersion-units. Nonlinear locking resists
     linear birefringence: the ±δ decomposes into a self-double-boosted
     pair of coincident chirped solitons.
3. **Regime catalogue (τ = 5 ps, λ₀ = 1.55 µm)**: z₀ = 0.71 km; from
   Δn ∈ [5×10⁻⁹, 8×10⁻⁴] the total δ range is ≈ 1.3×10⁻³ – 200,
   concentrated in **0.3–3.0**; R = 1.4×10⁶ ⇒ Rδ ≫ 1 for every δ in that
   range ⇒ the FWM terms can be neglected throughout. Fixed numbers: the
   fundamental locked state of a 5-ps pulse survives walk-offs that would
   naively split it over tens of kilometers.
4. **Short pulses (τ = 250 fs)**: z₀ = 1.8 m, δ ∈ [7×10⁻⁵, 0.10], and
   **R = 700**, so Rδ ≤ 1 only for **δ ≤ 1.4×10⁻³** — below the typical
   PM range but well inside the allowed range of Δn; the paper says that
   in this very low birefringence regime the oscillating terms "will play
   a role", but "will be ignored in most cases".
5. **Splitting vs locking (p. 175)**, the practical conclusion of the
   letter:
   - *Linear criterion*: if the nonlinearity is neglected, the pair splits
     linearly over L = 20 km when the walk-off separation Δβ₁·L reaches
     ~2 × FWHM, which happens at **δ ≳ 0.04** — "which is typically the
     case". This is a *kinematic* (distance-based) criterion.
   - *Nonlinear criterion*: "we expect that when δ ≤ 1, the nonlinearity
     will stabilize the pulse against splitting under the influence of
     linear birefringence. Numerical solution of (8) supports this
     conclusion."
6. **Modulational instability of the CW background** (Eqs. (11)-(17)):
   for u₀, v₀ the instability growth rate is K₀ = −i[Ω²(u₀²+v₀²−Ω²/4)]¹½
   — the scalar MI law on the *sum* power — shifted in wavenumber by δ and
   β (= 5/9) and slightly lowered by βδ. Not covered by this reproduction.

## What we reproduce, exactly

All checks run in `reproduce.py::validate()`, mapped onto
`photonics_helper.vector_gnlse.VectorSplitStepEngine`, with every physical
number traced to a page of the paper. The key convention to read the engine
from the paper is one line:

```text
   launch (paper mean frame)                → engine (x-retarded frame)
   u = v = √(3/5·P₁)·sech(t/t₀)e^{∓iδt/t₀}   → same chirps,
                                                engine walkoff = +Δβ₁.
```

The engine's ± walk-off splits naturally between the two chirped pulses:
each chirp contributes exactly half the walk-off, so the coincident lock
common-drifts at δ|β₂|z/t₀ and carries phase ½(1+δ²)·xi. Details are in the
"Conventions note" section below.

**Status** `[x]` — full reproduction with all measured metrics (see the
outcome table below); runtime ~14 s including figures.

## Reproduced checks (all in the paper's own scaling)

- **Instrument anchors** (paper Sec. II scaling map):
  - `t₀ = τ/1.763` (paper's t₀ = 0.568 τ, sech convention);
  - **β₂ = −π t₀²/(2 z₀)** from the paper's stated soliton periods
    z₀ = 0.71 km (τ = 5 ps), z₀ = 1.8 m (τ = 250 fs); both → D(1.55 µm) ≈
    **14 ps/(nm·km)** — a standard 1987 dispersion-shifted telecom fiber
    (cross-checked in `validate()` to ±2). The printed D(λ) = 6.5×10⁻³ s/m
    (p. 175) has ambiguous digitized units and is not used numerically; z₀
    is the paper's stated value.
  - γ = n₂ω₀/(cA_eff) = 2.027×10⁻³ W⁻¹ m⁻¹ (same profile as the Manakov
    reproduction: n₂ = 2.6×10⁻²⁰ m²/W, A_eff = 52 µm²);
  - P₁ = |β₂|/(γ t₀²) = **1.092 W** (τ = 5 ps) and **430.6 W** (τ = 250 fs);
    Eq. (10) per axis: P_axis = 3/5·P₁ = 0.655 W / 258.4 W
    (paper: total intensity 6/5 of the scalar soliton);
  - Walk-off Δβ₁ = 2|β₂|δ/t₀ (engine `walkoff`, β₁ᵧ − β₁ₓ in SI s/m, the
    relative drift rate of the two axes); anchor: δ = 0.3, τ = 5 ps →
    **3.76 fs/m = 3.76 ps/km** — a plausible PM-fiber PMD.
  - FWM mismatch: the paper's phase `Rδξ = Rδπz/(2 z₀)` enters the engine as
    `e^{±2i·Δβ·z}` with **Δβ = Rδπ/(4 z₀)**.

### Checks and measured outcomes

| Check | Ground truth | Measured |
|---|---|---|
| Eq. (9), δ=0 stationary filament, 5 z₀ | sech, P₁ = 1.092 W | shape L2 **1.2×10⁻⁶**, peak −1.4×10⁻⁶ |
| Eq. (9), δ=0.5 boosted filament | translation δ·&#124;β₂&#124;·z/t₀ = 11.1 ps; phase rate ½(1+δ²)ξ | center err **1.3×10⁻⁶ ps**; shape L2 1.3×10⁻⁶; phase residual 8×10⁻⁴ rad |
| Eq. (10) vector lock, τ = 5 ps, δ = 1.0, 5 z₀, engine walk-off 12.55 fs/m | coincident, common drift δ&#124;β₂&#124;·z/t₀ = 22.3 ps, phase ½(1+δ²)ξ, per-axis 0.655 W | coincidence err **3.4×10⁻⁶ ps**; shape L2 7×10⁻⁶; phase between axes 6×10⁻³ rad; phase-rate residual 3×10⁻³ rad |
| Eq. (10) *with* FWM (coherent), τ = 250 fs, δ = 0.1, Rδ = 70, FWM beat 10.3 cm, 5 z₀ | filament survives with O(1/Rδ) perturbation | shape L2 **3.9–4.9×10⁻³** ✓ O(1/70); peak ±1.4–1.6 % |
| Very small δ: Rδ = 0.7 (δ = 10⁻³), coherent vs incoherent | paper: the oscillating terms "will play a role in fibers with very low birefringence" | both locks coincident (< 7×10⁻⁵ ps); fields differ **83 %** — the FWM term decisively active |
| (i) Nonlinear stabilization (paper: "when δ ≤ 1 … the nonlinearity will stabilize … Numerical solution of (8) supports this") | δ ∈ [0.01, 1.0], fixed-power (3/5 P₁ per axis) coincident chirp-less launches over 20 km | x–y overlap > 0.87 through δ = 0.5; nonlinear split first appears at δ ≈ 0.56–0.61 (overlap < 0.5) — at the paper's quoted δ ≤ 1 border |
| (ii) Linear splitting (paper: "if nonlinearity is neglected, the pulse will split linearly over 20 km when δ ≳ 0.04") | γ = 0 run at δ = 0.04 | Δβ₁·L = **10.04 ps = 2 × FWHM** ✓ — the paper's 0.04 threshold reproduced to 0.4 % |

The coherent-coupling checks also extend the validated coherent-RK4
contract of `tests/test_vector_gnlse.py` from CW fields to pulsed soliton
fields, with the paper's FWM mismatch `Δβ = Rδπ/(4z₀)` step-resolved by the
substep rule (`2Δβ·dz` ≲ 0.05 rad per substep).

## Engine bug found and fixed (REPORT)

The reproduction exposed **two real engine bugs** in the coherent-coupling
path of `vector_gnlse.VectorSplitStepEngine` — both invisible to the
existing CW unit test (which runs tiny powers / small Δβ per step):

1. **Missing second Strang half of the diagonal** —
   `_coupled_nonlinear_step` (coherent branch) applied only
   `exp(+½iγPx·dz)` before the FWM substeps and returned without the
   completing half step. Coherent runs therefore evolved with *half* the
   nonlinearity: the vector-soliton launch decayed monotonically (peak
   258 → 98 W over 5 z₀) in every coherent run, **independently of Δβ down
   to 10⁵ rad/m** — the smoking gun. Fixed by completing the Strang
   composition.
2. **Substep count ignored the mismatch rate** — `n_sub` was sized by the
   mixing rate `γ|Ax||Ay|/3` alone; the frozen `e^{∓2iΔβ·z_mid}` phase
   factors also need `2Δβ·dz ≲ 0.05 rad` per substep, which is now included
   in `n_sub` (cap raised 200 → 2000). At the paper's Δβ = 30.5 rad/m the
   old rule left a 1.2 rad/step oscillation unresolved.

## Conventions note (paper frame ↔ engine frame)

The paper's exact locked soliton (10) is stationary in the *mean* group
frame with opposite chirps ∓iδs. The engine propagates in the
x-retarded frame, where the lock launches as
u, v = √(3/5·P₁)·sech(t/T₀)·e^{∓iδt/T₀} with `walkoff = +Δβ₁` on the y
channel: each chirp contributes exactly half of the walk-off, so the lock
stays coincident, common-drifts at δ|β₂|z/T₀, and carries the phase
½(1+δ²)·ξ (all three verified to the numbers in the table). Chirp signs
are pinned by the passing tests: chirp_x = +δ, walkoff positive on y.

## Files

- `parameters.json` — inputs, conventions and derived/measured metrics,
  with derivation notes (all to a page number of the paper).
- `reproduce.py` — Checks 1–4 (`validate()` returns all metrics), the
  locking/splitting phase diagram (Check 5), and two figures:
  `fig_eq9_eq10_filaments.png`, `fig_lock_split_map.png`.
- `jqe.1987.1073308.pdf`, `pages/` — source (local only, do not commit).
