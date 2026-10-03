# Poletti & Horak 2008 — the multimode GNLSE overlap coefficients

**Reference:** F. Poletti and P. Horak, "Description of ultrashort pulse
propagation in multimode optical fibers", *J. Opt. Soc. Am. B* **25**(10), 1645
(2008) · [10.1364/JOSAB.25.001645](https://doi.org/10.1364/JOSAB.25.001645).
Local PDF: `JOSAB.25.001645.pdf`.

**What this paper is.** The extended (multimode) GNLSE itself — the
specification for this repository's `multimode_gnlse` engine: polarisation,
high-order dispersion, Kerr and Raman, self-steepening, and wavelength-dependent
mode coupling and nonlinear coefficients, all carried by the four-mode overlap
coefficients

```
Q^(1)_plmn = C/(N_p N_l N_m N_n) INT [F_p* . F_l ][F_m . F_n*] dx dy      (Eq. 7)
Q^(2)_plmn = C/(N_p N_l N_m N_n) INT [F_p* . F_l*][F_m . F_n ] dx dy
```

Sections 4 and 5 are the analytical core, and they are what this deck
reproduces: the symmetry properties of those coefficients (Eq. 16, 18, 19), the
real-valued-mode-function statement of Sec. 3, and the computational-complexity
argument built on how many coefficients survive.

**Deck:** `parameters.json` (all inputs, with page references) ·
`overlap.py` (mode solver + Eq. (7) integrals + rule checkers) ·
`reproduce.py` (`validate()`, ~10 s) → `poletti_2008_overlaps.png`.

```bash
python reproductions/poletti_2008_multimode/reproduce.py
```

## The mode set

The paper's Fig. 1 fibre: 6 µm core radius, NA = 0.17 at 1.5 µm, so
`V = 2πa·NA/λ = 4.2726`. Exactly three LP families are guided (LP31 cuts off at
`j_{2,1} = 5.52 > V`):

| mode | radial root `u` | azimuthal copies | polarisations | modes |
|---|---|---|---|---|
| LP01 | 1.934196 | `m = 0` | σ± | 2 |
| LP11 | 3.044173 | `m = ±1` | σ± | 4 |
| LP21 | 3.998326 | `m = ±2` | σ± | 4 |

Ten modes, hence `2 × 10⁴` coupling coefficients over the two types of Eq. (7) —
the count the paper quotes for "the first ten modes". The roots come from the
analytic step-index eigenvalue equation
`u J_{l-1}(u)/J_l(u) + w K_{l+1}(w)/K_l(w) = 2l`, and every profile is
normalised to unit power (`N_k = 1`), to 0.0 on this grid.

Both azimuthal conventions the paper uses are implemented and carry the same ten
modes in the same order:

- **helical** `F = R(r) e^{i m φ}` (Eq. 17) — the basis Eq. (18) is written in;
- **real** `F = R(r) cos(m φ)`, `R(r) sin(m φ)` — the physical LP basis, in
  which `F_l* = F_l`.

Nothing about the selection rules is hard-coded. The angular factor of Eq. (7)
is evaluated by quadrature (256 azimuthal points) and the polarisation dot
product `e_{σp}* . e_{σl} = δ_{σp σl}` is applied as the exact identity of the
circular basis it is. The rules are then *predictions* checked against what the
integral produces.

## What was reproduced

| # | Claim | Paper | Measured |
|---|---|---|---|
| 1 | Eq. (18), spatial selection rules, **both** types | rules as printed | 1360 of 10⁴ quadruples survive per type (= the combinatorial prediction, exactly); the largest `|Q|` among the 8640 forbidden quadruples is **7.8e-17** |
| 2 | Eq. (19), polarisation selection rules | `Q = 0` for `σp ≠ σl` or `σm ≠ σn` | 1360 → **340** survivors per type; the 1020 quadruples the rule discards carry `|Q|` up to **1.0**, so the rule does real work; the masked tensor is *exactly* zero outside the combined rule (7500 cross-σ quadruples all exactly 0) |
| 3 | Eq. (16), the permutation / conjugation identities | 4 for `Q^(1)`, 7 for `Q^(2)` | all eleven hold, worst residual **2.2e-16** |
| 4 | Sec. 3: "for real-valued mode functions we have `Q^(1) = Q^(2)`" | stated | **exactly equal** in the real LP basis (max abs diff 0.0), and **not** an identity in the helical basis (max abs diff 0.672) — both halves of the statement |
| 5 | Fig. 1: log-log magnitude distribution with the `|Q| < 1 % max` band | three nested curves | 17 280 / 2×10⁴ zeros from Eq. (18) alone, 680 survivors after Eq. (18)+(19); the "all" and "after Eq. (18)" curves coincide, i.e. every zero in the raw set is the spatial rule |
| 6 | Sec. 5: the `O(M⁴N)` term "if no constraints on the coupling coefficients are imposed" | qualitative, plus Fig. 3(a) timings | surviving count fits `M^3.65` over M = 6…18, a **31–47×** reduction against `2M⁴` |
| 7 | Sec. 3.B / Eq. (14): "if a pulse is launched exclusively into one circular polarisation mode, no light is coupled into the orthogonal polarization … any numerical simulation can in fact be restricted to a single-mode GNLSE" | stated | on `MultimodeSplitStepEngine` fed the Eq. (7) tensor: opposite-σ power fraction **0.0** (exactly) over 0.5 m with all ten channels present, five of them launched empty. Energy conserved to **4.8e-8**. An isotropic-weight control, which ignores Eq. (19), leaks **1.09** (109 %) into the orthogonal polarisation — the check is not vacuous |

### The engine link

`overlap.engine_weights()` reads the two coefficient slots straight off
Eq. (7):

- `xpm_weights[i, j] = Q^(1)_{i i j j}` — the instantaneous Kerr coefficient of
  channel `j`'s intensity on channel `i`. Sanity anchors that fall out: the two
  circular states of **one** spatial mode couple in full (`xpm = 1`, the
  `|A_1|² + |A_2|²` of Eq. (14)), while LP01–LP11 is 0.3223 and LP01–LP21 is
  0.2025; the σ+ and σ− copies of every spatial shape have identical coupling
  rows/columns, as they must.
- `fwm_weights = Q^(2)_{m n n q}` — the degenerate (pump-driven) slice the
  engine's `A_n² A_q*` FWM term implements. The non-degenerate arms `l ≠ p` of
  Eq. (6) are outside the engine's scalar-channel geometry; that is a
  documented engine limitation, not a gap in the coefficients.

Raman is off in the engine runs, because the paper itself notes the delayed
Raman term breaks the per-mode energy conservation of Eq. (15) by design.

## Caveats and findings

1. **The paper's Fig. 1 counts are not reproduced, and cannot be with a scalar
   model.** The paper evaluates 17 872 of the 2×10⁴ coefficients as vanishing
   *identically* and a further 936 as lying below 1 % of the maximum, using the
   **exact complex-valued vectorial** mode functions of the step-index fibre.
   This deck reproduces the selection *rules* exactly but counts 17 280 zeros
   and no surviving coefficient below 1 % of the maximum (the smallest
   surviving one is 0.065 in the real basis, 0.405 in the helical basis). The
   small-coefficient tail of Fig. 1 is a vectorial-polarisation effect that a
   weakly-guiding scalar LP model has no way to produce. Closing this needs a
   full HE/EH/TE/TM vectorial mode solver, which is out of scope here. The
   qualitative reading of Fig. 1 (log-log distribution, plateau, threshold
   band) *is* reproduced.
2. **The engine's `oam_l` gate is exactly Eq. (18) type 2 — verified, not
   assumed.** This corrects an earlier draft of this README, which wrongly
   called the gate a *superset* of the paper's rule. Reading Eq. (6) properly
   settles it: the paper's `Q^(2)` term is `Q^(2)_plmn A_l* A_m A_n` feeding
   channel `p` with **`l` the conjugated field**, so momentum balance is
   `m_m + m_n = m_p + m_l` — precisely their printed rule. Under the index map
   `(p, l, m, n)_paper = (m, q, n, p)_engine` it becomes `ℓ_m + ℓ_q = ℓ_n + ℓ_p`,
   which with `p = n` is the engine's `ℓ_m = ℓ_n + ℓ_p − ℓ_q`. Checked
   element-wise over all 10⁴ quadruples: **0 mismatches**. The "64 quadruples"
   in the earlier draft were entries excluded by the *polarisation* rule
   Eq. (19), which the spatial gate correctly does not cover. See
   `../ISSUES.md` #15. What the gate does *not* carry is Eq. (19) and the exact
   magnitudes — those come in through `fwm_weights` / `xpm_weights`.
3. **The gate is inert for uniform labels.** `wright_2015_self_organized_instability`
   passes `oam_l=[0, 0, 0]`, which satisfies the condition for every triplet, so
   the gate is exactly the isotropic fallback of `oam_l=None`. It is a selection
   filter, never a work reduction — the "use `oam_l=` to keep costs sane" note in
   that folder's README gets no cost control from it. Wright-2015's results are
   unaffected (analytic 2×2 Kerr block + `fwm_weights`).
4. **The engine's coefficient slots are real matrices.** A complex `Q` (the
   helical basis, where the Eq. (16) conjugations matter) would lose its phase
   there. The deck therefore drives the engine from the real LP basis, where
   `Q^(1) = Q^(2)` is real by construction, and exercises the complex
   identities of Eq. (16) on the tensors directly.
5. **Energy drift on the engine deck is 4.8e-8, not machine zero.** It is the
   engine's split-step round-off, scaling as `1/N` and as `β₂²` (2.0e-10 at
   `β₂ = −5 ps²/m` on the same grid) — five orders below the engine's own 5 %
   drift monitor. The span is kept at 0.5 m deliberately: over a longer fibre the
   dispersed pulse wraps the periodic window, and the apparent power then
   wobbles at the 1e-3 level for purely window-aliasing reasons.
6. **Only the step-index case (Sec. 4.A) is reproduced.** The paper also treats
   cylindrically symmetric fibres of arbitrary profile (Sec. 4.B, same rules by
   the coupled-wave equations) and `C6v` microstructured fibres (Sec. 4.C,
   rules (23)/(24) over the eight symmetry classes of Table 2). Both need a mode
   solver this repository does not have; the counts the paper quotes there
   ("about 4000 nonzero coefficients" for ten modes of the MOF of Fig. 2(b))
   are therefore not addressed.
7. **Hexagonal-fibre complexity claim not checked.** Sec. 5's Fig. 3(a) runtime
   exponents (1.40 x^3.43 for the `A_l A_m A_n*` term, 4.86 x^2.22 for Raman,
   −0.59 offset + 6.11 x for self-steepening) are wall-clock measurements on the
   authors' implementation; the algebraic claim they support — that the
   `O(M⁴N)` term dominates for modest M — is what check 6 measures instead.
