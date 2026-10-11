# CHANGELOG — the formal specification

Every finding raised by the Lean development, and which side moved.

Format: what the statement said, why it was wrong, what replaced it.

## 2026-10-10 — Stage A

### Three statements were false as written

Found while proving, not by inspection. Recorded because the *pattern* matters:
each was a plausible-sounding generalisation that overreached.

#### 1. `beta_flip` — generalised to an arbitrary axis, where it is false

**Was:**
```lean
theorem beta_flip (s : AxisSign) (k : ℕ) (b : ℕ → ℝ) (h : validAxis s) :
    betaAxis s k b = (-1 : ℝ) ^ k * betaAxis 1 k b
```

**Wrong because:** at `s = 1` and odd `k` this asserts `b k = -(b k)`.

**Now:**
```lean
theorem beta_flip (k : ℕ) (b : ℕ → ℝ) :
    betaAxis (-1 : AxisSign) k b = (-1 : ℝ) ^ k * betaAxis 1 k b
```

The relation is between the **reversed** axis and the **identity** axis. Note
that `beta_flip_even` *does* take an arbitrary `s`, and is true there — so the
pair reads: even order is orientation-independent, odd order depends on it, and
the size of that dependence is exactly `±1` with no further scaling. Which
version you need is now visible from the signature rather than discovered by
counterexample.

#### 2. `lambda0_of_omega0` — dropped a factor of π

**Was:** `c / omega0 lam₀ * 2 = 2 * lam₀`, which requires `2π = 2`.

**Now:** `omega0 lam₀ * lam₀ = 2 * Real.pi * c`.

The product form is what the library actually uses, and unlike the quotient
form it does not divide by `π` — so it needs no `π ≠ 0` hypothesis.

#### 3. `axis_reversal_is_involution` — malformed

**Was:** a nested application `betaAxis (-1) (betaAxis (-1) k b) k`, applying a
real-valued expression where a `ℕ` order index was expected. It did not type;
it was never a statement.

**Now:** states the involutivity on the coefficient sequence itself,
`k ↦ betaAxis (-1) k b`.

### Two `sorry`s in `Conventions.lean` were in *definitions*, and moved to theorems

`dftMatrix` and `idftMatrix` were `def ... := by sorry`. With those, `dft_pair`
is **vacuously true** — the theorem would have "passed" while proving nothing,
which is worse than leaving it unproved.

Both matrices are now defined concretely (`Complex.exp (I·ω_i·t_j)`, with
`gridDt`/`gridTime`/`gridOmega` spelled out), so `dft_pair` is a real claim
about real definitions.

The three Fourier theorems (`dft_pair`, `conj_dft`, `hermitian`) remain
`sorry` and are marked **open (Stage B)** in their docstrings. They are
statements about sums of roots of unity — genuinely Stage B work per
`PLAN.md` §5.2, not something to fake with `decide` on a small case and call
finished.

### Note on API drift in Mathlib v4.34.1

Recorded because it will recur for anyone continuing this work:

- `Complex` moved to `Mathlib/Basic/Complex/`; `conj` is now the notation
  `star` in the scope `ComplexConjugate` (needs `open scoped`).
- `Matrix.one` is not applicable as a function here.
- `rw [← pow_add]` does **not** match exponents containing a numeric literal,
  so exponent arithmetic needs `ring`-normalisation first or an induction.

## 2026-10-11 — Stage B, continued

### Two more false statements of mine, caught before they reached the repo

#### 4. `kernel_factors` ignored the spectral-axis orientation

**Was:** `exp(I·s·π·(i−j)·(1−2k/N)) = exp(I·s·π·(i−j)) · rootOfUnity (i−j) n ^ k`

**Wrong because:** `rootOfUnity` carries no orientation factor, but reversing the
axis reverses the sign of the `k`-dependent phase. The identity fails for
`s = −1`.

**Now:** the root's argument is `(i−j)·s`. Verified numerically first (max
deviation 3e-15 over all bins and both orientations), then this is the version
to formalise.

This is the third time an axis-orientation error has appeared, in three
different guises (β coefficients, the frequency grid, and now the DFT kernel).
The pattern is strong enough to be worth stating as a rule in `PLAN.md`: *any*
formula that accumulates over the frequency index must be checked against
`omega_neg`, not just against the real-valued algebra.

#### 5. `conj_dft` is false at bin 0 — **WITHDRAWN, see finding 12**

On the `fftshift`ed grid the conjugate-symmetry pairing is `i ↦ −i mod n`. That
map sends bin 0 to itself, and bin 0 sits at `−πn/T`, **not** at zero, so
`ω_{−0} = ω_0 ≠ −ω_0`. Bins `0` and `n/2` are the self-paired ones (Nyquist and
DC), and for those conjugate symmetry says the bin is *real*, not that it
negates.

So `conj_dft` needs a hypothesis `i ≠ 0`, plus separate statements for the two
self-paired bins. Same structural point as finding F6 in `CONFORMANCE.md`:
symmetry on this grid holds everywhere except at the two distinguished bins,
and the exceptions are not defects, they are the Nyquist and DC frequencies.

### The assembly is blocked on coercion placement

`exponent_closed_formC` and `kernel_term` are proved. The next lemma must split
the summand as `C · ζ^k`, and lifting the real splitting identity into the
complex goal does not go through:

- `congr 1` leaves the integer subtraction in a different place than the
  `congrArg`-lifted hypothesis (`↑↑↑i - ↑↑↑j` versus `↑↑i - ↑↑j`);
- `rw [map_sub]`, `rw [Int.cast_sub]`, `push_cast` on both sides, and
  `convert ... using 1` each repair one side and break the other;
- the `s = −1` variant then hit `isDefEq` timeouts at 800k heartbeats while
  normalising coercions.

**The fix, for whoever continues.** Avoid the ℝ/ℂ boundary in the statements
rather than bridging it. Define the kernel once in complex form,
`exponent_closed_formC` becomes the only place casts appear, and state
`kernel_factors` and the entrywise theorem over that complex form, handling `k`
with `Complex.exp_nat_mul` instead of rewriting `↑k`. That removes every cast
mismatch hit above. It is a rewrite of the last third of `Spectrum.lean`, not a
new mathematical idea.

#### 6. `hermitian` is false as stated — **WITHDRAWN, see finding 12**

For real-valued `A`, the hypothesis `conj (A j) = A j` makes the right-hand side
equal to the transform of `A` itself, so the statement reduces to
`conj (X i) = X i`: that **every** bin of the spectrum of a real field is real.
A real pulse has a complex spectrum, so this is false.

The correct statement pairs bin `i` with bin `-i`, which is precisely
`conj_dft` restricted to real `A`. So of the three Stage B theorems nominally
open, `hermitian` and `conj_dft` are **one** theorem, not two, and the `i ≠ 0`
exception (Nyquist and DC bins are self-paired) applies to both.

The false statement is kept in `Conventions.lean` as a marker block, explicitly
flagged, so it is not re-introduced by someone reading the file for the first
time.

### 12. Withdrawal: findings 5 and 6 were wrong

I reported that `conj_dft` is false at bin 0 and that `hermitian` is false as
stated. **Both were wrong, and I should have checked numerically before
asserting them**, which is the same mistake I had already made once in this file.

The reasoning was: bin 0 sits at `-πn/T ≠ 0`, and negation maps `0` to `0`, so
`ω_{-0} = ω_0 ≠ -ω_0`. The first half is right; the conclusion does not follow,
because conjugate symmetry of the *transform* is not the statement
`ω_{-i} = -ω_i`. The `-Tmax/2` offset on the time grid contributes exactly the
missing factor.

Checked numerically on the exact definitions in `Conventions.lean`, `n = 4, 8, 16`:
`X_{(-i) mod n} = conj (X_i)` holds at **every** bin, including `0`, with zero
failures. And the arithmetic behind it is clean:

```
(ω_{-i} + ω_i)·t_j = 0                   for i ≠ 0
(ω_0 + ω_0)·t_j      = 2π·(n/2 − j)      for i = 0,  an exact multiple of 2π
```

So both statements are **true as written**, `hermitian` is the real-`A` special
case of `conj_dft` rather than a separate theorem, and no `i ≠ 0` hypothesis is
needed. The false-form marker block added to `Conventions.lean` for finding 6 has
been removed.

What this changes: Stage B's three open theorems are two (`dft_pair`,
`conj_dft`), and both are blocked on the same thing — the ℝ/ℂ coercion placement
— not on a mathematical obstacle. Three reindexing lemmas needed for the second
are now proved (`sum_neg_eq`, `neg_eq_sub`, `omega_reflect`).

The lesson, which I have now learned the hard way twice: a claim about a
*transform* is not a claim about its *kernel entries*. Check it numerically
first.

### Stage D, completed

Two new files, 24 theorems, no new `sorry`:

- `Nonlinear/Shock.lean` — the first-order shock expansion's validity. The key
  result is `shock_truncation_error`, an **exact** identity
  `1/(1−x) − (1+x) = x²/(1−x)`, from which `_SHOCK_TAYLOR_LIMIT = 0.2` is shown
  to admit at most 5 % truncation error. That replaces the code's docstring
  justification, which rested on three measured drift values.
- `Nonlinear/Raman.lean` — causality of `h_R`, the drive vanishing before the
  delay, and non-negativity of the drive. The gain bound `ramanDrive_le_max`
  is stated and remains `sorry`; its `Fin`-indexed bookkeeping was not finished.

### Practical notes for this Mathlib revision

- Lean distributes `ℝ → ℂ` coercions differently in a complex goal than in the
  corresponding real one. This is the single biggest time sink in the project.
- `congrArg (fun x : ℝ => Complex.I * (x : ℂ)) h` followed by
  `push_cast at h; convert h using 1 <;> ring` works when both sides are written
  with explicit `((i : ℤ) : ℝ)` casts. It failed on statements using bare `(i : ℤ)`.
- `Complex.exp_congr` does not exist at this revision.
- `omega` fails on goals containing `x + x` (§ F7 in `CONFORMANCE.md`).
- `mul_pos` / `div_pos` unify their implicit arguments from the **expected
  type**, so `mul_pos (mul_pos hn2 hω0) hG` fails when the goal is written
  `n2 * (ω0 * G)`. State an intermediate `have hnum : 0 < ... := ...` instead.
- `div_le_div_iff` and `le_div_iff` do not exist at this revision. Use
  `div_le_div_iff₀` and `le_div_iff₀`, which take **strict positivity**, and
  mind the direction: `le_div_iff₀ : a ≤ b/c ↔ a*c ≤ b` while
  `div_le_iff₀ : b/c ≤ a ↔ b ≤ a*c`.
- `Conventions.exp_add` / `Complex.exp_nat_mul` are the two lemmas that turn an
  exponent sum into a product of exponentials; everything in `kernel_factors`
  rests on them.
- **A coerced theorem is often better than a rewritten goal.** The ℝ/ℂ
  coercion problem that blocked `kernel_factors` for a whole session was not
  solved by the coercion rewrite the plan called for. It was solved by
  rewriting the summand with `kernel_term`, an already-proved theorem of
  exactly the right shape, so the two differently-placed coercions never arose.
  Reach for a bridging lemma before reaching for `congrArg`.
- `Equiv.neg (Fin n)` gives negation on `Fin n` as a permutation for free, and
  `Equiv.sum_comp` then gives `∑ f (-j) = ∑ f j` in one line. Do not write a
  `sum_bij` argument for this.
- `Fin.neg_def : -a = ⟨(n - ↑a) % n, _⟩` is the lemma that turns modular
  negation into ordinary reflection.
- `rcases (validAxis s) with rfl | rfl` substitutes inside `rootOfUnityS`'s
  body, which makes a `simp [heq]` close the goal outright; an explicit
  `rootOfUnity_pow` afterwards is a "no goals" error.
- `Finset.sum_eq_single` returns the summand, not zero, so a goal of the form
  `∑ = explicit value` needs a `calc` with the value spelled out.
- A markdown code fence inside a Lean docstring can break the parser; prefer a
  `/- -/` module comment when the text is not attached to a declaration.

## 2026-10-12 — Stage B closed: `dft_pair` proved

### The `↑1` cast artefact, solved

Recorded in `CONFORMANCE.md` F20. Summary: `rcases (validAxis s) with rfl | rfl`
injects a bare `1` into `rootOfUnityS`'s body and it lands as a *natural* cast
inside the exponent, after which `rw` and `ring` stop matching. Four routes
failed on this before the fifth worked.

**The fix: never substitute.** State each orientation with an explicit
`(1 : ℝ)` literal and assemble with `simpa [rootOfUnityS]`. For the
non-triviality half, take the imaginary-part argument *directly* on the
hand-substituted statement rather than routing through `rootOfUnityS`.

Four routes that failed, for the record:

1. `congr 1` then `ring` — the obligation contains `↑1`, unprovable.
2. `rw [← Complex.exp_add]` with a single `key` about the whole exponent —
   `key` closes, the following `rw` misses.
3. multiply through with `div_mul_cancel₀` — the cast on `N` is an `Int` cast
   where the lemma wants the `ℕ` one.
4. give `rootOfUnityS`'s axis argument an explicit `(s : ℝ)` ascription — no
   effect, because the substitution itself is the problem.

Two smaller findings from the same work:

- **`field_simp` on the imaginary-part identity already clears the
  denominator.** `him : -(m:ℝ)/N = k·(2π)` becomes `-m = N*k` in one step; the
  earlier version cleared the *wrong* `N`, because an `Int` cast was passed where
  an `ℕ` one was needed. Passing the right non-vanishing hypothesis is the whole
  fix.
- **`dft_pair` needed a `validAxis s` hypothesis.** Not a proof artefact: the
  Fourier pairing is a statement about one of two specific grids, so requiring
  the orientation to be `±1` is the correct statement, and it is now visible.

### Process note

Two sessions of churn this period came from editing Lean files through shell
heredocs: a `ℤ` (U+2124) silently became a Greek zeta, and a `sed` line-range
delete cut into an adjacent theorem. Both were recovered, but the correct tool
is the editor, and that is now recorded in `README.md`.

---

**Nothing in the Python library moved as a result of this stage** — no Python
symbol was found to disagree with a proved theorem. Stage G is where that
comparison is made systematically; this stage only built the theory to compare
against.
---

## 2026-10-24 — Stage B/D closure

All three previously-open statements are now proved. The repository now has
**no `sorry` at all**; `lake build` is green.

### 1. `conj_dft` — proved, after three real defects in the existing attempt

The theorem was not merely unproved; the draft proof did not compile, for
three distinct reasons. All three are recorded because each would recur.

#### (a) `Matrix.dotProduct` does not exist in this Mathlib revision

The draft unfolded `mulVec` with `simp only [Matrix.dotProduct]`. That constant
is unknown here, so the simp set silently did nothing on that lemma and the
goal stayed in `⋅`-form; the later `Finset.sum_congr` then had no sum to fire
on. `simp only` does **not** fail on an unknown identifier in the set — it
just skips it. The fix is root-namespace `dotProduct`.

This is worth flagging as a process finding: a silently-skipped simp lemma
presents as a confusing downstream error, not as "unknown identifier".

#### (b) `Complex.exp_conj` is oriented the other way in this revision

The draft used `rw [Complex.exp_conj]` to turn `conj(exp z)` into `exp(conj z)`.
Here `Complex.exp_conj : exp (conj x) = conj (exp x)`, so that rewrite needs
`←`. The draft also used `star_sum` with one argument; in this revision its
function argument is explicit (`star_sum (s) (f)`), and `star_sum _` leaves a
Pi type that `.symm` cannot be projected from.

#### (c) `origin_bin_exp` was ill-typed, not just unproved

The old `hcast` stated `((s * (n/2 - j) : ℤ) : ℝ)` with `s : AxisSign = ℝ`. That
coercion `ℝ → ℤ` does not exist, so the statement did not elaborate. It is
replaced by a `Complex.exp_eq_one_iff` argument that supplies the integer
witness directly, branching on `validAxis s` so that the witness is genuinely
integral.

Separately, `gridOmega n s Tmax 0` needs `OfNat (Fin n) 0`, which needs
`NeZero n` — and that instance is **not available while the statement is being
elaborated**, so a `haveI` inside the proof body is too late. The DC bin is
now written `⟨0, by omega⟩` with an explicit `hn0 : 0 < n` hypothesis.

### 2. `hermitian` — proved as a corollary, not independently

As `README.md` predicted, it is `conj_dft` restricted to real input. Realness
is stated as `A j = ((A j).re : ℂ)` (using the available `Complex.conj_ofReal`)
rather than a coercion hypothesis, because `(A j : ℝ)` does not typecheck as a
statement about `A j : ℂ`.

### 3. `ramanDrive_le_max` — proved

The `Fin`-indexed bookkeeping the previous session left unfinished. The route
that works is a termwise bound rather than a sum rearrangement: a single
response weight `h_R(j−i)` is one term of a non-negative sum, hence at most the
whole sum, so each surviving summand is `≤ (∑ h_R)·P(i)`. Summing and applying
`∑ h_R ≤ 1` closes it. No max, no monotonicity argument, no reindexing.

**Which side moved:** only `formal/`. No Python symbol was touched, and no
physics claim changed — `ramanDrive_le_max` was always true as stated; what was
missing was a proof.

### 4. `DBR.lean`'s header overstated *why* `R + T = 1` was unproved

The header said energy conservation "additionally needs the media to be
lossless", implying lossless media would settle it. That is not the whole
story, and the distinction is worth recording because it changes what the
remaining work is.

Writing the standard formulas and reducing `r² + T = 1` by hand:

- `r = (n₀B − C)/(n₀B + C)`, `T = 4n₀n_s/(n₀B + C)²`
- `B = M₀₀ + M₀₁n_s`, `C = M₁₀ + M₁₁n_s`

the identity is equivalent to `B·C = n_s`. That does **not** follow from
`det M = 1`. Nor is it automatic for real media, because a layer's
off-diagonal entries are `∓i·sin δ`, so `B` and `C` remain complex even when
`n`, `δ`, `η` are all real. The physically correct lossless statement needs the
conjugate and a real part — a different argument, not a longer version of this
one.

`DBR.lean`'s header now says so explicitly rather than leaving the impression
that only the lossless hypothesis is missing.

### 5. `stackOrder_ne` — layer-order sensitivity, added

`PLAN.md` §8.4 asks for the claim that swapping two layers of unequal `δ`
changes the stack. Stated as a criterion rather than a fixed example:

> `stackMatrix [A, B] ≠ stackMatrix [B, A]` whenever `A * B ≠ B * A`.

A concrete pair would prove one stack and generalise to nothing; the criterion
is what a caller actually needs. Its contrapositive is the useful direction —
layers that commute may be reordered freely, which is why quarter-wave periods
can be permuted without changing `R(λ)`.

---

## 2026-10-25 — Stage E/F/H coverage: MI gain, dispersive-wave sign, SHG mismatch and overlap, RK4IP, operator ordering

New theorems in four files (`Optics/FWM.lean`, `Optics/Chi2.lean`,
`Solvers/RK4.lean`, `Solvers/Strang.lean`) plus one new file
(`Optics/DBREnergy.lean`). No Python symbol was changed; every row closed as
`matches` against the code as written.

### 1. `DBREnergy.lean` — `R + T = 1`, correcting the 2026-10-24 entry

The previous entry recorded that `r² + T = 1` "does not follow from
`det M = 1`" and is "a different argument". Both halves of that were wrong in
the way that matters: in the lossless regime the conjugate statement is the
same argument with one more hypothesis, and it is short.

Writing the stack's reflection numerator and denominator as

```
D = (η₀a + η_x δ) + i(η₀η_x β + γ)
N = (η₀a − η_x δ) + i(η₀η_x β − γ)
```

for `M = [[a, iβ], [iγ, δ]]` (the lossless shape), the arithmetic is
`|D|² − |N|² = 4·η₀·η_x·(a·δ + β·γ)`, and `a·δ + β·γ = det M` because `i² = −1`.
So `det = 1` plus the shape plus positive real admittances closes it, exactly
the "regime as a hypothesis" pattern of `PLAN.md` §7. Recorded as F21.

The practical lesson, the third one of its kind in this project: the previous
session derived the *index-form* reduction (`B·C = n_s`) and stopped there,
without noticing that the correct conjugate form (`B·C̄ + B̄·C = 2n_s`) is what
the lossless shape gives. Same shape as F9: check the claim against the
structure you have, not against the form you assumed.

### 2. `FWM.lean` — the MI gain spectrum, peak location, and peak height

`mi_gain_peak` evaluates `g` at `Ω = √(2γP/|β₂|)` and gets exactly `2γP`; the
peak *location* `Ω² = Ω_c²/2` falls out of `miPeakSq_half`. `mi_gain_le_max`
bounds the gain everywhere by `√(β₂²·x·(Ω_c²−x)) ≤ 2γP`, i.e. AM–GM through
`0 ≤ (2x − Ω_c²)²`.

The extended-Δ form closes the docstring's "reduces exactly" sentence:
`miGainExt_reduces` proves `√[−(β₂Ω²)(β₂Ω²+4γP)] = |β₂Ω|·√(Ω_c²−Ω²)` for
quadratic dispersion. Note the `Ω ≠ 0` hypothesis: at `Δ = 0` the extended band
is open by definition, and the equality still holds there trivially (both
sides are 0) — the proof routes through the band statement, so the hypothesis
is stated rather than hidden.

### 3. `FWM.lean` — the dispersive-wave sign, and a magnitude open (F22)

`dwResidual_at_soliton` pins the residual at `ωₛ` at exactly `−qₛ`, so the
soliton frequency is a root exactly when `qₛ = 0`; the quadratic case is a
double root, which is the algebraic content of the `f'(ωₛ) = 0` comment in the
root finder's Newton fallback. `dwDetuning_blue` proves the sign invariant
`PLAN.md` §8.2 asks for: blue-shifted exactly when `β₂·β₃ < 0`.

**Open (F22):** the exact cubic-truncation root is `−3β₂/β₃`
(`dwResidual_cubic_root`); the `soliton.py` fast fallback uses `−2β₂/β₃`. Same
sign, factor `3/2` in magnitude. The fallback is labelled an estimate in its
own docstring; the magnitude question is routed to the soliton-RDW literature
because the correct factor depends on the `qₛ` convention, which Lean cannot
settle.

### 4. `Chi2.lean` — SHG mismatch, QPM cancellation, and the overlap reduction

`deltaK_qpm_cancels` is the crisp one: with `Λ = 2π/|Δk|`, first-order QPM
satisfies `Δk − 2π/Λ = 0` *exactly* for `Δk > 0` — a cancellation, not an
approximation. `overlapCoupling_planeWave` proves the Wang et al. (2017) overlap
collapse for a uniform field to the Boyd `shg_coupling` formula, with all
`E`-, `Es`- and `d₀`-dependence cancelling (the docstring's scale-invariance
claim). One wrinkle worth recording: the docstring writes `√(2Z₀/n³A)` while
the code computes `(ωd/(nc))·√(2Z₀/(nA))`; these are equal for `n > 0`, and the
theorem is stated against the code's form.

### 5. `RK4.lean` — the interaction picture, factorised exactly

`ipStep_factors`: the IP step is `exp(λh)·(1 + μh + (μh)²/2 + (μh)³/6 + (μh)⁴/24)·y`
— the exact propagator times the stability polynomial, with no further
approximation. `ipStep_linear` (pure exponential when `μ = 0`) and
`ipStep_nonlinear_only` (pure RK4 when `λ = 0`) say each half is exact when the
other is off. Together with `rk4_error_le`, this is the full content of the
library's "RK4IP" label.

A draft theorem claiming the two IP orderings coincide was removed before
commit: `exp(λh)·R₄(μh) ≠ exp(μh)·R₄(λh)` in general (they agree only to
`O(h⁴)`), which is precisely why the split-step ordering is a choice with a
cost, not a gauge change.

### 6. `Strang.lean` — the ordering theorem the split-step carries

`conv_diag_ne_commute`: a two-bin convolution with `R 1 0 ≠ 0` and an unequal
diagonal `d 0 ≠ d 1` do not commute, witnessed by the one-bin field. That is
the formal content of the ISSUES.md #0 convention swap — Raman and the
propagator had to change *together* because the operators do not commute, and
it is why the Raman term is the piece of `gnlse.py` that can move the norm
`splitSeq_norm` proved is otherwise fixed.

### Practical notes added to this revision's folklore

- **`rw` postpones elaboration of inline `(by tac : T)` arguments.** An inline
  `rw [← Real.sqrt_sq (by linarith : (0:ℝ) ≤ 2 * gamma * P)]` evaluates the
  `by` block *after* postponing, against a metavariable context where linarith
  fails with a phantom `a✝ : 2 * gamma * P < 0` and goal `False`. Fix: hoist the
  proof into a named `have` first. Cost: two hours.
- **`ring` over `ℂ` does not know `I² = −1`.** It normalises products to `I^2`
  and then fails. The fixed pattern is
  `push_cast; ring_nf; rw [show (I : ℂ) ^ 2 = -1 by norm_num]; ring` —
  `ring_nf` canonicalises so the `I^2` occurrence is always in the same place.
- **`linarith` cannot multiply hypotheses; `nlinarith` can, but not always by
  variables.** For a hypothesis-times-variable step, derive the two ends
  separately (`have h1 : eta0 * a = -etaX * d := by linarith [hU]`) and
  rewrite — deterministic beats clever.
- **`Matrix.mulVec` unfolds to a root-namespace `dotProduct`** at this revision;
  `Fin.sum_univ_two` is the 2-point sum splitter.
- **`Complex.normSq_ofReal : normSq ↑r = r * r`** is the real-cast norm; for
  `normSq (2 * ↑eta0)` rewrite the cast inside first.

## 2026-10-26 — the last three open statements closed (E5/E6, H6, F19)

Eighteen new theorems and one example across three files; no Python symbol
changed (the `np.clip` and admissibility rows describe the code as written).
`lake build` green, zero `sorry`.

### 1. `DBR.lean` — the admittance branches and the Snell/obliquity derivation (E5, E6)

- `snellSine` / `snell_invariant`: the layer sine times the layer index
  recovers the conserved tangential component `n₀·sin θ_i`. This is why the
  Python is entitled to pass the *configured incident* index into every layer.
- `snell_admissible`: `|sin θ_l| ≤ 1` is *equivalent* to
  `|n₀·sin θ_i| ≤ n_l` — so the no-clip set of `_admittance`'s `np.clip` guard
  is a theorem, not folklore.
- `clip_no_op` / `clip_lower` / `clip_upper` / `clip_fires_iff`: the clip is
  the identity exactly on the admissible set; outside it, `|sin θ_l| > 1` iff
  `sin θ_l < −1 ∨ sin θ_l > 1`, and the clip saturates to `∓1` — the
  total-internal-reflection branch the guard marks rather than repairs.
- `etaTE` / `etaTM` / `admittance_normal_incidence` / `admittance_product` /
  `etaTE_nonneg` / `etaTM_nonneg`: the branches agree at normal incidence and
  their product is `n²` — the polarisation contrast lives entirely in
  `cos θ_l`, never in the material. Both are nonnegative for real `n ≥ 0`
  with the principal root, so a negative admittance is unreachable.
- `deltaPhase` / `deltaPhase_nonneg` / `deltaPhase_normal_pos`: with
  `δ = 2π n d cos θ_l / λ`, `δ ≥ 0` for real media (through the principal
  root) and strictly positive at normal incidence.

Plus the degenerate E7 witness `layerMatrix 0 e = 1` (a `δ = 0` layer is the
identity matrix, hence commutes with anything — the trivial direction of the
order-sensitivity criterion).

### 2. `Solvers/RK4.lean` — the analytic bound H6, without the carrying hypothesis

Earlier H6 claimed the bound `Σ_{k≤4} zᵏ/k! ≤ exp z` under a hypothesis that
supplied the Taylor expansion, on the stated belief that this Mathlib revision
lacked the tools. That belief was wrong: `taylor_mean_remainder_lagrange_iteratedDeriv`
exists. What closed it:

- `iteratedDeriv_exp` — `iteratedDeriv k exp = exp`, by induction (Mathlib
  has `iter_deriv_exp` for `deriv^[k]` but nothing directly in iteratedDeriv
  form for the real exponential);
- `exp_taylor_abs_error_le` — `|exp z − Σ| ≤ exp|z|·|z|⁵/5!` for every
  `z ≠ 0`, including negative `z` where the truncation *overshoots*; the
  Lagrange point `x'` is bounded by `|z|` by case-splitting on the sign of
  `z`;
- `rk4_error_le` now has no expansion hypothesis at all — its proof is one
  call to `Real.sum_le_exp_of_nonneg`.

Recorded because the lesson is the third repeat of the pattern from
2026-10-25: *check whether the lemma exists before assuming it does not*.

### 3. `Dispersion/Beta.lean` — F19, the phase-truncation remainder

`phase_matching.py::_taylor_phase_max_error` numerically estimates the max of
`|φ_full − φ_Taylor|`. The statement underneath it is now proved:

- `phaseRemainder_lagrange` — Taylor's theorem with Lagrange remainder applied
  to an arbitrary sufficiently smooth dispersion `f` at a carrier `x₀`: the
  truncation error is exactly `f^(n+1)(x')·Δ^(n+1)/(n+1)!` for some
  intermediate `x'`;
- `phaseRemainder_bound` — with `M` a bound on `|f^(n+1)|` across the
  interval, the error is `≤ M·|Δ|^(n+1)/(n+1)!`.

Both stated in `Set.uIcc/uIoo` form directly over the Pythons' grid units.
Not a full formal correspondence — `f^(k)(x₀)` in the theorem is the abstract
iterated derivative while the Python's `betas` array is data — but the
statement the numeric estimate is checking is now a theorem.

### Meta

- `Finset.sum_eq_single` supplied the degenerate-case split for the bound at
  `x₀ = x` (every non-carrier term vanishes); `div_le_div_of_nonneg_right`
  closed the final `/ (n+1)!` step.
- `Real.contDiff_exp` is `ContDiff` for *every* order (`ContDiff ℝ n exp` with
  `n` universally quantified), which simplified the smoothness arguments.
- `rw [abs_mul, abs_pow, abs_of_nonneg h]` chains fail silently when the first
  `|·|` occurrence is not the intended one; factoring the numerator
  normalisation into a `hnum`/`hden` rewrite pair was the fix used repeatedly
  this session.

## 2026-10-27 — the −14 % shock drift was the wrong invariant (§17.5, side-quest)

**Finding.** The side-quest's Stage S1 left one item open: a −14 % photon-number
drift on fissioning Raman+shock runs, identical for both composition schemes and
invariant under `dz` refinement, exceeding the "≤0.3 %" the `gnlse.py` header
quoted. It had been attributed to the first-order `ω/ω₀ ≈ 1 + Ωτ` Taylor
truncation of the shock term ("a grid artifact, not a property of the model").

**Resolution: neither code nor model was wrong — the measurement was.** Blow &
Wood (1989), read from `formal/refs/blow.pdf` p. 2672 (Appendix): the conserved
quantity of the GNLS is **not** the unweighted energy but the *frequency-weighted
photon number*

```
Q = ∫ |Û(ω)|² / (ω₀ + ω) dω            (their Eq. (A5)),
```

and with `τ_shock = 1/ω₀` the `n_eff/N` prefactor may be taken outside the
integral, at which point "(A5) is then the obvious classical analog of the total
photon number in the optical wave". Their abstract: "Optical energy is not
conserved by the equation, as indeed it is not conserved in stimulated Raman
scattering". The operator symbol `(1 + Ωτ)` **is** `ω/ω₀` bin-by-bin — there is
no expansion in this realization to truncate.

**Measured (both sides moved; the Python side gained an accessor, the narrative
was rewritten):**

| configuration | Δ unweighted energy | Δ Q (A5 photon number) |
|---|---|---|
| N=3 soliton, 1550 nm, Raman+shock, τΩ=0.88 | −14.42 % | −0.0011 % |
| same deck, τΩ = 0.44 … 0.08 | −12.4 … −1.0 % | ≤ 5×10⁻⁵ % |
| Hult/Dudley 835 nm PCF deck, `strang` | −0.07 % | −0.015 % |
| same deck, `yoshida4` | −0.07 % | −0.014 % |
| shock only (no Raman) / Raman only | −0.04 % / exact | n/a (energy is the invariant there) |

The drift requires Raman **and** shock together, is scheme- and `dz`-independent,
and vanishes under the correct invariant — the textbook case where the
disagreement was between the code and the *question*, not the code and the
paper. The energy change is the physical conversion: `Σ|A|²dω = ⟨ω⟩·Q`, so a
net red shift of the fission products lowers the unweighted norm while the
photon count is constant.

**Python side moved.**

- `SplitStepEngine.photon_number_vs_z` / `_photon_number` (and the
  `GNLSESolver` / `TaperedGNLSESolver` passthroughs) record the B&W (A5)
  invariant; `energy_vs_z` docstrings now state plainly that it is not an
  invariant when self-steepening is on.
- `_emit_energy_drift_warning` now monitors the conserved quantity: the photon
  number when self-steepening is on, the unweighted energy otherwise. It
  previously false-alarmed (>5 %) on legitimate Raman+shock runs.
- `_validate_shock_grid` / the `_SHOCK_TAYLOR_LIMIT` header were rewritten: the
  limit is re-justified as Blow & Wood's small-bandwidth derivation regime
  ("bandwidth of U(ω') less than ω₀/3"), not a truncation bound. The constant
  `0.2` is kept (bandwidth ≤ ω₀/5, inside ω₀/3 with margin); the hard `Ω_max < ω₀`
  bin-positivity check is unchanged.
- New test file `tests/test_gnlse_photon_number.py`.

**Lean side: not implicated.** No theorem claimed energy conservation of the
shock operator; `Nonlinear/Shock.lean`'s statements are about the sign
convention and the Taylor-remainder *representation* `x²/(1−x)`, which remains
valid. The §17.5 table row "‖L‖ = 1 (lossless linear step)" is unaffected.

## 2026-10-(S3 session) — Stage S3: ETDRK4 implemented (`scheme="etdrk4"`)

The remaining open stage of the §17 side-quest. `SplitStepEngine` gained a
third `scheme`, `"etdrk4"` (Cox & Matthews 2002, Eqs. (26)–(29); diagonal
per-bin case of Kassam–Trefethen 2005 §4): the whole semilinear step
`∂_zA = L·A + N(A)` is advanced with the fourth-order exponential RK4, with
`L` diagonal in the Fourier basis — no splitting, no shock substepping
machinery. `TaperedGNLSESolver`/z-dependent hooks, TPA and free carriers are
explicitly out of scope for this path (Stage S3's conditional gate) and raise
`NotImplementedError`.

### Two implementation findings, both the classic traps for this scheme

**1. The frames.** The nonlinear stage values live in *time*; the per-bin
φ-function coefficients live in *frequency*. Multiplying a time-domain array
by a frequency-domain coefficient array — the natural first coding, and the
one every write-up warns against — produces a bound, `dz`-independent residual
that self-converges to machine precision while showing zero per-halving
improvement: "a converged but wrong flow" (measured: field rel-error locked
at 6.65e-01 across `dz → dz/8`, with a γ-independence at the 9th digit
confirming the nonlinear channel was frame-mixed). Every `N` must be FFT'd
before the weights are applied; the corrected implementation matches the
finely-substepped splitting reference at ~1e-12 per coarse step.

**2. The contour mean is only `real()`-safe for real operators.** The
Kassam–Trefethen diagonal recipe averages the defining function over a
circle centred at `hλ` — but their published sampling is a *half* circle
(`exp(1iπ((1:M)−.5)/M)`), whose mean is exact for real `L` (their `real()`
then merely clears round-off). For our complex `L` the half-circle residue is
physics-sized (order of the coefficient at the GNLSE's dispersion phases).
A *full*-circle mean is exact for the entire coefficient functions
(verified 1e-16 in `tests/test_gnlse_etdrk4.py`), but the shipped evaluation
does not need a contour at all: the coefficients are entire with removable
singularities, and the honest DC-bin fix is Cox & Matthews' own
recommendation — the Taylor series near `|x| < 1/2`, the defining formulas
away from it. Both behaviours are pinned by tests (DC cancellation breakdown
reproduced from the naive formula; contour/full-circle agreement).

### Verification

- `tests/test_gnlse_etdrk4.py` (12 tests): scalar order-4 on `u' = λu − u²`
  (halving ratios ≈ 1/16 vs the exact Riccati solution); DC-bin exactness;
  naive-formula DC breakdown reproduced; full-circle contour agreement and
  half-circle failure; two-term Taylor identities; full-field agreement with
  a finely-split `yoshida4` reference on the N=3 deck; (A5) photon-number
  conservation with Raman+shock (2.2e-14); energy drift is O(dz⁴) as theory
  requires for the non-split method.
- `GNLSESolver` was found missing the `photon_number_vs_z` passthrough that
  `SplitStepEngine` and `TaperedGNLSESolver` already had
  (`test_solver_passthrough` caught it) — added.
- `Photonics.Solvers.Etdrk4` (Lean, sorry-free): `Qcoef_mul_id` — the stage-1
  carry is the exact φ₁ of the half argument (`xQ = exp(x/2) − 1`) — and
  `f1coef_phi_combo` — the `N₀` coefficient is `φ₁ − 3φ₂ + 4φ₃`. 248 theorems
  repo-wide. **`Nonlinear/Shock.lean` was found not compiling in the working
  tree of the 2026-10-27 side-quest session**: a duplicated
  "derivation-regime" block left an unterminated block comment (silent until
  the next build), and `accuracy_implies_existence` was a false statement —
  `x ≤ 1/5` alone does not imply `x > −1`; the bin-positivity existence
  condition is `x < 1` given `0 ≤ x`, and the theorem now carries the missing
  non-negativity hypothesis.

### S3 gate: measured (coarse ladder, texts in `PLAN.md` §17.6)

| Configuration | `strang` | `yoshida4` | `etdrk4` |
|---|---|---|---|
| A. Kerr only | 2.007 | 4.136 | **3.971** |
| B. + Raman | 2.005 | 4.144 | **3.971** |
| C. + Raman + shock | 2.005 | 4.147 | **3.970** |

The 4th-order figure is ensemble-consistent on the finest pair (3.99); the
metric's finer ladder collapses into the round-off floor (~1e-13) for both
4th-order schemes — measured and then deliberately not reported as an order.
The Raman+shock run's A5 photon number is conserved to 2.2e-14 under ETDRK4.
(The finer S0/S1 ladder is retained in §17.6 for the *2nd-order* schemes,
where it remains informative.)
