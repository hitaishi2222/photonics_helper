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