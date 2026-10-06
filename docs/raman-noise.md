# Raman noise sources

!!! warning "Scope"
    This page describes work that **does not exist yet**. It is written before the code, on
    purpose, so that the boundary is documented before anyone has an incentive to blur it.

## What exists today

`photonics_helper.raman_transfer` models a continuous-wave Raman cascade under an imposed pump
modulation, and computes how that modulation appears at the signal. It is a **transfer
function**: it answers "given this perturbation on the pump, how much perturbation do I get
at the signal". It never generates a perturbation on its own.

See [Raman cascade: power flow and RIN transfer](raman-cascade.md) for what is implemented and
which published targets it reproduces.

## What does not exist yet

What is missing is every *source* of noise. The steady state in this module is noiseless by
construction; perturb it only by perturbing a pump. The sources to be implemented are:

### Amplified spontaneous emission (ASE)

Each Raman order generates spontaneous Raman noise over a bandwidth set by the Raman gain
spectrum, and that noise is amplified along the span like any other Stokes wave. In a
distributed amplifier the ASE power in order `k` is comparable to the signal power itself for a
moderate gain, which makes the amplifier noise floor the dominant term in a real link budget.

This needs a spontaneous-emission source term in the power equations. Note that it also
changes what "small-signal Stokes gain" means: without a spontaneous term, a Stokes channel
launched at a small seed grows as `P_s(0) exp(g P_p L_eff)` and can never reach the
seedless-limit form `P_s = P_p (exp(g P_p L_eff) − 1)`. The current test suite gates the
exponential gain law for exactly this reason, and a stage-1 change that adds the spontaneous
term should extend that test to the seedless form rather than silently change it.

The natural closed form is Zhu 2007 Eq. 7's DC behaviour plus the standard Raman gain spectrum,
and the natural gate is Zhu 2007 Fig. 1 — whose DC level is currently recorded as
**unverifiable**, because Zhu states its fiber loss and effective area in external references
[11] and [12]. Obtaining those two references would turn an unverifiable target into a real
one, and is worth doing before stage 1 inherits the gap.

### Spontaneous Raman scattering between orders

Each Stokes wave is itself a pump for the next order, so the noise generated in one order
becomes noise in the next. This is a genuinely multi-order effect: the noise in order `k`
arrives at order `k+1` modulated by the noise in order `k`. It is also why a dual-order
amplifier is not simply two single-order amplifiers in series, and it is the mechanism behind
Mermelstein's first-order pump response being much smaller than the second-order one.

### Stimulated Brillouin scattering

Not Raman, and not modelled here. It matters at narrow linewidths and high power over km-scale
spans, and it is usually suppressed with a distributed FBG. Listed because it is the obvious
question when someone asks why a narrow-linewidth pump is not used.

### Pump laser noise

The physical source feeding a real system: relaxation oscillations, mode competition, and
drive-current noise in the pump laser diode. This module takes pump RIN as an input; it does
not model the laser. Estimating the pump RIN a design actually requires is exactly the kind of
backward problem Mermelstein 2003 Section III approaches with its measured transfer functions.

## Why the boundary is where it is

Three reasons, in order of weight.

**The transfer function is a correctness oracle that happens to exist.** The steady-state
cascade can be validated against eight published DC levels and corner frequencies and two
interaction lengths, with no experiment of our own and no random sampling. A noise model has no
such oracle: ASE spectra depend on the Raman gain spectrum and the fiber's spontaneous
emission bandwidth, and comparing against a published spectrum requires trusting that paper's
normalization as much as this one's. Adding the sources before the transfer function is
validated would mean debugging two things at once.

**The physics that determines the noise shape is not in the papers held.** The cascade transfer
is in Eqs. 5 to 7 of Mermelstein 2003. The ASE spectral shape needs the Raman gain spectrum and
a spontaneous-emission bandwidth convention that none of the three held papers states. Guessing
it and then fitting is the failure mode the benchmark-fixture discipline exists to prevent.

**Machine learning and inverse design depend on the inputs being trustworthy.** A surrogate
trained on a noise model with a wrong spontaneous-emission constant is worse than no surrogate,
because it is fast and confidently wrong.

## Conventions the future work must keep

- **Photon-count, not power.** Raman conversion converts photons and dumps the energy
  difference into a phonon. Use `P·λ` weighting for photon budgets, not `P/λ`. The
  `photon_consistent` flag on `CWWCascade` selects which of the two conserved quantities the
  cascade respects; see [raman-cascade.md](raman-cascade.md).
- **Loss coefficients are never converted implicitly.** Every public function takes documented
  units and converts nothing.
- **Every published target arrives with its DOI, its figure or table number, and whether it was
  read from a table or eyeballed off a plot.** Figure-transcribed values are marked
  `kind: "figure"` and should not be held to table precision.
- **Unverifiable targets are recorded as unverifiable, not fitted.** Zhu 2007's DC level is the
  worked example: the honest move was to gate the corner frequency and the roll-off slope, which
  do not depend on the missing fiber parameters, and to leave the DC marked unavailable.

## Planned stage 1 scope

- Spontaneous-emission source term in the power equations, per order, over the Raman gain
  spectrum.
- Multi-order noise propagation, so that noise generated in one order seeds the next.
- An amplifier noise floor and an output SNR against the Mermelstein and Zhu configurations.
- Obtaining Zhu 2007 refs [11] and [12], which would convert the recorded-unverifiable DC level
  into a gate.

## References

- R01. Mermelstein, Brar, Headley 2003, *J. Lightwave Technol.* **21**(6), 1518. [10.1109/JLT.2003.812461](https://doi.org/10.1109/JLT.2003.812461)
- R08. Zhu, Zhang, Zhang 2007, *J. Lightwave Technol.* **25**(6), 1458. [10.1109/JLT.2007.895532](https://doi.org/10.1109/JLT.2007.895532)
- R05. Babin, Churkin, Fotiadi et al. 2005, *IEEE Photonics Technol. Lett.* **17**(12), 2553. [10.1109/LPT.2005.859547](https://doi.org/10.1109/LPT.2005.859547)
---

## Raman cascade noise sources

Three sources, kept separate because they do different work.

### 1. Pump amplitude noise (`pump_noise_field`)

A boundary condition: the input envelope is multiplied by `(1 + m(t))`, with
the modulation index built from a supplied RIN spectrum.

The supplied spectrum is the engineering **relative intensity noise** in Hz⁻¹
referred to mean power. Since `A = √P (1 + m)` implies `P = P(1 + m)²` and
therefore `δP/P = 2m`, the modulation index carries a **quarter** of it:
`S_m = S_RIN / 4`. Without that factor every level is 6 dB high.

### 2. Pump phase noise (`pump_phase_noise_field`)

A second, **independent** boundary condition, `A → A exp(iφ)`, specified by a
phase-noise PSD in rad²/Hz.

It is kept separate so the Kerr conversion from phase noise to intensity noise
at twice the offset frequency plus an offset stays *emergent* inside the
solver. Pre-computing that conversion analytically would make it impossible to
get wrong and equally impossible to verify.

### 3. Spontaneous Raman scattering (`spontaneous_raman_noise_field`)

A **distributed** Langevin term, accumulated per step, not a boundary
condition.

`noise.raman_noise_field` already carries the correct per-bin quantum
weighting — one photon per mode, scaled by the positive Raman gain `Im h̃_R` and
by the spontaneous-plus-stimulated emission factor `n_th + 1`. What it lacks is
per-order structure, so it is called **once per order** with that order's own
`h̃_R` and ω₀ rather than being reimplemented. Two calls into one correct
generator beat a second generator that has to be re-verified separately.

`n_sp` defaults to 1 (the standard quantum limit) and is applied as `√n_sp` on
the amplitude, because the seeded *power* scales linearly with it. At silica's
440 cm⁻¹ shift `ħΩ_R/k_B = 96 K`, so at 300 K the Bose factor is a **35%**
correction and is not negligible.

## Normalization

Two normalizations in this module are easy to get wrong by a factor of two or
four, and neither error is visible in the output — they read as a few dB on a
noise floor.

**Injection.** The generated field is rescaled so that `mean(n²)` equals the
one-sided integrated variance `∫_band S_f df` exactly, computed directly from
the supplied spectrum. Deriving the constant instead means carrying the FFT
scaling convention, the bin-ordering symmetry, and whether the field is taken
real through three layers of algebra.

The real part must be taken of the **field**, not of the spectrum. A real part
of a complex Gaussian is still Gaussian and a real field's spectrum is
automatically Hermitian; taking the real part of the spectrum instead leaves a
non-Hermitian array whose inverse transform is complex, and
`exp(i(a+ib)) = e⁻ᵇeⁱᵃ` would then let a phase modulation silently change `|A|`.

**Extraction.** The intensity PSD is `S_f = 2|F|²/(dt·N)` per positive bin.
Select on the *sign* of ω and then order — folding the negative bins onto the
positive ones with `abs()` duplicates every bin, halves the apparent spacing,
and makes the median spacing zero.

Integrating the PSD over the returned band must include **half a bin at each
edge**: the band is a set of bin centres, so a plain trapezoid drops ~5% on a
narrow band, which is enough to break the 0.2 dB RIN/g₂ gate for no physical
reason.

## Time-grid resolution

The grid's frequency spacing is `df = 1/T`. Because the Kerr term converts
phase noise to intensity noise at **twice** the injected offset, a grid sized
for the injected band puts the converted component beyond Nyquist, where it
wraps instead of appearing — silently. `required_window()` returns
`1/(2 f_max)`, and the injector **refuses** a band containing no grid bins
rather than returning silence.

## `g2` and what it is not

`second_order_coherence` returns `⟨|E|⁴⟩/⟨|E|²⟩²` per bin, which is identically
**2** for a circular complex Gaussian field — reached to within 0.05 dB here.

A strong *carrier* does **not** push it below 2: a carrier times independent
per-bin noise is still circular Gaussian bin by bin. What drives it down is a
component whose bin-to-bin phase is *locked* across realizations.

`g2` must come from the intensity statistics, **not** from `coherence_g12`.
`coherence_g12` measures the *mutual* coherence between different
realizations; independent seeds make it vanish while the second-order statistics
stay finite, so substituting one for the other would report `g2 = 1` for a field
that is plainly not Poissonian. Both are exposed, and the distinction is
guarded by a test.

## Ensemble convergence

RIN estimated from *M* realizations has a standard error falling as
`1/√M`, but the constant depends on the number of bins and on their correlation,
so it is **measured** rather than assumed. `ensemble_convergence()` returns the
count needed as a function of accuracy *and* offset band; 0.5 dB over 1 kHz to
10 MHz needs **under 200** realizations. `warn_if_undersampled()` names the
specific shortfall when fewer were used.
