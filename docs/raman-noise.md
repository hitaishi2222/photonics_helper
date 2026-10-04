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