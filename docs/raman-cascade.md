# Raman cascade: power flow and RIN transfer

This page documents `photonics_helper.raman_transfer`, which answers one question with as
little machinery as possible: how does relative intensity noise on a pump laser appear at
the signal of a Raman amplifier, and how does that transfer function depend on fiber loss,
gain, length, dispersion and geometry?

Three published results constrain the answer, and all three are implemented:

| Paper | What it pins down |
| --- | --- |
| Mermelstein, Brar, Headley 2003, JLT 21(6) 1518, [10.1109/JLT.2003.812461](https://doi.org/10.1109/JLT.2003.812461) | Eqs. 2, 5, 6, 7, 10 and Table I: the dual-order CW cascade and its noise response |
| Keita, Delaye, Frey, Roosen 2006, JOSA B 23(12) 2479, [10.1364/JOSAB.23.002479](https://doi.org/10.1364/JOSAB.23.002479) | Eqs. 4.5–4.7, 4.9, 4.10: closed-form transfer expressions |
| Zhu, Zhang, Zhang 2007, JLT 25(6) 1458, [10.1109/JLT.2007.895532](https://doi.org/10.1109/JLT.2007.895532) | Eq. 7: the single-pump frequency-dependent transfer |

## Scope: an amplifier by default, with opt-in cavity boundaries

The default configuration describes a **distributed Raman amplifier**. That is unchanged, and
every reproduction below was produced with the cavity options absent — passing them or not passing
them gives a byte-identical result, which the test suite asserts rather than assumes.

What is now additionally modelled, and only when asked for:

- **Resonant cavity boundaries.** Per-wavelength, per-end mirror reflectivity, a round-trip
  boundary applied between propagation passes, and an iteration to convergence. See
  [Cavity boundaries](#cavity-boundaries-opt-in) below.
- Threshold, as a *derived* quantity from fiber gain, length and cavity loss, rather than a
  hand-set operating point.

What remains excluded, and is not deferred to a later stage of this module:

- **Backward-propagating Stokes waves.** The cavity iteration closes a single forward pass on
  itself. It models a resonator whose circulating field is taken as forward-going throughout,
  which is the standard first-order treatment of a short Fabry-Perot fiber laser. It does not
  model the counter-propagating wave that Rota-Rodrigo 2020 identify as a precondition for
  anomalous RIN transfer, and results near that regime are not validated.
- **Distributed Rayleigh feedback**, above- and below-threshold relaxation dynamics, cavity
  mirror design, and pump-relay schemes.

If you are looking for threshold behaviour, use `RamanCascadeConfig` with reflectivity set; if
you are looking for random-laser physics, this is still the wrong module.

Amplifier noise *sources* — ASE, spontaneous Raman scattering, the amplifier noise floor — are
also out of scope. See [Raman noise sources](raman-noise.md), which documents that boundary
before the code exists.

## Unit conventions

Every public function takes quantities in documented SI units and performs **no** implicit
conversion. The loss-coefficient normalization is the single most likely transcription error in
this physics, and implicit conversion hides it rather than surfacing it at the call site.

| Quantity | Unit | Note |
| --- | --- | --- |
| Fiber loss | `dB/km` on `RamanChannel`, `1/m` in the closed forms | `1/m = dB/km × ln(10)/10 / 1000` |
| Raman gain coefficient | `(W km)⁻¹` on `RamanChannel` | The literature and data-sheet convention; rescaled to `(W m)⁻¹` internally against the per-metre integration variable |
| `gain_coefficient` in the Keita forms | `1/m` | This is `G`, **not** a linear gain ratio. Keita's Fig. 1 inset lists `G = 4.74 /km` with `L = 1 km`, so `G L = 4.74` |
| `Length` | metres internally | Keita 2006 works in inverse length throughout Eqs. 4.5–4.10 |
| `Wavelength` | as constructed | `Wavelength(1375.0, "nm")` |

One more convention worth stating: **Keita 2006 defines `ρ_dB = 10 ln(ρ)/ln 10`, a power ratio**,
whereas most of this library uses the amplitude convention `20 log10`. Pass
`db_from_linear(rho, amplitude=False)` to follow the paper. The transfer values returned by
`rin_transfer_from_net_gain` are already on the paper's convention.

## The closed forms

### Keita 2006 Eq. 4.6, the low-frequency limit

```python
rho = rin_transfer_low_frequency(G, alpha_p, L)
```

with `rho = (G/α_p)² (1 - e^{-α_p L})²`. This is the exact `Δ_k → 0` limit of Eq. 4.5, and the
two agree to 1e-12 at the Fig. 1 inset reference point `G = 4.74 /km`, `L = 1 km`,
`α = 0.046 /km`, which yields 13.31 dB against the paper's stated ~13 dB. That check is what
fixes the loss normalization, and it is worth doing before writing any ODE code: a unit error
found in a formula costs an afternoon, and the same error found inside a solver costs days.

Note which figure is the gate. Keita's Fig. 2 is the prominent plot and is **unusable** as a
gate because the paper never states the fiber length for it. Fig. 1's inset is fully specified
and carries a stated answer, so that is what the test suite uses.

### Keita 2006 Eq. 4.7, against net gain

`rin_transfer_from_net_gain(g_net_db, alpha_p, L)` maps the paper's **net** gain onto the
transfer. Note `g_net` is not `20 log10(G L)`: the Fig. 1 inset gives `g_net = 40 dB`
alongside `G = 4.74 /km` and `L = 1 km`, for which `20 log10(G L)` is 13.5 dB. The mapping is
the paper's Eq. 3.7, which is not transcribed here, so this function is gated only at the
single point where the paper states both quantities.

### Zhu 2007 Eq. 7, and what is verifiable

```python
h2 = single_pump_transfer(f_hz, g_on_off_db=9.1, alpha_p=..., v_signal=..., l_eff=...)
corner = single_pole_corner(alpha_p, v_signal)   # alpha_p * v_signal / (4 pi)
```

Zhu 2007 states "the other measured parameters of the fiber are given in [11] and [12]", so
its Fig. 1 **DC levels cannot be evaluated from the paper alone**. Rather than fit the unknown
pump loss to reproduce the figure and then present Eq. 7 as verified, the fixture marks the DC
unverifiable and gates the corner frequency and the roll-off slope, neither of which depends on
it. A test guards the record so that a later change cannot quietly "verify" it by fitting.

The roll-off is 20 dB per decade, within 3 dB, one decade past the corner.

## The steady-state cascade

`CWWCascade` solves the coupled power equations of Mermelstein 2003 Eqs. 5a–5c:

```python
from photonics_helper.base import Length, Wavelength
from photonics_helper.raman_transfer import CWWCascade, Geometry, RamanChannel, build_cascade

channels = build_cascade([...], group_index_zero=1.466, slope_ps_per_km_nm2=0.088, lambda0_nm=1312)
cascade = CWWCascade(channels, Length(60.0, "km"), geometry=Geometry.COUNTER_PROPAGATING)
result = cascade.solve()
result.on_off_gain_db(cascade.loss_per_m)   # -1: about 13.41 dB
```

**Integration order.** Mermelstein integrates all the power equations together. This module
splits them into passes, because the counter-propagating geometry makes the signal travel
against the pump direction, which is a two-point problem rather than an initial-value problem:

1. pumps forward from `z = 0` to `z = L`;
2. the signal steady state, forward for co-propagating and backward from `z = L` for
   counter-propagating;
3. the modulation indices, with `m₁` and `m₂` following the pumps forward and `m₃` following
   the signal.

The counter-propagating case is solved by fixed-point iteration between the two passes. The
iteration is well conditioned here because the signal is roughly four orders of magnitude
weaker than the pumps, so the loop gain is the fraction of pump power drained into the signal.

### Two corrections to the printed equations

Both are recorded in the fixture rather than applied silently.

**Eq. 5b.** As printed, it depletes `P₂` by `γ₂₃ P₁ P₃` while Eq. 5c gains the signal by
`γ₂₃ P₂ P₃`. The same Raman interaction cannot be `P₁ P₃` on one side and `P₂ P₃` on the other,
and Eqs. 5a and 5c are self-consistent for both other gain pairs. This module implements the
self-consistent `−γ₂₃ P₂ P₃`.

**Eq. 6.** The printed modulation equations carry no `m_i` self term and no `α_i m_i` term,
and re-deriving the linearization confirms they are exact, not an omission. Linearizing Eqs. 5
for the *relative* modulation index M = δP/P̄, the steady-state net gain of channel i appears
on both sides of the equation — multiplying P̄ on the mean side and δP = P̄ M on the
fluctuation side — and cancels identically. Only cross terms and the walk-off retardation
survive: exactly the printed Eqs. 6a to 6c. An early revision of this module "corrected" the
printed equations by adding the diagonal `−(α_i + Σ_j γ_ij P̄_j) m_i`, which is the correct
equation for the *absolute* perturbation but wrong for a relative one: it re-amplified each
pump's own noise by the whole Raman net gain, collapsed the published ~15 dB gap between the
two pumps' DC transfers to under 1 dB, and pushed every DC level 2 to 23 dB high. Reading the
paper literally resolves all four DC levels and the 15 dB structure at once; that resolution
is recorded in the `reproduction_status` block of the Mermelstein fixture.

### Group index

Mermelstein 2003 prints `n_i = n_0 + (S_0/8)(λ_i − λ_0²/λ_i)²`. That form is dimensionally
inconsistent — the bracket carries units of length squared while the dispersion slope carries
time per length cubed — and substituting Table I directly gives a group index of 2293, which is
not a physical value. This module instead uses `dn_g/dλ = c D(λ)`, which gives 1.46681 at
1560 nm, consistent with the standard single-mode value near 1.468 at 1550 nm. Both the broken
printed form and the resolution are recorded in the fixture and guarded by tests.

### Which quantity is conserved, and why there is a flag

Raman conversion is photon-count preserving: a 1375 nm photon becomes a 1465 nm photon and the
energy difference goes into a phonon. Two consequences follow, and the model has to pick one:

- **Mermelstein's Eqs. 5 move equal _power_** between a coupled pair, `dP_i = −g P_i P_j`,
  `dP_j = +g P_i P_j`. Their conserved quantity is therefore **power**.
- Real stimulated Raman scattering moves equal **photon number**, so the Stokes power gain is
  reduced by `λ_pump/λ_stokes`.

`CWWCascade(..., photon_consistent=True)` selects the second. The default is the first, because
that is what the paper's curves were produced with and it is what the 13 dB gate is stated
against. The distinction is not cosmetic: photon flux drifts by about 3 percent of the
transferred power in the default model, which is a property of those equations and not a solver
error.

Note the weighting. A photon count is `P λ/(hc)`, so the conserved sum is `Σ P_i λ_i`, weighted
by `P·λ`, **not** `Σ P_i/λ_i` — the latter weights by photon *energy* and is conserved by
nothing physical. `photon_flux()` returns the true photon rate.

```python
result.power_balance_drift     # < 1e-6 in both geometries for the default model
result.photon_flux_drift       # < 1e-6 only with photon_consistent=True
```

Both diagnostics subtract the analytic fiber loss before reporting, and both are evaluated on a
grid 100 times finer than the returned one: at 601 points over 60 km the trapezoidal
integration of the loss rate alone contributes about 5e-4, which would swamp the target and
read as a physics error. A drift beyond 1e-6 raises a `RuntimeWarning` rather than being
returned as if it were a result.

The counter-propagating signal needs **no** special handling in the budget, which is worth
stating because it is easy to get backwards. A signal travelling against `z` satisfies
`dP_s/dz = rhs` with the same right-hand side as a co-propagating channel, and `rhs` carries
`−α_s P_s` and a positive gain term; integrating it over a decreasing `z` makes the power grow
towards `−z` on its own. So the budget is the plain sum over all channels, and only the input
and output bookkeeping in `result.input_w` and `result.output_w` needs to know about the
reversal.

## Walk-off geometry

`Geometry.CO_PROPAGATING` and `Geometry.COUNTER_PROPAGATING` are the same equations with
opposite walk-off sign, selectable on the same solver. There are no separate code paths,
because there is no separate physics.

Mermelstein's Eq. 2 gives the walk-off between a pump and the signal as

```
d = 1/v_signal  -/+  1/v_pump
```

with the upper sign for co-propagating beams and the lower sign for counter-propagating ones.
For co-propagating channels of nearly equal group velocity this nearly cancels, which is the
physical statement that they do not separate; for counter-propagating channels the two
inverse velocities add.

In the modulation equations that walk-off multiplies the pump's **own** modulation index,
`i Ω d m_i`, as in Mermelstein's Eqs. 6a and 6b. Rotating `m_i` against the gain envelope turns
the transfer into a Fourier-type integral whose magnitude falls as `1/|A_i − i Ω d_i|` — a
clean single pole with a 6 dB corner at `A_i/(2π d)`. The corner therefore scales as the inverse
of the walk-off, which is why the co-propagating corner sits about four orders of magnitude
above the counter-propagating one.

**Putting the retardation on the coupling term instead, as a `z`-weighted phase, produces a
resonance rather than a pole.** The phase then oscillates inside the integration and averages
out, and the response rises with frequency instead of falling. That was measured, not assumed.

An inverted geometry sign swaps the two silently, so `test_corner_frequencies_follow_the_walk_off_geometry`
asserts that the two geometries are separated by orders of magnitude.

## The noise response

`noise_response(frequencies_hz, source=1)` returns the complex modulation-index ratio from one
perturbed pump to the signal, integrated along the converged steady state.

```python
response = cascade.noise_response(np.logspace(1, 4, 60), source=1)
response.db               # 20 log10 |H|, the amplitude convention Mermelstein reports
response.double_pole()    # (dc_db, corner_hz) via Eq. 10
```

`double_pole_fit` is the trial function of Mermelstein Eq. 10. It is fitted on the linear
amplitude by nonlinear least squares over `(dc, corner)`, not on the decibels, because the
trial function is rational and a fit in log space biases the corner.

`modulation_indices(z, frequency_hz, source)` keeps the same solution as a profile along the
span. `interaction_length_km` measures the full width at `1/e` of the **absolute** power
fluctuation `dP_2 = |m_2(z)| P_2(z)`, which is what Mermelstein Fig. 7 plots.

## Reproduced results, and what does not reproduce

Running `examples/44_cw_cascade_rin_transfer.py` gives:

| Target | Published | Model | Verdict |
| --- | --- | --- | --- |
| Co-propagating on-off gain | ~13 dB | 13.41 dB | reproduced |
| DC, counter, 2nd order | 15.6 dB | 13.9 dB | reproduced, within 1.8 dB |
| DC, counter, 1st order | 0.04 dB | −0.4 dB | reproduced, within 0.5 dB |
| DC, co, 2nd order | 15.4 dB | 14.0 dB | reproduced, within 1.5 dB |
| DC, co, 1st order | 0.7 dB | −0.3 dB | reproduced, within 1.1 dB |
| Counter 6 dB corner, 2nd order | 1.33 kHz | 1.52 kHz | reproduced, 14% |
| Counter 6 dB corner, 1st order | 1.59 kHz | 2.04 kHz | reproduced, 28% |
| Co-propagating 6 dB corner, 2nd order | 11.2 MHz | 8.8 MHz | reproduced, 21% |
| Co-propagating 6 dB corner, 1st order | 18.5 MHz | 16.1 MHz | reproduced, 13% |
| Direct interaction length, Fig. 7 | 20.5 km | 20.6 km | reproduced (0.5%) |
| Indirect interaction length, Fig. 7 | 25.5 km | 27.8 km | reproduced (9%) |

Corners are read as the 6 dB drop below the response's own DC level rather than through the
double-pole fit, because the model response is not exactly double-pole and Mermelstein's own
published corners come from a double-pole fit of measured data — comparing fit parameter to fit
parameter was over-reading both.

The **DC levels reproduce once the printed Eqs. 6 are taken literally.** The one structural
detail that matters is that the modulation equations carry no self term: for a relative
modulation index the channel's own net gain multiplies mean and fluctuation identically and
cancels, so each pump's noise reaches the signal only through the other channels. That is what
puts the first-order transfer about 15 dB below the second-order, as published, instead of
nearly equal. The residual is second-order 1.5 to 1.8 dB low in both geometries and first-order
0.3 to 1 dB low, one direction for all four, consistent with the residual uncertainty in the
fiber parameters rather than a structural error of the model.

The **corners track the paper to 13–28 percent** in both geometries, with the order-to-order
ratios close (0.74 vs 0.84 counter-propagating, 0.55 vs 0.61 co-propagating). A wrong walk-off
sign or a unit slip would move them by orders of magnitude rather than tens of percent.

The **Fig. 7 interaction lengths are reproduced once the plotted quantity is measured
correctly**: direct 20.6 km against 20.5 published and indirect 27.8 km against 25.5. The
figure plots the *absolute* 1465 nm pump power fluctuation `dP_2 = m_2 P_2` in mW, not the
relative modulation index, and its arrows span the **full width at 1/e** of the peak. An
earlier implementation measured the relative index from the peak and got 17.4 km direct
and an unbounded indirect; that was a measurement-definition error, not a physics one.

## Benchmarks and provenance

Every reproduced target lives in `benchmarks/raman_noise/<paper>/fixture.json` with its source,
DOI, unit, and whether it was read from a table, a figure, or the running text:

```python
from photonics_helper.raman_transfer import load_benchmark
fx = load_benchmark("mermelstein2003")
fx.value("parameters", "second_order_pump_power")     # 0.780
fx.is_figure_transcribed("targets", "counter_propagating", "dc_db", "second_order")   # True
```

`load_benchmark` fails loudly with `ProvenanceError` on a missing DOI, a blank DOI, an
unknown `kind`, or a value that is not a provenance record at all. A benchmark that silently
loses its provenance is worse than no benchmark.

### Benchmark index

| Fixture | Paper | DOI | Used by |
| --- | --- | --- | --- |
| `mermelstein2003` | Mermelstein, Brar & Headley 2003 | [10.1109/JLT.2003.812461](https://doi.org/10.1109/JLT.2003.812461) | CW cascade, stage 0 |
| `keita2006` | Keita, Delaye, Frey & Roosen 2006 | [10.1364/JOSAB.23.002479](https://doi.org/10.1364/JOSAB.23.002479) | closed forms, dispersion sweep |
| `zhu2007` | Zhu, Zhang & Zhang 2007 | [10.1109/JLT.2007.895532](https://doi.org/10.1109/JLT.2007.895532) | Eq. 7, six-pump reproduction |
| `krause2006` | Krause, Cierullies, Renner & Brinkmeyer 2006 | [10.1016/j.optcom.2005.10.077](https://doi.org/10.1016/j.optcom.2005.10.077) | RFL cavity reproduction |
| `rizzelli2016` | Rizzelli et al. 2016 | [10.1364/OE.24.029170](https://doi.org/10.1364/OE.24.029170) | ultra-long amplifier (attempted) |
| `ma2012` | Ma & Fathpour 2012 | [10.1364/OE.20.017962](https://doi.org/10.1364/OE.20.017962) | silicon material coupling |
| `rotarodrigo2020` | Rota-Rodrigo et al. 2020 | [10.1364/OE.403684](https://doi.org/10.1364/OE.403684) | measured fiber parameter sets |
| `babin2005` | Babin et al. 2005 | [10.1109/LPT.2005.859547](https://doi.org/10.1109/LPT.2005.859547) | beat spectrum, cavity gate |

Every fixture also carries a top-level `tuned_parameters` list. It must be empty for a target
to pass; `assert_not_tuned` raises `ReproductionError` naming any offending parameter. This is
the failure mode that produced the original rejected proposal — a reproduction that silently
tunes until the curve matches is indistinguishable from one that genuinely matches, until
someone checks.

### Recorded contradictions

**Mermelstein's power discrepancy.** The running text of Section III states launched powers of
760 mW and 22 mW; Table I states 780 mW and 20 mW. The fixture uses the **table** values,
records the text values, and records why: the table is what produced the published simulation
curves. Silently picking one would hide a real ambiguity from a future reader and from a
reviewer.

**Mermelstein's group index.** As described above, the printed Sellmeier form is dimensionally
inconsistent and yields 2293. The fixture stores the broken form, the observed value, and the
resolution.

**Zhu's DC level.** Marked unverifiable rather than fitted, with references [11] and [12] named
as the resolution.

**Keita's dB convention.** `ρ_dB = 10 ln(ρ)/ln 10` is a power ratio, not the amplitude convention
used elsewhere in this library. Recorded in the fixture and in the module docstring.

## Benchmark fixtures, not inline constants

Sixteen numbers plus two interaction lengths is too much to hold in test bodies, and a value
retyped into a test each time drifts from the paper it came from. The targets are data a test
reads. If a later change breaks the physics, the assertion fires against the recorded value.

## References

- R01. Mermelstein, Brar, Headley 2003, *J. Lightwave Technol.* **21**(6), 1518. [10.1109/JLT.2003.812461](https://doi.org/10.1109/JLT.2003.812461)
- R02. Keita, Delaye, Frey, Roosen 2006, *JOSA B* **23**(12), 2479. [10.1364/JOSAB.23.002479](https://doi.org/10.1364/JOSAB.23.002479)
- R08. Zhu, Zhang, Zhang 2007, *J. Lightwave Technol.* **25**(6), 1458. [10.1109/JLT.2007.895532](https://doi.org/10.1109/JLT.2007.895532)
---

## Stage 1: split-step cascade and noise extraction

Stage 0 solved the cascade as nine coupled ODEs. Stage 1 propagates it as a
split-step Fourier GNLSE with genuine inter-order Raman coupling, and adds the
stochastic sources that make a RIN spectrum computable at all.

### The coupling tensor

`MultimodeSplitStepEngine` now accepts two optional tensors, `coupling_tensor`
(N×N complex) and `fwm_coupling_tensor` (rank-4 complex), both in
W⁻¹ m⁻¹. With neither supplied the engine is **bit-identical** to its previous
behaviour — verified by stashing the change and comparing outputs with
`np.array_equal`, not by tolerance.

The tensors carry **absolute** coefficients rather than dimensionless
multipliers of a single scalar γ. That is deliberate: a Raman gain-transfer arm
needs a complex entry (an imaginary antisymmetric pair is a
photon-number-conserving energy exchange) and a per-order scale, neither of
which a real symmetric overlap weight can express.

For an adjacent pair (donor `j`, acceptor `j+1`),

```
Im C[j,   j+1] = +g/2          donor loses
Im C[j+1, j   ] = -e * g/2     acceptor gains
```

with `e = λ_j / λ_{j+1}`. One pump photon becomes one Stokes photon and the
energy difference goes into the phonon, so `e` is the photon-consistency
factor. `e = 1` conserves **power** (Mermelstein 2003 Eqs. 5a–5c); `e = λ_j/λ_{j+1}`
conserves **photon number** exactly. Note the split: the pump loses the full
`g P_j P_{j+1}` and the Stokes receives `e` of it. Dropping `e` on the Stokes
arm is a 6.5% error, wide enough to hide behind a loose cross-check tolerance.

The diagonal carries the Kerr arm, `C[k,k] = γ_k = n₂ω_k Γ/(c A_eff,k)`,
evaluated at each order's own wavelength and area — pump and Stokes commonly
differ by more than an order of magnitude in `A_eff`, and that ratio often sets
the whole cascade.

### Why the gain is data, not a spectrum

The gain coefficients are read from published measurements with DOI provenance
(`RamanGainTable.from_fixture`), the same discipline stage 0 applies to
Mermelstein 2003 Table I.

This is a considered decision, not a shortcut. A two-parameter gain spectrum
**cannot** reproduce a measured multi-order gain table. Against Mermelstein's
three measured gains — 0.527, 0.419 and 0.038 (W km)⁻¹ for 1375→1465,
1465→1560 and 1375→1560 — the implied normalization constant `g / shape(offset)`
spreads by a factor of **23**, whether the shape is `Im H̃` from the library's
causal `h_R`, its `Re H̃`, or a Lorentzian built from `RamanSpec`'s own 440 cm⁻¹
shift and 45 cm⁻¹ linewidth. `Re H̃` additionally flips sign across the
resonance. The cause is physical: silica's fused-fiber gain spectrum has broad
sub-resonance structure that a two-parameter model does not carry. In
particular the measured second-order gain at 25.9 THz is ~9× higher than any
Lorentzian wing predicts.

`raman_overlap_integral` still exposes the spectral arm for inter-order pairs
that no measurement covers, as a **shape** rather than an absolute coefficient.

### Cross-validation against stage 0

The split-step cascade and the stage-0 continuous-wave ODE cascade are two
independently implemented integrators. In the undepleted regime with matched
fiber they agree to **0.2% on the pump and 1% on the Stokes** — the strongest
check in the change, and the real replacement for the spectral gate that was
originally written for task 2.3 and is not physically achievable.

A deliberately wrong gain (1.5×) is caught by the same comparison, which is the
property that makes the check worth having.

### Conservation

Photon flux `Σ P_k λ_k` is conserved to **2 × 10⁻⁹** relative over a deep-
depletion 60 km cascade, against a 1 × 10⁻⁶ gate.

### Measured runtime

x86_64, 24 logical cores, CPython 3.13.12, 60 km span, coupling tensor active:

| orders | grid N | outer steps | s/run | ms/step |
| ------ | ------ | ----------- | ----- | ------- |
| 2      | 1024   | 2000        | 0.81  | 0.41    |
| 3      | 1024   | 2000        | 1.75  | 0.88    |
| 3      | 2048   | 4000        | 3.08  | 0.77    |
| 4      | 2048   | 4000        | 6.00  | 1.50    |

A single 3-order cascade is ~3 s, **not** the "minutes" the stage 1 proposal
assumed. The number that matters for stage 3 is the *ensemble* cost: a RIN
spectrum needs many realizations per parameter point, so at 200 realizations ×
3 s one sweep sample is ~10 min of a core and a 10³-sample dataset is ~7
core-days, or roughly 7 hours wall-clock across 24 cores. Budget from the
ensemble figure, not the per-run figure.

### Keita 2006 dispersion benchmark

At the Fig. 1 reference point (G = 4.74 km⁻¹, α = 0.046 km⁻¹, L = 1 km), the
transfer is **40.1 dB** lower for D = 200 than for D = 2 ps/(nm km) at 10¹¹ Hz,
monotone in D at every offset, with the roll-off corner moving one decade per
decade of D. That is Fig. 1's structure.

Precisely: it is the transfer **at a fixed offset** that falls by two orders,
not the DC value — all three curves share the same −0.4 dB plateau at low
frequency, because dispersion enters through the walk-off term and vanishes as
ν_s → 0.

The walk-off is `Δk = D(λ_p − λ_s) ν_s`, where **ν_s is the angular offset** —
the paper defines `f_s = ν_s/(2π)`. The unit chain matters: D in ps/(nm·km)
times Δλ in nm is ps/km = 10⁻¹⁵ s/m, converting to 1/km multiplies by 1000,
and the 2π converts the offset frequency to the angular frequency. Omitting
either factor makes every D look identical and the sweep flat.

**That sweep is a closed-form result and does not by itself validate the
solver.** See the next section for the solver cross-check, which does not yet
pass.

### Validity of the closed forms

Keita 2006 Eq. 4.8 requires

```
z >> 2π / [(1/v_p -/+ 1/v_s) Δω_p]
```

For the co-propagating case, where `(1/v_p − 1/v_s) ≈ D(λ_p − λ_s)`, this is
**z >> 3 × 10⁻³ km** for a 10 nm pump linewidth *even at D = 2 ps/(nm·km)*.
Counter-propagating needs only z >> 10⁻⁷ km, because the velocities nearly
cancel. The paper states explicitly that the condition would **not** be met for
a 1 km fibre pumped by a 0.1 nm bandwidth source.


### Solver vs. closed form (resolved)

The numbers above come from the stage 0 analytic functions. Cross-checking the
split-step solver against them — `benchmarks/raman_noise/keita2006/dispersion_solver_vs_closed.py`
— **now passes**: the solver agrees with Keita 2006 Eq. 4.5 to about **1 dB**
over `Δk·L = 0.2` to `66` rad.

**Method.** Deterministic linear response, no ensemble: a coherent pump
amplitude modulation (Keita Eq. 4.1) at a single offset, the Stokes modulation
demodulated at the same offset, `ρ = (m_s/m_p)²`, compared against
`rin_transfer_monochromatic_pump`.

An earlier version of this benchmark reported a systematic **−17.6 dB** offset
and a roll-off that started a factor of 2π too early. That was **not** the
large-bandwidth approximation's validity boundary, as first hypothesised. It
was three convention errors, all on the benchmark's side, all fixed together:

- **The wrong equation.** The solver drives a modulated *monochromatic* pump,
  which is Eq. 4.5 (Fig. 1). The benchmark compared it against Eq. 4.9/4.10,
  the *large-bandwidth* pump case (Fig. 3). One modulation tone is
  monochromatic however fast it is.
- **A missing 2π in the walk-off.** Keita's `Δk = (1/v_p − 1/v_s)·ν_s` uses the
  **angular** offset: the paper defines `f_s = ν_s/(2π)`. The benchmark passed
  `f` in Hz, so the closed form's walk-off was 2π too small and its roll-off
  appeared 2π late. The solver's own walk-off comes from the group delays and
  was already angular.
- **A demodulation factor of 4.** The field-based `ρ` of Eq. 2.16 is recovered
  with `|F|/(P₀·T)`. The `4|F|/(P₀·T)` form recovers `2·m_s` for a sine
  (verified on a synthetic signal) and lands 12 dB high; `2|F|/(P₀·dt)` is off
  by exactly `N` (`+72.25 dB` at `N = 4096`).

With those fixed, the sweep is:

| D (ps/nm/km) | f (Hz) | Δk·L (rad) | Eq. 4.5 (dB) | solver (dB) | diff (dB) |
| --- | --- | --- | --- | --- | --- |
| 2 | 10⁸ | 0.22 | 13.30 | 12.25 | −1.05 |
| 2 | 10⁹ | 2.20 | 11.49 | 10.43 | −1.06 |
| 2 | 3×10⁹ | 6.60 | −13.07 | −12.18 | +0.89 |
| 2 | 10¹⁰ | 22.0 | −7.51 | −8.27 | −0.76 |
| 20 | 3×10⁹ | 66.0 | −17.05 | −17.76 | −0.71 |
| 200 | 10⁸ | 22.0 | −7.51 | −8.27 | −0.76 |

At the low-frequency end the solver also matches Eq. 4.6, the paper's stated
"about 13 dB" noise addition.

The solver reproduces the closed form's **exact invariance in the product
`D·f`**, bin for bin — D = 2 at 10¹¹ Hz, D = 20 at 10¹⁰ Hz and D = 200 at 10⁹ Hz
all give identical solver values. That is the structural claim Eq. 4.8 rests on.

**The strongly walk-off-limited region is ill-conditioned, not validated.**
Once `Δk·L ≳ 220` rad the transfer is below −45 dB and sits on a `sinc²` null,
where the closed form is hypersensitive to the exact walk-off (`cos(Δk·L)`
swings through a full period for a sub-percent change in `Δk`). Those points are
reported by the benchmark and are not gated. The test gates `Δk·L ≤ 66` rad at a
1.5 dB tolerance.

**Consequence for stage 3.** The walk-off term and the spatial averaging are now
cross-validated in the well-conditioned regime, which was the blocker. The
strongly walk-off-limited regime remains untested against an independent
solver; treat results there as unvalidated.

## Cavity boundaries (opt-in)

Set `cavity_reflectivity_input` and `cavity_reflectivity_output` on
`RamanCascadeConfig` (one power reflectivity per order per end) and
`MultimodeSplitStepEngine.cavity_iterate` closes the propagation into a resonator.
Leaving them `None` — the default — is the amplifier above, unchanged.

### Formulation

One cavity iteration is: propagate every channel through the fiber, apply the
round-trip boundary, re-inject the pump, repeat.

The boundary is applied to the **Stokes channels only** (`1 .. N-1`). Channel 0 is the
pump; it is re-injected at its launch value on every iteration rather than reflected,
because a Raman laser is pumped from outside and the pump mirror is not part of the
round trip in the configurations this gate is written against. For each Stokes channel
`k` the boundary is

```
A_k(0) <- sqrt(R_in[k]) * exp(i * beta_k * L_rt) * sqrt(R_out[k]) * A_k(L)
```

with `beta_k = 2 pi n_g / lambda_k` and `L_rt = cavity_round_trip_length`. The
reflectivities are **power** reflectivities, hence the square roots on amplitudes. The
phase `beta_k * L_rt` is the one-way round trip at the group velocity; it is not
arbitrary — it is the representation of the round-trip delay `T_rt = n_g L_rt / c`, and
it is what generates the longitudinal-mode beat structure at `f_beat = c / (2 n_g L_rt)`
through the frequency dependence of the Stokes spectra. The group index `n_g` defaults
to 1.466 (silica near 1.5 um, and the value Babin 2005 quote) and is settable per
configuration as `cavity_group_index`.

### Convergence criterion

Convergence is judged on the **circulating intensity**, not on the complex field:

```
change = || |A|_new|^2 - |A|_old|^2 ||_2 / || |A|_old|^2 ||_2
```

This is deliberate and the reason is worth stating, because the obvious criterion is
wrong here. A complex-field residual `|| F(A) - A || / || A ||` cannot go to zero for
*any* converged cavity: the round-trip map carries an absolute phase `exp(i beta_k L_rt)`
whose argument is of order `1e8` rad, whose residue mod `2 pi` is arbitrary, and whose
magnitude never settles. The residual therefore plateaus at `|1 - m e^{i phi}|` — a
nonzero constant even for a perfectly converged, contracting cavity. A solver written on
that residual raises on exactly the configurations it is meant to accept; this was
observed directly during development, where the reported relative change sat at 1.552
forever on a cavity that was in fact relaxing cleanly at 0.448 per round trip. The
intensity envelope is phase-insensitive and is the observable that every consumer of a
cavity result actually reads.

### Iteration bound

`max_iterations` defaults to **100**. Exceeding it raises `RuntimeError`; a partially
converged field is never returned. This is not a formality: a cavity above its Stokes
threshold genuinely has no fixed point under plain fixed-point iteration, and the test
suite asserts that such a configuration raises rather than silently reporting the last
iterate as a converged state. Raising is the correct answer there, not a bug.

### Beat period

`cavity_beat_period()` returns `c / (2 n_g L_rt)`. For the Babin 2005 pump cavity
(`L_rt = 16 m`, `n_g = 1.466`) that is 6.39 MHz. Babin quotes "about 6 MHz" because 16 m
is itself a rounded round number; the gate is the 10 % agreement with `c / 2 L n`,
which is the figure derived from the cavity rather than quoted from the paper.

## Stage 2: multiple pumps, material coupling, and published reproductions

This section documents `photonics_helper.raman_multipump`, `photonics_helper.raman_materials`,
and `photonics_helper.raman_reproductions`, which extend the cascade to an arbitrary set of
pumps and to materials other than silica.

### Multiple-pump bidirectional cascade

A cascade is a set of channel descriptors, each carrying its own wavelength, power and
propagation direction (`PropagationDirection.FORWARD` or `BACKWARD`). Pump count is not capped.
Inter-channel coupling includes **pump-to-pump** and **signal-to-signal** pairs as well as
pump-to-signal, because Zhu 2007 shows that both increase the magnitude of the pump-to-signal
transfer.

The current continuous-wave solver supports two geometries: every channel forward
(co-propagating), or pumps forward with the signal backward (counter-propagating). A mixture
that is neither raises rather than silently picking one.

### Incoherent versus coherent pump combination

When several pumps drive one signal, their RIN combines differently depending on whether the
pumps are mutually coherent:

- **Coherent** (Zhu 2007's "virtual one pump"): `m_eq = |sum_j P_j m_j| / P_eq`. The equivalent
  pump modulation adds as a complex amplitude.
- **Incoherent** (the real multiple-wavelength case): `m_eq = sqrt(sum_j |P_j m_j|^2) / P_eq`,
  the square root of the summed mean squares. Cross terms average to zero for independent
  random phases.

**Which is physically typical.** Real WDM pump sources are mutually incoherent, so the
incoherent rule is the physically important one and the default. Zhu 2007's central finding is
that the real six-pump case transfers *more* than the coherent equivalent — the paper's Fig. 1
shows the real case about 8 dB above the virtual one-pump case. Note that the equivalent pump
*modulation index* is larger for the coherent sum, but the mean *RIN transfer*
`|m_signal|^2 / |m_eq|^2` is smaller, because the coherent denominator is larger. The stage 2
task list and the delta spec state the opposite ("coherent summation produces higher transfer");
that wording is inconsistent with the paper's own Fig. 1 rationale and with the algebra, and the
code follows the paper. This discrepancy is recorded rather than papered over.

### Material-parameterized Raman coupling

Raman coupling for a non-silica material is built from the `materials.db` `raman_specs` row, not
from an inline constant. `load_material_raman_gain` requires a non-empty reference and a DOI and
raises `ProvenanceError` otherwise. The DOI is supplied explicitly because the table stores a
citation string rather than a DOI.

`raman_shift_ratio("Si", "Silica")` returns **1.18** (520 cm⁻¹ / 440 cm⁻¹). The design document
and tasks 3.3/3.5/7.4 claim the silicon Raman shift is "roughly three times" the silica value;
that claim is **not supported** by the database rows and the reproduction records the measured
ratio instead of adjusting it. The silicon gain coefficient (9 m/GW, Ma 2012) is supplied from
the fixture because the `Si` row has no `gain_coeff`.

### The four reproductions

Every reproduction registers its tolerance, its parameters, and whether any were tuned. A target
that required tuned parameters cannot pass: `assert_not_tuned` raises `ReproductionError` naming
the offending parameter. Targets are reported as **passed** or **attempted**, and an attempted
target names the missing parameter.

| Reproduction | DOI | Verifiable targets | Attempted targets (missing parameter) |
| --- | --- | --- | --- |
| Zhu 2007 six-pump | [10.1109/JLT.2007.895532](https://doi.org/10.1109/JLT.2007.895532) | Eq. 7 corner within 1 kHz; roll-off within 3 dB/decade of −20; real-versus-virtual gap positive; pump-count trend | Absolute DC levels — the inter-pump gain matrix and fiber loss are deferred to refs [11],[12] |
| Krause 2006 RFL | [10.1016/j.optcom.2005.10.077](https://doi.org/10.1016/j.optcom.2005.10.077) | FSR comb spacing = `v_g/(2L)`; mean transfer falls with pump power; output-coupler, gain and length design rules | — (reproduced) |
| Rizzelli 2016 ultra-long | [10.1364/OE.24.029170](https://doi.org/10.1364/OE.24.029170) | — | Reflectivity and forward-pump trends — distributed-feedback FBG boundary (stage 3) |
| Ma 2012 silicon | [10.1364/OE.20.017962](https://doi.org/10.1364/OE.20.017962) | Silicon/silica Raman-shift ratio | Low-frequency RIN transfer — silicon Raman laser cavity (stage 3) |

**Krause 2006 is now reproduced** by the Raman fiber laser cavity model in
`photonics_helper.raman_rfl` (see [Stage 2b](#stage-2b-the-raman-fiber-laser-cavity)):
the FSR comb spacing, the pump-power trend, and the output-coupler, gain and length design
rules all pass. Rizzelli 2016 and Ma 2012 still need a *second-order* cavity (Rizzelli) or
a silicon cavity with pump mirrors (Ma), and are marked attempted rather than tuned. The
Zhu absolute DC is not verifiable because the paper defers the fiber loss and the inter-pump
gain matrix to external references; the model uses a Lorentzian lineshape normalized to the
measured silica peak, and that model choice is recorded in the fixture.

### Four-wave-mixing decision

Inter-channel four-wave mixing is **not** enabled in the Raman continuous-wave path. The
computable reproduction (Zhu 2007) runs through the continuous-wave cascade, which carries only
the Raman gain arm, and the cavity-limited targets are gated by the cavity boundary rather than
by four-wave mixing. The four-wave-mixing tensor slot remains present in
`MultimodeSplitStepEngine` for the split-step cascade, where it belongs.

### Measured ensemble cost

The four reproduction reports and their test suite run in about **10 s** on one CPU core
(`tests/test_raman_reproduction.py`, 29 tests). The Zhu six-pump transfer evaluation is the
dominant cost: each frequency requires one linearized integration per pump. A full frequency
grid of 24 points over six pumps takes a few seconds. This is the number stage 3 should budget
from, and it is well within the amplifier regime; the expensive part of stage 3 is the ensemble
average, not the per-configuration solve.

### Example

`examples/46_cascade_rin_vs_published.py` plots the Zhu real-versus-virtual transfer against the
published DC levels and prints every reproduction report, separating passed from attempted.

## Stage 2b: the Raman fiber laser cavity

`photonics_helper.raman_rfl` models a Raman fiber laser: one pump and one Stokes order,
with the Stokes closed into a cavity by two fiber Bragg gratings. This is what the
Krause 2006 reproduction needs, and it is the capability the stage 2 amplifier-only scope
explicitly deferred.

### Steady state

Forward pump and forward/backward Stokes:

```
dP_p /dz  = [-alpha_p - gR (lambda_s/lambda_p) (P_s^+ + P_s^-)] P_p
dP_s^+/dz = [-alpha_s + gR P_p] P_s^+
dP_s^-/dz = [+alpha_s - gR P_p] P_s^-
```

with `P_p(0) = P_0`, `P_s^+(0) = R_l P_s^-(0)` and `P_s^-(L) = R_r P_s^+(L)`. The
`lambda_s/lambda_p` factor on the pump depletion makes the exchange conserve photon
number rather than power, the same convention as `raman_transfer`.

**Why shooting and not collocation.** The trivial state `P_s = 0` is always a solution, so
a collocation BVP solve converges to it and returns a dark cavity. The solver instead
shoots on the single unknown `P_s^-(0)`: the whole three-component system is then an
initial value problem, and the right-hand boundary gives one scalar residual whose
non-trivial root is the lasing state. Below threshold there is no non-trivial root and the
trivial one is returned.

### Pump-to-Stokes RIN transfer

Fourier transforming the linearized equations gives a linear boundary value problem. It is
solved **by superposition**, not by fixed-point iteration: `y(z) = y_a(z) + v y_h(z)`,
where `y_a` is the response to the pump perturbation and `y_h` the homogeneous response to
a unit backward-Stokes launch. The boundary condition gives

```
v = (R_r y_a[1] - y_a[2]) / (y_h[2] - R_r y_h[1])     at z = L
```

The denominator is the round-trip resonance, and it is why the FSR comb appears. A
fixed-point iteration of the same problem converges only for a sub-threshold round trip and
returns the flat, non-resonant solution at threshold, which is the physically interesting
case.

**The advection sign is the whole resonance.** A forward wave `f(t - z/v)` contributes
`-iΩ/v`; a backward wave `f(t + z/v)` contributes `+iΩ/v`. If both carry the same sign the
round-trip phase cancels and the transfer is *exactly* frequency-independent — a flat
line with no comb. The opposite sign accumulates `2ΩL/v` per round trip and puts the peaks
at multiples of `v_g/(2L)`. This is not a detail that can be checked by inspection; it is
checked by asserting the peak spacing equals the FSR.

### Krause 2006 reproduction

For the experimental RFL (`L = 5 km`, `v_g = 2e8 m/s`) the FSR is 20 kHz and the model
puts the transfer peaks at 20, 40, 60, 80, 100 kHz. The mean level falls from about
+4.8 dB at 2 W to about +1 dB at 4 W, matching the published trend. The three design rules
of Section 4.2 are reproduced through the Q-factor penalty of Eqs. 24-25:

- the penalty falls monotonically with output-coupler reflectivity (Fig. 6);
- higher Raman gain lowers the penalty and the required pump power (Fig. 7);
- the penalty has an interior optimum in fiber length (Fig. 8).

## Stage 3b: the sweep dataset

`benchmarks/raman_noise/generate_raman_dataset.py` builds a coarse 60-sample dataset over
the RFL parameter box (pump power, length, output reflectivity) using the resumable sweep
driver, storing the transfer at four probe frequencies including the 20 kHz FSR point.
`tests/test_raman_dataset.py` gates the properties the stage 3 spec requires:

- shards are written atomically and are never left truncated or zero-byte;
- an interrupted sweep resumes and reproduces an uninterrupted run byte-for-byte;
- the resume skips completed points (verified with an evaluation counter, not just by
  comparing bytes);
- a corrupted shard is discarded and recomputed;
- the Krause 2006 experimental configuration is recovered from the dataset alone, matching
  the direct computation to 1e-9.

Three real driver bugs were found and fixed while writing those gates:

1. **Zero-byte shards.** `np.savez` appends `.npz` to any name lacking it, so a temporary
   named `*.tmp` wrote its data to `*.tmp.npz` while `Path.replace` moved the empty `*.tmp`
   into place. The temporary now carries an explicit `.npz` suffix.
2. **Pickled results.** Results were stored as a NumPy object array, which
   `allow_pickle=False` refuses to load, so every shard looked corrupt and the resume
   silently recomputed everything. Results are now stored as one JSON string.
3. **Trusting the manifest.** The resume recomputed only the indices the manifest was
   missing, so a truncated shard whose indices the manifest still listed was never
   repaired. Every shard the manifest claims is now re-loaded and verified, and any that
   fails has its indices dropped and recomputed in full.
