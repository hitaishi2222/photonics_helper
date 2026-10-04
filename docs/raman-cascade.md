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

## Scope: an amplifier, not a laser

Everything here describes a **distributed Raman amplifier**. Cavity boundaries, above-threshold
dynamics, distributed Rayleigh feedback and pump-relay schemes are out of scope and belong to
the later cascade stages. If you are looking for threshold behaviour, this is the wrong module.

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

**Eq. 6.** The printed modulation equations carry no `m_i` self term and no `α_i m_i` term.
Linearizing Eqs. 5 about the same steady state necessarily produces the diagonal term
`−(α_i + Σ_j γ_ij P̄_j) m_i`, which is the net linear gain plus loss of channel `i`. Without it
the equations are not the derivative of the power equations and the response has no
low-frequency limit at all. A related sign point: the pair term that couples `m_i` to `m_j`
enters the diagonal with the *opposite* sign, because a gain term grows a modulation index
while a loss term shrinks it. Writing the diagonal as `α` plus the coupling sum turns every
gain into a decay and lands the first-order pump 38 dB too low.

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
span, which is what Mermelstein Fig. 7 plots, and `interaction_length_km` measures the e-folding
decay of that profile.

## Reproduced results, and what does not reproduce

Running `examples/44_cw_cascade_rin_transfer.py` gives:

| Target | Published | Model | Verdict |
| --- | --- | --- | --- |
| Co-propagating on-off gain | ~13 dB | 13.41 dB | reproduced |
| Counter-propagating 6 dB corner, 2nd order | 1.33 kHz | 1.31 kHz | reproduced, 1.2% |
| Counter-propagating 6 dB corner, 1st order | 1.59 kHz | 1.33 kHz | reproduced, 18% |
| Co-propagating 6 dB corner, 2nd order | 11.2 MHz | 8.0 MHz | **not** reproduced, 28% low |
| Co-propagating 6 dB corner, 1st order | 18.5 MHz | 8.5 MHz | **not** reproduced, 54% low |
| DC, counter, 2nd order | 15.6 dB | 17.8 dB | **not** reproduced, 2.2 dB high |
| DC, counter, 1st order | 0.04 dB | 18.6 dB | **not** reproduced |
| DC, co, 2nd order | 15.4 dB | 23.1 dB | **not** reproduced, 7.7 dB high |
| DC, co, 1st order | 0.7 dB | 23.9 dB | **not** reproduced |
| Direct interaction length, Fig. 7 | 20.5 km | 23.8 km | **not** reproduced |
| Indirect interaction length, Fig. 7 | 25.5 km | 23.7 km | **not** reproduced |

The **counter-propagating corners are the strongest evidence that the model is right in the
way that matters.** The corner is `A_i/(2π d)` with `d = 1/v_s + 1/v_p ≈ 9.8 ns/m`, so
reproducing 1.33 kHz to 2.5 percent is simultaneously a check on the walk-off sign, on the group
indices, on the gain-per-unit-length normalization, and on the retardation structure. A wrong
sign or a per-kilometre/per-metre slip moves it by orders of magnitude, not by 2 percent.

The **DC levels are not reproduced**, and the failure is structured rather than random: the
second-order value lands within about 2 dB, while the first-order value is 18.6 dB too high.
The two pumps come out nearly equal, where the paper reports the first-order transfer 15 dB
*below* the second-order. In this model both pumps reach the signal through much the same
path — the cascaded `p₂ → p₃` coupling — so the direct `p₁ → p₃` coupling is not weighted the
way the paper's chain ordering implies. That is the discrepancy to chase next, and the
cascaded-versus-direct question is exactly what the change's design note flagged as needing
confirmation before encoding.

The **co-propagating corners are low by a similar factor in both cases** (1.4× and 2.2×), which
points at the co-propagating group-delay difference being underestimated rather than at the
walk-off sign, since the sign is pinned by the counter-propagating corners above.

The **Fig. 7 interaction lengths are not separated by the model at all**: 23.8 and 23.7 km,
where the paper separates them by 5 km. The measurement is defined as the e-folding decay of
the first-order pump's modulation profile measured from its peak, and both the direct and the
induced profiles have the same shape here.

Every one of these misses is a `pytest.mark.xfail(strict=False)` test that names the paper, the
figure, the expected value and the observed value, rather than a loosened threshold. If a later
stage fixes one, the test reports XPASS and the mark can be removed.

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