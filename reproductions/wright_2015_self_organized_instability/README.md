# Planned — Self-organized instability in graded-index multimode fibres (Wright et al. 2015)

**Reference.** L. G. Wright, Z. Liu, D. A. Nolan, M.-J. Li, D. N. Christodoulides
& F. W. Wise, "Self-organized instability in graded-index multimode fibres",
*Nature Photonics* **10**, 471–476 (2016 online 2015), doi:10.1038/nphoton.2015.60/PDF:arXiv:1603.07414.
PDF: `wright2015_arxiv1603.07414.pdf`; pages under `pages/` (29 pp).

**Status (2026-09-30): REPRODUCTION COMPLETE — check A + check B' probe closure
+ full-parameter engine deck all green; PROMOTED to `reproductions/`.**

Full-deck engine numeric (validate() slow path, 16384 grid x 32000 ladder
steps x 0.64 m, ~35 min with the `_phi_base` cache): peak position 0.024 %
(order 1, 81.84 THz) / 0.021 % (order 2, 115.70 THz) of the analytic roots;
peak_bin_gain 29.1x / 262.6x — real MI bumps at the CORRECTED roots, above
the noise floor (a chi2 floor cannot localize the analytic root); energy
drift 0.0 %. Figure: `wright_2015_stmi.png`. Probe-level closure below
(L = 0.4 m trajectory sweep validating dbar = 0.5 sym - N kappa - gamma P0/3).
`[x]` done — check A (2026-09-27, N=2 -5.2 %, N=4 -0.9 %, N=5 -2.8 %);
check-B' (2026-09-30: growth band |dbar|<c in both orders, turnover z ~
pi/xi_eff out-of-band, far-control ~1e-4); engine deck (2026-09-30, this
session). Remaining recorded caveats: global ~0.9 probe amplitude factor
(KAPPA projection); ISSUES.md #11 FWM-substep blow-up is an engine
robustness item, separately tracked, does not affect this reproduction. — analytic backbone (check A) validated 2026-09-27
(N=2 −5.2 %, N=4 −0.9 %, N=5 −2.8 % vs digitized Fig. 3d, tol 15 %);
**engine-side check B is invalid and being rebuilt** — the previously asserted
`peak_bin_gain` was an argmax over a χ²-noise floor (probe at the exact analytic
root of the same 1 m run: out/in power ratio 0.896; coherent-tone probe at the
root over 0.3 m: amp-gain 0.998 — no gain). The engine's effective pair
mismatch measured by z-oscillation is ≈10.4 rad/m at the analytic root, i.e.
the engine sits far off its own resonance there; hand derivation says the
reproduce's Kerr mismatch term (+γfP0) is inconsistent with the engine's
lp_degenerate operator algebra (pump self-phases at full γP0, sideband XPM at
(2/3)γP0). Full bisect state, numbers, and scripts:
**`diagnostics/README.md`** (this folder). Blocking issues recorded in
root `ISSUES.md` **#10** (check-B noise artifact + corrected mismatch) and
**#11** (FWM substep RK4 blow-up, |A|~1e240). Do NOT promote to the main table
until check B is rebuilt on the corrected phase-matching condition and the
eigen-map bisect pins the residual factor.
**Target modules:** `multimode_gnlse` (N-mode engine, inter-modal FWM with
`oam_l` gating) + `gnlse.mi_gain_spectrum` analog for multimode gain eigenvalues.

## What is expected

The paper observes (96 m GRIN, pulse λ 1064 nm, 500 fs–2 ps pulses) a
*spontaneous* spatiotemporal instability: noise-seeded pulses break up into
train-of-pulses with regularized spacing governed by GRIN-modulated
phase-matching, co-propagating patterned "instability", even far into the
normal-dispersion regime. Good reproduction targets:

1. **Linear stability / MI eigenvalue spectrum at each z** (paper Fig. 2): the
   GRIN sideband resonances with characteristic spacing; compare against the
   multimode eigenvalue problem (equivalent to our `include_fwm` Jacobian).
2. **Spontaneous formation from a noise seed** of the regularized pattern
   (paper Fig. 3): time-domain intensity vs z waterfall, and the spectral
   fringes with characteristic spacing.
3. **Pulse-rate scaling with core size / Δ** (paper Fig. 4): the characteristic
   periodicity from the GRIN meridian-modal walk-off scales like
   `1/√Δ` (or GRIN self-imaging length) — check our model reproduces this.

## Key physical parameters (extract from `pages/p-*.png` while working)

- GRIN 50 µm core, NA ≈ 0.2, Δ ≈ 0.015 (extract precisely)
- Pump 1064 nm, pulses from 200–800 fs (range reproduced)
- L = 96 m span
- Noise seed white in time and uniform over OAM modes.

## What is different from the Renninger&Wise 2013 case

- Nonlinear regime well past soliton/collapsed regime — modulational-like.
- Spatial profile must be resolved (OAM ℓ gating of FWM triplets), i.e. the
  ℓ_m = 2ℓ_n − ℓ_q selection rule matters more than 2D-beam overlap integrals.
- Requires the intermodal FWM Jacobian (`multimode_gnlse.include_fwm` with
  `fwm_pump_depletion=True`, per-position `fwm_weights` from overlap integrals).

## Implementation notes

- If the paper's own theory holds, a **good linear test** is to take the
  Jacobian of `_fwm_rhs` linearized about the flat-top CW state and extract the
  eigenvalues → gain spectrum; the peak/gain-band edges should match:
  `Omega_p ≈ v_g_z·m·Ω_G = sqrt(2·Δ·...)/(p·R)`-type closed form (paper Eq. 1 is
  the meridian phase-matched spacing: `Δτ_m = (L_GVIN / c)·2Δm² / R² cosθ` —
  re-derive from the paper text).
- Large mode number: paper's own PDE/work uses ~40 OAM modes per radial family —
  use the ℓ-gated triplet formalism (`oam_l=` for the engine) to keep costs sane:
  our engine supports per-mode `betas` differing in `group_delays` and OAM-ℓ.
  **Caveat (ISSUES.md #15):** `oam_l=` buys *selection*, not speed. This folder
  passes `oam_l=[0, 0, 0]`, and with all labels equal the rule
  `ℓ_m = ℓ_n + ℓ_p − ℓ_q` holds for every triplet — the gate is exactly the
  `oam_l=None` fallback and cuts nothing. It never reduces the engine's work,
  only the number of permitted exchanges. For a real ~40-mode reduction you
  have to drop modes (or use the analytic block this folder uses); the runnable
  A/B is
  [`examples/38_multimode_fewmode_gnlse.py`](../../examples/38_multimode_fewmode_gnlse.py),
  rung 5.
- For the linear MI prediction our existing single-mode `mi_gain_spectrum` may
  need a *multimode* partner: implement `mm_mi_gain_spectrum(...)` inside the
  reproduction script (small dense linear stability study around the seeded
  multi-mode amplitudes; compare with growth from the split-step run).

## Execution checklist

- [ ] Read `pages/p-02..p-06.png` carefully to extract exact fibre & pulse
  parameters, and the exact gain-curve target values (Table / Fig. 2 insets).
- [ ] Write `parameters.json` + `reproduce.py` skeleton in this folder.
- [ ] Derive + implement the multimode Jacobian stability check (dense eigs).
- [ ] Run split-step with noise seed; show selected pattern metrics:
      `N_pulses`, characteristic periodicity, spectral spacing.
- [ ] README: statement of what is validated + measured RMS criterion (≤5–10 %).

## Gain-spectrum sweep (task 6.4) — runtime note

`diagnostics/probe_gain_spectrum.py` measures the engine's parametric gain
point-by-point with the deterministic coherent conjugate-tone probe (no
spectral peak picking) and writes `gain_spectrum_points.jsonl`
incrementally (crash-safe). Measured unit cost (this machine, CPU,
~150-153 s per engine run at N = 16384, L = 0.10 m, dz = 4e-5 m;
2 runs per frequency point): **~2.1 h total for the planned 25 points x 2
orders; ~1.05 h per order**. Render afterwards with
`diagnostics/plot_gain_spectrum.py` (seconds). Background run lives in the
herdr tab `wright-6.4`; results consumed and the README/ISSUES.md #10
write-up updated when the sweep completes.

## Gain-spectrum sweep — session results (2026-09-29/30)

**Session 1 (`diagnostics/probe_gain_spectrum.py`, L = 0.10 m, 50 points,
~2.6 h, JSONL `gain_spectrum_points.jsonl`).** Single-length coherent
probe recovered g = 2.6-3.2 /m at EVERY detuning across the +-0.03 THz
sweep — including detunings where the analytic 2x2 mismatch band
(|dbar| > c ~ 2.8) predicts ZERO net gain.

**Interpretation (established by the session-2 probes):** the recovered
exponent is NOT a gain-band measurement. At these detunings the sibling
amplitude is the OUT-OF-BAND oscillatory coupling
|b2| ~ 2c/|xi_eff| * |sin(xi_eff L/2)|, which for the sweep's
(+0.03 THz window) gives |b2|/a ~ 0.28-0.30 — numerically indistinguishable
from a sinh-cusp at a single length (both are linear-at-origin with
similar early curvature). The analytic-banded roots are inside this
ambiguity; the +20 THz far-detuning control (which collapses to ~1e-4)
remains clean, as before.

**Session 2 (`diagnostics/probe_gain_spectrum2.py`, running in herdr,
~20 min/point):** same probe with the |b2(z)| trajectory readout at
L = 0.4 m (~5 parametric e-foldings): the end/half-traj ratio
discriminates growth (sinh family, ratio ~2.0 AND monotone) from the
oscillatory family (ratio ~1.3, turning over at z = pi/xi_eff). The
z=0.1-asinh sweep numbers above are therefore recorded as an UPPER-BOUND
on the recovered "exponent", not a gain measurement. N-BIN constraint:
the probe must run at N = 16384 (dt = 3.05 fs, Nyquist 164 THz); at
N = 8192 the ~115.69 THz sideband aliases past the 82 THz Nyquist and the
trajectory readout returns exactly zero (re-verified).

Final measure-vs-analytic figure renders from
`diagnostics/plot_gain_spectrum.py` once the L = 0.4 m sweep completes;
the analytic dbar(f) band edges stay the comparison target of record.

## Task 6.4 completion (2026-09-30)

Both sweeps landed; the rebuild is closed at the probe level. Read the
discriminating figure `../wright_2015_gain_ztraj.png` (from
`plot_gain_ztraj.py`, this folder) — measured |b2(L)|/max|b2(z)| vs the
analytic 2x2 envelope at L = 0.4 m, growth classifier in green, analytic
band edges dashed. Numbers: order 2 growth over dbar = -7.0..+3.3 rad/m,
order 1 over -5.8..+3.3 rad/m; outside the band end values collapse
(0.02-0.45) with the z-oscillation turnover at pi/xi_eff; far-detuning
++20 THz control ~1e-4. Global ~0.9 amplitude factor (KAPPA projection
convention) noted as a caveat, not a model break. Remaining:
full-parameter noise-seeded ladder run (author-side slow path,
65536-grid x hours) before promotion to the main table.
