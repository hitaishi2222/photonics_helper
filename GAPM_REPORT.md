# LANE Gap-M report (`lane/gap-m`, base `dev@b7bd01d`)

## Changes
- **M1** `gnlse.py`: `SplitStepEngine.__init__` warns (`UserWarning`) when `include_tpa=True`,
  `fiber.beta_tpa > 0`, `fiber.sigma_tpa == 0` and `include_free_carriers` is off, naming
  `include_free_carriers=True` as the flag that carries β_TPA; `tpa_step` now tests `sigma_tpa <= 0`
  before the `omega0 <= 0` demand, so an inactive call needs no carrier frequency.
- **M2** option **(a)**, opt-in hook `Fiber(..., a_eff=Area(...))`: then `beta_tpa`/`sigma_tpa` are
  physical (m/W, m²/W) and `three_photon_alpha_3` physical (m³/W²), applied as `β/A_eff`, `σ/A_eff`,
  `α₃/A_eff²`. Conversion sits in the consumers (`Fiber.beta_tpa_engine`, `sigma_tpa_engine`, α₃
  scaling in `SplitStepEngine.__init__`), never on the fields, so `fiber.beta_tpa` reads back
  unchanged and `a_eff=None` multiplies by exactly `1.0` -> bit-identical. A *new* field, not
  `A_eff` (which already feeds `n₂`->γ), so existing `Fiber(n2=, A_eff=, beta_tpa=)` keeps its
  numbers; 3PA keeps its exact closed form and its `w3pa`/`w3pa_energy` ledger.
- **M3** `tests/test_gnlse_tpa_units.py` (11 tests): closed-form TPA via `a_eff` (1.3e-14 vs the
  1e-3 gate), the guard warning plus the `include_free_carriers` twin that must not warn and must
  absorb, exact `A_eff`/`A_eff²` ratios, `alpha3=0` + `a_eff` bit-identical to `a_eff=None`.
- **N1** both wrappers accept and forward `gamma_omega`, `gamma_omega_naive`, `three_photon_alpha_3`;
  the test asserts `array_equal` vs `SplitStepEngine` on one deck.
- **N3** public `SplitStepEngine.photon_number(A=None)`; `_photon_number` stays the implementation.
- **N4** ETDRK4 probes `gamma_fn`/`alpha_fn`/`a_eff_fn` at `z = 0, L/2, L` and warns (not refuses), a
  static hook measuring order 3.99.
- **N2** one sentence in the `scheme` docstring and in `examples/45_gnlse_scheme_comparison.py`'s
  scheme list (regenerated `docs/examples/45*.md`); integrator untouched.
- **Residual of the same class** (orchestrator review): `gamma_omega` silently overrode `gamma_fn` /
  `a_eff_fn`; the engine now warns naming the ignored hook(s), and both wrapper docstrings say so.
  Test: `test_gamma_omega_warns_that_hooks_are_ignored`.
- Housekeeping: `CHANGELOG.md`, `docs/gnlse-physics.md` units table. No `photonics_helper.core.*`
  symbol touched, so no API-snapshot regeneration. `openspec/changes/fix-gap-m-tpa-units/proposal.md`
  exists on disk but `openspec/` is **gitignored** (`.gitignore:25`), so it is not in the commit;
  the author moves it by hand if it should be tracked.
## Reproducer (`GAP.md` §17.2: 400 steps, 100 pJ/200 fs, 10 mm)
| call | before `b7bd01d` | after |
|---|---|---|
| `include_tpa=True`, `beta_tpa=1e-3`, `sigma_tpa=0` | 1.000000, silent | 1.000000 + warning naming `include_free_carriers` |
| `include_free_carriers=True`, same fiber | 0.996691 | 0.996691 (unchanged) |
| physical `beta_tpa=9.3e-14` m/W, no `a_eff` | 1.000000 (β=9.3e-14) | 1.000000 (unchanged) |
| physical `beta_tpa=9.3e-14`, `a_eff=0.28 µm²` | n/a (no kwarg) | 0.502697 (β=0.3321 W⁻¹m⁻¹) |
Compatibility: a 6-deck probe (Strang+Raman+shock, ETDRK4+`gamma_omega`+α₃, free carriers, legacy σ-TPA,
full-β callable, photon-number path) is `array_equal` bit-identical to `b7bd01d`, `w3pa` too.
## Gates (`PYTHONPATH` proven inside `wt-gapm`)
| gate | exit | result |
|---|---|---|
| `ruff check .` | 0 | All checks passed |
| `mypy photonics_helper` | 0 | no issues in 41 source files |
| `pytest -q` | 1 | **1 failed, 1388 passed, 2 skipped** |
The failure is `tests/test_inverse_design_autodiff.py::test_autodiff_fit_recovers_truth` (the `torch`
here is a stub: no `manual_seed`, no `__version__`) and it fails identically on a pristine `b7bd01d`
worktree, so it is pre-existing and unrelated. `scripts/check_docs.py` reports the same 4 pre-existing
em-dash hits in `CHANGELOG.md` before and after. The project's `scripts/check_library_path.py`
(cwd = project repo) reports `"pass": true` on every gate and rewrites `runs/library_path_check.json`.
## New keyword names
`Fiber(..., a_eff=Area(...))`, `Fiber.from_gamma(..., a_eff=...)`, `Fiber.from_n2(..., a_eff=...)`,
read-only `Fiber.beta_tpa_engine` / `Fiber.sigma_tpa_engine`,
`GNLSESolver(..., gamma_omega=, gamma_omega_naive=, three_photon_alpha_3=)`,
`TaperedGNLSESolver(...,` the same three `)`, `SplitStepEngine.photon_number(A=None)`.
## Not verified
No V5/Karim or project physics run (library closed form only). The multimode engine builds per-mode
`Fiber`s without `a_eff`, so its `beta_tpa`/3PA stay engine-normalised (not extended).
3PA-generated carriers/FCA still unmodelled. ETDRK4 hook detection is a 3-point probe (a hook flat at
`0, L/2, L` but varying between would not warn); no external reference beyond the closed form.