# Audit report: photonics_helper

**What this repo does:** A Python library for photonics. It converts units and looks up material data in a bundled SQLite file (`materials.db`). Its main part is a family of split-step solvers for pulses in fibers and waveguides: the nonlinear Schrödinger equation (GNLSE), including vector and multimode variants, Raman response, FROG, TMM and χ² solvers. There is also a small local Dash web app.

**Load I assumed:** One scientist running scripts or notebooks, sometimes with several processes. The Dash app runs on localhost. Speed matters little. Wrong numbers matter most.

**How I checked:**
- Full test suite: 1557 passed, 8 skipped, **1 failed**.
- `ruff check .`: clean.
- I wrote small scripts to confirm findings 1–5 and 8. I read the code behind the rest.

---

## Must fix

### 1. Loss or gain set by `alpha_fn`, or negative `alpha`, is silently ignored on the default scheme (`photonics_helper/gnlse.py:1799-1800, 1904-1905`)
- **What this is:** Each split-step step applies the fiber loss `exp(-α·dz/2)` in `SplitStepEngine._linear_step`.
- **Problem:** The two non-tapered branches use `self.fiber.alpha` and only apply it when `alpha > 0`. So:
  - `alpha_fn=` is ignored unless `dispersion_profile=` is also set.
  - Gain (`alpha < 0`) is dropped.
  - The ETDRK4 scheme (`_etdrk_linear`) does use `_get_alpha` and applies both.
  - I ran it with `Fiber(alpha=-0.2)` over 10 m:
    - Strang, the default: energy ratio **1.0**.
    - ETDRK4: **7.39**, which is e².
  - With `alpha_fn=lambda z: 0.5` over 10 m:
    - Strang: **1.0**.
    - ETDRK4: **0.0067**, which is correct.
  - Two schemes disagree on the same input, and no warning is shown. The docs say `alpha_fn` combines with `full_beta` and the other options.
- **Fix:** In `_linear_step`, use `alpha = self._get_alpha(self._current_z)` in all three branches and test `alpha != 0.0` instead of `> 0`. Add a test that Strang and ETDRK4 give the same energy for `alpha_fn` and for negative `alpha`.
- **If we skip it:** A user who models a gain fiber or a z-dependent loss gets a lossless run with no warning.

### 2. The free-carrier model integrates carriers along z, not along time (`gnlse.py:774-850`, esp. L843-844)
- **What this is:** `include_free_carriers=True` is meant to follow the cited Soref–Bennett and Yin–Agrawal models. There the carrier density obeys `∂N/∂t = βI²/(2ħω) − N/τ` at a fixed position. Later parts of a pulse then see carriers made by earlier parts, so the trailing edge is absorbed more.
- **Problem:** The code evolves `N` per time sample with `dN/dz = S − N/(τ·v)`. Each sample only sees its own intensity. I ran a 50 fs sech pulse with τ_c = 1 ns:

  | Time from pulse centre | Carrier density N |
  |---|---|
  | −200 fs | 4e6 |
  | 0 | 2.8e17 |
  | +200 fs | 4e6 |
  | +1 ps | 2.5e-40 |

  A real 1 ns carrier lifetime would leave a plateau of carriers after the pulse. Here N is symmetric and falls to zero. The units are also not closed: the source term has no `1/A_eff` and no `1/v`, and `docs/gnlse-physics.md` only notes "σ_FCA (no area)".
- **Fix:** Either do the time-ordered recursion `N[i] = N[i-1]·e^{-dt/τ} + S[i]·dt` (a cumulative sum over the T axis) with SI-consistent units, or rename the model and document that it is not Yin–Agrawal. The first test should assert that N after the pulse is non-zero for a long τ_c.
- **If we skip it:** Silicon-waveguide results with free-carrier absorption are quantitatively wrong and have no asymmetry. The docs and citations say otherwise.

### 3. First use of any material function fails on a read-only install (`photonics_helper/core/data.py:147-290`, `_init_db`)
- **What this is:** Every `MaterialsDatabase()` runs `_init_db` on first use. It does `ALTER`, `UPDATE`, `INSERT OR IGNORE … SELECT` and `commit` on the **bundled** `materials.db` inside `site-packages`.
- **Problem:** I copied the DB to a read-only folder and called `list_materials()`. It raised `sqlite3.OperationalError: attempt to write a readonly database`. This happens whenever the package dir is not writable:
  - a root-owned conda or system install used by a normal user;
  - a read-only container image;
  - an HPC module.

  Worse, `RefractiveIndex.from_material_database` catches `sqlite3.Error` and reports "the file is missing or corrupt, reinstall". That sends the user the wrong way.
- **Fix:** Check `PRAGMA table_info` first and only write when a column or table is actually missing. Open the bundled file with `file:…?mode=ro` when nothing needs migrating. The bundled DB already has all columns.
- **If we skip it:** `material("Silica")` and `RamanSpec.from_database` do not work on shared machines.

### 4. Saved snapshots are labelled with a z they were not taken at (`gnlse.py:2981-2990, 3108`)
- **What this is:** `propagate(nsaves=N)` promises evenly spaced snapshots and exposes `z_array`.
- **Problem:** A snapshot is taken at the first step that passes `save_z[i]`. At the end, `z_array` is overwritten with the nominal `save_z`. I ran 7 steps with `nsaves=5` over 10 m:
  - the snapshot labelled 2.5 m was really at 2.857 m;
  - its energy ratio was 0.2397, which is `exp(-α·2.857)`, not the 0.2865 expected at 2.5 m.

  If the step is larger than the save spacing, several labels get the same field.
- **Fix:** Store the real `z` in `_append_snapshot` (`self._z_positions.append(z)`). Drop the overwrite at L3108. Optionally clip the step so it lands on `save_z`.
- **If we skip it:** Waterfall plots and "field at z = x" comparisons are off by up to one step.

---

## Should fix

### 5. `SplitStepEngine.propagate()` called a second time continues from the old field (`gnlse.py:2926`, with `self.A` set only in `__init__` at L1561)
- **What this is:** `propagate` resets `evolution`, `_z_positions` and the 3PA ledger.
- **Problem:** It does not reset `self.A`, `_U` (TPA carrier) or `_N`. I ran `propagate(50)` twice. The second run's `evolution[0]` is the output of the first run, labelled z = 0. `GNLSESolver` builds a fresh engine each call, so only direct engine users are hit.
- **Fix:** Reset `self.A`, `self._U` and `self._N` at the top of `propagate` from `self.pulse`. Or raise if called twice.
- **If we skip it:** Sweeps that reuse one engine get silently chained runs.

### 6. `RamanSpec.from_database` hides every database error (`photonics_helper/raman/spec.py:559-571`)
- **What this is:** It reads a material row from the DB, then falls back to the hardcoded `RAMAN_MATERIALS`.
- **Problem:** `except Exception: pass` covers the read-only failure (finding 3), a wrong `db_path`, a corrupt file and a pydantic validation error from a bad row. In all those cases it returns built-in numbers with no message. A user who passes `db_path=` to their own DB can get the built-in values instead of their own.
- **Fix:** Catch only `sqlite3.Error` when no `db_path` was given. If a `db_path` was given, let the error through.
- **If we skip it:** A wrong DB path looks like a working run on different data.

### 7. `seed_phonon_data` can wipe the shipped phonon table (`core/data.py:538-575`, uncommitted change)
- **What this is:** The new idempotent seeding commits `DELETE FROM phonon_modes …` for every material, then writes the modes one by one, each through its own `add_phonon_mode` connection.
- **Problem:** A crash or Ctrl-C between the delete and the last insert leaves the bundled `materials.db` with missing modes. The root cause stays open: the primary key `(material, shift_cm, symmetry)` has a nullable `symmetry`, and SQLite treats NULLs as distinct. So a user calling `add_phonon_mode` twice with `symmetry=None` still gets a duplicate row. The Raman response sums over modes, so that doubles the mode's strength.
- **Fix:**
  - Do the delete and inserts in one transaction.
  - Add `CREATE UNIQUE INDEX … (material, shift_cm, COALESCE(symmetry,''))`.
  - Keep the new test.
- **If we skip it:** A half-seeded DB, or silently doubled phonon strength.

### 8. Dashboard turns an explicit 0 into a default (`photonics_helper/dashboard.py:256-262`)
- **What this is:** The GNLSE viewer callback fills defaults with `value or default`.
- **Problem:** `gamma=0` (a linear-dispersion check) becomes 0.01. `p0=0` becomes 1000 W, and `length=0` becomes 0.1 m. The same code handles `beta2` correctly with `is not None`.
- **Fix:** Use `x if x is not None else default` for every field. Also cap `steps` and `length` so a typo cannot start a very long run.
- **If we skip it:** The plot shows a nonlinear run when the user asked for none.

### 9. One committed test fails in a plain install (`tests/test_gnlse_etdrk4.py:423-439`)
- **What this is:** `test_etdrk4_full_beta_gamma_omega_raman_is_fourth_order` demands a fitted order of at least 3.5.
- **Problem:** It fails every time here (numpy 2.2.6, scipy 1.15.3, Python 3.13, no pyfftw): the measured order is **3.4976**. The docstring says it should be 3.98. The drift near 6e-12 is close to the rounding level, so the fit is fragile. CI runs with and without pyfftw, so this probably flips between runs.
- **Fix:** Fit the order only on points above a round-off floor, for example 1e-10, or loosen the bound. Check whether the order really dropped on this FFT backend.
- **If we skip it:** A red CI gate that people learn to ignore.

### 10. Releases are not gated by tests (`.github/workflows/publish.yml`)
- **What this is:** Pushing a `v*` tag builds and publishes to PyPI.
- **Problem:** The `publish` job only `needs: build`. The CI jobs (test, ruff, mypy) are not required, and the tag is not compared with `pyproject.toml`'s version. A tag on a red commit ships.
- **Fix:** Add a job that runs `ruff`, `mypy` and `pytest` first. Add `needs:` on it. Compare `${GITHUB_REF_NAME}` with the package version.
- **If we skip it:** A broken release can reach users.

### 11. SQL built from caller keys in `update_material` (`core/data.py:570-592`)
- **What this is:** `update_material(name, **kwargs)` formats the keys into `SET {k} = ?`.
- **Problem:** The values are bound, but the keys are not. `update_material("X", **{"crystal='hacked', n2": 5})` ran as written. A typo key raises `OperationalError`. Nothing in the package calls it today, so it is a trap rather than a live hole.
- **Fix:** Check `kwargs` against `PRAGMA table_info(raman_specs)` before running.
- **If we skip it:** Any future caller that forwards user keys becomes injectable.

---

## Nice to have

### 12. `delete_material` leaves orphan rows (`core/data.py:593-610`)
It removes `raman_specs` and `nk_data` rows only. `sellmeier` and `phonon_modes` rows stay. Delete from all four tables, or document it. The bundled DB also has 1712 `nk_data` rows for `GaP` with no `raman_specs` row, so foreign keys are not enforced anywhere.

### 13. Every `MaterialsDatabase()` re-runs the migration (`core/data.py:147`, `_ensure_initialized`)
The `_initialized` flag is per instance. `RamanSpec.nk`, `from_database` and `material()` each build a new instance. I measured about 10 ms per instance. Cache the result per path in a module-level set. That also reduces finding 3.

### 14. `seed_db.py --nk` deletes before it validates (`seed_db.py:537-575`)
`clear_all_tabulated_nk()` commits first. Datasets that then fail validation are "skipped", so they vanish from the shipped DB. `nk_datasets/` is not in git, so the DB file is the only copy. Validate everything first and write in one transaction.

### 15. Dash `debug=True` (`dashboard.py:21`, `raman/dashboard.py:80,581`)
Fine on localhost. If someone later passes `host="0.0.0.0"`, the Werkzeug debugger gives remote code execution. Make debug opt-in.

### 16. Repo size
`.git` has 16 copies of the 11 MB `materials.db` and a 59 MB HTML figure in history. A clone is large. Keep new DB versions as release assets or generate them in CI. Not urgent.

### 17. Small cleanups
- `fiber.py:85`: `raise Exception(f"{e}")` throws away the original exception type. Use `raise … from e`.
- `ConvergenceDiagnostics.converged` (`gnlse.py`): `self.invariant == "n/a"` never matches the strings that the gate returns ("n/a (…)"). It works only because the drift is 0.0 then.
- `pyproject.toml` ignores all `UserWarning`s in pytest. The library's own conservation and "too coarse" warnings can therefore never fail a test.

---

**Verdict:** The numerics are heavily tested and mostly careful. Fix 1 (loss silently dropped) and 3 (read-only install fails) first. Then decide whether finding 2 (free carriers) should be fixed or documented as a non-physical model, and fix 4 (z labels).

**Not checked:**
- The four background reviewers (`pulse.py`, `base.py`, `phase_matching.py`, `raman_transfer.py`, `fiber.py`, multimode, vector, `chi2.py`, `dbr.py`, `structured.py` and others) ended with errors or were stopped before returning results. I did **not** read these files line by line:
  - `pulse.py`, `base.py`, `phase_matching.py`, `raman_transfer.py`
  - `multimode_gnlse.py`, `vector_gnlse.py`, `mode_analysis.py`, `vector_modes.py`, `chi2.py`, `inverse_design.py`
  - `dbr.py`, `structured.py`, `breathers.py`, `soliton.py`, `phonon.py`, `noise.py`
  - most of `gnlse.py` beyond the engine core, `propagate_auto` and the tapered solver
- I did not run the examples, reproductions or docs snippets.
- CI behaviour with `pyfftw` and Python 3.12 was not run.
