# Photonics Helper — Gap Analysis for A1 Implementation

**Goal:** What's missing in `photonics_helper` to run A1 (Soliton Dynamics & Fission in Chalcogenide Waveguides) end-to-end.

---

## What's Already Working ✅

| Module | Status | What It Does |
|--------|--------|--------------|
| `Dispersion` | ✅ solid | `from_neff()`, `from_propagation_constant()`, `get_beta2()`, `get_betas()` (polynomial fit in ω) |
| `RefractiveIndex` | ✅ solid | Sellmeier, tabulated n+k, group index, group velocity |
| `RamanSpec` + DB | ✅ solid | 30+ materials, Sellmeier data, Stokes/anti-Stokes, Raman database (SQLite) |
| `RamanResponse` | ✅ solid | Time-domain h_R(t), Silica model, auto-derived τ₁/τ₂ |
| `GNLSESolver` | ✅ working | SSFM with Kerr + Raman + self-steepening + TPA |
| `Wave` / `Envelope` / `TemporalGrid` | ✅ solid | Pulse representation, grid, shapes (gaussian, sech, etc.) |
| Visualization | ✅ basic | Waterfall plot, spectrum vs distance, intensity metrics |
| `FiberProfile` | ✅ working | n₂, α, A_eff, σ_tpa, carrier_lifetime, raman_response |

---

## Gaps That Block A1 🔴

### 1. Missing Material: GeAsSe in Database

**Problem:** As₂Se₃ is in `RAMAN_MATERIALS` (n₂=3.5e-18, fR=0.55, Raman=310 cm⁻¹). GeAsSe is **not** listed.

**Fix:** Add GeAsSe to `RAMAN_MATERIALS` dict and/or seed into SQLite DB:
```python
"GeAsSe": {
    "name": "GeAsSe",
    "crystal": "Amorphous",
    "bandgap_eV": 1.6,
    "n2": 6.0e-18,        # ~2× As₂Se₃
    "raman_shift_cm": 250,
    "raman_linewidth_cm": 50,
    "fR": 0.50,
    "gain_coeff": None,
    ...
}
```

Also need **Sellmeier coefficients** for both As₂Se₃ and GeAsSe across 1–12 μm range (for dispersion calculation from COMSOL neff data or direct Sellmeier use).

---

### 2. Waveguide-Aware GNLSE (Not Just Fiber)

**Problem:** `FiberProfile` uses `A_eff` (effective area) which is a fiber concept. For rib waveguides, the nonlinearity depends on **mode confinement factor** Γ, not just A_eff. The γ calculation is:

```
γ = n₂ · ω₀ · Γ / (c · A_mode)
```

where A_mode is the mode area (not "effective area" in the fiber sense) and Γ is the confinement factor.

**Fix:** Add a `WaveguideProfile` class (or extend `FiberProfile`):
```python
@dataclass
class WaveguideProfile:
    """Waveguide parameters for GNLSE propagation."""
    n2: float                    # m²/W
    alpha: float                 # 1/m (linear loss)
    A_mode: Area                 # mode area (m²)
    confinement_factor: float = 1.0  # Γ — fraction of mode in nonlinear core
    length: Length               # m
    sigma_tpa: float = 0.0
    carrier_lifetime: Time | None = None
    raman_response: RamanResponse | None = None

    @property
    def gamma(self, omega0: float) -> float:
        """Nonlinear coefficient γ = n₂·ω₀·Γ / (c·A_mode)"""
        return self.n2 * omega0 * self.confinement_factor / (C_MS * self.A_mode.as_m2)
```

Then adapt `SplitStepEngine` to accept either `FiberProfile` or `WaveguideProfile`.

**Why this matters for A1:** Your rib waveguides have Γ < 1 (some field in cladding), and A_mode ≠ A_eff. Using the fiber formula with wrong A_eff gives incorrect γ → wrong soliton number N → wrong fission length.

---

### 3. Soliton Diagnostics Module (Doesn't Exist)

**Problem:** The GNLSE solver gives you spectra vs z, but there's **no built-in analysis** to extract:
- Soliton order N at input
- Fission length L_fiss
- Number of solitons at output
- Raman self-frequency shift (RSFS) rate
- Dispersive wave (Cherenkov) wavelength and efficiency
- Soliton trajectories (peak wavelength vs. z)

**Fix:** Create `photonics_helper/soliton.py`:

```python
class SolitonAnalyzer:
    """Analyze GNLSE propagation results for soliton dynamics."""

    def __init__(self, solver: GNLSESolver, gamma: float, beta2: float):
        ...

    def soliton_order(self, P_peak: float) -> float:
        """N = √(γ · P_peak · L_eff) / |β₂|"""
        ...

    def fission_length(self, N: float, L_D: float) -> float:
        """L_fiss ≈ L_D / N for fundamental soliton fission."""
        ...

    def dispersive_wave_wavelength(self, beta2, beta3, beta2_dw: float) -> float:
        """Phase-matching: β₂(ωDW) = -2·β₃·(ωDW - ω₀) / ... """
        ...

    def count_solitons(self, spectrum: NDArray, omega: NDArray) -> int:
        """Count soliton peaks in output spectrum using peak finding."""
        ...

    def raman_shift_rate(self, solver: GNLSESolver) -> float:
        """Track peak wavelength vs z to get RSFS rate (nm/mm)."""
        ...

    def soliton_trajectories(self, solver: GNLSESolver) -> list:
        """Extract individual soliton trajectories (λ_peak vs z)."""
        ...
```

**Also need:**
- `dispersion_length(T0, beta2)` — helper
- `nonlinear_length(P_peak, gamma)` — helper
- `chernikov_wavelength(beta2, beta3, beta2_external)` — dispersive wave phase matching

---

### 4. Dispersion.get_betas() Unit Consistency

**Problem:** Looking at the code, `get_betas()` returns betas in `ps^k/m` and converts omega to `rad/ps`. The GNLSE solver's `_linear_step()` does the same conversion. But the adaptive step size in `SplitStepEngine._adaptive_step_size()` converts beta2 to `s²/m` via `* 1e-24` which is **wrong** — it should be `* 1e-24` only if betas are in ps²/m, but the code is mixing unit systems inconsistently.

**Fix:** Standardize on one unit system throughout. Either:
- **(a)** Keep everything in SI (s²/m for β₂, rad/s for ω) — cleaner, no conversion bugs
- **(b)** Keep everything in fs/ps/μm units — faster, what most SC codes use

Recommend **(a)** — SI units. Change `get_betas()` to return SI values and remove all the `* 1e-24` / `* 1e-12` conversions scattered around.

---

### 5. Self-Steepening Performance

**Problem:** `self_steepening_step()` uses `scipy.integrate.solve_ivp` (RK45) for every nonlinear step. This is **extremely slow** — it's an adaptive ODE solver called thousands of times per propagation. For a typical A1 simulation (100+ steps × 2 materials × multiple pulse durations), this could take hours.

**Fix:** Use the **explicit exponential** form instead of solve_ivp:
```python
# Current (slow):
sol = solve_ivp(_ss_rhs, ...)

# Faster (exact for constant P_NL over step):
# ∂A/∂z = iγ(1 - ω/ω₀) · P_NL · A  →  A(z+dz) = A(z) · exp(iγ(1-ω/ω₀)·P_NL·dz)
```

Or use a 4th-order Runge-Kutta directly (not adaptive) for the nonlinear step — much faster than solve_ivp for fixed-step SSFM.

**Priority:** Medium — doesn't block A1, but makes simulations impractically slow.

---

### 6. Soliton Visualization Tools

**Problem:** Current visualization is generic (waterfall, spectrum vs z). For A1 you need:
- Soliton trajectory plot (wavelength of each soliton vs propagation distance)
- Soliton count vs N plot
- Dispersive wave phase-matching diagram
- RSFS rate plot

**Fix:** Add to `soliton.py` or extend `gnlse.py` visualization:
```python
def plot_soliton_trajectories(solver, omega0, betas, ax=None):
    """Plot individual soliton peak wavelengths vs propagation distance."""
    ...

def plot_fission_dynamics(solver, N, L_D, ax=None):
    """Show soliton fission process: N solitons → fewer solitons + DW."""
    ...

def plot_raman_shift(solver, ax=None):
    """Track soliton peak wavelength drift due to RSFS."""
    ...

def plot_parametric_study(results_dict, ax=None):
    """Heatmap: bandwidth / RSFS / soliton_count vs pulse_duration × peak_power."""
    ...
```

---

## Optional / Nice-to-Have 🟡

### 7. Multi-Pump Wavelength Support in GNLSE

**Problem:** A1 simulations use pump wavelengths at 1.5 μm, 2.0 μm, and 4.0 μm. The current solver handles this fine (just change `omega0`), but the `Dispersion.get_betas()` polynomial fit needs to cover wide wavelength ranges. A utility to auto-select the best polynomial order for a given wavelength range would help.

### 8. Coherence Analysis

**Problem:** For publication-quality SC results, spectral coherence (first-order coherence function) is often expected. This requires running multiple simulations with noise realizations.

**Fix:** Add optional noise term to GNLSE:
```python
def noise_step(A, dz, gamma, hbar, omega0, grid):
    """Add quantum noise: seed for soliton number fluctuations."""
    ...
```

**Priority:** Low — can be added after first paper is submitted.

---

## Summary: Build Priority

| Priority | What to Build | Why |
|----------|--------------|-----|
| **🔴 P0** | `WaveguideProfile` class | Core physics — wrong γ = wrong N = wrong everything |
| **🔴 P0** | `SolitonAnalyzer` class | The whole point of A1 — extract dynamics metrics |
| **🔴 P0** | Add GeAsSe to materials DB | Can't compare materials without it |
| **🟡 P1** | Fix dispersion unit consistency | Prevents subtle bugs in β coefficients |
| **🟡 P1** | Add soliton visualization functions | Needed for paper figures |
| **🟢 P2** | Fix self-steepening performance | Makes simulations practical |
| **🟢 P3** | Coherence analysis | Nice for paper, not blocking |

---

## Recommended Implementation Order

1. **Fix dispersion units** (P1) — quick win, prevents future bugs
2. **Add GeAsSe to DB** (P0) — 5 minutes, just data entry
3. **Add `WaveguideProfile`** (P0) — ~100 lines, adapt existing FiberProfile
4. **Add `SolitonAnalyzer`** (P0) — ~200 lines, the science engine for A1
5. **Add soliton plots** (P1) — ~150 lines, needed for paper
6. **Fix self-steepening** (P2) — ~50 lines, performance boost

**Total estimated effort:** ~500 lines of new code across 2-3 new files.
