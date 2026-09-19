# Photonics Helper

A comprehensive helper library for photonics and optics calculations, providing easy-to-use tools for wavelength, frequency, and angular frequency conversions.

📖 **Documentation:** <https://hitaishi2222.github.io/photonics_helper/> — built
with [Zensical](https://zensical.org/) from the docstrings, one page per API
module.

# Installation

```bash
pip install photonics-helper
```

For the **FFTW3-accelerated GNLSE / Raman solver** (recommended for large grids
and long propagation runs):

```bash
pip install "photonics-helper[fftw]"
```

Other optional extras: `plotting`, `webapp`, `extras`, `mode-export` — e.g.
`pip install "photonics-helper[webapp]"` for the unified Dash dashboard.

When `pyfftw` is present, every FFT in `gnlse.py` and `raman.py` executes on
the system FFTW3 library with cached plans; otherwise the solver transparently
falls back to `numpy.fft`. See `photonics_helper._fftw` (env vars
`PHOTONICS_FFTW_PLANNER`, `PHOTONICS_FFTW_THREADS`).

For very large grids an optional **cupy (GPU)** backend can be forced with
`PHOTONICS_FFT_BACKEND=cupy`. It is never auto-selected and falls back to the
CPU chain with a warning when cupy or a GPU is unavailable. Benchmark it with
`python benchmarks/fft_backend_benchmark.py`.

# Key Features

- **Type Safety**: Full inline type hints (PEP 561 `py.typed`)
- **Unit Conversions**:
  - Wavelength (nm, μm, m)
  - Frequency (THz, GHz, MHz, Hz)
  - Angular Frequency (rad/s, rad/ps)
- **Array Operations**: NumPy-based array operations for batch processing
- **Pulse Visualization**: Interactive 2D/3D pulse envelope plots
  - Gaussian, sech, chirped, Airy, and custom pulse shapes
  - Temporal intensity, spectral intensity, phase, and polar plots
  - FWHM markers and pulse width (T₀) annotations
  - Light/dark theme support
  - Interactive Plotly HTML or static Matplotlib PNG export
- **DBR Simulation**: Transfer Matrix Method for multilayer stacks
- **FROG**: SHG-FROG trace generation and PCGPA pulse retrieval
  - Generate FROG traces from electric fields
  - Retrieve pulse shape, chirp, and phase from measured traces
  - Fidelity metric for retrieval quality assessment
- **Raman Modeling**: Full Raman response physics for 40+ materials
  - Time-domain response (electronic Kerr + delayed lattice oscillation)
  - Frequency-domain gain spectrum
  - Stokes / anti-Stokes wavelength calculation
  - Pump-wavelength explorer and material comparison overlays
  - SQLite material database (`materials.db`) with 44 materials
  - Interactive Dash dashboard for comparing Raman properties
- **Comprehensive Documentation**: Clear documentation with examples
- **Easy to Use**: Intuitive API design

# Units are classes — an intentional, opinionated choice

> **TL;DR:** Passing units as bare `float`s is how silent factor-of-1000 bugs are
> born. This library deliberately routes every physical quantity through typed
> unit classes in `photonics_helper/base.py`.

This is a **design decision, not a bug**. The maintainer's stance is simple: you
should not have to keep unit conversions in your head (or in your comments)
while doing photonics. Construct a quantity once in whatever unit is natural,
and the library converts, stores, and returns it in the units you ask for.

Every module — dispersion (`fiber.py`), pulses (`pulse.py`), the GNLSE solver
(`gnlse.py`), materials (`materials.py`), phase matching (`phase_matching.py`),
and the DBR/TMM stack (`dbr.py`) — accepts and returns these classes at its
public boundaries instead of raw numbers.

| Quantity | Class (scalar / array) | Example |
|----------|-----------------------|---------|
| Wavelength | `Wavelength` / `WavelengthArray` | `Wavelength(1550, "nm")` |
| Frequency | `Frequency` / `FrequencyArray` | `Frequency(193.4, "THz")` |
| Angular frequency | `AngularFrequency` / `AngularFrequencyArray` | `AngularFrequency(1.2, "rad/ps")` |
| Wavenumber | `Wavenumber` / `WavenumberArray` | `Wavenumber(6450, "1/cm")` |
| Length | `Length` | `Length(5, "mm")` |
| Time | `Time` | `Time(50, "fs")` |
| Energy | `Energy` | `Energy(1.24, "eV")` |
| Power | `Power` | `Power(100, "mW")` |
| Area | `Area` | `Area(0.2, "um^2")` |

The pattern is always the same — construct in any unit, read out in any unit:

```python
from photonics_helper.base import Wavelength

wl = Wavelength(1550, "nm")   # constructed in nm
wl.as_um                       # 1.55      -> view in μm
wl.as_m                        # 1.55e-06  -> internally stored in SI (m)
wl.to_freq().as_THz            # 193.41    -> convert and read out in THz
```

Internally everything is normalised to SI on construction, so downstream
arithmetic and comparisons stay consistent; the `.as_*` properties are simply
views on the same value. `*Array` variants mirror the scalars for vectorised
work (dispersion tables, spectral grids, …), and MEEP-style conversions are
available through `from_meep` / `as_meep`.

**The trade-off, stated honestly:** you write `Length(5, "mm")` instead of
`5e-3`, and functions return `Time` objects rather than floats. In exchange,
unit mismatches surface as explicit conversions instead of propagating silently
through a simulation — which, for photonics, is almost always the better deal.

# Foundation core (`photonics_helper.core`)

`photonics_helper.core` is the **stable, dependency-light foundation** the rest
of the library — and your own projects — build on:

```python
from photonics_helper.core import units, constants, grids, materials
```

It bundles typed units, physical constants, `TemporalGrid`, and the
`OpticalMaterial` interface, and importing it pulls in **only numpy, scipy and
pydantic** — no matplotlib, plotly, dash, or solver modules. The package itself
is imported lazily (PEP 562), so `import photonics_helper` costs nothing extra.

```python
import photonics_helper          # lazy: loads nothing heavy
from photonics_helper import Wavelength   # imports only .base
```

Every existing import path is preserved: `photonics_helper.base`, 
`photonics_helper.pulse.TemporalGrid` and friends still work and refer to the
same objects. See the [foundation core docs](https://hitaishi2222.github.io/photonics_helper/core/).

What's frozen and what can change is spelled out in the
[stability contract](https://hitaishi2222.github.io/photonics_helper/stability/):
`core` is the stable surface, the solvers are provisional, and stable symbols
are only removed after a deprecation cycle. A worked external tool built on the
core alone is in the
[building on the core guide](https://hitaishi2222.github.io/photonics_helper/building-on-core/).

# Quick Start

To get started, import the library and use its functions:

```python
from photonics_helper import Wavelength

# Convert wavelength to frequency
wl = Wavelength(1550, "nm")
freq = wl.to_freq()
print(f"Frequency: {freq.as_THz:.2f} THz")

# Convert frequency to angular frequency
omega = freq.to_omega()
print(f"Angular frequency: {omega.as_rad_ps:.2f} rad/ps")
```

# Pulse Visualization

Visualize pulse envelopes with interactive 2D/3D plots:

```python
from photonics_helper.pulse import Envelope

# Create a 50 fs Gaussian pulse
pulse = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=50e-15)

# Generate interactive Plotly HTML
fig = pulse.visualize_2d(backend="plotly", theme="light")
fig.write_html("pulse.html")

# Or generate static Matplotlib PNG
fig = pulse.visualize_2d(backend="matplotlib", figsize=(14, 10))
fig.savefig("pulse.png", dpi=150, bbox_inches="tight")

# 3D spectrogram
fig_3d = pulse.visualize_3d()
fig_3d.write_html("spectrogram.html")
```

The visualization shows:
- **Temporal intensity** with FWHM and T₀ markers
- **Spectral intensity** (Fourier transform)
- **Instantaneous phase** (for chirped pulses)
- **Polar plot** (Re vs Im of the field)
- **Parameters box**: shape, T₀, FWHM, chirp

# Pulse Trains

Model mode-locked laser output with pulse trains:

```python
from photonics_helper.pulse import Envelope, Wave
from photonics_helper.base import Wavelength, Frequency

# Single pulse
pulse = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=50e-15)

# Mode-locked laser: 1 GHz rep rate, 10 pulses
wave = Wave.from_pulse_train(
    envelope=pulse,
    central_wavelength=Wavelength(1550, "nm"),
    grid=Envelope._make_grid(pulse, N=2**14),
    repetition_rate=Frequency(1, "GHz"),
    n_pulses=10,
)

# Visualize
fig = wave.visualize(t_unit="ns", w_unit="THz", t_scale=1e9, w_scale=1e12)
fig.savefig("pulse_train.png", dpi=150, bbox_inches="tight")
```

# Physical Power & Energy

`Wave.peak_power()` and `Wave.pulse_energy()` operate on the **normalized**
envelope `A(t)` by default, so their values are in field-units², not watts or
joules. Calling them without an effective area emits a one-time `UserWarning`,
and the `Wave.visualize()` summary labels the value as `(normalized units)`.

To get physical units, attach an effective mode area:

```python
from photonics_helper import Area, Frequency, Wavelength, Time
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

A0 = 2.0          # envelope peak amplitude (V/m)
T0 = 50e-15       # pulse width
env = Envelope(shape="gaussian", peak_amplitude=A0, pulse_width=Time(T0, "s"))
grid = TemporalGrid(N=2**14, Tmax=Time(20 * T0, "s"))
wave = Wave(
    grid=grid,
    envelope=env,
    central_wavelength=Wavelength(800, "nm"),
    refractive_index=1.44,
)

wave = wave.with_effective_area(Area(80, "um^2"))

wave.peak_power()                          # W = ½·n·c·ε₀·A_eff·A₀²
wave.pulse_energy()                        # J = ∫ P dt
wave.average_power(Frequency(80, "MHz"))   # W
```

For one-shot conversions without a `Wave`, use
`PeakPower.from_envelope(A, A_eff, n, lambda0)`:

```python
from photonics_helper import Area, PeakPower, Wavelength

PeakPower.from_envelope(A0, Area(80, "um^2"), n=1.44, lambda0=Wavelength(800, "nm"))
# → Power -> ... W
```

The conversion follows the plane-wave intensity relation
`I = ½·n·c·ε₀·|A|²`, evaluated at the envelope peak.

# Waveguide Mode Import (FEM)

Bring an effective-index table from an external eigenmode solver (Lumerical
MODE, COMSOL Wave Optics, …) into the library's dispersion objects with
`WaveguideMode`.

**CSV** — comma-separated, optional header, columns `wavelength_um, neff`
(optionally a third `ng` column):

```csv
wavelength_um, neff
1.50, 2.4310
1.55, 2.4205
1.60, 2.4102
```

**NPZ** — `np.savez` archive with `wavelength_um` (µm) and `neff`, optionally
`ng` and `central_wavelength_nm`:

```python
import numpy as np
np.savez("mode.npz", wavelength_um=wl_um, neff=neff, central_wavelength_nm=1550.0)
```

```python
from photonics_helper import WaveguideMode, Wavelength

mode = WaveguideMode.from_csv("mode.csv")   # or .from_npz("mode.npz")
pc = mode.to_propagation_constant()         # β = n_eff·ω/c
D = mode.to_dispersion()                    # D(λ) = -λ/c · d²n_eff/dλ²

mode.neff_at(Wavelength(1550, "nm"))        # interpolated n_eff
pc.beta2(Wavelength(1550, "nm"))            # d²β/dω² (s²/m)
```

The table is validated on load: a finite, positive, strictly increasing
wavelength grid with at least four points (the cubic-spline minimum), mirroring
`RefractiveIndex`.

Runnable example: [`examples/29_waveguide_mode_import.py`](examples/29_waveguide_mode_import.py)
(fabricates a synthetic FEM export, round-trips CSV/NPZ, and checks `beta2`
against the analytic derivative).

### Real solver exports (femwell + Tidy3D)

[`examples/generate_waveguide_mode_data.py`](examples/generate_waveguide_mode_data.py)
solves a canonical Si strip (500 × 220 nm on SiO₂) with **femwell** (FEM) and
**Tidy3D**'s local mode solver, and writes `n_eff(λ)` tables in the conventions
above to `examples/data/`. The committed exports let the consumer example run
without either solver installed:

```bash
pip install femwell tidy3d          # or: pip install -e '.[mode-export]'
python examples/generate_waveguide_mode_data.py   # regenerate examples/data/
python examples/31_waveguide_mode_import_femwell_tidy3d.py
```

The two independent solvers agree on `n_eff` to **0.76 %** over 1.5–1.6 µm;
the import example cross-checks them, builds `PropagationConstant` /
`Dispersion`, and feeds the result into `phase_matching`.

# χ⁽²⁾ Nonlinear Optics (SHG / SFG / DFG)

`photonics_helper.chi2` integrates scalar, long-pulse three-wave mixing in a
waveguide with a fourth-order Runge–Kutta step in the interaction picture
(RK4IP). Envelopes are normalized so `|A|²` is power in watts. At perfect
phase matching the pump-depleted SHG solution is exact,
`η = tanh²(κL)` with `κ = σ√P₀` (see `reproductions/shg_textbook/`).

```python
from photonics_helper import Area, Wavelength
from photonics_helper.chi2 import Lambda_qpm, shg_coupling, solve_shg

wl = Wavelength(1550, "nm")
sigma = shg_coupling(wl, d_eff=10e-12, n=2.0, A_eff=Area(1.0, "um^2"))

# v0.1.1: σ follows the Boyd convention (ω d_eff/(n c))·√(2 Z₀/(n³ A_eff)),
# which is 2× smaller than the v0.1.0 release — see CHANGELOG [Unreleased].
```
result = solve_shg(length=4e-3, P0=0.1, sigma=sigma, n_steps=4000)
result.efficiency()[-1]     # η at the output
result.power("sh")          # SH power vs z (W)

# Per-field propagation loss (dB/cm) — lossy coupled-mode limit:
solve_shg(length=4e-3, P0=0.1, sigma=sigma, loss_db_per_cm=(3.0, 63.5))

# Quasi-phase-matching: Λ = 2π/|Δk| from the phase mismatch
period = Lambda_qpm(delta_k)
result = solve_shg(length=4e-3, P0=0.1, sigma=sigma, delta_k=delta_k, qpm_period=period)

# SFG / DFG (generic ω₃ = ω₁ + ω₂)
from photonics_helper.chi2 import solve_sfg, solve_dfg
solve_sfg(length=5e-3, P1=0.1, P2=0.1, sigma=sigma)
solve_dfg(length=5e-3, Ppump=0.1, Psignal=1e-3, sigma=sigma)
```

Scope: scalar envelopes, no group-velocity mismatch, dispersion, walk-off or
loss — suitable for CW / long-pulse efficiency estimates.
`delta_k_shg(beta_fn, omega)` computes `Δk = β(2ω) − 2β(ω)` from any
`phase_matching` adaptor, and `qpm_grating(z, Λ)` gives the square-wave poling
sign.

# FROG (Frequency-Resolved Optical Gating)

Generate FROG traces and retrieve ultrashort pulse shapes using PCGPA.

```python
import numpy as np
from photonics_helper import generate_trace, retrieve, fidelity

# Create a chirped pulse
T0 = 50e-15  # 50 fs
N = 2**10
dt = 10 * T0 / N
t = np.arange(N) * dt - N * dt / 2
E = np.exp(-t**2 / (2 * T0**2)) * np.exp(1j * 0.5 * 2.0 * (t / T0)**2)

# Generate FROG trace
trace = generate_trace(E, dt=dt)

# Retrieve pulse using PCGPA
result = retrieve(trace, max_iter=100)
f = fidelity(trace, result)  # → ~1.0 for good retrieval

# Visualize trace + retrieved field
fig = trace.visualize(retrieved=result)
fig.savefig("frog.png", dpi=150, bbox_inches="tight")
```

# Raman Material Database

Load Raman material parameters from the built-in SQLite database (44 materials):

```python
from photonics_helper.raman import RamanSpec

# Load from bundled materials.db (falls back to hardcoded dict)
silica = RamanSpec.from_database("Silica")
print(silica.summary())
# Material: Silica
# Raman shift: 440.0 cm⁻¹ = 13.19 THz
# Linewidth: 45.0 cm⁻¹ = 1.35 THz
# fR: 0.18
# n₂: 3.2e-20 m²/W
```

All 44 available materials:

| Category | Materials |
|----------|-----------|
| Glasses | Silica, GeO₂, As₂S₃, As₂Se₃, ZBLAN |
| Semiconductors | Si, Ge, GaAs, GaN, AlN, InP, InGaAs, AlGaAs, SiC, Si₃N₄, Si₃N₄-Ligentec |
| II-VI | CdS, CdTe, ZnO, ZnSe |
| Oxides | Ga₂O₃, Al₂O₃ (sapphire), BaTiO₃, LiNbO₃, LiTaO₃, KTP, YVO₄ |
| Chalcogenides | GeAsSe |
| Crystals & Hosts | Diamond, YAG, YLF, Zerodur |
| NLO Crystals | LBO, AgGaS₂, AgGaSe₂ |
| Optical substrates | BaF₂, CaF₂, MgF₂, KBr, F₂, N-BK7, N-F2, N-SF11, PMMA |

```python
# Stokes / anti-Stokes for a given pump
from photonics_helper.base import Wavelength

pump = Wavelength(800, "nm")
stokes = silica.stokes_wavelength(pump)
anti = silica.anti_stokes_wavelength(pump)
print(f"Stokes: {stokes.as_nm:.1f} nm, Anti-Stokes: {anti.as_nm:.1f} nm")
```

## What data ships (catalogue)

List every dataset the library bundles — tabulated `n`/`k` spectra and
Sellmeier equations — with its wavelength range, DOI and licence:

```python
from photonics_helper import material_catalog, print_material_catalog

print_material_catalog("sil")       # rich table, case-insensitive name filter

rows = material_catalog("LiNbO3")   # programmatic: MaterialDataset objects
for d in rows:
    print(d.material, d.kind, d.axis, d.wavelength_range_um, d.doi, d.license)
```

`material_catalog()` with no argument returns everything (69 datasets across 44
Raman materials, 39 Sellmeier equations and 30 tabulated datasets; 28 with a
DOI). The `axis` field labels the LiNbO₃ ordinary/extraordinary sub-rows.

# Soliton Analysis

Analyze soliton dynamics from GNLSE simulation results:

```python
from photonics_helper.gnlse import GNLSESolver, FiberProfile
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.soliton import SolitonAnalyzer, plot_soliton_trajectories
from photonics_helper.base import Wavelength, Time, Area, Length
import numpy as np

# Create pulse and fiber (waveguide with confinement factor)
grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
pulse = Wave(
    grid=grid,
    envelope=Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(100e-15, "s")),
    central_wavelength=Wavelength(1550, "nm"),
)
fiber = FiberProfile(
    n2=6.0e-18,  # GeAsSe
    alpha=0.0,
    A_eff=Area(0.2e-12, "m^2"),
    length=Length(5e-3, "m"),
    confinement_factor=0.8,  # waveguide
)

# Run GNLSE
solver = GNLSESolver(
    pulse, fiber,
    betas=np.array([-0.2, 0.001]),  # β₂=-0.2 ps²/m, β₃=0.001 ps³/m
    include_raman=True,
)
solver.propagate(num_steps=200)

# Analyze soliton dynamics
analyzer = SolitonAnalyzer(
    pulse, fiber, solver.betas, solver.z_array, solver.spectra_vs_z
)
print(f"Soliton order: {analyzer.soliton_order():.2f}")
print(f"Fission length: {analyzer.fission_length():.2f} mm")
print(f"DW wavelength: {analyzer.dispersive_wave_wavelength()*1e9:.1f} nm")

# Plot results
fig = plot_soliton_trajectories(solver)
fig.savefig("trajectories.png", dpi=150)
```

# Breathers, Noise and Wave Breaking

Three modules cover physics that deterministic pulse solvers cannot express on
their own: exact NLSE breather solutions, stochastic seeding, and the
normal-dispersion wave-breaking regime.

## Analytic breathers (`breathers.py`)

Exact soliton-on-finite-background solutions of the focusing NLSE — the
Akhmediev breather, the Peregrine soliton and the Kuznetsov–Ma soliton — plus a
physical-parameter mapping that turns fibre parameters into an exact initial
field:

```python
from photonics_helper.base import Time
from photonics_helper.breathers import (
    SolitonOnBackground,
    peregrine_soliton,
    akhmediev_breather,
    general_sfb,
)
from photonics_helper.pulse import TemporalGrid

# SMF-28 at 1550 nm, 0.7 W background (Kibler et al. 2012 parameters)
sob = SolitonOnBackground(beta2=-21.8e-27, gamma=1.3e-3, P0=0.7)
print(f"L_NL = {sob.L_NL:.1f} m, T0 = {sob.T0*1e12:.3f} ps")
print(f"KM period = {sob.spatial_period_m(0.66)/1e3:.3f} km")
print(f"KM peak   = {sob.peak_power(0.66):.3f} W")

grid = TemporalGrid(N=4096, Tmax=Time(80e-12, "s"))
wave = sob.initial_wave(grid, a=0.66)          # exact Kuznetsov-Ma field

# Or use the dimensionless solutions directly:
psi = peregrine_soliton(xi=0.0, tau=grid.t / sob.T0)   # peak |psi|^2 = 9
psi_ab = akhmediev_breather(xi=0.0, tau=grid.t / sob.T0, a=0.25)
psi_any = general_sfb(xi=0.0, tau=grid.t / sob.T0, a=0.66)
```

## Stochastic noise (`noise.py`)

Reproducible (seedable) noise sources, needed for spontaneous MI and
supercontinuum coherence studies:

```python
from photonics_helper.noise import add_noise, add_ase_noise, complex_gaussian_noise

noisy = add_noise(cw_wave, rms_relative=0.01, seed=1)       # ~1 % amplitude noise
seeded = add_ase_noise(cw_wave, level_dB=-50.0, seed=1)     # -50 dB ASE background
n = complex_gaussian_noise(grid, rms=1e-3, seed=0)          # raw time-domain noise
```

`add_ase_noise` handles the `TemporalGrid.fft`/`ifft` `dt` scaling internally, so
the requested dB level is exact.

## Wave breaking (`wave_breaking.py`)

Analytic wave-breaking distance and diagnostics for normal dispersion:

```python
from photonics_helper.wave_breaking import WaveBreaking, wave_breaking_distance

wb = WaveBreaking(beta2=20e-27, gamma=1.5e-3, P0=10.0, T0=10e-12)
print(f"z_WB = {wb.z_WB:.1f} m, sqrt(L_D L_NL) = {wb.sqrt_LD_LNL:.1f} m")
# after propagating a Gaussian:
# result = wb.analyze(solver.z_array, [w.envelope_field for w in solver.evolution], grid.t)
# result["z_onset_m"], result["z_oscillation_m"], result["peak_steepness"]
```

# Structured light — Laguerre-Gaussian / OAM (`structured.py`)

Analytic Laguerre-Gaussian transverse modes carrying orbital angular momentum
(OAM), Gaussian-beam propagation helpers and discrete overlap integrals. Modes
are normalized to unit power, so ``overlap`` gives the modal overlap directly:

```python
from photonics_helper.structured import (
    LaguerreGaussianMode,
    overlap,
    rayleigh_range,
    beam_waist,
)

w0, lam = 1e-3, 1064e-9                      # 1 mm waist at 1064 nm
lg01 = LaguerreGaussianMode(p=0, l=1, w0=w0, wavelength=lam)  # OAM = hbar
lg02 = LaguerreGaussianMode(p=0, l=2, w0=w0, wavelength=lam)

print(overlap(lg01, lg01))                    # 1.0  (normalized)
print(overlap(lg01, lg02))                    # ~ 0  (orthogonal OAM)

z_r = rayleigh_range(w0, lam)                 # pi w0^2 / lambda
print(beam_waist(w0, z_r, lam))               # sqrt(2) * w0

field = lg01.structured()                     # StructuredField on (x, y)
print(field.power, field.second_moment_radius())
fig = field.plot()                            # intensity + phase panels
```

``plot_transverse_profile`` mirrors the pulse backend pattern
(``backend="matplotlib"`` default, ``backend="plotly"`` when plotly is
installed). The ``OAM`` winding of a mode is ``2 pi l`` around any loop that
encloses the optical axis; see `examples/28_structured_light.py`.


# Vector / polarization-coupled GNLSE

`photonics_helper.vector_gnlse` propagates the field as a two-component
polarization vector — the physics the scalar engine structurally cannot
express (details: `docs/vector-gnlse.md`):

- per-axis Taylor dispersion, PMD walk-off (`walkoff`, s/m) and loss,
- XPM `2/3` + coherent polarization FWM (`coupling="coherent"`, Agrawal §6.3,
  energy-conserving mixing pair advanced with an RK4IP substep),
- the Manakov polarization-averaged mode (`coupling="manakov"`, the `8/9`
  coefficient of Wai & Menyuk 1996) for randomly birefringent fiber — a
  scalar engine overstates the effective nonlinearity by 12.5% there,
- `RandomBirefringenceEngine`: SU(2) random-frame evolution whose ensemble
  converges to the Manakov limit (validated),
- exact scalar reduction: with `A_y ≡ 0` the vector engine equals the scalar
  `SplitStepEngine` to machine precision, so every published scalar
  reproduction remains untouched.
- exact scalar reduction: with `A_y ≡ 0` the vector engine equals the scalar
  `SplitStepEngine` to machine precision, so every published scalar
  reproduction remains untouched.

`photonics_helper.multimode_gnlse` generalizes the same machinery to **N
simultaneously guided spatial modes** (`MultimodeSplitStepEngine`): per-mode
dispersion/group-delay/loss, LP (`1, 2/3`) or isotropic SPM-XPM models, and
opt-in pump-driven inter-modal FWM (`include_fwm=True`) gated by the
angular-momentum rule `ℓ_m = 2ℓ_n − ℓ_q` when OAM indices are supplied —
so `structured.py`'s LG/OAM portraits become a genuinely nonlinear
modal-propagation layer. Details: `docs/multimode-gnlse.md`.
so `structured.py`'s LG/OAM portraits become a genuinely nonlinear
modal-propagation layer. Details: `docs/multimode-gnlse.md`.

**Phase 4 items 4–5** — `chi2.solve_cascaded_shg` couples the χ⁽³⁾ Kerr
terms into the degenerate SHG three-wave integrator (limit contracts:
pure-quadratic = `solve_shg` exactly; pure-Kerr = analytic SPM phase;
cascaded-Kerr limit `γ_φ = σ²P₀/Δk` recovered <5%), and
`inverse_design.fit_two_wave` / `design_efficiency` provide the
least-squares parameter-identification / device-design layer on the exact
forward solvers (identifiable invariants κ = σ√P₀ and |Δk| recovered
exactly; tanh²(κL) design length analytic; loud failures on unreachable
targets / degenerate directions). Details: `docs/cascaded-chi23.md`.

# Solver physics hardening

Four pieces of GNLSE physics, each stated against the literature and tested
against analytic ground truth (details and derivations: `docs/gnlse-physics.md`):

- **Interaction-picture self-steepening.** The shock step factors out the
  exact Kerr/Raman phase and advances only the shock correction with RK4 in
  the interaction picture (Hult 2007). No spectral bin is clamped, and a
  spectral-validity guard requires the grid to resolve only positive absolute
  frequencies (`Ω_max < ω₀`), failing with the remedy when it does not. The
  photon number is conserved exactly in the constant-drive limit and the step
  matches a fully-resolved reference of the same flow to ≈1e-5.
- **Multi-phonon Raman response.** `PhononResponse.h_R(t)` is a causal,
  unit-integral superposition of damped oscillators (Hollenbeck & Cantrell
  2002), and the solver dispatches over it: crystalline materials propagate
  with their full multi-mode response, silica keeps the classic Blow–Wood form.
- **Time-resolved TPA / free carriers (opt-in).**
  `SplitStepEngine(include_free_carriers=True)` resolves the carrier density
  `N(t)` on the retarded-time grid instead of spatially averaging it:
  exact TPA attenuation `I₀/(1+βI₀z)`, carrier generation/recombination with
  a lifetime, and free-carrier absorption (Soref & Bennett 1987; Cowan et al.
  2003).
- **Convergence and validation harness.**

```python
from photonics_helper import convergence_study, check_soliton, ValidationFailure

report = convergence_study(
    build,                       # your solver factory → propagated solver
    refinements=[
        {"N": 8192,  "Tmax_s": 8e-12, "num_steps": 1000},
        {"N": 16384, "Tmax_s": 8e-12, "num_steps": 2000},
        {"N": 32768, "Tmax_s": 8e-12, "num_steps": 2000},
    ],
    observables=["peak_intensity", "rms_bandwidth"],
    tolerance=2e-3,
)
assert report.converged, report.summary()

# cited checks raise ValidationFailure on a closed-form mismatch:
check_soliton(beta2=-21e-27, gamma=1.0, t0=1e-12, wavelength_m=1550e-9, grid=grid)
```

`check_spm` (Stolen & Lin 1978), `check_mi` (Agrawal Eq. 5.1.9),
`check_soliton` (Agrawal §5.2) and `check_gordon_ssfs` (Gordon 1986) each
compare a *propagation result* against the closed form, so a configuration is
checked against physics and not against another code (Sinkin et al. 2003 for
the convergence criteria).

# Reproductions

The `reproductions/` directory validates the library against published results,
not against other software. Each folder has a `parameters.json`, a script that
asserts its result against an analytic/closed-form reference, a figure, and a
README with the DOI, findings and open issues.

| Reproduction | Reference (DOI) |
|---|---|
| `stolen_lin_1978_spm` | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) · [10.1103/PhysRevA.17.1448](https://doi.org/10.1103/PhysRevA.17.1448) |
| `macleod_quarter_wave_dbr` | Macleod, *Thin-Film Optical Filters* (textbook) |
| `gordon_1986_ssfs` | Gordon, *Opt. Lett.* **11**, 662 (1986) · [10.1364/OL.11.000662](https://doi.org/10.1364/OL.11.000662) |
| `dudley_2006_cherenkov_dw` | Akhmediev & Karlsson, *Phys. Rev. A* **51**, 2602 (1995) · [10.1103/PhysRevA.51.2602](https://doi.org/10.1103/PhysRevA.51.2602) |
| `dudley_2006_scg` | Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006) · [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135) |
| `kuznetsov_ma_2012_breather` | Kibler et al., *Sci. Rep.* **2**, 463 (2012) · [10.1038/srep00463](https://doi.org/10.1038/srep00463) |
| `narhi_2016_mi_breathers` | Närhi et al., *Nat. Commun.* **7**, 13675 (2016) · [10.1038/ncomms13675](https://doi.org/10.1038/ncomms13675) |
| `tomlinson_1985_wave_breaking` | Tomlinson, Stolen & Johnson, *Opt. Lett.* **10**, 457 (1985) · [10.1364/OL.10.000457](https://doi.org/10.1364/OL.10.000457) |

See [`reproductions/README.md`](reproductions/README.md) for the full status,
cross-cutting findings and issues, and
[`reproductions/dudley_2006_scg/PAPER_ANALYSIS.md`](reproductions/dudley_2006_scg/PAPER_ANALYSIS.md)
for the figure-by-figure reproducibility map of the Dudley review.

```sh
python -m pytest tests/test_reproductions.py   # all reproduction regressions
python reproductions/dudley_2006_scg/fig05_ideal_soliton_period.py   # one figure
```

# Dashboard & API docs

One Dash app hosts both the Raman explorer and a GNLSE result viewer:

```python
from photonics_helper.dashboard import app

app().run(debug=False, port=8050)   # Raman Explorer | GNLSE Viewer
```

Requires the `webapp` (Dash) and `plotting` (Plotly) extras; see
`examples/32_unified_dashboard.py`. The standalone Raman app remains available
as `photonics_helper.raman.app()`.

API documentation is built with [Zensical](https://zensical.org/) +
mkdocstrings from the docstrings (one page per module in the API section):

```bash
pip install -e ".[docs]"
zensical serve      # http://127.0.0.1:8000
```

(Zensical reads the repo's `mkdocs.yml`, so the classic `mkdocs serve`
workflow also works — both builders stay supported.)

`.github/workflows/` also provides the CI matrix, the docs build/deploy, and
the PyPI Trusted Publishing release (cut a `v*` tag to publish).

# Development

To install for development:

```sh
git clone https://github.com/hitaishi2222/photonics_helper
cd photonics_helper
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
pip install -e .
```

# Roadmap

The feature status and evolution tracker lives in **[ROADMAP.md](ROADMAP.md)** —
shipped features (✅), how they got built, and what's coming next.

# License & data

The source code is released under the **MIT License** (see [`LICENSE`](LICENSE)).

The bundled material database (`photonics_helper/materials.db`) is a
compilation of numerical data from the published literature and from
[refractiveindex.info](https://refractiveindex.info/) (CC0). Every row carries
its own `source`/`citation`/`references` provenance **and a `license` column**;
the `provenance` registry joins nk datasets to their citation. See [`NOTICE`](NOTICE),
the [data provenance](https://hitaishi2222.github.io/photonics_helper/data-provenance/)
page, and the [database schema](https://hitaishi2222.github.io/photonics_helper/data-schema/)
for the full audit.
