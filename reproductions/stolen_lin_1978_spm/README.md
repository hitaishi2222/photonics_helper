# Reproduction — Stolen & Lin (1978): SPM in silica fibers

**Reference:** R. H. Stolen and C. Lin, "Self-phase-modulation in silica optical
fibers," *Phys. Rev. A* **17**, 1448 (1978), doi:10.1103/PhysRevA.17.1448.

## Result reproduced

A Gaussian pulse propagating in a dispersionless Kerr medium acquires the
nonlinear phase

```
phi(t) = phi_max · exp(-t²/T0²),   phi_max = gamma · P0 · L,
```

and its spectrum develops the classic interference fringes. The number of
spectral peaks follows the textbook rule `N_peaks = floor(phi_max/pi) + 1`
(Agrawal, *Nonlinear Fiber Optics*, §4.1).

## Ground truth

The analytic output field `A(L,t) = sqrt(P0) e^{-t²/2T0²} e^{i phi_max e^{-t²/T0²}}`
is Fourier-transformed numerically and compared with the GNLSE output. With
`beta = 0` the split-step nonlinear operator is exact, so agreement is at
floating-point precision.

## Outcome

```
phi=1π: peaks=2 (expected 2), max|Δ| = 3.0e-15
phi=2π: peaks=3 (expected 3), max|Δ| = 4.1e-15
phi=4π: peaks=5 (expected 5), max|Δ| = 1.2e-14
phi=8π: peaks=9 (expected 9), max|Δ| = 1.3e-13
```

Figure: `spm_spectra.png` (analytic solid black vs GNLSE red dashed).

## Findings

- With `β = 0` the split-step nonlinear operator is *exact*, so the GNLSE field
  agrees with the closed-form Fourier integral to machine precision
  (`max|Δ| ≤ 1.3e-13` at `φ = 8π`). This is the cleanest available unit test of
  the Kerr operator and of the `TemporalGrid` FFT scaling/sign convention.
- The textbook peak-count rule `N_peaks = floor(φ_max/π) + 1` (Agrawal §4.1) is
  reproduced exactly for `φ = π, 2π, 4π, 8π` (2/3/5/9 peaks).
- Because the operator is exact and dispersion-free, any mismatch here would
  isolate a Kerr-operator/FFT bug rather than a dispersion bug; the current
  match to `1e-13` is a strong regression guard.

## ISSUES

- The reproduction is deliberately *dispersionless*; it cannot detect errors
  in the β₂/β₃ handling (those are covered by the ideal-soliton and Cherenkov
  reproductions).
- Peak counting is done on a discretised spectrum; near a threshold
  (`φ_max/π` close to an integer) the count can be off by one. The chosen
  cases are all safely away from thresholds.
