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
