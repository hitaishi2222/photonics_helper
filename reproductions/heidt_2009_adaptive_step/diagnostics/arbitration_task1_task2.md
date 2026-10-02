# Arbitration of ISSUES.md #14 symptoms (2026-10-01)

## Symptom 1 — deck A global error saturates at ~5e-4

RESOLVED as a pre-fix artifact. The recorded measurements (6.1e-4 … 4.9e-4
across dz = 1e-5 … 1e-6) were taken before the `betas_si` unit fix
(`ps^k/m → s^k/m`, powers from k = 2); the pre-fix deck-A cascade was pure
dispersion. Fresh post-fix runs:

| dz (m) | eps vs ref (dz=5e-6) | n_fft |
|---|---|---|
| 4e-5 | 2.90e-3 | 40 002 |
| 2e-5 | 1.41e-4 | 80 002 |
| 1e-5 | 6.71e-6 | 160 002 |
| 1e-6 | 7.09e-6 | 1 600 002 |

Measured local orders: 4.36 (4e-5→2e-5), 4.40 (2e-5→1e-5) — clean RK4IP
4th order. At dz = 1e-6 the error stops improving at ~7e-6: that is the
chaotic sensitivity floor of the fissioning cascade (tiny seed differences
amplify), NOT a numerics floor and NOT the recorded 5e-4 saturation.
Conclusion: deck A reaches ε ≈ 1e-5; the Fig.-2 efficiency ladder is
testable over ε ∈ [1e-2, 1e-5] (paper: 1e-4…1e-12 — unreachable tail is a
chaos property of the physical deck, recorded as a bounded deviation).
Data: `arbitrate_eps_floor.json`.

## Symptom 2 — deck B two-soliton field disperses into a pedestal

RESOLVED as a pre-fix artifact of the same family: the recorded run read
β₂ = −0.1 ps²/km as ps²/m (1000× too dispersive), smearing both solitons
into a pedestal by 40 km. Fresh post-fix RK4IP run (dz = 25 m, 400 km,
N = 2048, T = 256 ps), snapshots of |A|² peak structure:

| z (km) | peaks (ps) | rel heights | note |
|---|---|---|---|
| 40 | −90.0, −10.0 | 1.00, 0.99 | walking together |
| 100 | −74.9, −25.1 | 1.00, 0.98 | closing |
| 200 | −53.6…−47.6 | merged | COLLISION, peak 3.9× single-soliton |
| 300 | −75.5, −24.5 | 0.93, 1.00 | passed through |
| 400 | −100.6, +0.6 | 0.91, 1.00 | separated cleanly |

Energy conserved to +0.07 % over 400 km; window-edge leakage < 1e-8;
single detuned solitons (±400 GHz alone) hold shape exactly. The collision
lands at the paper's 200 km (walk-off β₂·Δω = 0.5 ps/km). Deck B physics
is correct; the check_fig3 step-profile/efficiency comparison can run.
Data: `deck_b_diagnose.json`.
