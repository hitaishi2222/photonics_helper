# Post-#0 sweep (2026-09-28, openspec fix-audit-issues-batch §7)

Engine deck re-run after the ISSUES.md #0 convention fix:

- Eq-(12) log-averaged amplification readout, L = 5 m, 2 seeds, 65536 grid:
  band |ν| ∈ (0.05, 0.65) max = 0.233 / edge (0.7–1.15) mean = 0.230 with
  NOISE_POWER_W = 1e-7 (flat, as before the fix).
- Seed-level sweep (ranked suspect 1): 1e-7 → 0.233/0.230; 1e-5 → 0.188/0.184;
  1e-3 → 0.141/0.138. Raising the seed does NOT produce the paper's banded
  morphology (paper: band ≈ 0.64 at 5 m); the readout is seed-robust flat,
  i.e. the flatness is dynamics, not floor saturation.

Remaining open (task 7.3): per-channel pump–sideband walk-off at the mode-2
detuning in the channel-0 retarded frame vs Eq. (11); possible dedicated
per-channel-delay arm. Recorded; outstanding.
