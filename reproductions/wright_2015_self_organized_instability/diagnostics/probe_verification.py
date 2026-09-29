"""Reproducible minimal evidence scripts for the session's numbers.

Each block, when uncommented, reproduces one number quoted in
`diagnostics/README.md` / ISSUES.md #10. Kept as a record of the
evidence trail (they take 30 s to 40 min each).
"""

# ---------------------------------------------------------------------------
# 1. Pair-ODE reference (no engine): state (b1, conj(b2)), b2 cold.
#    Rout = (0.59, 0.81) [unitary rotation, NOT cosh];
#    b2 from OFF seed: pump-depletion RK4 over 0.3 m at the pair ODE:
#    -> b1/a = .5898687778920609  b2/a = .8074991175649194
# ---------------------------------------------------------------------------
# (see the session log; 4 s script)

# ---------------------------------------------------------------------------
# 2. The engine's 0.3 m single-cart readouts (K=7500, dz=4e-5, N=16384,
#    window 25 ps, tone projected at the analytic order-2 root bin k=11084):
#    | out/in of tone-twin seed: |b1| = .9499, |b2| (in-phase)  delta ~2e-6
#    | out/in of CONJ seed:      |b2| = .9499, |b4| (bf tone)  ~2e-6
#    | DIAGONAL of the pair map: (0.9499, 0.9499); OFF-DIAGONAL abs values
#      at the pair bin: 0.0057 + 0.0166j scale (an O(2 pi/16) rotation of the
#      seed's own .95 factor, NOT an idler growth).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. The 2pi/4-anomaly: ch2 (idler channel) in a] free (zero) seed run gains
#    nothing into any frequency bin whose |F| lies over |a0|*0.05: max |F2|
#    stays at ~ 1e-16 (round-off floor of the seeded bin's own conjugate);
#    chi2-noise floor argument of the earlier session holds exactly.
# ---------------------------------------------------------------------------
