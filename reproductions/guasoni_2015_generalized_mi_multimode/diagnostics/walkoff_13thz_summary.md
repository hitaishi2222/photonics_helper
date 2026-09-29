# Task 7.3 — per-channel walk-off audit (probe_walkoff_13thz.py)

**STATUS (2026-09-30, second run): the multimode linear layer is
VINDICATED.** The v1 "sign-inverted / window-dependent walk-off" finding
was a probe artifact: v1's probe envelopes were centered at tt = T/2,
which is the circular FFT seam (array edge), so every time-shift
measurement was biased by wrap spill across the seam — reproducing
exactly the reported pathology (magnitude wrong by the wrap distance,
window-dependent, sign-odd). v1's tone tests were additionally aliased
(T = 400 ps / N = 8192 -> dt = 48.8 fs -> Nyquist 10.24 THz < 13 THz).

The fixed probe (envelope centered at t = 0, T = 200 ps -> dt = 24.4 fs,
Nyquist 20.5 THz, tones +-5/&13 THz unaliased) gives:

| n | tone code-Om (THz) | GVM | tau_meas (ps) | verdict |
|---|---|---|---|---|
| 1 | 0    | True  | +5.400  | = GVM_1*0.5 = +5.4 EXACT |
| 1 | 0    | False |  0.000  | reference exact |
| 2 | 0    | True  | +3.550  | = GVM_2*0.5 EXACT |
| 3 | 0    | True  | +6.800  | = GVM_3*0.5 EXACT |
| 1 | -13  | False | -6.032  | = beta2_1*Omega*L = -6.032 EXACT |
| 1 | +13  | False | +6.032  | mirrored EXACT |
| 1 | -13  | True  | -0.632  | = 5.4 + (-6.03) EXACT |
| 1 | +13  | True  | +6.032+5.4 = 11.432 EXACT |
| 2 | -13  | False | +1.483  | = beta2_2*Omega (+1.48) |
| 3 | -13  | False | -0.143  | beta2_3 ~ microscopic |
| 1 | b3-only isolated | — | -13.01 | beta3*Om^2/2*0.5 + b2 arm = -13.02 (7 %) |

Interpretation for the audit question (deck vs Eq. (11)):
- The group-delays arm = GVM_n * L, sign and magnitude exact in the
  channel-0 frame (post-#0 e^{+i}-kernel convention).
- The beta2 group-velocity arm realized VERBATIM (dphi/dOmega = beta2 * W).
- The beta3 Omega^2/2 arm realized in dphi/dOmega (beta3-only probe
  matches the analytic 2x2 within ~7 %, limited by the tone packet's
  spectral width and the dz = 1 mm phase rotation).
- Tone-bin bookkeeping (post-#0): a tone exp(+i*Om*t) lands at bin
  w = -Om (red under the lambda map) — the hypotheses column must use
  the same bin-mapped side. The v1 hypotheses (hyp GVD/hyp SUM) used
  the un-mapped side, so their "mismatch" was an artifact too.

CONSEQUENCE for ISSUES.md #8: the multimode linear layer (the exact
armlist delta_beta^(p,s)_n = GVM*Om + b2*Om^2/2 + b3*Om^3/6) is NOT the
cause of the flat Eq.-12 banded readout. The walk-off explanation family
is CLOSED; the flat Eq.-12 readout needs a different mechanism (deck
noise statistics / seed level / windowing of the paper's own readout).
