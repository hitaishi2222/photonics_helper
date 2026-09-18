"""
Example: FROG Retrieval of Chirped Pulses (Robust PCGPA)
========================================================

Demonstrates the corrected PCGPA retrieval in
``photonics_helper.pulse.retrieve``.

What changed relative to the earlier algorithm:

- The extraction step is now the **generalized-projections least-squares**
  update ``E_new(t) = Σ_τ G'(t,τ)·E*(t+τ) / Σ_τ|E(t+τ)|²`` for the SHG signal
  model ``E_sig(t,τ) = E(t)E(t+τ)`` (DeLong, Trebino, Hunter & White,
  JOSA B 11, 2206 (1994)) — not a fixed-time slice of the time-domain gated
  signal, which returned a time-*reversed* copy of the pulse and therefore
  failed on chirped pulses (symmetric pulses hid the bug).
- Retrieval is initialized from the dominant SVD component plus seeded
  random complex fields (``seed``, ``n_restarts``); the lowest-FROG-error
  result across restarts is returned.
- Convergence tracks the normalized FROG error of the *current* estimate
  against ``tol``, so a stalled restart is not mistaken for convergence.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from photonics_helper.pulse import fidelity, generate_trace, retrieve

N = 512
T0 = 50e-15  # 50 fs
Tmax = 10 * T0
dt = Tmax / N
t = np.arange(N) * dt - N * dt / 2


def chirped_gaussian(c: float) -> np.ndarray:
    return np.exp(-(t**2) / (2 * T0**2)) * np.exp(1j * 0.5 * c * (t / T0) ** 2)


print("FROG retrieval of chirped pulses (updated PCGPA)")
print("=" * 60)

# ── 1. Retrieval across chirp values, with ship-shape defaults ───────────
for chirp in [0.0, 2.0, 3.0]:
    trace = generate_trace(chirped_gaussian(chirp), dt=dt)
    result = retrieve(
        trace, max_iter=150, tol=1e-5, verbose=False, seed=0, n_restarts=5
    )
    f = fidelity(trace, result)

    # Determinism: the same call with the same seed returns identical fields
    result2 = retrieve(
        trace, max_iter=150, tol=1e-5, verbose=False, seed=0, n_restarts=5
    )
    identical = bool(np.allclose(result.field, result2.field))

    # Compare intensity envelopes (SHG-FROG cannot distinguish E(t) from
    # E*(-t) or a global phase, so trace-based fidelity is the right metric).
    print(
        f"  chirp C = {chirp:.1f} | fidelity {f:.4f} | seed-deterministic: {identical}"
    )


# ── 2. What the trace uniquely determines (and what it does not) ─────────
# The corrected PCGPA no longer returns a time-reversed field, and the
# envelope profile is recovered. But SHG-FROG itself leaves the
# E(t) ↔ E*(−t), E(t−t0) delays and global phase undetermined, so raw
# field comparisons must first align the time axis. Compare the intensity
# envelope after finding the best (shift, reversal) alignment:
def align_env(true_env2, rec_env2):
    """Best matched-shift |error| over circular shifts and reversal."""
    A = true_env2 / true_env2.max()
    best = np.inf
    for s in range(len(A)):
        roll = np.roll(rec_env2 / rec_env2.max(), s)
        best = min(best, np.max(np.abs(A - roll)), np.max(np.abs(A - roll[::-1])))
    return best


for chirp in [2.0, 3.0]:
    trace = generate_trace(chirped_gaussian(chirp), dt=dt)
    result = retrieve(
        trace, max_iter=150, tol=1e-5, verbose=False, seed=0, n_restarts=5
    )
    env_err = align_env(np.abs(chirped_gaussian(chirp)) ** 2, np.abs(result.field) ** 2)
    print(
        f"  chirp C = {chirp:.1f}: envelope error (best shift/reversal) "
        f"= {env_err:.4f}; trace fidelity = {fidelity(trace, result):.4f}"
    )

# ── 3. Visual: retrieved vs. true real field for the chirped cases ───────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, chirp in zip(axes, [2.0, 3.0]):
    E_true = chirped_gaussian(chirp)
    trace = generate_trace(E_true, dt=dt)
    result = retrieve(trace, max_iter=150, verbose=False, seed=0, n_restarts=5)
    ax.plot(t / T0, E_true.real, alpha=0.6, label="true Re E(t)")
    ax.plot(t / T0, result.field.real, alpha=0.6, label="retrieved Re E(t)")
    ax.set_title(f"chirp C = {chirp:.0f}, fidelity = {fidelity(trace, result):.4f}")
    ax.set_xlabel("t / T0")
    ax.set_ylabel("Re E(t)")
    ax.legend()
plt.tight_layout()
fig.savefig("examples/images/frog_chirped_retrieval.png", dpi=150, bbox_inches="tight")
print("  Saved examples/images/frog_chirped_retrieval.png")
plt.close(fig)

print("\n✓ FROG retrieval example complete")
