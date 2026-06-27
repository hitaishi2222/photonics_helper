"""
Example: FROG (Frequency-Resolved Optical Gating)
===================================================

Demonstrates SHG-FROG trace generation and PCGPA retrieval.
Shows how to generate a FROG trace from a pulse, retrieve the field,
and assess retrieval quality.
"""

import numpy as np
from photonics_helper.frog import generate_trace, retrieve, fidelity
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG export
import matplotlib.pyplot as plt

# --- Setup: create test pulses ---
N = 2**10
T0 = 50e-15  # 50 fs pulse width
Tmax = 10 * T0
dt = Tmax / N
t = np.arange(N) * dt - N * dt / 2

pulses = {
    "gaussian": np.exp(-t**2 / (2 * T0**2)),
    "sech": 1.0 / np.cosh(t / T0),
    "chirped_gaussian": np.exp(-t**2 / (2 * T0**2)) * np.exp(
        1j * 0.5 * 2.0 * (t / T0) ** 2
    ),
}

print("FROG Trace Generation & PCGPA Retrieval")
print("=" * 50)

for name, E in pulses.items():
    print(f"\n{name}:")

    # Generate FROG trace
    trace = generate_trace(E, dt=dt)
    print(f"  Trace shape: {trace.trace.shape}")

    # Retrieve using PCGPA
    result = retrieve(trace, max_iter=100, verbose=False)
    f = fidelity(trace, result)
    print(f"  Retrieval fidelity: {f:.6f}")

    # Visualize
    fig = trace.visualize(retrieved=result)
    fig.savefig(f"examples/images/frog_{name}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved frog_{name}.png")
    plt.close(fig)

print("\n✓ All FROG examples saved to examples/images/")
