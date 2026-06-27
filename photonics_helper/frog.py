"""
SHG-FROG (Frequency-Resolved Optical Gating) module.

Generates FROG traces from electric fields and retrieves pulses from
measured traces using the PCGPA (Principal Component Generalized Phase
Retrieval) algorithm.

SHG-FROG trace:
    I(ω, τ) = |F{E(t) · E(t - τ)}|²

where E(t) is the electric field, τ is the delay, and ω is the angular frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from photonics_helper.pulse import TemporalGrid, Envelope, Wavelength, Frequency


@dataclass
class FROGTrace:
    """Represents a FROG trace I(ω, τ).

    Attributes
    ----------
    trace : 2D array of shape (N_omega, N_tau)
        The FROG trace (normalized if normalize=True in from_field).
    unnormalized_trace : 2D array of shape (N_omega, N_tau)
        The raw trace before normalization (used for retrieval).
    omega : angular frequency axis (rad/s), centered at 0.
    tau : delay axis (s), centered at 0.
    dt : time step (s).
    dw : frequency step (rad/s).
    field : reconstructed field E(t) (set after retrieval).
    """

    trace: NDArray
    unnormalized_trace: NDArray
    omega: NDArray
    tau: NDArray
    dt: float
    dw: float
    field: Optional[NDArray] = field(default=None, repr=False)

    @classmethod
    def from_field(
        cls,
        E_field: NDArray,
        dt: float,
        normalize: bool = True,
    ) -> "FROGTrace":
        """Generate a FROG trace from a complex electric field E(t).

        Parameters
        ----------
        E_field : complex 1D array, the electric field on a uniform time grid.
        dt : time step in seconds.
        normalize : normalize trace to [0, 1] (default True).

        Returns
        -------
        FROGTrace
        """
        N = len(E_field)
        # Create delay axis: tau = [-N/2*dt, ..., N/2*dt - dt]
        tau = np.arange(-N // 2, N // 2) * dt

        # Build the gated signal G(t, tau) = E(t) * E(t - tau)
        # Use FFT-based shift for efficiency.
        # G(t, tau) = IFFT{ FFT{E} * conj(FFT{E}) * exp(i*omega*tau) }
        # Actually, simpler: G(t, tau) = E(t) * E(t - tau) directly.

        # For each tau, shift E and multiply
        E_shifted = np.zeros((len(tau), N), dtype=complex)
        for i, t_val in enumerate(tau):
            # Shift: E(t - tau) means E shifted by +tau in time
            shift = int(round(t_val / dt))
            if shift >= 0:
                E_shifted[i] = np.roll(E_field, -shift)
            else:
                E_shifted[i] = np.roll(E_field, -shift)

        # G(t, tau) = E(t) * E(t - tau)
        G = E_field[np.newaxis, :] * E_shifted  # shape: (N_tau, N)

        # FFT along time axis
        E_hat = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G, axes=1), axis=1), axes=1)

        # FROG trace: I(omega, tau) = |E_hat|²
        trace = np.abs(E_hat) ** 2

        # Frequency axis
        omega = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(N, d=dt))

        raw_trace = trace.copy()
        if normalize and trace.max() > 0:
            trace = trace / trace.max()

        dw = omega[1] - omega[0] if len(omega) > 1 else 1.0

        return cls(
            trace=trace,
            unnormalized_trace=raw_trace,
            omega=omega,
            tau=tau,
            dt=dt,
            dw=dw,
        )

    def visualize(
        self,
        retrieved: Optional["FROGTrace"] = None,
        figsize: tuple[float, float] | None = None,
        save_path: str | None = None,
    ):
        """Plot the FROG trace and optionally the retrieved pulse.

        Parameters
        ----------
        retrieved : optional FROGTrace with .field set (from retrieve()).
        figsize : figure size.
        save_path : if given, save figure to this path.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if figsize is None:
            figsize = (14, 6) if retrieved is None else (16, 7)

        if retrieved is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()

        # Trace heatmap
        ax_trace = axes[0]
        omega_THz = self.omega / (2 * np.pi * 1e12)
        tau_ps = self.tau * 1e12

        im = ax_trace.pcolormesh(
            tau_ps,
            omega_THz,
            self.trace,
            shading="gouraud",
            cmap="inferno",
        )
        ax_trace.set_xlabel("Delay (ps)")
        ax_trace.set_ylabel("Frequency (THz)")
        ax_trace.set_title("FROG Trace")
        fig.colorbar(im, ax=ax_trace, label="Normalized intensity")

        if retrieved is not None and retrieved.field is not None:
            E = retrieved.field
            t = np.arange(len(E)) * self.dt * 1e12  # ps

            # Intensity
            ax_int = axes[1]
            intensity = np.abs(E) ** 2
            intensity /= intensity.max() if intensity.max() > 0 else 1
            ax_int.plot(t, intensity, color="#00d4ff", linewidth=1.2)
            ax_int.set_xlabel("Time (ps)")
            ax_int.set_ylabel("Intensity (arb.)")
            ax_int.set_title("Retrieved Intensity")
            ax_int.grid(True, alpha=0.3)

            # Phase
            ax_phase = axes[2]
            phase = np.unwrap(np.angle(E))
            ax_phase.plot(t, phase, color="#ff6b6b", linewidth=1.2)
            ax_phase.set_xlabel("Time (ps)")
            ax_phase.set_ylabel("Phase (rad)")
            ax_phase.set_title("Retrieved Phase")
            ax_phase.grid(True, alpha=0.3)

        fig.suptitle(
            "FROG Analysis" + (" — Retrieved" if retrieved is not None else ""),
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


def generate_trace(
    E_field: NDArray,
    dt: float,
    normalize: bool = True,
) -> FROGTrace:
    """Generate a SHG-FROG trace from an electric field.

    Parameters
    ----------
    E_field : complex 1D array, electric field on uniform time grid.
    dt : time step in seconds.
    normalize : normalize trace to [0, 1] (default True).

    Returns
    -------
    FROGTrace
    """
    return FROGTrace.from_field(E_field, dt=dt, normalize=normalize)


def fidelity(trace1: FROGTrace, trace2: FROGTrace) -> float:
    """Compute FROG fidelity between two traces.

    fidelity = 1 - ||I1 - I2|| / ||I1||

    Returns
    -------
    float in [0, 1], where 1 = perfect match.
    """
    # Ensure same shape
    t1 = trace1.trace
    t2 = trace2.trace

    # Crop or pad to same shape
    min_rows = min(t1.shape[0], t2.shape[0])
    min_cols = min(t1.shape[1], t2.shape[1])
    t1 = t1[:min_rows, :min_cols]
    t2 = t2[:min_rows, :min_cols]

    diff = t1 - t2
    norm_diff = np.linalg.norm(diff)
    norm_ref = np.linalg.norm(t1)

    if norm_ref == 0:
        return 1.0 if norm_diff == 0 else 0.0

    return float(1.0 - norm_diff / norm_ref)


def retrieve(
    trace: FROGTrace,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = True,
) -> FROGTrace:
    """Retrieve the electric field E(t) from a FROG trace using PCGPA.

    Parameters
    ----------
    trace : FROGTrace — the measured (or generated) trace.
    max_iter : maximum iterations (default 100).
    tol : convergence tolerance on fidelity change (default 1e-4).
    verbose : print iteration progress (default True).

    Returns
    -------
    FROGTrace with .field set to the retrieved E(t).
    """
    N_tau, N_omega = trace.trace.shape
    dt = trace.dt
    dw = trace.dw

    # Use unnormalized trace if available (preserves amplitude info)
    measured_trace = trace.unnormalized_trace if trace.unnormalized_trace is not None else trace.trace

    # --- Step 1: SVD initialization ---
    # The trace is real and non-negative. We use its SVD to get an initial guess.
    # I(omega, tau) = sum_k s_k * u_k(omega) * v_k(tau)
    # Initial guess: E(t) proportional to the first right singular vector.
    U, S, Vt = np.linalg.svd(measured_trace, full_matrices=False)

    # Take the singular vector with largest singular value
    # E_init(tau) = sqrt(s_0) * v_0(tau)
    E_init = np.sqrt(S[0]) * Vt[0]

    # Normalize
    E_init = E_init / np.linalg.norm(E_init)

    # --- Step 2: PCGPA iteration ---
    E = E_init.copy()

    prev_fidelity = 0.0

    for iteration in range(max_iter):
        # a. Form gated signal: G(t, tau) = E(t) * E(t - tau)
        #    We need to be careful with indexing. tau ranges from -N/2 to N/2.
        #    E(t - tau) for tau > 0 means E shifted to the right.

        N = len(E)
        G = np.zeros((N_tau, N), dtype=complex)

        for i, tau_val in enumerate(trace.tau):
            # Shift amount in samples
            shift = int(round(tau_val / dt))
            # E(t - tau) = E shifted by +shift samples
            E_shifted = np.roll(E, -shift)
            G[i] = E * E_shifted

        # b. FFT along time axis
        G_hat = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(G, axes=1), axis=1), axes=1
        )

        # c. Replace magnitude with sqrt(I_meas), keep phase
        measured_mag = np.sqrt(measured_trace)
        G_new = measured_mag * np.exp(1j * np.angle(G_hat))

        # d. Inverse FFT along frequency axis
        G_new_time = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G_new, axes=1), axis=1), axes=1)

        # e. Extract new E from G_new_time at tau = 0 (center column)
        #    tau=0 is at index N_tau // 2
        E_new = G_new_time[:, N_tau // 2].copy()

        # Normalize
        E_new = E_new / np.linalg.norm(E_new)

        # f. Check convergence
        # Recompute trace from new E
        G_check = np.zeros((N_tau, N), dtype=complex)
        for i, tau_val in enumerate(trace.tau):
            shift = int(round(tau_val / dt))
            E_shifted = np.roll(E, -shift)
            G_check[i] = E * E_shifted

        G_check_hat = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(G_check, axes=1), axis=1), axes=1
        )
        trace_calc = np.abs(G_check_hat) ** 2
        if trace_calc.max() > 0:
            trace_calc /= trace_calc.max()

        curr_fidelity = fidelity(trace, FROGTrace(
            trace=trace_calc,
            unnormalized_trace=trace_calc,
            omega=trace.omega,
            tau=trace.tau,
            dt=trace.dt,
            dw=trace.dw,
        ))

        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            print(f"  Iter {iteration:3d}: fidelity = {curr_fidelity:.6f}")

        if abs(curr_fidelity - prev_fidelity) < tol and iteration > 5:
            if verbose:
                print(f"  Converged at iteration {iteration}, fidelity = {curr_fidelity:.6f}")
            break

        prev_fidelity = curr_fidelity
        E = E_new

    # Build result FROGTrace with retrieved field
    result = FROGTrace(
        trace=trace.trace,
        unnormalized_trace=trace.unnormalized_trace,
        omega=trace.omega,
        tau=trace.tau,
        dt=trace.dt,
        dw=trace.dw,
        field=E,
    )

    return result
