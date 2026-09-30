"""β₁-aware in-engine absolute-arrival readout (Brahms & Travers 2021).

Post-#0 validation addendum (ISSUES.md #0, "follow-up" row): the v1
reproduction measured RDW arrival in the engine's carrier-group-velocity
frame (≤ 0.2 fs envelope-frame imprint) and reconstructed the physical
walk-off analysis-side via Eq. (11)/(12) with the analytically modelled L_f.
Any future *in-engine* absolute-arrival observable needs a β₁-aware readout
helper — this module ships it.

Physical model of the helper
----------------------------
The engine propagates in the pump's carrier group-velocity frame, so the
pump envelope stays at τ = 0 at the exit. A resonant dispersive wave born
at the fission/compression point z_f at (simulated) wavelength λ_RDW walks
off from the pump frame with the group-delay difference of the guide at
that frequency:

    tau_abs = ∫_{z_f}^{L} [ beta1(omega_RDW; z) - beta1(omega0; z) ] dz ,

with omega_RDW taken from the ENGINE-MEASURED RDW spectral centroid (not an
analytic phase-matching root) and beta1(omega; z) evaluated on the same
dispersion profile family the engine used (constant or gradient decks).

Sign convention: τ > 0 means the RDW arrives LATER than the pump.
"""
from __future__ import annotations

import importlib.util as ilu
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
C_MS = 299792458.0


def _load_rep():
    spec = ilu.spec_from_file_location("btrep", HERE / "reproduce.py")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def omega0_from_lambda(lambda_nm: float) -> float:
    return 2 * np.pi * C_MS / (lambda_nm * 1e-9)


def lambda_from_omega(omega: float | np.ndarray) -> float | np.ndarray:
    return 2 * np.pi * C_MS / np.asarray(omega, dtype=float) * 1e9


def absolute_arrival_time_fs(
    beta1_fn: Callable[[float], Callable[[float], float]],
    *,
    length_m: float,
    z_fission_m: float | np.ndarray = 0.0,
    omega0: float,
    omega_rdw: float | np.ndarray,
    lambda0_nm: float | None = None,
    n_samples: int = 50,
) -> float | np.ndarray:
    """Absolute RDW arrival relative to the pump (fs), β₁-integrated.

    Parameters
    ----------
    beta1_fn : callable z_frac -> beta1(omega)  [s/m]
        Group-delay profile of the SAME dispersion the engine propagated
        (constant decks: any z-independent callable; gradient decks: pass
        the z-dependent profile built the same way ``build_engine`` does).
    z_fission_m : float or ndarray
        Distance at which the RDW is born (compression point); the walk-off
        leg runs from there to the exit. Default 0 (full-guide walk-off).
        Pass a per-seed array (e.g. the engine-measured compression
        distance or the Eq.-12 L_f(E)) when that is the observable.
    omega_rdw : float or ndarray
        RDW angular frequency, engine-measured (per-seed array supported).
    lambda0_nm : float
        Pump wavelength used to build omega0 if ``omega0`` not given.

    Returns
    -------
    float or ndarray
        tau in fs; positive = later than the pump.
    """
    if omega0 is None and lambda0_nm is not None:
        omega0 = omega0_from_lambda(lambda0_nm)
    w = np.atleast_1d(np.asarray(omega_rdw, dtype=float))
    zf = np.broadcast_to(np.asarray(z_fission_m, dtype=float), w.shape)
    w_all = np.concatenate([w, [omega0]])
    n_samples = max(n_samples, 8)
    tau = np.zeros_like(w)
    for k in range(w.size):
        zs = np.linspace(zf[k], length_m, n_samples + 1)
        dz = float(zs[1] - zs[0])
        z_mid = 0.5 * (zs[:-1] + zs[1:])
        for zmm in z_mid:
            b1 = np.asarray(beta1_fn(zmm / length_m)(w_all))
            tau[k] += (b1[k] - b1[-1]) * dz
    tau_fs = tau * 1e15
    scalar = np.asarray(omega_rdw).ndim == 0 and np.asarray(z_fission_m).ndim == 0
    return float(tau_fs[0]) if scalar else tau_fs
