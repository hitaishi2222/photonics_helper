"""Stochastic noise sources for GNLSE / NLSE simulations.

Deterministic split-step solvers cannot reproduce the noise-driven physics of
spontaneous modulation instability, supercontinuum coherence, or Raman-induced
spectral fluctuations.  This module provides reproducible (seedable) stochastic
sources that can be added to a :class:`~photonics_helper.pulse.Wave`.

Two conventions are supported:

* **Amplitude contrast** (:func:`complex_gaussian_noise` / :func:`add_noise`) —
  time-domain complex Gaussian noise with a standard deviation fixed as a
  fraction of the field amplitude.  This is the natural way to write "a CW with
  < 5 % intensity contrast".
* **Spectral level** (:func:`ase_noise_field` / :func:`add_ase_noise`) — flat
  amplified-spontaneous-emission (ASE) background a given number of dB below the
  pump, with random spectral phase.  This mirrors the noise model used in
  supercontinuum and MI experiments (e.g. Närhi et al. 2016, whose ASE
  background is ``-50 dB`` relative to the pump).

.. note::

   Fields are built with :meth:`~photonics_helper.pulse.TemporalGrid.fft` /
   :meth:`~photonics_helper.pulse.TemporalGrid.ifft`, which are a scaled pair
   (``fft(A) = raw_fft(A)·dt``, ``ifft(A_w) = raw_ifft(A_w)/dt``).  A spectrum
   constructed with arbitrary units and mapped back with ``grid.ifft`` is
   therefore scaled by ``1/dt``.  :func:`ase_noise_field` handles this
   internally; if you build a spectrum yourself, carry the ``dt`` factor (or
   construct the field in the time domain with :func:`complex_gaussian_noise`).

References
----------
- G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Sec. 5.2 (noise and MI).
- J. M. Dudley, G. Genty, S. Coen, *Rev. Mod. Phys.* **78**, 1135 (2006), Eq. 5.
- M. Närhi et al., *Nat. Commun.* **7**, 13675 (2016) (ASE seeding).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .pulse import TemporalGrid, Wave

__all__ = [
    "add_ase_noise",
    "add_noise",
    "ase_noise_field",
    "coherence_g12",
    "complex_gaussian_noise",
    "raman_noise_field",
]


def _make_rng(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass at most one of rng and seed, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def complex_gaussian_noise(
    grid: TemporalGrid,
    rms: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    remove_mean: bool = True,
) -> NDArray:
    """White complex-Gaussian noise with standard deviation ``rms``.

    The noise is generated in the **time domain** (so no FFT scaling enters) and
    is unit-variance per quadrature: ``Re(n)`` and ``Im(n)`` each have rms
    ``rms / sqrt(2)``.

    Parameters
    ----------
    grid : TemporalGrid — defines the number of samples ``N``.
    rms : float — target RMS amplitude in the same units as the envelope field
        (i.e. ``sqrt(W)`` for power-based ``Wave`` objects).
    rng, seed : optional generator / seed for reproducibility.
    remove_mean : bool — subtract the mean so no DC pump is added.

    Returns
    -------
    ndarray — complex noise samples, shape ``(grid.N,)``.
    """
    generator = _make_rng(rng, seed)
    n = (
        generator.standard_normal(grid.N) + 1j * generator.standard_normal(grid.N)
    ) / np.sqrt(2.0)
    if remove_mean:
        n = n - n.mean()
    n = n / np.std(n) * float(rms)
    return np.asarray(n, dtype=complex)


def add_noise(
    wave: Wave,
    rms_relative: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    remove_mean: bool = True,
) -> Wave:
    """Return a copy of ``wave`` with additive complex-Gaussian noise.

    The noise RMS is ``rms_relative * sqrt(peak_power)``, i.e. ``rms_relative``
    is the fractional **amplitude** contrast relative to the peak of the field.
    For a CW background of power ``P0`` and ``A = sqrt(P0)(1 + n)``, an
    intensity contrast of ~5 % corresponds to ``rms_relative ≈ 0.025``.

    The input ``wave`` is not modified.
    """
    from .pulse import Wave

    field = np.asarray(wave.envelope_field, dtype=complex)
    scale = np.sqrt(wave.peak_power()) if field.size else 0.0
    if scale == 0.0:
        raise ValueError("Cannot scale noise for a zero field.")
    noise = complex_gaussian_noise(
        wave.grid,
        float(rms_relative) * scale,
        rng=rng,
        seed=seed,
        remove_mean=remove_mean,
    )
    out = Wave(
        grid=wave.grid,
        envelope=wave.envelope,
        central_wavelength=wave.central_wavelength,
        refractive_index=wave.refractive_index,
        repetition_rate=wave.repetition_rate,
    )
    out.with_field(field + noise)
    return out


def ase_noise_field(
    grid: TemporalGrid,
    reference_power: float,
    level_dB: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> NDArray:
    """Flat amplified-spontaneous-emission background with random phase.

    Builds a spectral field whose per-bin amplitude is ``level_dB`` below the DC
    line of a continuous wave of power ``reference_power``, then returns the
    corresponding time-domain field using the grid's (scaled) inverse FFT.  The
    DC bin is removed so the returned field carries no net pump.

    Parameters
    ----------
    grid : TemporalGrid
    reference_power : float — CW power in W whose DC line sets the reference.
    level_dB : float — noise level relative to that line (e.g. ``-50``).
    rng, seed : optional generator / seed.

    Returns
    -------
    ndarray — complex time-domain noise field, shape ``(grid.N,)``.
    """
    generator = _make_rng(rng, seed)
    pump = np.full(grid.N, np.sqrt(float(reference_power)), dtype=complex)
    pump_spec = np.asarray(grid.fft(pump), dtype=complex)
    dc = abs(pump_spec[grid.N // 2])
    if dc == 0.0:
        raise ValueError("Reference spectrum has a zero DC component.")
    amplitude = dc * 10.0 ** (float(level_dB) / 20.0)
    phase = generator.uniform(0.0, 2.0 * np.pi, size=grid.N)
    spec = amplitude * np.exp(1j * phase)
    spec[grid.N // 2] = 0.0
    return np.asarray(grid.ifft(spec), dtype=complex)


def add_ase_noise(
    wave: Wave,
    level_dB: float,
    *,
    reference_power: float | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> Wave:
    """Return a copy of ``wave`` with an ASE background added.

    Parameters
    ----------
    wave : Wave — input field.
    level_dB : float — background level relative to the pump DC line.
    reference_power : float, optional — CW reference power; defaults to the
        peak power of ``wave``.
    rng, seed : optional generator / seed.

    Notes
    -----
    The input ``wave`` is not modified.
    """
    from .pulse import Wave

    field = np.asarray(wave.envelope_field, dtype=complex)
    ref = (
        float(reference_power)
        if reference_power is not None
        else float(wave.peak_power())
    )
    noise = ase_noise_field(wave.grid, ref, level_dB, rng=rng, seed=seed)
    out = Wave(
        grid=wave.grid,
        envelope=wave.envelope,
        central_wavelength=wave.central_wavelength,
        refractive_index=wave.refractive_index,
        repetition_rate=wave.repetition_rate,
    )
    out.with_field(field + noise)
    return out


def raman_noise_field(
    grid: TemporalGrid,
    h_R_fft: NDArray,
    seed: int | None = None,
    *,
    omega0: float,
    temperature: float = 300.0,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Spontaneous-Raman noise field for one GNLSE step (Dudley Eq. 5 ``Γ_R``).

    Semi-classical one-photon-per-mode seed: each frequency bin gets a complex
    Gaussian with variance ``σ² = (ℏω₀/2)·Im[h̃_R]·(n_th+1)·(2π/Δω)``, shaped by the
    (positive) Raman gain ``Im[h̃_R]`` — bins with ``Im[h̃_R] ≤ 0``
    (anti-Stokes side) carry no spontaneous Stokes seed. ``n_th`` is the
    thermal phonon occupation ``1/(exp(ℏΩ_R/k_B T) − 1)`` (Boer, *Quantum
    Optics* / Shen, *The Principles of Nonlinear Optics*, Ch. 7);
    ``n_th + 1 = 1/(1 − exp(−ℏΩ_R/k_B T))`` is the spontaneous + stimulated
    emission factor of the quantum Langevin treatment (Drummond & Hardman,
    *Eur. Phys. J. D* **21**, 49 (2003)). At optical frequencies and room
    temperature ``n_th ≪ 1`` for all Raman shifts of interest, so the
    semi-classical limit ``n_th + 1 → 1`` is accurate; the factor is kept
    explicit (parameter ``temperature``) so mid-IR or cryogenic studies can
    resolve it.  The ``2π/Δω``
    factor is the grid's FFT-pair normalization (``fft(A) = raw_fft(A)·dt``,
    ``ifft(A_w) = raw_ifft(A_w)/dt``): it makes the time-domain noise
    variance ``ℏω₀/(2·dt)·Im[h̃_R]``, i.e. PSD ``ℏω₀/2`` per unit angular
    frequency.  The caller scales the returned field by ``sqrt(dz)``
    (Langevin ``δ(z−z′)`` discretization) and adds it to the envelope once
    per step.

    Parameters
    ----------
    grid : TemporalGrid — defines ``N``, ``dt`` and the ``fft``/``ifft`` pair.
    h_R_fft : array — FFT of the Raman response on the grid (same ordering as
        ``grid.fft`` output), shape ``(grid.N,)``.
    seed : int, optional — reproducibility seed (mutually exclusive with rng).
    omega0 : float — carrier angular frequency in rad/s (keyword-only; sets
        the photon energy ``ℏω₀``).
    temperature : float, optional — lattice/medium temperature in kelvin
        (keyword-only; default 300). Sets the thermal factor ``n_th + 1``.
    rng : Generator, optional — externally threaded generator for per-step
        draws (mutually exclusive with seed).

    Returns
    -------
    ndarray — complex time-domain noise field, shape ``(grid.N,)``.
    """
    from scipy.constants import hbar
    from scipy.constants import k as k_B

    generator = _make_rng(rng, seed)
    h_R = np.asarray(h_R_fft, dtype=complex)
    if h_R.shape != (grid.N,):
        raise ValueError(f"h_R_fft must have shape ({grid.N},), got {h_R.shape}")
    if not omega0 > 0.0:
        raise ValueError(f"omega0 must be positive, got {omega0!r}")
    if not temperature > 0.0:
        raise ValueError(f"temperature must be positive (kelvin), got {temperature!r}")
    dw = float(grid.dw)
    gain = np.clip(h_R.imag, 0.0, None)
    # Thermal factor n_th + 1 = 1/(1 − exp(−ℏΩ_R/k_B T)), evaluated per bin
    # at the Raman shift |Ω| carried by Im[h̃_R].  For bins with zero gain the
    # factor is irrelevant; guard the exponent against overflow.
    omega_shift = np.abs(grid.w)
    expo = np.clip(hbar * omega_shift / (k_B * temperature), 0.0, 700.0)
    with np.errstate(divide="ignore"):
        thermal = np.where(expo > 0.0, 1.0 / (1.0 - np.exp(-expo)), 1.0)
    thermal = np.where(gain > 0.0, thermal, 1.0)
    gauss = (
        generator.standard_normal(grid.N) + 1j * generator.standard_normal(grid.N)
    ) / np.sqrt(2.0)
    spec = (
        np.sqrt(
            hbar * float(omega0) / 2.0 * gain * thermal * (2.0 * np.pi / dw)
        )
        * gauss
    )
    return np.asarray(grid.ifft(spec), dtype=complex)


def coherence_g12(runs: NDArray) -> NDArray:
    """First-order ensemble coherence ``g₁₂`` over stochastic realizations.

    implements the Dudley coherence formalism (Dudley, Genty & Coen, *Rev.
    Mod. Phys.* **78**, 1135 (2006), Eq. (23)): for each frequency bin,

    ``g₁₂ = |⟨E_m* · E_n⟩_{m≠n}| / ⟨|E|²⟩``,

    where the pair average is taken over all distinct realization pairs and
    the **modulus of the complex pair average** is used (not its real part),
    matching the convention of Dudley Fig. 19.  Identical runs give
    ``g₁₂ = 1``; partially decohered ensembles give ``g₁₂ < 1``.

    Parameters
    ----------
    runs : array — complex spectra, shape ``(n_runs, n_bins)`` with
        ``n_runs >= 2``.

    Returns
    -------
    ndarray — ``g₁₂`` per bin, shape ``(n_bins,)``, values in ``[0, 1]``.
    """
    arr = np.asarray(runs, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"runs must have shape (n_runs>=2, n_bins), got {arr.shape}")
    n = arr.shape[0]
    # Accumulate the complex off-diagonal pair sum directly:
    #   Σ_{m≠n} conj(E_m)·E_n  per frequency bin
    # (the |ΣE|²−Σ|E|² shortcut discards the pair-average phase, so it can
    # only give Re⟨conj(E_m)E_n⟩, not |⟨…⟩|; n_runs is small so O(n²) is fine).
    cross = np.zeros(arr.shape[1], dtype=complex)
    total = arr.sum(axis=0)
    for m in range(n):
        cross += np.conj(arr[m]) * (total - arr[m])
    mean_cross = cross / (n * (n - 1))
    mean_auto = np.mean(np.abs(arr) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g12 = np.where(mean_auto > 0.0, np.abs(mean_cross) / mean_auto, 0.0)
    return np.clip(np.asarray(g12, dtype=float), 0.0, None)
