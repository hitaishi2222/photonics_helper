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
    "complex_gaussian_noise",
    "add_noise",
    "ase_noise_field",
    "add_ase_noise",
]


def _make_rng(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass at most one of rng and seed, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def complex_gaussian_noise(
    grid: "TemporalGrid",
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
    n = (generator.standard_normal(grid.N) + 1j * generator.standard_normal(grid.N)) / np.sqrt(2.0)
    if remove_mean:
        n = n - n.mean()
    n = n / np.std(n) * float(rms)
    return np.asarray(n, dtype=complex)


def add_noise(
    wave: "Wave",
    rms_relative: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    remove_mean: bool = True,
) -> "Wave":
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
        wave.grid, float(rms_relative) * scale, rng=rng, seed=seed, remove_mean=remove_mean
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
    grid: "TemporalGrid",
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
    wave: "Wave",
    level_dB: float,
    *,
    reference_power: float | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> "Wave":
    """Return a copy of ``wave`` with an ASE background added.

    Parameters
    ----------
    wave : Wave — input field.
    level_dB : float — background level relative to the pump DC line.
    reference_power : float, optional — CW reference power; defaults to the
        peak power of ``wave``.
    rng, seed : optional generator / seed.

    The input ``wave`` is not modified.
    """
    from .pulse import Wave

    field = np.asarray(wave.envelope_field, dtype=complex)
    ref = float(reference_power) if reference_power is not None else float(wave.peak_power())
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
