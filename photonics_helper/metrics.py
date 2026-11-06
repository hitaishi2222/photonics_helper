"""Spectral span metrics on a stated numeraire.

Every group that quotes "N octaves at −M dB" writes the number slightly
differently: which density (per Hz or per nm), which level reference, and how the
threshold crossing is interpolated. The metric is only meaningful when all three
travel with it, so this module makes them arguments and returns them in the
result.

The one number this exists for: a supercontinuum's span is a property of
``S_λ = S_ν·|dω/dλ|`` when the axis is wavelength. Skipping the Jacobian moves a
−50 dB crossing by up to ``10·log10((9/2)²) ≈ 13 dB`` between 2 µm and 9 µm,
i.e. it changes the octave count.

Example
-------
>>> from photonics_helper.metrics import span_at_level
>>> span = span_at_level(wl_nm, S_nu, level_db=-50.0, numeraire="frequency")
>>> f"{span.octaves:.2f} octaves over {span.short_nm:.0f}-{span.long_nm:.0f} nm"

Reference: Dudley, Genty & Coen, Rev. Mod. Phys. 78, 1135 (2006), §IV
(spectral densities and the level that must travel with a bandwidth claim).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .core.constants import C_MS

__all__ = ["Span", "span_at_level", "to_wavelength_density"]


@dataclass(frozen=True)
class Span:
    """A measured spectral span, with everything needed to quote it.

    Attributes
    ----------
    short_nm, long_nm : float
        The two ``level_db`` crossings, in nm. ``nan`` when the spectrum does not
        reach the level on that side inside the supplied axis.
    octaves : float
        ``log2(long_nm / short_nm)``; ``nan`` unless both crossings exist.
    level_db : float
        The level the crossings were taken at, relative to the peak.
    numeraire : str
        ``"wavelength"`` or ``"frequency"`` — which density the input was.
    interpolation : str
        How the crossing was placed (``"linear-in-db"``).
    """

    short_nm: float
    long_nm: float
    octaves: float
    level_db: float
    numeraire: str
    interpolation: str = "linear-in-db"


def to_wavelength_density(
    wavelength_nm: NDArray, spectrum: NDArray
) -> NDArray:
    """``S_ν → S_λ``: multiply by ``|dω/dλ| = 2πc/λ²`` (λ in metres)."""
    wl_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    return np.asarray(spectrum, dtype=float) * (2.0 * np.pi * C_MS / wl_m**2)


def _crossing(wl: NDArray, db: NDArray, i_in: int, i_out: int, level: float) -> float:
    """Linear interpolation of the ``level`` crossing between two samples."""
    x0, x1 = wl[i_in], wl[i_out]
    y0, y1 = db[i_in], db[i_out]
    if y0 == y1:  # pragma: no cover - degenerate flat pair
        return float(x0)
    t = (level - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def span_at_level(
    wavelength_nm: NDArray,
    spectrum: NDArray,
    level_db: float = -50.0,
    *,
    numeraire: str = "wavelength",
    reference: str = "peak",
) -> Span:
    """Span of a spectrum at ``level_db``, in octaves, on a stated numeraire.

    Parameters
    ----------
    wavelength_nm : 1-D array — wavelength axis in nm, strictly increasing.
    spectrum : 1-D array — spectral density of the same length. Its scale is
        irrelevant: it is normalised to its own peak.
    level_db : float — the level, relative to the reference (negative, e.g. −50).
    numeraire : {"wavelength", "frequency"} — what ``spectrum`` is a density
        *of*. ``"frequency"`` applies the Jacobian, so the answer is the one a
        wavelength axis should show.
    reference : {"peak"} — what the level is measured from. Only ``"peak"`` is
        implemented; a "total energy" reference is a different (and rarely
        stated) convention that should be added with its own test.

    Returns
    -------
    Span
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    if wl.ndim != 1 or wl.size < 8:
        raise ValueError(f"need a 1-D axis with >= 8 points, got shape {wl.shape}")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if reference != "peak":
        raise ValueError(f"only reference='peak' is implemented, got {reference!r}")

    if numeraire == "frequency":
        density = to_wavelength_density(wl, spectrum)
    elif numeraire == "wavelength":
        density = np.asarray(spectrum, dtype=float)
    else:
        raise ValueError(
            f"numeraire must be 'wavelength' or 'frequency', got {numeraire!r}"
        )

    peak = float(np.max(density))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("spectrum has no positive finite peak")
    db = 10.0 * np.log10(np.clip(density / peak, 1e-300, None))
    i_peak = int(np.argmax(db))

    short = np.nan
    long = np.nan
    # Walk out from the peak to the first sub-level sample on each side.
    left = np.nonzero(db[: i_peak + 1] < level_db)[0]
    if left.size:
        i = int(left[-1])
        short = _crossing(wl, db, i, i + 1, level_db)
    right = np.nonzero(db[i_peak:] < level_db)[0]
    if right.size:
        i = int(right[0]) + i_peak
        long = _crossing(wl, db, i - 1, i, level_db)

    octaves = (
        float(np.log2(long / short))
        if np.isfinite(short) and np.isfinite(long) and short > 0 and long > short
        else float("nan")
    )
    return Span(
        short_nm=float(short),
        long_nm=float(long),
        octaves=octaves,
        level_db=float(level_db),
        numeraire=numeraire,
    )
