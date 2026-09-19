"""Build on the core: group-velocity dispersion from a material you define.

A worked example of using **only** ``photonics_helper.core`` to build something
useful — here, a calculator for the dispersive broadening of a Gaussian pulse in
a glass — and validating it against the closed-form result.

Everything comes from the stable foundation primitives:

- ``core.units``      — ``Wavelength``, ``Time`` and their conversions
- ``core.constants``  — ``C_MS``
- ``core.grids``      — ``TemporalGrid`` (time/frequency sampling + FFT)
- ``core.materials``  — the ``OpticalMaterial`` protocol (implemented below)

No solver, plotting or database module is imported: the material is defined by
implementing the small ``OpticalMaterial`` interface, which is the intended
extension point for downstream code.

The physics
-----------

For a field envelope ``A(0, t)`` the linear propagation step is a spectral phase

    ``A(z, w) = A(0, w) * exp(i * beta2 * w**2 * z / 2)``

with ``beta2 = d^2 beta / d omega^2`` at the carrier. A Gaussian pulse of field
half-width ``T0`` broadens by

    ``T(z) / T0 = sqrt(1 + (z / L_D)**2)``,   ``L_D = T0**2 / |beta2|``.

We compute ``beta2`` numerically from the material's ``n(lambda)`` and check the
simulated broadening against that formula.

Run it with ``python examples/33_build_on_core.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from photonics_helper.core import constants, grids, units
from photonics_helper.core.materials import OpticalMaterial


@dataclass
class CauchyGlass:
    """A minimal material implementing the ``OpticalMaterial`` protocol.

    ``n(lambda) = A + B / lambda**2`` (two-term Cauchy, lambda in um). The
    defaults approximate fused silica near 1.5 um.
    """

    name: str = "Cauchy glass"
    A: float = 1.4437
    B_um2: float = 0.00354
    source: str | None = "two-term Cauchy fit (illustrative)"
    license: str | None = None

    def n_func(self, wavelength_um: float) -> float:
        """Refractive index at ``wavelength_um`` (um)."""
        return self.A + self.B_um2 / wavelength_um**2

    def k_func(self, wavelength_um: float) -> float:
        """Extinction coefficient (lossless)."""
        return 0.0

    @property
    def wl(self) -> units.WavelengthArray:
        """Sampled wavelengths over which the arrays are tabulated."""
        return units.WavelengthArray(np.linspace(1.0, 2.0, 11), "um")

    @property
    def n(self) -> NDArray:
        """Tabulated refractive index."""
        return np.array([self.n_func(w) for w in self.wl.as_um])

    @property
    def k(self) -> NDArray:
        """Tabulated extinction coefficient."""
        return np.zeros_like(self.n)


def group_velocity_dispersion(
    material: OpticalMaterial, wavelength: units.Wavelength
) -> float:
    """Return ``beta2 = d^2 beta / d omega^2`` (s^2/m) at ``wavelength``.

    Uses a central finite difference of ``beta(omega) = n(omega) * omega / c``.
    """
    omega0 = wavelength.to_freq().to_omega().as_rad_s

    def beta(omega: float) -> float:
        # omega -> lambda (um) -> n, then beta = n * omega / c
        wavelength_um = (2.0 * math.pi * constants.C_MS / omega) * 1e6
        return material.n_func(wavelength_um) * omega / constants.C_MS

    step = omega0 * 1e-3
    return (beta(omega0 + step) - 2.0 * beta(omega0) + beta(omega0 - step)) / step**2


def _gaussian_pulse(grid: grids.TemporalGrid, t0: float) -> NDArray:
    return np.exp(-grid.t**2 / (2.0 * t0**2)).astype(complex)


def _rms_width(field: NDArray, t: NDArray) -> float:
    intensity = np.abs(field) ** 2
    moment0 = float(np.sum(intensity))
    mean = float(np.sum(t * intensity) / moment0)
    variance = float(np.sum((t - mean) ** 2 * intensity) / moment0)
    return math.sqrt(variance)


def propagate(
    grid: grids.TemporalGrid, field: NDArray, beta2: float, length_m: float
) -> NDArray:
    """Apply the linear dispersive step over ``length_m`` metres."""
    spectral = grid.fft(field)
    phase = np.exp(1j * beta2 * grid.w**2 * length_m / 2.0)
    return grid.ifft(spectral * phase)


def run() -> dict[str, object]:
    """Compute the broadening table; returns the results for testing."""
    glass = CauchyGlass()
    carrier = units.Wavelength(1550, "nm")
    beta2 = group_velocity_dispersion(glass, carrier)

    t0 = units.Time(50, "fs").as_s
    dispersion_length = t0**2 / abs(beta2)

    grid = grids.TemporalGrid(N=1 << 14, Tmax=units.Time(20, "ps"))
    initial = _gaussian_pulse(grid, t0)
    initial_width = _rms_width(initial, grid.t)

    rows: list[tuple[float, float, float, float]] = []
    for ratio in (0.5, 1.0, 2.0, 4.0):
        length_m = ratio * dispersion_length
        final = propagate(grid, initial, beta2, length_m)
        broadening = _rms_width(final, grid.t) / initial_width
        analytic = math.sqrt(1.0 + ratio**2)
        rows.append((ratio, broadening, analytic, abs(broadening - analytic)))

    return {
        "beta2_s2_per_m": beta2,
        "dispersion_length_m": dispersion_length,
        "rows": rows,
        "max_error": max(row[3] for row in rows),
        "is_optical_material": isinstance(glass, OpticalMaterial),
    }


def main() -> int:
    results = run()
    print("Build-on-core example: dispersive broadening of a Gaussian pulse")
    print("  material          : CauchyGlass (Cauchy two-term)")
    print(f"  beta2             : {results['beta2_s2_per_m'] * 1e27:.2f} ps^2/km")
    print(f"  L_D               : {results['dispersion_length_m'] * 1e3:.3f} mm")
    print(f"  OpticalMaterial   : {results['is_optical_material']}")
    print()
    print("  z/L_D    numeric    analytic   |error|")
    for ratio, numeric, analytic, error in results["rows"]:  # type: ignore[union-attr]
        print(f"  {ratio:4.1f}    {numeric:8.6f}   {analytic:8.6f}   {error:.2e}")
    print()
    print(f"  max error vs sqrt(1+(z/L_D)^2): {results['max_error']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
