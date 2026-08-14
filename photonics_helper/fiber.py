"""Fiber dispersion and propagation constant calculations."""

from scipy.interpolate import BSpline, RegularGridInterpolator
from photonics_helper.base import (
    C_MS,
    PI,
    AngularFrequencyArray,
    Wavelength,
    WavelengthArray,
)

from functools import cached_property
from numpy.typing import NDArray
from typing import Literal, Self
from pydantic.dataclasses import dataclass
from pydantic import model_validator

import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.interpolate import make_splrep
from rich.traceback import install

install()


@dataclass(config={"arbitrary_types_allowed": True})
class Dispersion:
    """Wavelength-dependent dispersion D(λ) with spline interpolation.

    Values are stored internally in s/m² and can be queried in ps/(nm·km).

    Attributes
    ----------
    wavelengths : wavelength array.
    values : dispersion values in s/m².
    central_wavelength : design central wavelength.
    """

    values: NDArray
    unit: Literal["ps/nm.km", "s/m^2"]
    wavelengths: WavelengthArray
    central_wavelength: Wavelength

    @model_validator(mode="after")
    def _normalize_units(self) -> "Dispersion":
        """Store values canonically in s/m².

        ps/(nm·km) → s/m² requires x1e-6 (1 ps/(nm·km) = 1e-6 s/m²).
        All internal accessors and get_betas()/get_beta2() consume the SI form.
        """
        if self.unit == "ps/nm.km":
            self.values = np.asarray(self.values) * 1e-6
            self.unit = "s/m^2"
        return self

    def __repr__(self):
        return f"Dispersion: from wl: {self.wavelengths.as_m.min()} to {self.wavelengths.as_m.max()}"

    def __str__(self):
        return f"Dispersion D(λ) in ps/(nm·km), range {self.wavelengths.as_nm.min():.0f}–{self.wavelengths.as_nm.max():.0f} nm"

    @cached_property
    def as_ps_nm_km(self) -> NDArray:
        """Dispersion values in ps/(nm·km)."""
        return self.values * 1e6

    @cached_property
    def as_s_m_m(self) -> NDArray:
        """Dispersion values in s/m² (internal units)."""
        return self.values

    def get_wls(self) -> WavelengthArray:
        """Return the wavelength array."""
        return self.wavelengths

    @cached_property
    def _disp_fn(self) -> BSpline:
        """Return a cubic spline interpolant for dispersion vs wavelength."""
        try:
            return make_splrep(self.wavelengths.as_m, self.as_s_m_m)
        except Exception as e:
            raise Exception(f"{e}")

    def _check_wl(self, wavelength: Wavelength):
        """Raise ValueError if wavelength is outside the interpolation range."""
        if (
            wavelength.as_m < self.wavelengths.as_m.min()
            or wavelength.as_m > self.wavelengths.as_m.max()
        ):
            raise ValueError(
                f"values of dispersion available between "
                f"{self.wavelengths.as_m.min():.3e} and {self.wavelengths.as_m.max():.3e} m"
            )

    def fn(self, wavelength: Wavelength) -> float:
        """Interpolated dispersion at wavelength (m)"""
        self._check_wl(wavelength)
        return self._disp_fn(wavelength.as_m).item()

    @classmethod
    def from_neff(
        cls,
        neff: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength: Wavelength,
        ignore_fit_error: bool = False,
    ) -> Self:
        """Compute dispersion D(λ) from effective index neff(λ).

        D = -λ/c × d²neff/dλ²
        """

        if not isinstance(wavelengths, WavelengthArray):
            raise TypeError("wavelengths should be an instance of `WavelengthArray`")
        # D = -lambda / C_MS * (d^2 neff/ d lambda^2)

        if len(neff) != len(wavelengths.value):  # type: ignore[arg-type]
            raise ValueError("Length of both neff and wavelengths should be same")
        wl = wavelengths.to_equally_spaced()
        interp = make_splrep(wavelengths.as_m, neff)(wl)

        spline = make_splrep(wl, interp)
        diff_2 = spline.derivative(2)

        dispersion: NDArray = -wavelengths.as_m / C_MS * diff_2(wavelengths.as_m)

        smooth_fit = np.all(np.diff(dispersion * 1e6) < 50)
        if not smooth_fit:
            warnings.warn(
                "Bad fitting of neff values. Consider building Disperison in other ways..."
            )
            if not ignore_fit_error:
                raise ChildProcessError(
                    "Can't perform numerical differentiation with small error..."
                )
        return cls(
            wavelengths=wavelengths,
            values=dispersion,
            unit="s/m^2",
            central_wavelength=central_wavelength,
        )

    @classmethod
    def from_propagation_constant(
        cls,
        beta: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength: Wavelength,
        ignore_fit_error: bool = False,
    ) -> Self:
        """Compute dispersion D(λ) from propagation constant β(ω).

        D = -(2πc)/λ² × d²β/dω²
        """
        # -(2*PI*C_MS) / lambda^2 * (d^2 beta/ d omega^2)

        if len(beta) != len(wavelengths.value):  # type: ignore[arg-type]
            raise ValueError("Length of both beta and wavelengths should be same")
        if not isinstance(wavelengths, WavelengthArray):
            raise TypeError(
                f"wavelengths cannot process the type: {type(wavelengths)}, required WavelengthArray"
            )

        omega = wavelengths.to_omega().to_equally_spaced()
        interp = make_splrep(wavelengths.to_omega().as_rad_s[::-1], beta[::-1])(omega)

        spline = make_splrep(omega[::-1], interp[::-1])
        diff_2 = spline.derivative(2)

        dispersion: NDArray = (
            -(2 * PI * C_MS)
            / wavelengths.as_m**2
            * diff_2(wavelengths.to_omega().as_rad_s)
        )

        smooth_fit = np.all(np.diff(dispersion * 1e6) < 50)
        if not smooth_fit:
            warnings.warn(
                "Bad fitting of neff values. Consider building Disperison in other ways..."
            )
            if not ignore_fit_error:
                raise ChildProcessError(
                    "Can't perform numerical differentiation with small error..."
                )
        return cls(
            wavelengths=wavelengths,
            values=dispersion,
            unit="s/m^2",
            central_wavelength=central_wavelength,
        )

    def get_beta2(self, wavelength_nm: float):
        """Return β₂ at wavelength_nm (s²/m) via spline interpolation."""
        _min_wl = self.wavelengths.as_nm.min()
        _max_wl = self.wavelengths.as_nm.max()
        if wavelength_nm > _max_wl or wavelength_nm < _min_wl:
            raise ValueError(
                f"values of disersion available between {_min_wl} and {_max_wl} nm."
            )
        beta2 = -self.wavelengths.as_m**2 / (2 * PI * C_MS) * self.as_s_m_m
        spline = make_splrep(self.wavelengths.as_nm, beta2)
        return float(spline(wavelength_nm))

    def get_betas(
        self,
        polyOrder,
        wavelength: Wavelength | None = None,
        make_plot=False,
        return_diagnostics=False,
    ):
        """Fit dispersion data to a polynomial in ω and return beta coefficients.

        Returns betas in array [beta2, beta3, ...].
        If return_diagnostics is True, returns (betas, fit_x_axis, data, fit).
        """

        if wavelength is None:
            wavelength = self.central_wavelength

        min_nm = self.get_wls().as_nm
        maximum = max(min_nm)
        minimum = min(min_nm)
        if wavelength.as_nm < minimum or wavelength.as_nm > maximum:
            raise ValueError(
                f"wavelength {wavelength.as_nm:.3f} nm is outside the dispersion range "
                f"[{minimum:.3f}, {maximum:.3f}] nm"
            )

        # omega - omega_0
        omegaAxis = 2 * np.pi * C_MS / (self.get_wls().as_m) - 2 * np.pi * C_MS / (
            wavelength.as_m
        )

        # Convert from D to beta via beta2 = -D * lambda^2 / (2*pi*c)
        betaTwo = -self.as_s_m_m * (self.get_wls().as_m) ** 2 / (2 * np.pi * C_MS)
        # The units of beta2 for the GNLSE solver are ps^2/m; convert
        betaTwo = betaTwo * 1e24
        # Also convert angular frequency to rad/ps
        omegaAxis = omegaAxis * 1e-12  #  s/ps

        # Fit beta2 with high-order polynomial
        polyFitCo = np.polyfit(omegaAxis, betaTwo, polyOrder)

        Betas = polyFitCo[::-1]

        polyFit = np.zeros((len(omegaAxis),))

        for i in range(len(Betas)):
            Betas[i] = Betas[i] * factorial(i)
            polyFit = polyFit + Betas[i] / factorial(i) * omegaAxis**i

        if make_plot:
            plt.plot(omegaAxis, betaTwo, "o")
            plt.plot(omegaAxis, polyFit)
            plt.show()
        if return_diagnostics:
            return Betas, omegaAxis, betaTwo, polyFit
        else:
            return Betas


@dataclass(config={"arbitrary_types_allowed": True})
class PropagationConstant:
    """Propagation constant β as a function of wavelength or angular frequency.

    Attributes
    ----------
    values : β values.
    x_values : wavelength or angular frequency array.
    """

    values: NDArray
    x_values: WavelengthArray | AngularFrequencyArray

    @model_validator(mode="after")
    def _setup(self) -> "PropagationConstant":
        if isinstance(self.x_values, WavelengthArray):
            self.wavelengths = self.x_values
            self.omegas = self.x_values.to_omega()
        elif isinstance(self.x_values, AngularFrequencyArray):
            self.omegas = self.x_values
            self.wavelengths = self.x_values.to_wl()
        return self

    @classmethod
    def beta_from_neff(
        cls, neff: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ):
        """Compute β = neff × ω / c from effective index."""
        if not isinstance(x_values, (WavelengthArray, AngularFrequencyArray)):
            raise TypeError(
                "x_values should be a type of either `WavelengthArray` or `AngularFrequencyArray`"
            )
        if len(neff) != len(x_values.value):  # type: ignore[arg-type]
            raise ValueError("both neff and x_values must be of same length.")

        if isinstance(x_values, WavelengthArray):
            omegas = x_values.to_omega()
        elif isinstance(x_values, AngularFrequencyArray):
            omegas = x_values

        beta2 = neff * omegas.as_rad_s / C_MS
        return beta2

    @classmethod
    def from_neff_omega(cls, neff: NDArray, omega: AngularFrequencyArray) -> Self:
        """Construct from effective index neff and angular frequency array."""
        if not isinstance(omega, AngularFrequencyArray):
            raise TypeError(
                f"omega should be a type of 'AngularFrequencyArray' : got {type(omega)}"
            )
        if len(neff) != len(omega.value):  # type: ignore[arg-type]
            raise ValueError(
                "both neff and angular frequency array must have same length"
            )

        betas = omega.as_rad_s * neff / C_MS
        return cls(values=betas, x_values=omega)


@dataclass(config={"arbitrary_types_allowed": True})
class ZDependentDispersion:
    """Z-dependent dispersion profile β(ω, z) for tapered/dispersion-managed waveguides.

    Holds a 2-D table of propagation constants β[ω_idx, z_idx] with axis arrays,
    plus a `fn(omega, z)` interpolant built via `RegularGridInterpolator`.

    Attributes
    ----------
    omegas : 1-D array of angular frequencies (rad/s).
    z_positions : 1-D array of propagation positions (m).
    beta : 2-D array of shape (n_omega, n_z) with β values.
    central_wavelength : design central wavelength (m).

    Notes
    -----
    The interpolant uses linear interpolation with `bounds_error=False` and
    `fill_value=None` so out-of-range queries return NaN (caller decides what to do).
    """

    omegas: NDArray
    z_positions: NDArray
    beta: NDArray
    central_wavelength: float

    def __post_init__(self) -> None:
        """Validate shapes and build the interpolant."""
        self.omegas = np.asarray(self.omegas, dtype=float)
        self.z_positions = np.asarray(self.z_positions, dtype=float)
        self.beta = np.asarray(self.beta, dtype=float)
        if self.beta.ndim != 2:
            raise ValueError(f"beta must be 2-D, got shape {self.beta.shape}")
        if self.beta.shape[0] != len(self.omegas):
            raise ValueError(
                f"beta.shape[0] ({self.beta.shape[0]}) must match len(omegas) ({len(self.omegas)})"
            )
        if self.beta.shape[1] != len(self.z_positions):
            raise ValueError(
                f"beta.shape[1] ({self.beta.shape[1]}) must match len(z_positions) ({len(self.z_positions)})"
            )
        # RegularGridInterpolator requires strictly ascending axes.
        # Always sort to ascending order, reordering beta accordingly.
        omega_idx = np.argsort(self.omegas)
        self.omegas = self.omegas[omega_idx]
        self.beta = self.beta[omega_idx, :]
        z_idx = np.argsort(self.z_positions)
        self.z_positions = self.z_positions[z_idx]
        self.beta = self.beta[:, z_idx]
        self._interpolator = RegularGridInterpolator(
            (self.omegas, self.z_positions),
            self.beta,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def __repr__(self) -> str:
        return (
            f"ZDependentDispersion: ω from {self.omegas[0]:.3e} to {self.omegas[-1]:.3e} rad/s, "
            f"z from {self.z_positions[0]:.3e} to {self.z_positions[-1]:.3e} m"
        )

    def fn(self, omega: float | NDArray, z: float) -> float | NDArray:
        """Interpolate β(ω, z) via the 2-D table.

        Parameters
        ----------
        omega : float or 1-D array — absolute angular frequency(ies) in rad/s.
        z : float — propagation position in m.

        Returns
        -------
        β interpolated at (omega, z). Scalar if omega is scalar, array otherwise.
        Out-of-range values return NaN (not an error).
        """
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        result = self._interpolator(np.column_stack([omega_arr, np.full_like(omega_arr, z)]))
        if np.isscalar(omega):
            return float(result[0])
        return result

    @property
    def n_omega(self) -> int:
        return len(self.omegas)

    @property
    def n_z(self) -> int:
        return len(self.z_positions)

    def _default_omega0(self) -> float:
        """Carrier angular frequency (rad/s) from central_wavelength."""
        return 2 * np.pi * C_MS / self.central_wavelength

    def get_betas_at_z(
        self,
        z: float,
        order: int = 7,
        omega0: float | None = None,
        halfwidth: float | None = None,
    ) -> NDArray:
        """Fit Taylor coefficients β₂…β_order near omega0 at position z.

        Returns [β₂, β₃, …, β_order] in SI units (s^k/m). ω is in rad/s.
        """
        if order < 2:
            raise ValueError(f"order must be >= 2, got {order}")
        omega0 = float(omega0 if omega0 is not None else self._default_omega0())
        omega_min, omega_max = float(self.omegas[0]), float(self.omegas[-1])
        if halfwidth is None:
            halfwidth = min(0.3e15, (omega_max - omega_min) / 2)
        omega_low = max(omega_min, omega0 - halfwidth)
        omega_high = min(omega_max, omega0 + halfwidth)

        mask = (self.omegas >= omega_low) & (self.omegas <= omega_high)
        omega_fit = self.omegas[mask]
        beta_at_z = self.fn(omega_fit, z)
        valid = ~np.isnan(beta_at_z)
        if valid.sum() < order + 1:
            return np.full(order - 1, np.nan)

        omega_offset = omega_fit[valid] - omega0
        beta_valid = np.asarray(beta_at_z[valid], dtype=float)
        coeffs = np.polyfit(omega_offset, beta_valid, order)
        return np.array(
            [factorial(k) * coeffs[order - k] for k in range(2, order + 1)],
            dtype=float,
        )

    def get_betas_vs_z(
        self,
        order: int = 7,
        omega0: float | None = None,
        halfwidth: float | None = None,
    ) -> NDArray:
        """Taylor coefficients β₂…β_order at every z position.

        Returns array of shape (n_z, order - 1) in SI units (s^k/m).
        """
        return np.array(
            [
                self.get_betas_at_z(z, order=order, omega0=omega0, halfwidth=halfwidth)
                for z in self.z_positions
            ],
            dtype=float,
        )

    @classmethod
    def from_arrays(
        cls,
        omegas: NDArray,
        z_positions: NDArray,
        beta: NDArray,
        central_wavelength: float,
    ) -> "ZDependentDispersion":
        """Construct from raw arrays.

        Parameters
        ----------
        omegas : 1-D array — angular frequencies (rad/s).
        z_positions : 1-D array — propagation positions (m).
        beta : 2-D array — shape (n_omega, n_z).
        central_wavelength : float — design central wavelength (m).
        """
        return cls(
            omegas=np.asarray(omegas, dtype=float),
            z_positions=np.asarray(z_positions, dtype=float),
            beta=np.asarray(beta, dtype=float),
            central_wavelength=float(central_wavelength),
        )

    @classmethod
    def from_npz(cls, path: str, central_wavelength: float | None = None) -> "ZDependentDispersion":
        """Load β(ω, z) from an NPZ file.

        Expected keys:
            - "omegas": 1-D array of angular frequencies (rad/s).
            - "z_positions": 1-D array of propagation positions (m).
            - "beta": 2-D array of shape (n_omega, n_z).
            - "central_wavelength" (optional): if present, used; otherwise `central_wavelength` arg required.

        Parameters
        ----------
        path : str — path to the .npz file.
        central_wavelength : float — required if not stored in the NPZ file.
        """
        data = dict(np.load(path))
        if central_wavelength is None:
            if "central_wavelength" in data:
                central_wavelength = float(data["central_wavelength"])
            else:
                raise ValueError(
                    f"central_wavelength not found in NPZ keys or as argument. "
                    f"Available keys: {list(data.keys())}"
                )
        obj = cls.from_arrays(
            omegas=data["omegas"],
            z_positions=data["z_positions"],
            beta=data["beta"],
            central_wavelength=central_wavelength,
        )
        if "betas_taylor" in data:
            obj.betas_taylor = np.asarray(data["betas_taylor"], dtype=float)
        if "beta_orders" in data:
            obj.beta_orders = np.asarray(data["beta_orders"], dtype=int)
        if "omega0" in data:
            obj.omega0_fit = float(data["omega0"])
        return obj
