from photonics_helper.base import AngularFrequencyArray, Wavelength, WavelengthArray

from functools import cached_property
from numpy.typing import NDArray
from typing import Literal, Self, Tuple
from scipy.interpolate import BSpline, RegularGridInterpolator
from pydantic import model_validator
from pydantic.dataclasses import dataclass

@dataclass(config={"arbitrary_types_allowed": True})
class Dispersion:
    """
    Represents optical fiber dispersion characteristics.

    This class handles chromatic dispersion data for optical fibers, providing
    methods to interpolate dispersion values at specific wavelengths and convert
    between different units.

    Attributes:
        wavelengths: wavelength array.
        values: dispersion values in s/m^2.
        unit: internal unit representation (always "s/m^2").
        central_wavelength: design central wavelength.
    """

    wavelengths: WavelengthArray
    values: NDArray
    unit: Literal["ps/nm.km", "s/m^2"]
    central_wavelength: Wavelength

    def __repr__(self) -> str:
        """Return string representation showing wavelength range."""
        ...

    @cached_property
    def as_ps_nm_km(self) -> NDArray:
        """Get dispersion values in ps/nm.km units."""
        ...

    @cached_property
    def as_s_m_m(self) -> NDArray:
        """Get dispersion values in s/m^2 units."""
        ...

    def get_wls(self) -> WavelengthArray:
        """Get the wavelength array for this dispersion data."""
        ...

    @cached_property
    def _disp_fn(self) -> BSpline:
        """Return a cubic spline interpolant for dispersion vs wavelength."""
        ...

    def _check_wl(self, wavelength: Wavelength) -> None:
        """Raise ValueError if wavelength is outside the interpolation range."""
        ...

    def fn(self, wavelength: Wavelength) -> float:
        """Interpolated dispersion at wavelength (m)"""
        ...

    @classmethod
    def from_neff(
        cls,
        neff: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength: Wavelength,
        ignore_fit_error: bool = False,
    ) -> Self:
        """
        Create Dispersion object from effective refractive index data.

        Calculates dispersion using the formula:
        D = -λ/c * (d²neff/dλ²)

        Args:
            neff: Array of effective refractive index values
            wavelengths: Corresponding wavelength array
            central_wavelength: design central wavelength
            ignore_fit_error: Gives output ignoring bad curve fitting (Default: false)

        Returns:
            Dispersion object calculated from neff data

        Raises:
            ValueError: If neff and wavelengths arrays have different lengths
            TypeError: If wavelengths is not a WavelengthArray
            ChildProcessError: If numerical differentiation fails
        """
        ...

    @classmethod
    def from_propagation_constant(
        cls,
        beta: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength: Wavelength,
        ignore_fit_error: bool = False,
    ) -> Self:
        """
        Create Dispersion object from propagation constant data.

        Calculates dispersion using the formula:
        D = -(2πc)/λ² * (d²β/dω²)

        Args:
            beta: Array of propagation constant values
            wavelengths: Corresponding wavelength array
            central_wavelength: design central wavelength
            ignore_fit_error: Gives output ignoring bad curve fitting (Default: false)

        Returns:
            Dispersion object calculated from beta data

        Raises:
            ValueError: If beta and wavelengths arrays have different lengths
            TypeError: If wavelengths is not a WavelengthArray
            ChildProcessError: If numerical differentiation fails
        """
        ...

    def get_beta2(self, wavelength_nm: float) -> float:
        """Return β₂ at wavelength_nm (s²/m) via spline interpolation."""
        ...

    def get_betas(
        self,
        polyOrder: int,
        wavelength: Wavelength | None = None,
        make_plot: bool = False,
        return_diagnostics: bool = False,
    ) -> NDArray | Tuple[NDArray, NDArray, NDArray, NDArray] | None:
        """
        Calculate beta coefficients from dispersion data.

        Fits a polynomial to the group velocity dispersion (GVD) parameter β₂
        and returns the series expansion coefficients (β₂, β₃, ...).

        Args:
            polyOrder: Order of the polynomial fit.
            wavelength: at which wavelengths data is calculated from.
            make_plot: If True, displays a plot of the fit.
            return_diagnostics: If True, returns detailed fit diagnostics.

        Returns:
            - If return_diagnostics is False: An array of beta coefficients [β₂, β₃, ...].
            - If return_diagnostics is True: A tuple containing:
                - betas: The array of beta coefficients.
                - fit_x_axis: The frequency axis (ω - ω₀) in THz.
                - data: The original β₂ data in ps²/m.
                - fit: The polynomial fit of β₂ data.
        """
        ...

@dataclass(config={"arbitrary_types_allowed": True})
class ZDependentDispersion:
    """Z-dependent dispersion profile β(ω, z) for tapered/dispersion-managed waveguides.

    Holds a 2-D table of propagation constants β[ω_idx, z_idx] with axis arrays,
    plus a `fn(omega, z)` interpolant built via `RegularGridInterpolator`.
    """

    omegas: NDArray
    z_positions: NDArray
    beta: NDArray
    central_wavelength: float

    def __post_init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def fn(self, omega: float | NDArray, z: float) -> float | NDArray: ...
    @property
    def n_omega(self) -> int: ...
    @property
    def n_z(self) -> int: ...
    def get_betas_at_z(
        self,
        z: float,
        order: int = 7,
        omega0: float | None = None,
        halfwidth: float | None = None,
    ) -> NDArray:
        """Return [β₂, β₃, …, β_order] in SI units (s^k/m) fitted near omega0."""
        ...
    def get_betas_vs_z(
        self,
        order: int = 7,
        omega0: float | None = None,
        halfwidth: float | None = None,
    ) -> NDArray:
        """Return shape (n_z, order - 1) Taylor coefficients in SI units (s^k/m)."""
        ...
    @classmethod
    def from_arrays(
        cls,
        omegas: NDArray,
        z_positions: NDArray,
        beta: NDArray,
        central_wavelength: float,
    ) -> Self: ...
    @classmethod
    def from_npz(path: str, central_wavelength: float | None = None) -> Self: ...

@dataclass(config={"arbitrary_types_allowed": True})
class PropagationConstant:
    """
    Represents propagation constant characteristics of optical fibers.

    This class handles propagation constant data and provides methods to
    calculate related parameters from effective refractive index or other
    optical properties.

    Attributes:
        values: propagation constant values.
        x_values: wavelength or angular frequency array.
    """

    values: NDArray
    x_values: WavelengthArray | AngularFrequencyArray

    @model_validator(mode="after")
    def _setup(self) -> "PropagationConstant": ...
    @classmethod
    def beta2_from_neff(
        cls, neff: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ) -> NDArray:
        """
        Calculate β₂ parameter from effective refractive index.

        Uses the relationship: β = neff * ω / c

        Args:
            neff: Array of effective refractive index values
            x_values: Wavelength or angular frequency array

        Returns:
            Array of β₂ values

        Raises:
            ValueError: If neff and x_values have different lengths
        """
        ...

    @classmethod
    def from_neff_omega(cls, neff: NDArray, omega: AngularFrequencyArray) -> Self:
        """
        Create PropagationConstant object from neff and angular frequency data.

        Calculates propagation constants using: β = neff * ω / c

        Args:
            neff: Array of effective refractive index values
            omega: Angular frequency array

        Returns:
            PropagationConstant object

        Raises:
            ValueError: If neff and omega have different lengths
            TypeError: If omega is not an AngularFrequencyArray
        """
        ...
