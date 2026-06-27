from photonics_helper.base import AngularFrequencyArray, Wavelength, WavelengthArray

from functools import cached_property
from numpy.typing import NDArray
from typing import Literal, Self, Tuple

class Dispersion:
    """
    Represents optical fiber dispersion characteristics.

    This class handles chromatic dispersion data for optical fibers, providing
    methods to interpolate dispersion values at specific wavelengths and convert
    between different units.

    Attributes:
        _values: Dispersion values in s/m^2
        _wavelengths: Wavelength array for the dispersion data
        _unit: Internal unit representation (always "s/m^2")
    """

    def __init__(
        self,
        wavelengths: WavelengthArray,
        values: NDArray,
        unit: Literal["ps/nm.km", "s/m^2"],
        central_wavelength: Wavelength,
    ) -> None:
        """
        Initialize a Dispersion object.

        Args:
            wavelengths: Array of wavelengths corresponding to dispersion values
            values: Dispersion values in the specified unit
            unit: Unit of the dispersion values ("ps/nm.km" or "s/m^2")
            central_wavelength: Central wavelength for the dispersion curve
        """
        ...

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

    def _check_wl(self, wavelength_m: float) -> None:
        """Check if wavelength (in meters) is within valid range."""
        ...

    def fn(self, wavelength: float) -> float:
        """
        Get interpolated dispersion value at a specific wavelength (in meters).

        Raises:
            ValueError: If wavelength is outside valid range
        """
        ...

    def fn_s_m_m(self, wavelength_nm: float) -> float:
        """Dispersion in s/m^2 at wavelength in nm."""
        ...

    def fn_ps_nm_km(self, wavelength_nm: float) -> float:
        """Dispersion in ps/nm.km at wavelength in nm."""
        ...

    @classmethod
    def from_neff(
        cls,
        neff: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength_nm: float,
        ignore_fit_error: bool = False,
    ) -> Self:
        """
        Create Dispersion object from effective refractive index data.

        Calculates dispersion using the formula:
        D = -λ/c * (d²neff/dλ²)

        Args:
            neff: Array of effective refractive index values
            wavelengths: Corresponding wavelength array
            central_wavelength_nm: Central wavelength in nanometers
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
    def from_propagation_constanant(
        cls,
        beta: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength_nm: float,
        ignore_fit_error: bool = False,
    ) -> Self:
        """
        Create Dispersion object from propagation constant data.

        Calculates dispersion using the formula:
        D = -(2πc)/λ² * (d²β/dω²)

        Args:
            beta: Array of propagation constant values
            wavelengths: Corresponding wavelength array
            central_wavelength_nm: Central wavelength in nanometers
            ignore_fit_error: Gives output ignoring bad curve fitting (Default: false)

        Returns:
            Dispersion object calculated from beta data

        Raises:
            ValueError: If beta and wavelengths arrays have different lengths
            TypeError: If wavelengths is not a WavelengthArray
            ChildProcessError: If numerical differentiation fails
        """
        ...

    def get_betas(
        self,
        polyOrder: int,
        wavelength: Wavelength | None = None,
        make_plot: bool = False,
        return_diagnostics: bool = False,
    ) -> NDArray | Tuple[NDArray, NDArray, NDArray, NDArray]:
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

class PropagationConstant:
    """
    Represents propagation constant characteristics of optical fibers.

    This class handles propagation constant data and provides methods to
    calculate related parameters from effective refractive index or other
    optical properties.

    Attributes:
        _values: Propagation constant values
        _wavelengths: Wavelength array (if initialized with wavelengths)
        _omegas: Angular frequency array (if initialized with frequencies)
    """

    def __init__(
        self, values: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ) -> None:
        """
        Initialize a PropagationConstant object.

        Args:
            values: Array of propagation constant values
            x_values: Either wavelength or angular frequency array
        """
        ...

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
