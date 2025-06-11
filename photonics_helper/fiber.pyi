from photonics_helper.base import AngularFrequencyArray, Wavelength, WavelengthArray

from numpy.typing import NDArray
from typing import Literal, Self

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

    @property
    def as_ps_nm_km(self) -> NDArray:
        """Get dispersion values in ps/nm.km units."""
        ...

    @property
    def as_s_m_m(self) -> NDArray:
        """Get dispersion values in s/m^2 units."""
        ...

    def get_wl(self) -> WavelengthArray:
        """Get the wavelength array for this dispersion data."""
        ...

    def check_wavelength_limit(
        self, wavelength: float, unit: Literal["nm", "m", "um"]
    ) -> None:
        """
        Check if a wavelength is within the valid range of the dispersion data.

        Args:
            wavelength: Wavelength value to check
            unit: Unit of the wavelength ("nm", "m", or "um")

        Raises:
            ValueError: If wavelength is outside the valid range
        """
        ...

    def fn(self, wavelength: float) -> float:
        """
        Get interpolated dispersion value at a specific wavelength.

        Args:
            wavelength: Wavelength in meters

        Returns:
            Dispersion value in s/m^2

        Raises:
            ValueError: If wavelength is outside valid range
        """
        ...

    def fn_s_m_m(self, wavelength_nm: float) -> float:
        """
        Get interpolated dispersion value in s/m^2 units.

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            Dispersion value in s/m^2

        Raises:
            ValueError: If wavelength is outside valid range
        """
        ...

    def fn_ps_nm_km(self, wavelength_nm: float) -> float:
        """
        Get interpolated dispersion value in ps/nm.km units.

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            Dispersion value in ps/nm.km

        Raises:
            ValueError: If wavelength is outside valid range
        """
        ...

    @classmethod
    def from_neff(
        cls,
        neff: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength_nm: float,
        ignore_fit_error: bool = True,
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

    def get_beta2(self, wavelength_nm: float) -> float:
        """
        Get the second-order dispersion parameter β₂ at a specific wavelength.

        β₂ is related to dispersion by: β₂ = -λ²/(2πc) * D

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            β₂ value in s²/m

        Raises:
            ValueError: If wavelength is outside valid range
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

