"""Fiber dispersion and propagation constant calculations."""

import warnings
from functools import cached_property
from math import factorial
from pathlib import Path
from typing import Any, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from pydantic.dataclasses import dataclass
from scipy.interpolate import BSpline, RegularGridInterpolator, make_splrep

from photonics_helper.base import (
    C_MS,
    PI,
    AngularFrequencyArray,
    Wavelength,
    WavelengthArray,
)


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
        return float(self._disp_fn(wavelength.as_m).item())

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
        wl_eq = wavelengths.to_equally_spaced()
        interp = make_splrep(wavelengths.as_m, neff)(wl_eq)

        spline = make_splrep(wl_eq, interp)
        diff_2 = spline.derivative(2)

        dispersion: NDArray = -wl_eq / C_MS * diff_2(wl_eq)

        smooth_fit = np.all(np.diff(dispersion * 1e6) < 50)
        if not smooth_fit:
            warnings.warn(
                "Bad fitting of neff values. Consider building Disperison in other ways..."
            )
            if not ignore_fit_error:
                raise ValueError(
                    "Can't perform numerical differentiation with small error..."
                )
        return cls(
            wavelengths=WavelengthArray(wl_eq, "m"),
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
                raise ValueError(
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
        beta2 = -(self.wavelengths.as_m**2) / (2 * PI * C_MS) * self.as_s_m_m
        spline = make_splrep(self.wavelengths.as_nm, beta2)
        return float(spline(wavelength_nm))

    def get_beta2_at(self, wavelength: Wavelength) -> float:
        """Return β₂ (s²/m) at a :class:`Wavelength` (unit-safe wrapper).

        This is the preferred entry point for callers that work with unit
        objects; it forwards the wavelength in nm to :meth:`get_beta2` so the
        unit convention cannot be mixed up.
        """
        return float(self.get_beta2(wavelength.as_nm))

    def get_betas(
        self,
        polyOrder,
        wavelength: Wavelength | None = None,
        make_plot=False,
        return_diagnostics=False,
    ):
        """Fit dispersion data to a polynomial in ω and return beta coefficients.

        Returns betas in array [beta2, beta3, ...] in ``ps^k/m`` (with ``Ω``
        in ``rad/ps``) — the native form expected by
        :class:`~photonics_helper.gnlse.GNLSESolver`. If you instead have SI
        coefficients (``s^k/m``), pass them with
        ``GNLSESolver(..., betas_unit="s^k/m")`` and they will be converted.
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

    def __call__(self, omega: "NDArray | float") -> "float | NDArray":
        """Alias for :meth:`beta` — makes the pc directly usable as ``beta_fn``."""
        return self.beta(omega)

    def beta(self, omega: "NDArray | float") -> "float | NDArray":
        """Callable β(ω) via spline interpolation of the stored values.

        Scalar input returns a Python ``float``; array input returns an
        ``NDArray`` of the same shape. Raises ``ValueError`` when ``omega``
        lies outside the stored ``n_eff(λ)``/ω table window.

        Bridges ``DispersionModel``-style usage (``beta_fn`` in
        :mod:`photonics_helper.phase_matching` and the χ⁽²⁾ solvers) without
        the :class:`~photonics_helper.phase_matching.PropagationConstantAdaptor`
        indirection.
        """
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        om = np.asarray(self.omegas.as_rad_s, dtype=float)
        beta = np.asarray(self.values, dtype=float)
        if len(om) < 2:
            raise ValueError(
                "beta(omega) requires at least 2 tabulated points, got "
                f"{len(om)}"
            )
        order = np.argsort(om)
        spline = make_splrep(om[order], beta[order])
        lo, hi = float(om.min()), float(om.max())
        if omega_arr.min() < lo - 1e-9 or omega_arr.max() > hi + 1e-9:
            raise ValueError(
                f"beta(omega) queried outside the stored table window "
                f"[{lo:.6g}, {hi:.6g}] rad/s"
            )
        result = spline(omega_arr)
        if np.isscalar(omega) or np.asarray(omega).ndim == 0:
            return float(result.reshape(-1)[0])
        return np.asarray(result, dtype=float)

    def beta2(self, wavelength: Wavelength) -> float:
        """Return the group-velocity dispersion ``d²β/dω²`` (s²/m).

        Differentiates a cubic spline through the tabulated ``β(ω)``. Requires
        at least four points (cubic-spline minimum).
        """
        omega = np.asarray(self.omegas.as_rad_s, dtype=float)
        beta = np.asarray(self.values, dtype=float)
        if len(omega) < 4:
            raise ValueError(
                "beta2 requires at least 4 tabulated points, got "
                f"{len(omega)}"
            )
        order = np.argsort(omega)
        spline = make_splrep(omega[order], beta[order])
        d2 = spline.derivative(2)
        return float(d2(wavelength.to_omega().as_rad_s))

    @classmethod
    def beta_from_neff(
        cls, neff: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ) -> Self:
        """Compute β = neff × ω / c and return a ``PropagationConstant``."""
        if not isinstance(x_values, (WavelengthArray, AngularFrequencyArray)):
            raise TypeError(
                "x_values should be a type of either `WavelengthArray` or `AngularFrequencyArray`"
            )
        if len(neff) != len(x_values.value):  # type: ignore[arg-type]
            raise ValueError("both neff and x_values must be of same length.")

        if isinstance(x_values, WavelengthArray):
            omegas = x_values.to_omega()
        else:
            omegas = x_values

        betas = neff * omegas.as_rad_s / C_MS
        return cls(values=betas, x_values=x_values)

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


def _read_table(
    path: str | Path, delimiter: str, skiprows: int
) -> NDArray:
    """Read a delimited numeric table with an optional header row.

    Blank lines and ``#`` comments are ignored; ``skiprows`` counts raw lines
    before that filtering. The first remaining row is treated as a header when
    any of its fields is non-numeric.
    """
    with open(path) as fh:
        lines = fh.readlines()
    data_lines = [
        ln
        for ln in lines[skiprows:]
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not data_lines:
        raise ValueError(f"no data rows found in {path}")
    fields = [f.strip() for f in data_lines[0].split(delimiter) if f.strip()]
    try:
        [float(f) for f in fields]
    except ValueError:
        data_lines = data_lines[1:]
    if not data_lines:
        raise ValueError(f"no data rows found in {path} after the header")
    rows = [[float(f) for f in ln.split(delimiter)] for ln in data_lines]
    return np.asarray(rows, dtype=float)


def _pick_key(data: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate key present in ``data`` (case-insensitive)."""
    lowered: dict[str, str] = {k.lower(): k for k in data}
    for name in candidates:
        if name in data:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


@dataclass(config={"arbitrary_types_allowed": True})
class WaveguideMode:
    """Imported waveguide mode: effective index ``n_eff(λ)`` from an FEM solver.

    Tabulated effective indices are converted into the library's dispersion
    objects. This is the documented bridge from an external eigenmode solver
    (Lumerical MODE, COMSOL Wave Optics, …) into
    :class:`PropagationConstant` / :class:`Dispersion`.

    File conventions
    ----------------
    **CSV** — comma separated, optional header, two or three columns::

        wavelength_um, neff
        1.50, 2.4310
        1.55, 2.4205
        ...

    or ``wavelength_um, neff, ng`` with the optional group index ``ng``. A
    header line is detected automatically (any non-numeric first row).

    **NPZ** — ``np.savez`` archive with keys:

    * ``wavelength_um`` (required) — 1-D vacuum wavelengths in **micrometres**.
    * ``neff`` (required) — 1-D effective indices, same length.
    * ``ng`` (optional) — group index.
    * ``central_wavelength_nm`` or ``central_wavelength_um`` (optional) —
      design wavelength; defaults to the mid-point of the grid.

    Validation mirrors :class:`~photonics_helper.materials.RefractiveIndex`:
    a finite, positive, strictly increasing wavelength grid with at least four
    points (cubic-spline requirement).

    Attributes
    ----------
    neff : 1-D effective index array.
    wavelengths : :class:`WavelengthArray` of the tabulated grid.
    central_wavelength : design wavelength.
    ng : optional 1-D group-index array.
    """

    neff: NDArray
    wavelengths: WavelengthArray
    central_wavelength: Wavelength | None = None
    ng: NDArray | None = None

    @model_validator(mode="after")
    def _validate(self) -> "WaveguideMode":
        neff = np.asarray(self.neff, dtype=float).ravel()
        wl_um = np.asarray(self.wavelengths.as_um, dtype=float).ravel()
        if neff.ndim != 1 or wl_um.ndim != 1:
            raise ValueError("neff and wavelengths must be one-dimensional")
        if len(neff) != len(wl_um):
            raise ValueError(
                "neff and wavelength arrays must have equal length, got "
                f"{len(neff)} and {len(wl_um)}"
            )
        if len(wl_um) < 4:
            raise ValueError(
                "cubic spline interpolation requires at least 4 wavelength "
                f"points, got {len(wl_um)}"
            )
        if not np.all(np.isfinite(neff)):
            raise ValueError("neff contains non-finite values")
        if not np.all(np.isfinite(wl_um)):
            raise ValueError("wavelength grid contains non-finite values")
        if np.any(wl_um <= 0):
            raise ValueError("wavelength grid must be positive")
        if not np.all(np.diff(wl_um) > 0):
            raise ValueError("wavelengths must be strictly increasing")
        self.neff = neff
        if self.ng is not None:
            ng = np.asarray(self.ng, dtype=float).ravel()
            if ng.shape != neff.shape:
                raise ValueError(
                    f"ng must have the same shape as neff, got {ng.shape} vs {neff.shape}"
                )
            self.ng = ng
        if self.central_wavelength is None:
            self.central_wavelength = Wavelength(float(wl_um[len(wl_um) // 2]), "um")
        return self

    def __repr__(self) -> str:
        return (
            f"WaveguideMode: neff from {self.wavelengths.as_um.min():.3f} to "
            f"{self.wavelengths.as_um.max():.3f} um, {len(self.neff)} points"
        )

    @cached_property
    def _neff_spline(self) -> BSpline:
        return make_splrep(self.wavelengths.as_um, self.neff)

    def neff_at(self, wavelength: Wavelength) -> float:
        """Interpolated ``n_eff`` at ``wavelength`` within the tabulated range."""
        wl_um = wavelength.as_um
        lo, hi = float(self.wavelengths.as_um.min()), float(self.wavelengths.as_um.max())
        if not (lo <= wl_um <= hi):
            raise ValueError(
                f"n_eff available only between {lo:.4f} and {hi:.4f} um, "
                f"got {wl_um:.4f} um"
            )
        return float(self._neff_spline(wl_um))

    def to_propagation_constant(self) -> PropagationConstant:
        """Build a :class:`PropagationConstant` with ``β = n_eff·ω/c``."""
        return PropagationConstant.beta_from_neff(self.neff, self.wavelengths)

    def to_dispersion(self, ignore_fit_error: bool = True) -> Dispersion:
        """Build a :class:`Dispersion` ``D(λ)`` from ``n_eff(λ)``.

        ``ignore_fit_error`` defaults to ``True`` because FEM exports are often
        coarse and the smoothness guard in :meth:`Dispersion.from_neff` can
        otherwise reject a perfectly usable table.
        """
        if self.central_wavelength is None:  # pragma: no cover - validator sets it
            raise ValueError("central_wavelength is not set")
        return Dispersion.from_neff(
            neff=self.neff,
            wavelengths=self.wavelengths,
            central_wavelength=self.central_wavelength,
            ignore_fit_error=ignore_fit_error,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        central_wavelength: Wavelength | None = None,
        delimiter: str = ",",
        skiprows: int = 0,
    ) -> "WaveguideMode":
        """Load ``wavelength_um, neff[, ng]`` from a CSV/text file."""
        data = _read_table(path, delimiter, skiprows)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                "expected at least 2 columns (wavelength_um, neff), got "
                f"shape {data.shape}"
            )
        ng = data[:, 2] if data.shape[1] >= 3 else None
        return cls(
            neff=data[:, 1],
            wavelengths=WavelengthArray(data[:, 0], "um"),
            central_wavelength=central_wavelength,
            ng=ng,
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        central_wavelength: Wavelength | None = None,
    ) -> "WaveguideMode":
        """Load an ``n_eff(λ)`` table from an ``np.savez`` archive.

        Required keys: ``wavelength_um`` (µm) and ``neff``. Optional: ``ng``,
        ``central_wavelength_nm``/``central_wavelength_um``.
        """
        data = dict(np.load(path, allow_pickle=False))
        wl_key = _pick_key(
            data, ("wavelength_um", "wavelengths_um", "wavelength", "wl")
        )
        neff_key = _pick_key(data, ("neff", "n_eff", "neff_um"))
        if wl_key is None or neff_key is None:
            raise ValueError(
                "NPZ must contain a wavelength key (wavelength_um) and an "
                f"effective-index key (neff); found {sorted(data)}"
            )
        ng_key = _pick_key(data, ("ng", "n_group", "group_index"))
        if central_wavelength is None:
            if "central_wavelength_nm" in data:
                central_wavelength = Wavelength(
                    float(data["central_wavelength_nm"]), "nm"
                )
            elif "central_wavelength_um" in data:
                central_wavelength = Wavelength(
                    float(data["central_wavelength_um"]), "um"
                )
        return cls(
            neff=np.asarray(data[neff_key], dtype=float),
            wavelengths=WavelengthArray(
                np.asarray(data[wl_key], dtype=float), "um"
            ),
            central_wavelength=central_wavelength,
            ng=None if ng_key is None else np.asarray(data[ng_key], dtype=float),
        )


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
    betas_taylor: NDArray | None = None
    beta_orders: NDArray | None = None
    omega0_fit: float | None = None

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
        result = self._interpolator(
            np.column_stack([omega_arr, np.full_like(omega_arr, z)])
        )
        if np.isscalar(omega):
            return float(result[0])
        return np.asarray(result)

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
        beta_at_z = np.asarray(self.fn(omega_fit, z), dtype=float)
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
        # Convert to meters if needed (detect by checking magnitude)
        cw = float(central_wavelength)
        if cw > 1.0:  # Likely in micrometers
            cw = cw * 1e-6  # Convert to meters
        return cls(
            omegas=np.asarray(omegas, dtype=float),
            z_positions=np.asarray(z_positions, dtype=float),
            beta=np.asarray(beta, dtype=float),
            central_wavelength=cw,
        )

    @classmethod
    def from_npz(
        cls, path: str, central_wavelength: float | None = None
    ) -> "ZDependentDispersion":
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
