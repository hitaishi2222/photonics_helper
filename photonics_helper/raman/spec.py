"""RamanSpec - material Raman scattering properties and derivations."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from ..base import (
    C_MS,
    PI,
    Energy,
    Time,
    Wavelength,
)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if TYPE_CHECKING:
    from ..materials import NKMaterial

from .db import RamanDatabase
from .reference import RAMAN_MATERIALS

# ─── RamanSpec ────────────────────────────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanSpec:
    """Material Raman scattering properties.

    Stores Raman parameters and auto-derives derived quantities.

    Attributes
    ----------
    name : Material name.
    crystal : Crystal structure (e.g. "Wurtzite", "Diamond cubic").
    bandgap_eV : Bandgap energy in eV.
    n2 : Nonlinear refractive index n₂ (m²/W).
    raman_shift_cm : Raman shift in cm⁻¹.
    raman_linewidth_cm : Raman linewidth (FWHM) in cm⁻¹.
    fR : Raman response fraction (0 to 1).
    gain_coeff : Raman gain coefficient (m/GW).
    tau1 : Oscillation period (s). Auto-derived if None.
    tau2 : Damping time (s). Auto-derived if None.
    alpha : Silica model parameter α (default 0.52).
    lo_phonon_cm : LO phonon wavenumber (cm⁻¹).
    to_phonon_cm : TO phonon wavenumber (cm⁻¹).
    references : Source citation.
    phonon_modes : List of PhononMode for multi-mode Raman materials.
    """

    name: str
    raman_shift_cm: float | None = None
    raman_linewidth_cm: float | None = None
    crystal: str | None = None
    bandgap_eV: Energy | None = None
    n2: float | None = None
    fR: float | None = None
    gain_coeff: float | None = None
    tau1: Time | None = None
    tau2: Time | None = None
    alpha: float = 0.52
    lo_phonon_cm: float | None = None
    to_phonon_cm: float | None = None
    references: str | None = None
    phonon_modes: list | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_types(cls, values) -> dict:
        # Handle ArgsKwargs from pydantic.dataclasses
        if hasattr(values, "kwargs"):
            values = values.kwargs
        if isinstance(values, dict):
            for key, target_type, unit in [
                ("bandgap_eV", Energy, "eV"),
                ("tau1", Time, "s"),
                ("tau2", Time, "s"),
            ]:
                v = values.get(key)
                if v is not None and not isinstance(v, target_type):
                    values[key] = target_type(v, unit)
        return dict(values)

    @model_validator(mode="after")
    def _validate(self) -> "RamanSpec":
        """Validate that essential Raman parameters are present.

        Note: raman_shift_cm and raman_linewidth_cm are optional in the
        constructor. They should be provided when creating from raw params,
        but can be None when using from_database() with fallback.
        """
        return self

    @cached_property
    def raman_shift_Hz(self) -> float:
        """Raman shift in Hz."""
        if self.raman_shift_cm is None:
            return 0.0
        return C_MS * self.raman_shift_cm * 100.0

    @cached_property
    def raman_shift_THz(self) -> float:
        """Raman shift in THz."""
        return self.raman_shift_Hz / 1e12

    @cached_property
    def raman_shift_omega(self) -> float:
        """Raman shift in rad/s."""
        return 2 * PI * self.raman_shift_Hz

    @cached_property
    def linewidth_Hz(self) -> float:
        """Raman linewidth (FWHM) in Hz."""
        if self.raman_linewidth_cm is None:
            return 0.0
        return C_MS * self.raman_linewidth_cm * 100.0

    @cached_property
    def linewidth_THz(self) -> float:
        """Raman linewidth (FWHM) in THz."""
        return self.linewidth_Hz / 1e12

    @cached_property
    def quality_factor(self) -> float:
        """Quality factor Q = shift / linewidth."""
        if self.linewidth_Hz == 0 or self.raman_shift_Hz == 0:
            return float("inf") if self.raman_shift_Hz != 0 else 0.0
        return self.raman_shift_Hz / self.linewidth_Hz

    def stokes_wavelength(self, pump_wl: Wavelength) -> Wavelength:
        """Stokes wavelength for a given pump wavelength.

        ν_stokes = ν_pump - ν_Raman
        λ_stokes = c / ν_stokes
        """
        if self.raman_shift_cm is None:
            raise ValueError("Raman shift not available")
        nu_pump = C_MS / pump_wl.as_m
        nu_stokes = nu_pump - self.raman_shift_Hz
        if nu_stokes <= 0:
            raise ValueError("Stokes frequency would be negative")
        return Wavelength(C_MS / nu_stokes, "m")

    def anti_stokes_wavelength(self, pump_wl: Wavelength) -> Wavelength:
        """Anti-Stokes wavelength for a given pump wavelength.

        ν_anti_stokes = ν_pump + ν_Raman
        λ_anti_stokes = c / ν_anti_stokes
        """
        if self.raman_shift_cm is None:
            raise ValueError("Raman shift not available")
        nu_pump = C_MS / pump_wl.as_m
        nu_anti_stokes = nu_pump + self.raman_shift_Hz
        return Wavelength(C_MS / nu_anti_stokes, "m")

    @property
    def phonon_response(self):
        """Multi-mode phonon response, or None if no modes configured.

        Returns
        -------
        PhononResponse or None
        """
        from ..phonon import PhononResponse

        if self.phonon_modes:
            return PhononResponse(self.phonon_modes, fR=self.fR)
        return None

    def multi_stokes_wavelengths(self, pump_wl) -> list:
        """Stokes wavelength for each phonon mode.

        Parameters
        ----------
        pump_wl : Wavelength
            Pump wavelength.

        Returns
        -------
        list[Wavelength]
            Stokes wavelengths, one per mode.
        """
        if not self.phonon_modes:
            return []

        from ..base import C_MS

        wavelengths = []
        for mode in self.phonon_modes:
            nu_pump = C_MS / pump_wl.as_m
            nu_stokes = nu_pump - C_MS * mode.shift_cm.as_1_cm * 100.0
            if nu_stokes > 0:
                wavelengths.append(Wavelength(C_MS / nu_stokes, "m"))
        return wavelengths

    def summary(self) -> str:
        """Text summary of material Raman properties."""
        lines = [
            f"Material: {self.name}",
            f"Crystal: {self.crystal or 'N/A'}",
            f"Bandgap: {self.bandgap_eV or 'N/A'} eV",
            f"n₂: {self.n2 or 'N/A'} m²/W",
            f"Raman shift: {self.raman_shift_cm} cm⁻¹ = {self.raman_shift_THz:.2f} THz",
            f"Linewidth: {self.raman_linewidth_cm} cm⁻¹ = {self.linewidth_THz:.2f} THz",
            f"fR: {self.fR or 'N/A'}",
            f"Gain coeff: {self.gain_coeff or 'N/A'} m/GW",
            f"Q factor: {self.quality_factor:.1f}",
        ]
        if self.lo_phonon_cm:
            lines.append(f"LO phonon: {self.lo_phonon_cm} cm⁻¹")
        if self.to_phonon_cm:
            lines.append(f"TO phonon: {self.to_phonon_cm} cm⁻¹")
        if self.phonon_modes:
            lines.append(f"Multi-mode: {len(self.phonon_modes)} phonon modes")
        return "\n".join(lines)

    def nk(self, wavelength_um: float, source: str | None = None) -> complex:
        """Interpolated complex refractive index at wavelength (μm).

        Returns n + i·k from the nk_data table, or falls back to Sellmeier
        interpolation if tabulated data is absent.

        Parameters
        ----------
        wavelength_um : Wavelength in μm
        source : Optional ``material–author`` provenance key. When the material
            has several tabulated datasets, one MUST be selected; when omitted
            and a single dataset exists it is used automatically.

        Returns
        -------
        complex : n + i·k

        Raises
        ------
        ValueError : If no data available or wavelength outside valid range
        """

        # Query tabulated data first
        db = RamanDatabase()
        if source is not None:
            wl, n_tab, k_tab = db.get_nk_by_source(self.name, source)  # type: ignore[arg-type]
        else:
            sources = db.list_nk_sources(self.name)  # type: ignore[arg-type]
            if len(sources) > 1:
                raise ValueError(
                    f"Multiple tabulated n/k datasets available for "
                    f"'{self.name}': {sources}. Select one via "
                    f"nk(wavelength, source='{sources[0]}')."
                )
            wl, n_tab, k_tab = db.get_nk_data(cast("NKMaterial", self.name))

        if len(wl) > 0:
            # Interpolate from tabulated data
            from scipy.interpolate import interp1d

            n_func = interp1d(wl, n_tab, kind="cubic", fill_value="extrapolate")
            k_func = interp1d(wl, k_tab, kind="cubic", fill_value="extrapolate")

            # Validate range
            if wavelength_um < wl[0] or wavelength_um > wl[-1]:
                raise ValueError(
                    f"Wavelength {wavelength_um} μm outside valid range [{wl[0]:.3f}, {wl[-1]:.3f}] μm"
                )

            n_val = float(n_func(wavelength_um))
            k_val = float(k_func(wavelength_um))
            return complex(n_val, k_val)

        # Fallback to Sellmeier
        sellmeier_data = db.get_sellmeier(self.name)  # type: ignore[arg-type]
        if sellmeier_data:
            return self.nk_from_sellmeier(wavelength_um, sellmeier_data)

        # No data available
        raise ValueError(f"No refractive index data available for {self.name}")

    def nk_from_sellmeier(
        self, wavelength_um: float, sellmeier_data: dict | None = None
    ) -> complex:
        """Compute n + ik from stored Sellmeier coefficients.

        Parameters
        ----------
        wavelength_um : Wavelength in μm
        sellmeier_data : Dict with keys: form, a0, coefficients, wavelengths, valid_from_um, valid_to_um
                       If None, fetches from database

        Returns
        -------
        complex : n + i·0 (k=0 for transparent region)

        Raises
        ------
        ValueError : If wavelength outside valid range or no data available
        """
        import numpy as np

        if sellmeier_data is None:
            db = RamanDatabase()
            sellmeier_data = db.get_sellmeier(self.name)  # type: ignore[arg-type]

        if sellmeier_data is None:
            raise ValueError(f"No Sellmeier data available for {self.name}")

        # Validate wavelength range
        if (
            wavelength_um < sellmeier_data["valid_from_um"]
            or wavelength_um > sellmeier_data["valid_to_um"]
        ):
            raise ValueError(
                f"Wavelength {wavelength_um} μm outside valid range "
                f"[{sellmeier_data['valid_from_um']:.3f}, {sellmeier_data['valid_to_um']:.3f}] μm"
            )

        # Compute n from Sellmeier equation
        form = sellmeier_data["form"]
        a0 = sellmeier_data["a0"]
        coefficients = sellmeier_data["coefficients"]
        wavelengths = sellmeier_data["wavelengths"]

        if form == "standard":
            # n² = A₀ + Σ Aᵢλ²/(λ² - Bᵢ)
            n_squared = a0
            for A, B in zip(coefficients, wavelengths):
                n_squared += A * wavelength_um**2 / (wavelength_um**2 - B)
            n_val = np.sqrt(n_squared)

        elif form == "alt":
            # n² = A₀ + Σ Aᵢ/(λ² - Bᵢ²)
            n_squared = a0
            for A, B in zip(coefficients, wavelengths):
                n_squared += A / (wavelength_um**2 - B**2)
            n_val = np.sqrt(n_squared)
        else:
            raise ValueError(f"Unknown Sellmeier form: {form}")

        # k = 0 for transparent region (no absorption data in Sellmeier)
        return complex(float(n_val), 0.0)

    def plot_spectrum(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        shift_range_cm: float = 200,
        n_points: int = 1000,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Raman intensity vs Raman shift.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        shift_range_cm : Range around 0 to plot (cm⁻¹)
        n_points : Number of points
        figsize : Figure size for matplotlib
        """
        if self.raman_shift_cm is None or self.raman_linewidth_cm is None:
            raise ValueError("Raman shift and linewidth required")

        shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
        center = self.raman_shift_cm
        width = self.raman_linewidth_cm / 2

        # Lorentzian lineshape
        intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
        intensity /= np.max(intensity)  # Normalize

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=shift, y=intensity, mode="lines", name="Raman"))
            fig.update_layout(
                title=f"Raman Spectrum: {self.name}",
                xaxis_title="Raman shift (cm⁻¹)",
                yaxis_title="Intensity (arb.)",
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize or (8, 5))
            ax.plot(shift, intensity, linewidth=2, label="Raman peak")
            ax.axvline(
                x=center,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Peak: {center} cm⁻¹",
            )
            ax.axvspan(
                center - width,
                center + width,
                alpha=0.2,
                color="orange",
                label=f"FWHM: {2 * width:.1f} cm⁻¹",
            )
            ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=12)
            ax.set_ylabel("Intensity (arb.)", fontsize=12)
            ax.set_title(f"Raman Spectrum: {self.name}", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

    def plot_phonons(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Raman-active phonon modes.

        Displays LO, TO, and other Raman-active modes if available.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            fig = go.Figure()
            if self.lo_phonon_cm:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=self.lo_phonon_cm,
                        title=dict(text="LO Phonon"),
                        gauge=dict(axis=dict(range=[0, self.lo_phonon_cm * 1.2])),
                    )
                )
            if self.to_phonon_cm:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=self.to_phonon_cm,
                        title=dict(text="TO Phonon"),
                        gauge=dict(axis=dict(range=[0, self.to_phonon_cm * 1.2])),
                    )
                )
            fig.update_layout(title=f"Phonon Modes: {self.name}")
            return fig
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (10, 4))
            if self.lo_phonon_cm:
                axes[0].barh(["LO"], [self.lo_phonon_cm], color="blue", alpha=0.7)
                axes[0].set_xlabel("Wavenumber (cm⁻¹)")
                axes[0].set_title("LO Phonon")
            else:
                axes[0].text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=axes[0].transAxes,
                )
                axes[0].set_title("LO Phonon")

            if self.to_phonon_cm:
                axes[1].barh(["TO"], [self.to_phonon_cm], color="red", alpha=0.7)
                axes[1].set_xlabel("Wavenumber (cm⁻¹)")
                axes[1].set_title("TO Phonon")
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )
                axes[1].set_title("TO Phonon")

            fig.suptitle(
                f"Raman-Active Phonons: {self.name}", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            return fig

    @classmethod
    def from_database(
        cls, name: str, fallback: dict | None = None, db_path: Path | None = None
    ) -> "RamanSpec":
        """Create RamanSpec from SQLite database or fallback.

        Queries materials.db first, then falls back to the hardcoded
        RAMAN_MATERIALS dict, and finally to the provided fallback dict.

        Parameters
        ----------
        name : Material name
        fallback : Optional dict of fallback values if not in DB
        db_path : Optional explicit path to SQLite database
        """
        # Try SQLite database first
        try:
            db = RamanDatabase(db_path=db_path)
            row = db.get_material(name)
            if row:
                return cls(**row)  # type: ignore[arg-type]
        except Exception:
            pass

        # Fall back to hardcoded materials
        if name in RAMAN_MATERIALS:
            return cls(**RAMAN_MATERIALS[name])  # type: ignore[arg-type]

        # Use fallback if provided
        if fallback:
            return cls(name=name, **fallback)

        raise ValueError(f"Material '{name}' not found in database or fallback")
