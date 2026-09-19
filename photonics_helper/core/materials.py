"""Optical-material interface — a foundation primitive.

Two things live here:

- :class:`OpticalMaterial`, a runtime-checkable protocol describing a
  wavelength-dependent material (a name, the tabulated ``n``/``k`` and
  ``wl`` arrays, the interpolating ``n_func``/``k_func``, and a provenance
  ``source``). Downstream code can type against the interface instead of the
  concrete :class:`~photonics_helper.materials.RefractiveIndex`.
- :func:`material`, a convenience lookup that builds a :class:`Material` from
  the bundled database via the existing
  :meth:`RefractiveIndex.from_material_database` path.

This module is additive: nothing here replaces the existing
``photonics_helper.materials`` API.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from ..base import WavelengthArray
from ..materials import RefractiveIndex

__all__ = ["Material", "OpticalMaterial", "material"]


@runtime_checkable
class OpticalMaterial(Protocol):
    """Wavelength-dependent optical material.

    Satisfied structurally by any object exposing these attributes/methods.
    """

    name: str
    source: str | None
    license: str | None
    wl: WavelengthArray
    n: NDArray
    k: NDArray

    def n_func(self, wavelength: float) -> float:
        """Refractive index n at ``wavelength`` (μm)."""
        ...

    def k_func(self, wavelength: float) -> float:
        """Extinction coefficient k at ``wavelength`` (μm)."""
        ...


@dataclass
class Material:
    """Concrete :class:`OpticalMaterial` wrapping a :class:`RefractiveIndex`.

    Parameters
    ----------
    name : material name (as looked up in the database).
    index : the underlying tabulated/interpolated refractive index.
    source : provenance string (citation/author key) when known, else ``None``.
    license : licence identifier/sentinel for the data when known, else ``None``.
    """

    name: str
    index: RefractiveIndex
    source: str | None = None
    license: str | None = None

    @property
    def wl(self) -> WavelengthArray:
        """Wavelength samples of the underlying table."""
        return self.index.wl

    @property
    def n(self) -> NDArray:
        """Real refractive index samples."""
        return self.index.n

    @property
    def k(self) -> NDArray:
        """Extinction-coefficient samples."""
        return self.index.k

    @property
    def nk(self) -> NDArray:
        """Complex refractive index samples (n + ik)."""
        return self.index.nk

    def n_func(self, wavelength: float) -> float:
        """Refractive index n at ``wavelength`` (μm)."""
        return self.index.n_func(wavelength)

    def k_func(self, wavelength: float) -> float:
        """Extinction coefficient k at ``wavelength`` (μm)."""
        return self.index.k_func(wavelength)

    def nk_func(self, wavelength: float) -> complex:
        """Complex index n + ik at ``wavelength`` (μm)."""
        return self.index.nk_func(wavelength)

    def group_index(self, wavelength: float) -> float:
        """Group index at ``wavelength`` (μm)."""
        return self.index.group_index(wavelength)


def _lookup_metadata(
    name: str, axis: str | None
) -> tuple[str | None, str | None]:
    """Best-effort ``(source, license)`` lookup in the bundled database.

    Returns ``(None, None)`` when no row (or no provenance string) is found;
    provenance is metadata, so a lookup miss must not break ``material()``.
    """
    try:
        from ..raman import RamanDatabase

        db = RamanDatabase()

        # A tabulated key ("material-author") is itself a provenance source
        # key, so look it up directly first.
        direct = db.get_provenance(name)
        direct_source = str(direct["citation"]) if direct and direct.get("citation") else None
        direct_license = str(direct["license"]) if direct and direct.get("license") else None

        candidates = [name]
        if axis is not None:
            suffix = "_or" if axis.lower() in ("ordinary", "o") else "_er"
            candidates.insert(0, f"{name}{suffix}")

        source: str | None = direct_source
        license_id: str | None = direct_license
        for candidate in candidates:
            row = db.get_sellmeier(candidate)  # type: ignore[arg-type]
            if row:
                if source is None and row.get("source"):
                    source = str(row["source"])
                if license_id is None and row.get("license"):
                    license_id = str(row["license"])

        spec = db.get_material(name)
        if spec:
            if source is None and spec.get("references"):
                source = str(spec["references"])
            if license_id is None and spec.get("license"):
                license_id = str(spec["license"])

        if license_id is None:
            for source_key in db.list_nk_sources(name):  # type: ignore[arg-type]
                prov = db.get_provenance(source_key)
                if prov and prov.get("license"):
                    license_id = str(prov["license"])
                    if source is None and prov.get("citation"):
                        source = str(prov["citation"])
                    break

        return source, license_id
    except (sqlite3.Error, ValueError, KeyError, TypeError, OSError):
        return None, None


def material(
    name: str, *, axis: str | None = None, n_points: int = 200
) -> Material:
    """Load a material from the bundled database.

    Parameters
    ----------
    name : material name or ``material-author`` tabulated key.
    axis : for birefringent crystals, ``"extraordinary"`` (default canonical
        row) or ``"ordinary"``.
    n_points : number of samples when the source is a Sellmeier equation.

    Returns
    -------
    Material
        A concrete :class:`OpticalMaterial`.
    """
    index = RefractiveIndex.from_material_database(name, n_points=n_points, axis=axis)
    source, license_id = _lookup_metadata(name, axis)
    return Material(name=name, index=index, source=source, license=license_id)
