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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from numpy.typing import NDArray

from ..base import WavelengthArray
from ..materials import RefractiveIndex

if TYPE_CHECKING:
    from .data import MaterialsDatabase

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


def _text(value: object) -> str | None:
    """``str(value)`` unless it is missing/empty, else ``None``."""
    return str(value) if value else None


def _fill_from_sellmeier(
    db: MaterialsDatabase,
    name: str,
    axis: str | None,
    source: str | None,
    license_id: str | None,
) -> tuple[str | None, str | None]:
    """Fill gaps from Sellmeier rows (axis variant first, then the base key)."""
    candidates = [name]
    if axis is not None:
        suffix = "_or" if axis.lower() in ("ordinary", "o") else "_er"
        candidates.insert(0, f"{name}{suffix}")
    for candidate in candidates:
        row = db.get_sellmeier(candidate)  # type: ignore[arg-type]
        if not row:
            continue
        if source is None:
            source = _text(row.get("source"))
        if license_id is None:
            license_id = _text(row.get("license"))
    return source, license_id


def _fill_from_spec(
    db: MaterialsDatabase, name: str, source: str | None, license_id: str | None
) -> tuple[str | None, str | None]:
    """Fill remaining gaps from the material-spec row."""
    spec = db.get_material(name)
    if not spec:
        return source, license_id
    if source is None:
        source = _text(spec.get("references"))
    if license_id is None:
        license_id = _text(spec.get("license"))
    return source, license_id


def _fill_from_nk_sources(
    db: MaterialsDatabase, name: str, source: str | None, license_id: str | None
) -> tuple[str | None, str | None]:
    """Fill from the first licensed tabulated-data source."""
    for source_key in db.list_nk_sources(name):  # type: ignore[arg-type]
        prov = db.get_provenance(source_key)
        if not prov or not prov.get("license"):
            continue
        license_id = str(prov["license"])
        if source is None:
            source = _text(prov.get("citation"))
        break
    return source, license_id


def _lookup_metadata(name: str, axis: str | None) -> tuple[str | None, str | None]:
    """Best-effort ``(source, license)`` lookup in the bundled database.

    Returns ``(None, None)`` when no row (or no provenance string) is found;
    provenance is metadata, so a lookup miss must not break ``material()``.
    Falls back through direct provenance, Sellmeier rows, the material spec,
    and tabulated-data sources in that order.
    """
    try:
        from .data import MaterialsDatabase

        db = MaterialsDatabase()
        direct = db.get_provenance(name) or {}
        source = _text(direct.get("citation"))
        license_id = _text(direct.get("license"))
        source, license_id = _fill_from_sellmeier(db, name, axis, source, license_id)
        source, license_id = _fill_from_spec(db, name, source, license_id)
        if license_id is None:
            source, license_id = _fill_from_nk_sources(db, name, source, license_id)
        return source, license_id
    except (sqlite3.Error, ValueError, KeyError, TypeError, OSError):
        return None, None


def material(name: str, *, axis: str | None = None, n_points: int = 200) -> Material:
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

    Raises
    ------
    ValueError
        With an actionable message when the material is unknown, the axis is
        unknown/unavailable, or the database is missing, empty or corrupt.
    """
    index = RefractiveIndex.from_material_database(name, n_points=n_points, axis=axis)
    source, license_id = _lookup_metadata(name, axis)
    return Material(name=name, index=index, source=source, license=license_id)
