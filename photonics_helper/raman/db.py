"""RamanDatabase - SQLite CRUD for Raman material data."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass

from ..phonon import PhononMode

if TYPE_CHECKING:
    from ..materials import NKMaterial

from .reference import RAMAN_MATERIALS, THORLABS_SUBSTRATE_MATERIALS

# Default/sentinel licences per data table. The sentinels point at the row's
# citation column rather than asserting a blanket licence for literature data;
# see docs/data-schema.md. Concrete identifiers (e.g. "CC0-1.0") are only used
# where the source genuinely declares them.
_LICENSE_DEFAULTS: dict[str, str] = {
    "nk_data": "CC0-1.0",
    "sellmeier": "see-source-publication",
    "raman_specs": "see-references",
    "phonon_modes": "see-note",
}

# ─── RamanDatabase ────────────────────────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanDatabase:
    """SQLite database for Raman material data.

    Schema:
    CREATE TABLE raman_specs (
        name              TEXT PRIMARY KEY,
        crystal           TEXT,
        bandgap_eV        REAL,
        n2                REAL,
        raman_shift_cm    REAL,
        raman_linewidth_cm REAL,
        fR                REAL,
        gain_coeff        REAL,
        tau1              REAL,
        tau2              REAL,
        lo_phonon_cm      REAL,
        to_phonon_cm      REAL,
        references        TEXT
    );

    CREATE TABLE nk_data (
        material       TEXT REFERENCES raman_specs(name),
        wavelength_um  REAL,
        n              REAL,
        k              REAL,
        source         TEXT,  -- material–author provenance key (e.g. "si-green")
        citation       TEXT  -- full reference the (n, k) values were taken from
    );
    """

    db_path: Path | None = None

    def __post_init__(self):
        if self.db_path is None:
            # Try bundled DB first, then next to module
            bundled = Path(__file__).parent.parent / "materials.db"
            if bundled.exists():
                object.__setattr__(self, "db_path", bundled)
            else:
                # Fall back to a user-home database (created on first use)
                object.__setattr__(
                    self,
                    "db_path",
                    Path.home() / ".photonics_helper" / "materials.db",
                )
        # NOTE: no connection/schema work here — see _ensure_initialized().

    @property
    def _path(self) -> Path:
        """Resolved database path (always set by ``__post_init__``)."""
        assert self.db_path is not None
        return self.db_path

    def _ensure_initialized(self) -> None:
        """Create/migrate the schema once, on first use."""
        if getattr(self, "_initialized", False):
            return
        # Set the flag before _init_db: the seeding path queries the database,
        # and must not recurse back into this method.
        object.__setattr__(self, "_initialized", True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Open the database, initialising the schema on first use."""
        self._ensure_initialized()
        return sqlite3.connect(self._path)

    def _init_db(self):
        """Create tables if they don't exist (idempotent migration)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raman_specs (
                name              TEXT PRIMARY KEY,
                crystal           TEXT,
                bandgap_eV        REAL,
                n2                REAL,
                raman_shift_cm    REAL,
                raman_linewidth_cm REAL,
                fR                REAL,
                gain_coeff        REAL,
                tau1              REAL,
                tau2              REAL,
                lo_phonon_cm      REAL,
                to_phonon_cm      REAL,
                "references"        TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nk_data (
                material       TEXT REFERENCES raman_specs(name),
                wavelength_um  REAL,
                n              REAL,
                k              REAL,
                source         TEXT,
                citation       TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sellmeier (
                material       TEXT PRIMARY KEY REFERENCES raman_specs(name),
                form           TEXT CHECK(form IN ('standard', 'alt')),
                a0             REAL,
                coefficients   TEXT,
                wavelengths    TEXT,
                valid_from_um  REAL,
                valid_to_um    REAL,
                source         TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phonon_modes (
                material           TEXT REFERENCES raman_specs(name),
                shift_cm           REAL,
                linewidth_cm       REAL,
                symmetry           TEXT,
                relative_strength  REAL DEFAULT 1.0,
                lo_phonon_cm       REAL,
                to_phonon_cm       REAL,
                note               TEXT,
                PRIMARY KEY (material, shift_cm, symmetry)
            )
        """)

        # Migration: add provenance columns to nk_data. Older/already-shipped
        # DBs lack these columns, so add them in place and backfill existing
        # rows with NULL.
        cursor.execute("PRAGMA table_info(nk_data)")
        cols = [r[1] for r in cursor.fetchall()]
        if "source" not in cols:
            cursor.execute("ALTER TABLE nk_data ADD COLUMN source TEXT")
        if "citation" not in cols:
            cursor.execute("ALTER TABLE nk_data ADD COLUMN citation TEXT")

        # Licence column on every data table, backfilled with a documented
        # default/sentinel (see docs/data-schema.md).
        for table, default_license in _LICENSE_DEFAULTS.items():
            cursor.execute(f"PRAGMA table_info({table})")
            table_cols = [r[1] for r in cursor.fetchall()]
            if "license" not in table_cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN license TEXT")
            cursor.execute(
                f"UPDATE {table} SET license = ? WHERE license IS NULL",
                (default_license,),
            )

        # Provenance registry, backfilled from the existing nk_data so that an
        # already-shipped database upgrades without the external manifest
        # (nk_datasets/ is not in the repository).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provenance (
                source_key TEXT PRIMARY KEY,
                kind       TEXT,
                citation   TEXT,
                doi        TEXT,
                url        TEXT,
                license    TEXT
            )
        """)
        cursor.execute(
            """
            INSERT OR IGNORE INTO provenance (source_key, kind, citation, license)
            SELECT source, 'nk', MAX(citation), ?
            FROM nk_data
            WHERE source IS NOT NULL
            GROUP BY source
            """,
            (_LICENSE_DEFAULTS["nk_data"],),
        )

        conn.commit()
        conn.close()
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        """Populate an empty user-home DB from RAMAN_MATERIALS (not test/temp paths)."""
        if self.list_materials():
            return
        db_path = Path(self._path).resolve()
        bundled = Path(__file__).parent.parent / "materials.db"
        if bundled.exists() and db_path == bundled.resolve():
            return
        home_db = Path.home() / ".photonics_helper" / "materials.db"
        if db_path != home_db.resolve():
            return
        for data in RAMAN_MATERIALS.values():
            self.add_material(data)
        for data in THORLABS_SUBSTRATE_MATERIALS.values():
            self.add_material(data)
        # Phonon modes live in the same database, so seed them for the
        # empty-home-DB fallback path too.
        self.seed_phonon_data()

    def add_material(self, spec: dict) -> None:
        """INSERT or REPLACE a material entry.

        Parameters
        ----------
        spec : dict with material properties
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO raman_specs
            (name, crystal, bandgap_eV, n2, raman_shift_cm, raman_linewidth_cm,
             fR, gain_coeff, tau1, tau2, lo_phonon_cm, to_phonon_cm, "references",
             license)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                spec.get("name"),
                spec.get("crystal"),
                spec.get("bandgap_eV"),
                spec.get("n2"),
                spec.get("raman_shift_cm"),
                spec.get("raman_linewidth_cm"),
                spec.get("fR"),
                spec.get("gain_coeff"),
                spec.get("tau1"),
                spec.get("tau2"),
                spec.get("lo_phonon_cm"),
                spec.get("to_phonon_cm"),
                spec.get("references"),
                spec.get("license", _LICENSE_DEFAULTS["raman_specs"]),
            ),
        )

        conn.commit()
        conn.close()

    def get_material(self, name: str) -> dict | None:
        """SELECT one material by name.

        Parameters
        ----------
        name : Material name

        Returns
        -------
        dict with material properties, or None if not found
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM raman_specs WHERE name = ?", (name,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return dict(row)
        return None

    def list_materials(self) -> list[str]:
        """SELECT DISTINCT name FROM raman_specs.

        Returns
        -------
        list of material names
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM raman_specs ORDER BY name")
        rows = cursor.fetchall()

        conn.close()
        return [row[0] for row in rows]

    def get_provenance(self, source_key: str) -> dict | None:
        """Fetch the provenance record for a source key.

        Parameters
        ----------
        source_key : the ``source`` key used by ``nk_data`` rows
            (e.g. ``"si-green"``).

        Returns
        -------
        dict | None
            The provenance record, or ``None`` when the key is unknown.
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM provenance WHERE source_key = ?", (source_key,)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row is not None else None

    def list_provenance(self) -> list[dict]:
        """List every provenance record, ordered by source key."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM provenance ORDER BY source_key")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def add_phonon_mode(
        self, material: str, mode: PhononMode, license: str | None = None
    ) -> None:
        """Insert a phonon mode for a material.

        Parameters
        ----------
        material : str
            Material name.
        mode : PhononMode
            Phonon mode to add.
        license : str | None
            Optional licence; defaults to the documented ``phonon_modes``
            sentinel (``see-note``).
        """
        from ..phonon import PhononMode

        if not isinstance(mode, PhononMode):
            raise TypeError("mode must be a PhononMode instance")

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO phonon_modes
            (material, shift_cm, linewidth_cm, symmetry, relative_strength,
             lo_phonon_cm, to_phonon_cm, note, license)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                material,
                mode.shift_cm.as_1_cm,
                mode.linewidth_cm.as_1_cm,
                mode.symmetry,
                mode.relative_strength,
                mode.lo_phonon_cm.as_1_cm if mode.lo_phonon_cm is not None else None,
                mode.to_phonon_cm.as_1_cm if mode.to_phonon_cm is not None else None,
                mode.note,
                license or _LICENSE_DEFAULTS["phonon_modes"],
            ),
        )

        conn.commit()
        conn.close()

    def get_phonon_modes(self, material: str) -> list[PhononMode]:
        """Get all phonon modes for a material.

        Parameters
        ----------
        material : str
            Material name.

        Returns
        -------
        list[PhononMode]
            Phonon modes for the material.
        """
        from ..phonon import PhononMode

        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM phonon_modes WHERE material = ? ORDER BY shift_cm",
            (material,),
        )
        rows = cursor.fetchall()

        conn.close()

        modes = []
        for row in rows:
            modes.append(
                PhononMode(
                    shift_cm=row["shift_cm"],
                    linewidth_cm=row["linewidth_cm"],
                    symmetry=row["symmetry"],
                    relative_strength=row["relative_strength"],
                    lo_phonon_cm=row["lo_phonon_cm"],
                    to_phonon_cm=row["to_phonon_cm"],
                    note=row["note"],
                )
            )
        return modes

    def list_phonon_materials(self) -> list[str]:
        """Get materials with phonon mode data.

        Returns
        -------
        list[str]
            Material names that have phonon_modes entries.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT material FROM phonon_modes ORDER BY material")
        rows = cursor.fetchall()

        conn.close()
        return [row[0] for row in rows]

    def seed_phonon_data(self) -> int:
        """Seed PHONON_MATERIALS data into the phonon_modes table.

        Returns
        -------
        int
            Number of modes seeded.
        """
        from ..phonon import PHONON_MATERIALS

        count = 0
        for material, modes in PHONON_MATERIALS.items():
            for mode in modes:
                self.add_phonon_mode(material, mode)
                count += 1
        return count

    def update_material(self, name: str, **kwargs) -> None:
        """UPDATE material fields.

        Parameters
        ----------
        name : Material name
        **kwargs : Fields to update
        """
        if not kwargs:
            return

        conn = self._connect()
        cursor = conn.cursor()

        # Build UPDATE query dynamically
        columns = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [name]

        cursor.execute(f"UPDATE raman_specs SET {columns} WHERE name = ?", values)

        conn.commit()
        conn.close()

    def delete_material(self, name: str) -> None:
        """DELETE material by name.

        Parameters
        ----------
        name : Material name
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM raman_specs WHERE name = ?", (name,))
        cursor.execute("DELETE FROM nk_data WHERE material = ?", (name,))

        conn.commit()
        conn.close()

    def add_nk_data(
        self,
        material: NKMaterial,
        wl_um: float,
        n: float,
        k: float,
        source: str | None = None,
        citation: str | None = None,
        license: str | None = None,
    ) -> None:
        """INSERT nk data point.

        Parameters
        ----------
        material : Canonical material name (must exist in raman_specs)
        wl_um : Wavelength in μm
        n : Real refractive index
        k : Extinction coefficient
        source : Optional ``material–author`` provenance key
            (e.g. ``"si-green"``). When omitted the row is unattributed.
        citation : Optional full reference the (n, k) values were taken from.
            Stored for traceability; when omitted the row has no citation.
        license : Optional licence identifier; defaults to the documented
            ``nk_data`` licence (``CC0-1.0`` from refractiveindex.info).
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO nk_data
            (material, wavelength_um, n, k, source, citation, license)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                material,
                wl_um,
                n,
                k,
                source,
                citation,
                license or _LICENSE_DEFAULTS["nk_data"],
            ),
        )

        conn.commit()
        conn.close()

    @overload
    def get_nk_data(
        self, material: NKMaterial, with_source: Literal[False] = ...
    ) -> tuple[NDArray, NDArray, NDArray]: ...

    @overload
    def get_nk_data(
        self, material: NKMaterial, with_source: Literal[True]
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]: ...

    def get_nk_data(
        self, material: NKMaterial, with_source: bool = False
    ) -> tuple[NDArray, NDArray, NDArray] | tuple[NDArray, NDArray, NDArray, NDArray]:
        """SELECT nk data for a material.

        Parameters
        ----------
        material : Material name
        with_source : If True, also return a source array alongside
            (wavelength_um, n, k, source); otherwise return the legacy
            3-tuple (wavelength_um, n, k).

        Returns
        -------
        (wavelength_um, n, k) or (wavelength_um, n, k, source) as numpy arrays
        """
        conn = self._connect()
        cursor = conn.cursor()

        cols = "wavelength_um, n, k"
        if with_source:
            cols += ", source"
        cursor.execute(
            f"SELECT {cols} FROM nk_data WHERE material = ? ORDER BY wavelength_um",
            (material,),
        )
        rows = cursor.fetchall()

        conn.close()

        if not rows:
            empty3 = (np.array([]), np.array([]), np.array([]))
            if not with_source:
                return empty3
            return empty3 + (np.array([]),)

        wl = np.array([r[0] for r in rows])
        n = np.array([r[1] for r in rows])
        k = np.array([r[2] for r in rows])
        if not with_source:
            return wl, n, k

        source = np.array([r[3] if r[3] is not None else "" for r in rows])
        return wl, n, k, source

    def list_nk_sources(self, material: NKMaterial) -> list[str]:
        """Return the distinct ``source`` keys stored for a material.

        Parameters
        ----------
        material : Canonical material name

        Returns
        -------
        sorted list of distinct source keys (may include None values)
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT source FROM nk_data WHERE material = ? ORDER BY source",
            (material,),
        )
        rows = cursor.fetchall()
        conn.close()

        sources = [r[0] for r in rows if r[0] is not None]
        return sorted(set(sources))

    def list_nk_citations(self, material: NKMaterial) -> dict[str, str]:
        """Return the citation for every stored source of a material.

        Parameters
        ----------
        material : Canonical material name

        Returns
        -------
        dict mapping each ``source`` key to its full citation text
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT source, citation FROM nk_data WHERE material = ? "
            "ORDER BY source",
            (material,),
        )
        rows = cursor.fetchall()
        conn.close()

        return {
            r[0]: (r[1] if r[1] is not None else "") for r in rows if r[0] is not None
        }

    def clear_all_tabulated_nk(self) -> int:
        """Delete every tabulated nk_data row (attributed by a ``source``).

        Used by the manifest seeder to make re-seeding idempotent: the
        tabulated ``nk_data`` table is owned entirely by the manifest, so all
        attributed rows are cleared and re-inserted on each ``seed_db.py --nk``
        run. Sellmeier data (a separate table) is untouched.

        Returns
        -------
        Number of rows removed.
        """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM nk_data WHERE source IS NOT NULL")
        removed = cursor.rowcount
        conn.commit()
        conn.close()
        return removed

    def get_nk_by_source(
        self, material: NKMaterial, source: str
    ) -> tuple[NDArray, NDArray, NDArray]:
        """SELECT tabulated nk data for a material attributed to a source.

        Parameters
        ----------
        material : Canonical material name
        source : ``material–author`` provenance key (e.g. ``"si-green"``)

        Returns
        -------
        (wavelength_um, n, k) as numpy arrays (empty if none match)
        """
        # Case-insensitive source match, done in Python to avoid SQLite's
        # unreliable case-folding. Per-material source counts are small, so
        # fetching the material's rows is cheap.
        wl_all, n_all, k_all, source_all = self.get_nk_data(material, with_source=True)
        if len(wl_all) == 0:
            return np.array([]), np.array([]), np.array([])

        mask = np.array(
            [str(s).lower() == source.lower() for s in source_all], dtype=bool
        )
        wl, n, k = wl_all[mask], n_all[mask], k_all[mask]
        if len(wl) == 0:
            return np.array([]), np.array([]), np.array([])
        return wl, n, k

    def list_nk_dataset_summaries(self) -> list[dict]:
        """Return a per-dataset summary of every tabulated ``nk_data`` entry.

        One row per distinct ``(material, source)`` pair. Used by the n/k
        dataset browser to list what is available. Returns dicts with keys:
        ``material``, ``source``, ``wl_min_um``, ``wl_max_um``, ``n_points``,
        ``n_central``, ``k_central`` and ``central_wl_um``. Central-wavelength
        n/k is obtained by linear interpolation onto a fine grid.
        """
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT material, source,
                   MIN(wavelength_um) AS wl_min, MAX(wavelength_um) AS wl_max,
                   COUNT(*)           AS n_points
            FROM nk_data
            WHERE source IS NOT NULL
            GROUP BY material, source
            ORDER BY material, source
            """
        ).fetchall()
        conn.close()

        summaries: list[dict] = []
        for material, source, wl_min, wl_max, n_points in rows:
            central = 0.5 * (wl_min + wl_max)
            n_central, k_central = None, None
            if n_points > 1 and wl_min > 0:
                try:
                    gw, gn, gk, _gs = self.get_nk_data(material, with_source=True)
                    grid = np.linspace(wl_min, wl_max, 4096)
                    sel = np.clip(np.searchsorted(gw, grid), 1, len(gw) - 1)
                    lo = sel - 1
                    w = (grid - gw[lo]) / (gw[sel] - gw[lo])
                    n_at = gn[lo] + w * (gn[sel] - gn[lo])
                    k_at = gk[lo] + w * (gk[sel] - gk[lo])
                    idx = int(np.argmin(np.abs(gw - central)))
                    n_central = float(n_at[idx])
                    k_central = float(k_at[idx])
                except Exception:
                    pass
            summaries.append(
                {
                    "material": material,
                    "source": source,
                    "wl_min_um": float(wl_min),
                    "wl_max_um": float(wl_max),
                    "n_points": int(n_points),
                    "central_wl_um": float(central),
                    "n_central": n_central,
                    "k_central": k_central,
                }
            )
        return summaries

    def add_sellmeier(
        self,
        material: NKMaterial,
        form: str,
        a0: float,
        coefficients: list[float],
        wavelengths: list[float],
        valid_from_um: float,
        valid_to_um: float,
        source: str,
        license: str | None = None,
    ) -> None:
        """INSERT Sellmeier coefficients.

        Parameters
        ----------
        material : Material name
        form : 'standard' or 'alt'
        a0 : Constant term A₀
        coefficients : List of Aᵢ coefficients
        wavelengths : List of Bᵢ wavelengths
        valid_from_um : Valid range start (μm)
        valid_to_um : Valid range end (μm)
        source : Citation
        license : Optional licence; defaults to the documented ``sellmeier``
            sentinel (``see-source-publication``).
        """
        import json

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO sellmeier (material, form, a0, coefficients, wavelengths, valid_from_um, valid_to_um, source, license)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                material,
                form,
                a0,
                json.dumps(coefficients),
                json.dumps(wavelengths),
                valid_from_um,
                valid_to_um,
                source,
                license or _LICENSE_DEFAULTS["sellmeier"],
            ),
        )

        conn.commit()
        conn.close()

    def get_sellmeier(self, material: NKMaterial) -> dict | None:
        """SELECT Sellmeier coefficients for a material.

        Parameters
        ----------
        material : Material name

        Returns
        -------
        dict with keys: material, form, a0, coefficients, wavelengths, valid_from_um, valid_to_um, source
        Or None if not found
        """
        import json

        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sellmeier WHERE material = ?", (material,))
        row = cursor.fetchone()

        conn.close()

        if not row:
            return None

        result = dict(row)
        result["coefficients"] = json.loads(result["coefficients"])
        result["wavelengths"] = json.loads(result["wavelengths"])

        return result

    def search_materials(self, query: str) -> list[dict[str, float]]:
        """LIKE search across name, crystal, references.

        Parameters
        ----------
        query : Search string

        Returns
        -------
        list of matching materials
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        search_pattern = f"%{query}%"
        cursor.execute(
            """
            SELECT * FROM raman_specs
            WHERE name LIKE ? OR crystal LIKE ? OR "references" LIKE ?
            ORDER BY name
        """,
            (search_pattern, search_pattern, search_pattern),
        )
        rows = cursor.fetchall()

        conn.close()
        return [dict(row) for row in rows]

    def export_to_dict(self) -> dict[str, dict]:
        """Export all materials as a Python dict.

        Returns
        -------
        dict mapping material names to property dicts
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM raman_specs ORDER BY name")
        rows = cursor.fetchall()

        conn.close()
        return {row["name"]: dict(row) for row in rows}

    def import_from_dict(self, data: dict[str, dict]) -> None:
        """Bulk INSERT from a Python dict.

        Parameters
        ----------
        data : dict mapping material names to property dicts
        """
        conn = self._connect()
        cursor = conn.cursor()

        for name, spec in data.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO raman_specs
                (name, crystal, bandgap_eV, n2, raman_shift_cm, raman_linewidth_cm,
                 fR, gain_coeff, tau1, tau2, lo_phonon_cm, to_phonon_cm, "references")
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    spec.get("name", name),
                    spec.get("crystal"),
                    spec.get("bandgap_eV"),
                    spec.get("n2"),
                    spec.get("raman_shift_cm"),
                    spec.get("raman_linewidth_cm"),
                    spec.get("fR"),
                    spec.get("gain_coeff"),
                    spec.get("tau1"),
                    spec.get("tau2"),
                    spec.get("lo_phonon_cm"),
                    spec.get("to_phonon_cm"),
                    spec.get("references"),
                ),
            )

        conn.commit()
        conn.close()
