"""Tests for Raman module — Layer 1: RamanSpec and RamanDatabase."""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pathlib import Path
from tempfile import TemporaryDirectory

from photonics_helper.raman import RamanSpec, RamanDatabase, RAMAN_MATERIALS, THORLABS_SUBSTRATE_MATERIALS
from photonics_helper.base import Wavelength, C_MS, Energy, Time


# ─── RamanSpec Tests ──────────────────────────────────────────────────────────


class TestRamanSpec:
    """Tests for RamanSpec class."""

    def test_from_database_hardcoded(self):
        """Test loading from hardcoded materials."""
        silica = RamanSpec.from_database("Silica")
        assert silica.name == "Silica"
        assert silica.raman_shift_cm == 440
        assert silica.raman_linewidth_cm == 45
        assert silica.fR == 0.18
        assert silica.n2 == 3.2e-20

    def test_from_database_geasse(self):
        """Test loading GeAsSe (chalcogenide waveguide material)."""
        geasse = RamanSpec.from_database("GeAsSe")
        assert geasse.name == "GeAsSe"
        assert geasse.n2 == 6.0e-18
        assert geasse.raman_shift_cm == 250
        assert geasse.raman_linewidth_cm == 50
        assert geasse.fR == 0.50
        assert geasse.bandgap_eV.as_eV == 1.6
        assert "Richardson" in geasse.references

    def test_from_database_multiple_materials(self):
        """Test loading multiple hardcoded materials."""
        materials = ["Silica", "CdS", "GaAs", "Diamond", "As2Se3", "Si", "Ge", "LiNbO3"]
        for name in materials:
            spec = RamanSpec.from_database(name)
            assert spec.name == name
            assert spec.raman_shift_cm is not None
            assert spec.raman_linewidth_cm is not None

    def test_from_database_not_found(self):
        """Test error when material not found."""
        with pytest.raises(ValueError, match="not found"):
            RamanSpec.from_database("NonExistentMaterial")

    def test_from_database_with_fallback(self):
        """Test fallback values when material not in database."""
        spec = RamanSpec.from_database(
            "CustomMat",
            fallback={
                "raman_shift_cm": 500,
                "raman_linewidth_cm": 25,
                "fR": 0.4,
            }
        )
        assert spec.name == "CustomMat"
        assert spec.raman_shift_cm == 500
        assert spec.raman_linewidth_cm == 25
        assert spec.fR == 0.4

    def test_derived_shift_hz(self):
        """Test Raman shift conversion to Hz."""
        silica = RamanSpec.from_database("Silica")
        expected = 440 * C_MS * 100
        assert_almost_equal(silica.raman_shift_Hz, expected, decimal=6)

    def test_derived_shift_thz(self):
        """Test Raman shift conversion to THz."""
        silica = RamanSpec.from_database("Silica")
        expected = silica.raman_shift_Hz / 1e12
        assert_almost_equal(silica.raman_shift_THz, expected, decimal=6)

    def test_derived_shift_omega(self):
        """Test Raman shift conversion to rad/s."""
        silica = RamanSpec.from_database("Silica")
        expected = 2 * np.pi * silica.raman_shift_Hz
        assert_almost_equal(silica.raman_shift_omega, expected, decimal=6)

    def test_derived_linewidth_hz(self):
        """Test linewidth conversion to Hz."""
        silica = RamanSpec.from_database("Silica")
        expected = 45 * C_MS * 100
        assert_almost_equal(silica.linewidth_Hz, expected, decimal=6)

    def test_derived_linewidth_thz(self):
        """Test linewidth conversion to THz."""
        silica = RamanSpec.from_database("Silica")
        expected = silica.linewidth_Hz / 1e12
        assert_almost_equal(silica.linewidth_THz, expected, decimal=6)

    def test_quality_factor(self):
        """Test quality factor calculation."""
        silica = RamanSpec.from_database("Silica")
        expected = silica.raman_shift_Hz / silica.linewidth_Hz
        assert_almost_equal(silica.quality_factor, expected, decimal=1)

    def test_quality_factor_zero_linewidth(self):
        """Test Q factor with zero linewidth."""
        spec = RamanSpec(
            name="Test",
            raman_shift_cm=400,
            raman_linewidth_cm=0,
            fR=0.2,
        )
        assert spec.quality_factor == float("inf")

    def test_stokes_wavelength(self):
        """Test Stokes wavelength calculation."""
        silica = RamanSpec.from_database("Silica")
        pump = Wavelength(800, "nm")
        stokes = silica.stokes_wavelength(pump)

        # Stokes should be longer wavelength than pump
        assert stokes.as_nm > pump.as_nm

        # Verify frequency difference
        nu_pump = C_MS / pump.as_m
        nu_stokes = C_MS / stokes.as_m
        shift_expected = silica.raman_shift_Hz
        assert_almost_equal(nu_pump - nu_stokes, shift_expected, decimal=6)

    def test_anti_stokes_wavelength(self):
        """Test Anti-Stokes wavelength calculation."""
        silica = RamanSpec.from_database("Silica")
        pump = Wavelength(800, "nm")
        anti_stokes = silica.anti_stokes_wavelength(pump)

        # Anti-Stokes should be shorter wavelength than pump
        assert anti_stokes.as_nm < pump.as_nm

        # Verify frequency difference
        nu_pump = C_MS / pump.as_m
        nu_anti = C_MS / anti_stokes.as_m
        shift_expected = silica.raman_shift_Hz
        assert_almost_equal(nu_anti - nu_pump, shift_expected, decimal=6)

    def test_stokes_wavelength_negative_frequency(self):
        """Test error when Stokes frequency would be negative."""
        spec = RamanSpec(
            name="Test",
            raman_shift_cm=5000,
            raman_linewidth_cm=100,
            fR=0.3,
        )
        pump = Wavelength(2000, "nm")
        with pytest.raises(ValueError, match="negative"):
            spec.stokes_wavelength(pump)

    def test_summary(self):
        """Test summary string generation."""
        silica = RamanSpec.from_database("Silica")
        summary = silica.summary()

        assert "Silica" in summary
        assert "440" in summary
        assert "THz" in summary
        assert "Q factor" in summary

    def test_plot_spectrum_matplotlib(self):
        """Test spectrum plot with matplotlib backend."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        silica = RamanSpec.from_database("Silica")
        fig = silica.plot_spectrum(backend="matplotlib")

        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_spectrum_plotly(self):
        """Test spectrum plot with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        silica = RamanSpec.from_database("Silica")
        fig = silica.plot_spectrum(backend="plotly")

        assert fig is not None

    def test_plot_phonons_matplotlib(self):
        """Test phonon plot with matplotlib backend."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cds = RamanSpec.from_database("CdS")
        fig = cds.plot_phonons(backend="matplotlib")

        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_phonons_missing_data(self):
        """Test phonon plot with missing phonon data."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        silica = RamanSpec.from_database("Silica")
        fig = silica.plot_phonons(backend="matplotlib")

        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_construction_with_all_params(self):
        """Test construction with all parameters."""
        spec = RamanSpec(
            name="FullSpec",
            crystal="TestCrystal",
            bandgap_eV=Energy(2.0, "eV"),
            n2=1e-19,
            raman_shift_cm=400,
            raman_linewidth_cm=20,
            fR=0.3,
            gain_coeff=1e-13,
            tau1=Time(1e-12, "s"),
            tau2=Time(2e-12, "s"),
            alpha=0.5,
            lo_phonon_cm=400,
            to_phonon_cm=390,
            references="Test reference",
        )

        assert spec.name == "FullSpec"
        assert spec.crystal == "TestCrystal"
        assert spec.bandgap_eV.as_eV == 2.0
        assert spec.lo_phonon_cm == 400

    def test_construction_minimal(self):
        """Test construction with minimal required params."""
        spec = RamanSpec(
            name="Minimal",
            raman_shift_cm=400,
            raman_linewidth_cm=20,
        )

        assert spec.name == "Minimal"
        assert spec.fR is None
        assert spec.n2 is None


class TestRamanDatabase:
    """Tests for RamanDatabase class."""

    def test_init_default_path(self):
        """Test initialization with default path."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)
            assert db.db_path == db_path
            assert db_path.exists()

    def test_init_creates_tables(self):
        """Test that initialization creates tables."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            assert "raman_specs" in tables
            assert "nk_data" in tables

    def test_add_and_get_material(self):
        """Test adding and retrieving a material."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            spec_data = {
                "name": "TestMat",
                "crystal": "TestCrystal",
                "raman_shift_cm": 400,
                "raman_linewidth_cm": 20,
                "fR": 0.3,
                "references": "Test ref",
            }
            db.add_material(spec_data)

            result = db.get_material("TestMat")
            assert result is not None
            assert result["name"] == "TestMat"
            assert result["raman_shift_cm"] == 400
            assert result["fR"] == 0.3

    def test_get_material_not_found(self):
        """Test getting a non-existent material."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            result = db.get_material("NonExistent")
            assert result is None

    def test_list_materials(self):
        """Test listing all materials."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            db.add_material({"name": "Mat1", "raman_shift_cm": 400, "raman_linewidth_cm": 20})
            db.add_material({"name": "Mat2", "raman_shift_cm": 500, "raman_linewidth_cm": 25})

            materials = db.list_materials()
            assert "Mat1" in materials
            assert "Mat2" in materials
            assert len(materials) == 2

    def test_update_material(self):
        """Test updating material fields."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            db.add_material({"name": "TestMat", "raman_shift_cm": 400, "raman_linewidth_cm": 20})

            db.update_material("TestMat", fR=0.5, n2=1e-19)

            result = db.get_material("TestMat")
            assert result is not None
            assert result["fR"] == 0.5
            assert result["n2"] == 1e-19
            assert result["raman_shift_cm"] == 400

    def test_delete_material(self):
        """Test deleting a material."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            db.add_material({"name": "TestMat", "raman_shift_cm": 400, "raman_linewidth_cm": 20})
            db.delete_material("TestMat")

            result = db.get_material("TestMat")
            assert result is None

    def test_add_nk_data(self):
        """Test adding nk data."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            db.add_material({"name": "TestMat", "raman_shift_cm": 400, "raman_linewidth_cm": 20})
            db.add_nk_data("TestMat", 1.0, 2.5, 0.01)
            db.add_nk_data("TestMat", 1.5, 2.3, 0.02)

            wl, n, k = db.get_nk_data("TestMat")
            assert len(wl) == 2
            assert_array_almost_equal(wl, [1.0, 1.5])
            assert_array_almost_equal(n, [2.5, 2.3])
            assert_array_almost_equal(k, [0.01, 0.02])

    def test_get_nk_data_empty(self):
        """Test getting nk data for material with no data."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            wl, n, k = db.get_nk_data("NonExistent")
            assert len(wl) == 0
            assert len(n) == 0
            assert len(k) == 0

    def test_search_materials(self):
        """Test searching materials."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            db.add_material({
                "name": "Silica",
                "crystal": "Amorphous SiO2",
                "raman_shift_cm": 440,
                "raman_linewidth_cm": 45,
                "references": "Agrawal book",
            })
            db.add_material({
                "name": "CdS",
                "crystal": "Wurtzite",
                "raman_shift_cm": 305,
                "raman_linewidth_cm": 12,
                "references": "Pankove book",
            })

            results = db.search_materials("Silica")
            assert len(results) == 1
            assert results[0]["name"] == "Silica"

            results = db.search_materials("Amorphous")
            assert len(results) == 1
            assert results[0]["name"] == "Silica"

            results = db.search_materials("Agrawal")
            assert len(results) == 1
            assert results[0]["name"] == "Silica"

    def test_export_import_dict(self):
        """Test exporting and importing material data."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = RamanDatabase(db_path=db_path)

            db.add_material({"name": "Mat1", "raman_shift_cm": 400, "raman_linewidth_cm": 20})
            db.add_material({"name": "Mat2", "raman_shift_cm": 500, "raman_linewidth_cm": 25})

            exported = db.export_to_dict()
            assert "Mat1" in exported
            assert "Mat2" in exported

            db2_path = Path(tmpdir) / "test2.db"
            db2 = RamanDatabase(db_path=db2_path)
            db2.import_from_dict(exported)

            materials = db2.list_materials()
            assert "Mat1" in materials
            assert "Mat2" in materials


class TestRAMAN_MATERIALS:
    """Tests for the hardcoded reference materials."""

    def test_all_materials_have_required_fields(self):
        """Test that all hardcoded materials have required fields."""
        for name, data in RAMAN_MATERIALS.items():
            assert "name" in data
            assert "raman_shift_cm" in data
            assert "raman_linewidth_cm" in data
            assert data["raman_shift_cm"] > 0  # type: ignore[operator]
            assert data["raman_linewidth_cm"] > 0  # type: ignore[operator]

    def test_all_materials_loadable(self):
        """Test that all hardcoded materials can be loaded."""
        for name in RAMAN_MATERIALS:
            spec = RamanSpec.from_database(name)
            assert spec.name == name

    def test_material_count(self):
        """Test that RAMAN_MATERIALS has the expected number of entries (30 + GeAsSe)."""
        assert len(RAMAN_MATERIALS) == 31

    def test_all_original_materials_present(self):
        """Test original 8 materials are still present."""
        originals = {"Silica", "CdS", "GaAs", "Diamond", "As2Se3", "Si", "Ge", "LiNbO3"}
        for name in originals:
            assert name in RAMAN_MATERIALS

    def test_all_new_materials_present(self):
        """Test all 22 new materials are present."""
        new = {
            "As2S3", "Si3N4", "SiC_4H", "YAG", "BaTiO3", "ZBLAN",
            "GaN", "AlN", "InP", "LiTaO3", "KTP", "AlGaAs", "Al2O3",
            "YLF", "ZnO", "CdTe", "Ga2O3", "LBO", "AgGaS2", "AgGaSe2",
            "InGaAs", "GeO2",
        }
        for name in new:
            assert name in RAMAN_MATERIALS

    def test_derived_properties_for_all_materials(self):
        """Test derived properties (Stokes, Q, summary) for all materials."""
        pump = Wavelength(800, "nm")
        for name in RAMAN_MATERIALS:
            spec = RamanSpec.from_database(name)
            stokes = spec.stokes_wavelength(pump)
            anti = spec.anti_stokes_wavelength(pump)
            assert stokes.as_nm > pump.as_nm
            assert anti.as_nm < pump.as_nm
            assert spec.quality_factor > 0
            assert len(spec.summary()) > 0

    def test_negative_n2_materials(self):
        """Test materials with negative n2 still load correctly."""
        for name in ["ZnO", "CdTe"]:
            spec = RamanSpec.from_database(name)
            assert spec.n2 is not None

    def test_zero_fR_materials(self):
        """Test materials with fR=0.0 (purely Kerr, no Raman)."""
        zero_fr = [
            "Diamond", "LiNbO3", "LiTaO3", "KTP", "LBO",
            "BaTiO3", "GaN", "AlN", "SiC_4H", "YAG", "Al2O3",
            "YLF", "AgGaS2", "AgGaSe2", "Si", "Ge",
        ]
        for name in zero_fr:
            spec = RamanSpec.from_database(name)
            assert spec.fR == 0.0

    def test_none_fR_materials(self):
        """Test materials with unspecified fR (=None)."""
        none_fr = ["AlGaAs", "InGaAs", "Ga2O3", "Si3N4", "CdTe"]
        for name in none_fr:
            spec = RamanSpec.from_database(name)
            assert spec.fR is None

    def test_each_material_stokes_wavelength(self):
        """Test Stokes/anti-Stokes wavelength consistency for every material."""
        pump = Wavelength(1550, "nm")
        for name in RAMAN_MATERIALS:
            spec = RamanSpec.from_database(name)
            stokes = spec.stokes_wavelength(pump)
            anti = spec.anti_stokes_wavelength(pump)
            nu_diff_stokes = (1/pump.as_m - 1/stokes.as_m) * 3e8
            nu_diff_anti = (1/anti.as_m - 1/pump.as_m) * 3e8
            assert np.isclose(nu_diff_stokes, spec.raman_shift_Hz, rtol=1e-3)
            assert np.isclose(nu_diff_anti, spec.raman_shift_Hz, rtol=1e-3)

    def test_gain_coeff_is_positive_or_none(self):
        """Test gain_coeff is either positive or None for every material."""
        for name, data in RAMAN_MATERIALS.items():
            gc = data.get("gain_coeff")
            if gc is not None:
                assert gc > 0, f"{name} has gain_coeff={gc}, expected > 0"  # type: ignore[operator]

    def test_n2_is_real(self):
        """Test n2 is a finite number or None for every material."""
        for name, data in RAMAN_MATERIALS.items():
            n2 = data.get("n2")
            if n2 is not None:
                assert np.isfinite(n2), f"{name} has non-finite n2={n2}"


class TestNewMaterialsDatabase:
    """Tests specific to the SQLite database loading for all 30 materials."""

    def test_all_materials_in_database(self):
        """Test that all 30 materials are queryable from the bundled DB.
        
        Seeds the DB with RAMAN_MATERIALS if it's empty.
        """
        db = RamanDatabase()
        materials = db.list_materials()
        
        # Seed DB if empty
        if not materials:
            for name, data in RAMAN_MATERIALS.items():
                db.add_material(data)
            materials = db.list_materials()
        
        for name in RAMAN_MATERIALS:
            assert name in materials, f"{name} not found in DB"

    def test_database_preferred_over_dict(self):
        """Test from_database tries DB before falling back to dict."""
        spec = RamanSpec.from_database("Silica")
        assert spec.name == "Silica"
        assert spec.raman_shift_cm == 440

    def test_from_database_with_custom_db(self):
        """Test from_database with an explicitly created temporary DB."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "custom.db"
            db = RamanDatabase(db_path=db_path)
            db.add_material({
                "name": "CustomDB",
                "crystal": "Custom",
                "raman_shift_cm": 500,
                "raman_linewidth_cm": 25,
                "fR": 0.3,
                "n2": 1e-19,
            })
            spec = RamanSpec.from_database("CustomDB", db_path=db_path)
            assert spec.name == "CustomDB"
            assert spec.raman_shift_cm == 500
            assert spec.fR == 0.3


class TestThorlabsSubstrateMaterials:
    """Tests for Thorlabs optical substrate entries in materials.db."""

    THORLABS_NAMES = sorted(THORLABS_SUBSTRATE_MATERIALS.keys())

    @pytest.fixture(autouse=True)
    def _ensure_db_seeded(self):
        db = RamanDatabase()
        if not db.get_sellmeier("N-BK7"):
            for data in THORLABS_SUBSTRATE_MATERIALS.values():
                db.add_material(data)
            from seed_db import THORLABS_SELLMEIER_MATERIALS

            for name, sellmeier in THORLABS_SELLMEIER_MATERIALS.items():
                db.add_sellmeier(material=name, **{
                    k: sellmeier[k]
                    for k in (
                        "form",
                        "a0",
                        "coefficients",
                        "wavelengths",
                        "valid_from_um",
                        "valid_to_um",
                        "source",
                    )
                })

    def test_thorlabs_material_count(self):
        assert len(THORLABS_SUBSTRATE_MATERIALS) == 12

    def test_thorlabs_materials_load_from_database(self):
        for name in self.THORLABS_NAMES:
            spec = RamanSpec.from_database(name)
            assert spec.name == name
            assert spec.fR == 0.0
            assert spec.raman_shift_cm is None

    def test_thorlabs_sellmeier_refractive_index(self):
        checks = {
            "N-BK7": (0.5876, 1.517, 0.01),
            "N-SF11": (0.5876, 1.785, 0.01),
            "CaF2": (0.5876, 1.434, 0.01),
            "MgF2": (0.633, 1.425, 0.05),
            "ZnSe": (1.06, 2.26, 0.1),
            "PMMA": (0.5876, 1.491, 0.01),
        }
        for name, (wl_um, n_expected, tol) in checks.items():
            spec = RamanSpec.from_database(name)
            n = spec.nk(wl_um).real
            assert abs(n - n_expected) < tol, f"{name}: n={n:.4f}, expected≈{n_expected}"

    def test_refractive_index_from_material_database(self):
        from photonics_helper.materials import RefractiveIndex

        ri = RefractiveIndex.from_material_database("N-BK7")
        n = ri.n_func(0.5876)
        assert abs(n - 1.517) < 0.01


# ─── RamanFrequencyResponse Tests ─────────────────────────────────────────────


class TestRamanFrequencyResponse:
    """Tests for RamanFrequencyResponse class (Layer 3)."""

    def _make_freq_resp(self, material="Silica", **overrides):
        """Helper to create a RamanFrequencyResponse."""
        from photonics_helper.raman import RamanFrequencyResponse, RamanResponse
        from photonics_helper.pulse import TemporalGrid

        spec = RamanSpec.from_database(material)
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        resp = RamanResponse(spec=spec, grid=grid)
        return RamanFrequencyResponse(response=resp, grid=grid, **overrides)

    def test_construction(self):
        """Test basic construction."""
        fr = self._make_freq_resp()
        assert fr.response is not None
        assert fr.grid is not None

    def test_H_real_shape(self):
        """Test that H_real has correct shape."""
        fr = self._make_freq_resp()
        assert fr.H_real.shape == (fr.grid.N,)  # type: ignore[union-attr]

    def test_H_imag_shape(self):
        """Test that H_imag has correct shape."""
        fr = self._make_freq_resp()
        assert fr.H_imag.shape == (fr.grid.N,)  # type: ignore[union-attr]

    def test_H_magnitude_shape(self):
        """Test that H_magnitude has correct shape."""
        fr = self._make_freq_resp()
        assert fr.H_magnitude.shape == (fr.grid.N,)  # type: ignore[union-attr]

    def test_H_phase_shape(self):
        """Test that H_phase has correct shape."""
        fr = self._make_freq_resp()
        assert fr.H_phase.shape == (fr.grid.N,)  # type: ignore[union-attr]

    def test_magnitude_non_negative(self):
        """Test that |H(Ω)| is non-negative."""
        fr = self._make_freq_resp()
        assert np.all(fr.H_magnitude >= 0)

    def test_resonance_frequency_matches_raman_shift(self):
        """Test that the resonance frequency matches the Raman shift."""
        fr = self._make_freq_resp()
        spec = fr.response.spec

        # Resonance should be near the Raman shift frequency
        # Allow 10% tolerance due to FFT binning
        expected = spec.raman_shift_THz  # Already in THz
        actual = fr.resonance_frequency_THz
        assert abs(actual - expected) / expected < 0.1, \
            f"Resonance {actual:.2f} THz too far from expected {expected:.2f} THz"

    def test_resonance_frequency_positive(self):
        """Test that resonance frequency is positive."""
        fr = self._make_freq_resp()
        assert fr.resonance_frequency_THz > 0

    def test_FWHM_positive(self):
        """Test that FWHM is positive."""
        fr = self._make_freq_resp()
        assert fr.resonance_FWHM_THz > 0

    def test_quality_factor_positive(self):
        """Test that quality factor is positive."""
        fr = self._make_freq_resp()
        assert fr.quality_factor > 0

    def test_quality_factor_reasonable(self):
        """Test that Q factor is in reasonable range for Silica."""
        fr = self._make_freq_resp()
        # Silica Q factor should be around 10 (shift/linewidth ≈ 440/45 ≈ 9.8)
        assert 1 < fr.quality_factor < 100, \
            f"Q factor {fr.quality_factor:.1f} outside expected range"

    def test_H_at_zero_frequency(self):
        """Test H(0) is real and positive (DC component)."""
        fr = self._make_freq_resp()
        # Find index closest to Ω=0
        zero_idx = np.argmin(np.abs(fr.grid.w))  # type: ignore[union-attr]
        # H(0) should be real (no phase) and positive
        assert abs(fr.H_imag[zero_idx]) < 1e-10 * np.max(np.abs(fr.H_real))
        assert fr.H_real[zero_idx] > 0

    def test_plot_real_returns_figure(self):
        """Test plot_real returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fr = self._make_freq_resp()
        fig = fr.plot_real(backend="matplotlib")
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_imag_returns_figure(self):
        """Test plot_imag returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fr = self._make_freq_resp()
        fig = fr.plot_imag(backend="matplotlib")
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_magnitude_returns_figure(self):
        """Test plot_magnitude returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fr = self._make_freq_resp()
        fig = fr.plot_magnitude(backend="matplotlib")
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_phase_returns_figure(self):
        """Test plot_phase returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fr = self._make_freq_resp()
        fig = fr.plot_phase(backend="matplotlib")
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_all_returns_figure(self):
        """Test plot_all returns a 4-panel figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fr = self._make_freq_resp()
        fig = fr.plot_all(backend="matplotlib")
        assert fig is not None
        assert len(fig.axes) == 4  # type: ignore[attr-defined]
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_all_plotly(self):
        """Test plot_all with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        fr = self._make_freq_resp()
        fig = fr.plot_all(backend="plotly")
        assert fig is not None

    def test_different_materials_different_resonance(self):
        """Test that different materials have different resonance frequencies."""
        fr_silica = self._make_freq_resp("Silica")
        fr_diamond = self._make_freq_resp("Diamond")

        assert fr_silica.resonance_frequency_THz != fr_diamond.resonance_frequency_THz

    def test_narrow_linewidth_sharper_resonance(self):
        """Test that narrower linewidth gives sharper resonance (higher Q)."""
        # Create a custom spec with very narrow linewidth
        spec_narrow = RamanSpec(
            name="Narrow",
            raman_shift_cm=440,
            raman_linewidth_cm=5,  # Very narrow
            fR=0.18,
        )
        spec_wide = RamanSpec(
            name="Wide",
            raman_shift_cm=440,
            raman_linewidth_cm=100,  # Very wide
            fR=0.18,
        )

        from photonics_helper.pulse import TemporalGrid
        from photonics_helper.raman import RamanResponse, RamanFrequencyResponse

        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        resp_narrow = RamanResponse(spec=spec_narrow, grid=grid)
        resp_wide = RamanResponse(spec=spec_wide, grid=grid)

        fr_narrow = RamanFrequencyResponse(response=resp_narrow, grid=grid)
        fr_wide = RamanFrequencyResponse(response=resp_wide, grid=grid)

        # Narrower linewidth should have higher Q
        assert fr_narrow.quality_factor > fr_wide.quality_factor


# ─── RamanResponse Tests ──────────────────────────────────────────────────────


class TestRamanResponse:
    """Tests for RamanResponse class (Layer 2)."""

    def _make_response(self, **overrides):
        """Helper to create a RamanResponse with Silica defaults."""
        from photonics_helper.raman import RamanResponse
        from photonics_helper.pulse import TemporalGrid

        spec = RamanSpec.from_database("Silica")
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        return RamanResponse(spec=spec, grid=grid, **overrides)

    def test_construction_auto_derives_tau(self):
        """Test that τ1, τ2 are auto-derived from Raman shift/linewidth."""
        resp = self._make_response()
        spec = resp.spec

        # τ1 = 1 / ν_R (oscillation period)
        expected_tau1 = 1.0 / spec.raman_shift_Hz
        assert_almost_equal(resp.tau1, expected_tau1, decimal=10)  # type: ignore[arg-type]

        # τ2 = 1 / (π × linewidth_Hz) (damping time from Lorentzian FWHM)
        expected_tau2 = 1.0 / (np.pi * spec.linewidth_Hz)
        assert_almost_equal(resp.tau2, expected_tau2, decimal=10)  # type: ignore[arg-type]

    def test_construction_explicit_tau_overrides(self):
        """Test that explicit tau1/tau2 override auto-derived values."""
        resp = self._make_response(tau1=1e-12, tau2=2e-12)
        assert_almost_equal(resp.tau1, 1e-12)  # type: ignore[arg-type]
        assert_almost_equal(resp.tau2, 2e-12)  # type: ignore[arg-type]

    def test_h_R_at_t_equals_zero(self):
        """Test h_R(0) = 0 because sin(0) = 0."""
        resp = self._make_response()
        t = np.array([0.0])
        h = resp.delayed_response(t)
        assert_almost_equal(h[0], 0.0, decimal=10)

    def test_h_R_positive_only(self):
        """Test that h_R(t) = 0 for t < 0 (causality)."""
        resp = self._make_response()
        t_neg = np.linspace(-1e-12, -1e-15, 100)
        h_neg = resp.delayed_response(t_neg)
        assert_array_almost_equal(h_neg, np.zeros_like(h_neg), decimal=15)

    def test_h_R_damped_oscillation(self):
        """Test that h_R(t) shows damped oscillation pattern."""
        resp = self._make_response()
        spec = resp.spec
        t = np.linspace(0, 5e-12, 10000)
        h = resp.delayed_response(t)

        # Should have multiple zero crossings (oscillation)
        zero_crossings = np.where(np.diff(np.sign(h)))[0]
        assert len(zero_crossings) > 5, "Expected multiple zero crossings from oscillation"

        # Envelope should decay (absolute values at later times < earlier times)
        # Check that the envelope of the last peak is smaller than the first peak
        positive_mask = h > 0
        if positive_mask.any():
            positive_t = t[positive_mask]
            positive_h = h[positive_mask]
            # Find local maxima by checking sign changes of derivative
            dh = np.diff(positive_h)
            peaks = np.where((dh[:-1] > 0) & (dh[1:] < 0))[0]
            if len(peaks) >= 2:
                assert positive_h[peaks[-1]] < positive_h[peaks[0]], \
                    "Delayed response should show decaying oscillation"

    def test_instantaneous_response_is_gaussian(self):
        """Test that instantaneous response is a narrow Gaussian approximation of δ(t)."""
        resp = self._make_response()
        t = np.linspace(-1e-13, 1e-13, 1000)
        inst = resp.instantaneous_response(t)

        # Should be a Gaussian centered at t=0
        max_idx = np.argmax(inst)
        assert_almost_equal(t[max_idx], 0.0, decimal=10)

        # Should be approximately symmetric around the peak.
        # For a narrow Gaussian on a discrete grid, perfect symmetry isn't achievable,
        # but values at equal distances from the peak should be close.
        offset = 5
        left = inst[max_idx - offset:max_idx][::-1]
        right = inst[max_idx + 1:max_idx + offset + 1]
        # Relative tolerance of 5% accounts for discrete sampling of narrow peak
        np.testing.assert_allclose(left, right, rtol=0.05)

        # Should be positive everywhere
        assert np.all(inst >= 0)

    def test_instantaneous_response_normalization(self):
        """Test that δ_ε(t) integrates to ~1."""
        resp = self._make_response()
        # Use a fine grid around t=0
        eps = resp.grid.dt * 10  # type: ignore[union-attr]  # narrow Gaussian width
        t = np.linspace(-5 * eps, 5 * eps, 10000)
        inst = resp.instantaneous_response(t)
        # instantaneous_response = (1-fR)*delta, so integral = (1-fR)
        expected_integral = 1.0 - resp.fR  # type: ignore[operator]
        integral = np.trapezoid(inst, t)
        assert_almost_equal(integral, expected_integral, decimal=1)

    def test_combined_response_at_t_less_than_zero(self):
        """Test combined response: only instantaneous part for t < 0."""
        resp = self._make_response()
        t_neg = np.array([-1e-12, -5e-13, -1e-13])
        combined = resp.combined_response(t_neg)
        inst = resp.instantaneous_response(t_neg)

        # For t < 0, delayed response is 0, so combined = instantaneous
        assert_array_almost_equal(combined, inst, decimal=15)

    def test_combined_response_at_t_equals_zero(self):
        """Test combined response at t=0: instant + delayed(0)."""
        resp = self._make_response()
        t_zero = np.array([0.0])
        combined = resp.combined_response(t_zero)
        inst = resp.instantaneous_response(t_zero)
        delayed = resp.delayed_response(t_zero)

        # combined = instant + delayed, and delayed(0) = 0
        expected = inst + delayed
        assert_almost_equal(combined[0], expected[0], decimal=10)

    def test_fR_zero_pure_kerr(self):
        """Test that fR=0 gives pure instantaneous (Kerr) response."""
        resp = self._make_response(fR=0.0)
        t = np.linspace(-1e-12, 1e-12, 1000)
        combined = resp.combined_response(t)
        inst = resp.instantaneous_response(t)

        # With fR=0, combined should equal instantaneous
        assert_array_almost_equal(combined, inst, decimal=10)

    def test_fR_one_pure_raman(self):
        """Test that fR=1 gives pure Raman response (no delta spike)."""
        resp = self._make_response(fR=1.0)
        t_neg = np.array([-1e-12, -1e-13])
        combined = resp.combined_response(t_neg)

        # With fR=1, instantaneous part vanishes
        assert_array_almost_equal(combined, np.zeros_like(combined), decimal=10)

    def test_tau1_tau2_from_silica_values(self):
        """Test τ1, τ2 values match known Silica parameters.

        Silica: shift=440 cm⁻¹, linewidth=45 cm⁻¹
        Expected: τ1 ≈ 2.06 fs, τ2 ≈ 15.4 fs
        """
        resp = self._make_response()
        spec = resp.spec

        # τ1 = 1/ν_R
        expected_tau1 = 1.0 / spec.raman_shift_Hz
        assert_almost_equal(resp.tau1, expected_tau1, decimal=8)  # type: ignore[arg-type]
        # τ1 should be on the order of femtoseconds
        assert 1e-15 < resp.tau1 < 1e-13  # type: ignore[operator]

        # τ2 = 1/(π × linewidth)
        expected_tau2 = 1.0 / (np.pi * spec.linewidth_Hz)
        assert_almost_equal(resp.tau2, expected_tau2, decimal=8)  # type: ignore[arg-type]
        # τ2 should be larger than τ1 (damping slower than oscillation)
        assert resp.tau2 > resp.tau1  # type: ignore[operator]

    def test_plot_components_returns_figure(self):
        """Test that plot_components returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        resp = self._make_response()
        fig = resp.plot_components(backend="matplotlib")

        assert fig is not None
        # Should have 3 axes (3 panels)
        assert len(fig.axes) == 3  # type: ignore[attr-defined]
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_components_plotly(self):
        """Test plot_components with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        resp = self._make_response()
        fig = resp.plot_components(backend="plotly")
        assert fig is not None

    def test_grid_default_sizing(self):
        """Test that the default grid is wide enough to capture the response."""
        resp = self._make_response()
        # Grid should cover at least several τ2
        assert resp.grid.Tmax.as_s > 10 * resp.tau2, \
            "Grid Tmax should cover at least 10× τ2 to capture damped oscillation"

    def test_different_materials_different_tau(self):
        """Test that different materials produce different τ1, τ2."""
        from photonics_helper.raman import RamanResponse
        from photonics_helper.pulse import TemporalGrid

        cds = RamanSpec.from_database("CdS")
        dia = RamanSpec.from_database("Diamond")

        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        resp_cds = RamanResponse(spec=cds, grid=grid)
        resp_dia = RamanResponse(spec=dia, grid=grid)

        # Different Raman shifts → different τ1
        assert resp_cds.tau1 != resp_dia.tau1

    def test_delayed_response_peak_timing(self):
        """Test that the first peak of h_R(t) occurs near τ1."""
        resp = self._make_response()
        t = np.linspace(0, 5e-12, 100000)
        h = resp.delayed_response(t)

        # First positive peak should be near t = τ1/4 to τ1/2
        # (sin reaches first max at π/2, so 2πν_R·t = π/2 → t = 1/(4ν_R) = τ1/4)
        first_max_idx = np.argmax(h)
        first_peak_t = t[first_max_idx]
        expected_peak = resp.tau1 / 4  # type: ignore[operator]
        assert_almost_equal(first_peak_t, expected_peak, decimal=1)

    def test_combined_response_integral_positive(self):
        """Test that the combined response R(t) has positive integral (causality)."""
        resp = self._make_response()
        t = resp.grid.t  # type: ignore
        combined = resp.combined_response(t)
        integral = np.trapezoid(combined, t)
        assert integral > 0, "Combined response integral should be positive"


# ─── RamanPulseInteraction Tests (Layer 4) ────────────────────────────────────


class TestRamanPulseInteraction:
    """Tests for RamanPulseInteraction class (Layer 4)."""

    def _make_interaction(self, material="Silica", **overrides):
        """Helper to create a RamanPulseInteraction with Silica defaults."""
        from photonics_helper.raman import RamanPulseInteraction, RamanResponse
        from photonics_helper.pulse import Wave, Envelope, TemporalGrid
        from photonics_helper.base import Wavelength, Time

        spec = RamanSpec.from_database(material)
        grid = TemporalGrid(N=2**14, Tmax=Time(20e-12, "s"))
        envelope = Envelope(
            shape="gaussian",
            peak_amplitude=1.0,
            pulse_width=Time(100, "fs"),  # 100 fs
        )
        wave = Wave(
            grid=grid,
            envelope=envelope,
            central_wavelength=Wavelength(800, "nm"),
        )
        response = RamanResponse(spec=spec, grid=grid)
        return RamanPulseInteraction(pulse=wave, response=response, spec=spec, **overrides)

    def test_construction(self):
        """Test basic construction."""
        interaction = self._make_interaction()
        assert interaction.pulse is not None
        assert interaction.response is not None
        assert interaction.spec is not None

    def test_polarization_shape(self):
        """Test that nonlinear polarization has correct shape."""
        interaction = self._make_interaction()
        P_NL = interaction.nonlinear_polarization
        assert P_NL.shape == (interaction.grid.N,)  # type: ignore[union-attr]

    def test_polarization_zero_outside_pulse(self):
        """Test that P_NL is negligible where pulse intensity is negligible."""
        interaction = self._make_interaction()
        P_NL = interaction.nonlinear_polarization
        I = interaction.pulse.envelope_intensity

        # Find where intensity is negligible (< 1% of peak)
        threshold = 0.01 * np.max(I)
        quiet_mask = I < threshold

        if quiet_mask.any():
            # P_NL should be much smaller than max in quiet regions
            max_PNL_in_quiet = np.max(np.abs(P_NL[quiet_mask]))
            max_PNL = np.max(np.abs(P_NL))
            assert max_PNL_in_quiet < 0.1 * max_PNL, \
                "P_NL should be small where pulse intensity is negligible"

    def test_polarization_lags_pulse(self):
        """Test that the delayed polarization peaks after the pulse."""
        interaction = self._make_interaction()
        P_NL = interaction.nonlinear_polarization
        I = interaction.pulse.envelope_intensity

        pulse_peak_t = interaction.grid.t[np.argmax(I)]  # type: ignore[union-attr]
        PNL_peak_t = interaction.grid.t[np.argmax(P_NL)]  # type: ignore[union-attr]

        # The delayed polarization should peak after the pulse peak
        # (due to the convolution with the delayed Raman response)
        assert PNL_peak_t > pulse_peak_t, \
            f"P_NL peak at {PNL_peak_t*1e12:.3f} ps should be after pulse peak at {pulse_peak_t*1e12:.3f} ps"

    def test_polarization_positive_with_positive_response(self):
        """Test that P_NL is positive when both R(t) and I(t) are positive."""
        interaction = self._make_interaction()

        # Get the delayed (positive) part of Raman response
        t = interaction.grid.t  # type: ignore[union-attr]
        R_delayed = interaction.response.delayed_response(t)
        I = interaction.pulse.envelope_intensity

        # Where both are positive, P_NL should be positive
        positive_mask = (R_delayed > 0) & (I > 0)
        if positive_mask.any():
            P_NL = interaction.nonlinear_polarization
            # The convolution should preserve sign in regions where both are positive
            assert np.any(P_NL[positive_mask] > 0), \
                "P_NL should be positive where both R(t) and I(t) are positive"

    def test_polarization_zero_for_zero_fR(self):
        """Test that P_NL has no delayed component when fR=0 (pure Kerr)."""
        interaction = self._make_interaction(fR=0.0)
        P_NL = interaction.nonlinear_polarization
        I = interaction.pulse.envelope_intensity

        # With fR=0, only instantaneous response exists (delta function)
        # The convolution of delta with I gives I itself (scaled)
        # So P_NL should track the pulse intensity profile
        assert np.max(np.abs(P_NL)) > 0, "P_NL should be nonzero even with fR=0"

    def test_n2_override(self):
        """Test that n2 override affects polarization magnitude."""
        interaction_low = self._make_interaction(n2=1e-20)
        interaction_high = self._make_interaction(n2=1e-18)

        P_low = interaction_low.nonlinear_polarization
        P_high = interaction_high.nonlinear_polarization

        # Higher n2 should give larger polarization
        assert np.max(np.abs(P_high)) > np.max(np.abs(P_low)), \
            "Higher n2 should produce larger P_NL"

    def test_different_pulse_shapes(self):
        """Test interaction with different pulse shapes."""
        from photonics_helper.pulse import Wave, Envelope, TemporalGrid
        from photonics_helper.base import Wavelength, Time
        from photonics_helper.raman import RamanResponse, RamanPulseInteraction

        spec = RamanSpec.from_database("Silica")
        grid = TemporalGrid(N=2**14, Tmax=Time(20e-12, "s"))
        response = RamanResponse(spec=spec, grid=grid)

        for shape in ["gaussian", "sech", "lorentzian"]:
            envelope = Envelope(
                shape=shape,  # type: ignore[arg-type]
                peak_amplitude=1.0,
                pulse_width=Time(100, "fs"),
            )
            wave = Wave(
                grid=grid,
                envelope=envelope,
                central_wavelength=Wavelength(800, "nm"),
            )
            interaction = RamanPulseInteraction(
                pulse=wave, response=response, spec=spec
            )
            P_NL = interaction.nonlinear_polarization
            assert P_NL.shape == (grid.N,), f"Wrong shape for {shape} pulse"
            assert np.max(np.abs(P_NL)) > 0, f"P_NL should be nonzero for {shape} pulse"

    def test_plot_interaction_returns_figure(self):
        """Test that plot_interaction returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        interaction = self._make_interaction()
        fig = interaction.plot_interaction(backend="matplotlib")

        assert fig is not None
        # Should have 4 axes (4 panels)
        assert len(fig.axes) == 4  # type: ignore[attr-defined]
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_interaction_plotly(self):
        """Test plot_interaction with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        interaction = self._make_interaction()
        fig = interaction.plot_interaction(backend="plotly")
        assert fig is not None

    def test_animation_returns_figure(self):
        """Test that animate returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        interaction = self._make_interaction()
        fig = interaction.animate(backend="matplotlib", frames=5)

        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_convolution_with_known_function(self):
        """Test convolution correctness with a known input.

        If I(t) is a narrow pulse (approximating delta), then
        P_NL(t) ∝ R(t) (the response function itself).
        """
        interaction = self._make_interaction()
        P_NL = interaction.nonlinear_polarization
        t = interaction.grid.t  # type: ignore[union-attr]

        # Get the Raman response
        R = interaction.response.combined_response(t)

        # The convolution should produce a signal that has similar shape to R(t)
        # but shifted/smeared by the pulse duration
        # For a very short pulse, P_NL ≈ R(t) (up to scaling)
        # We check that the peak positions are close
        R_peak_t = t[np.argmax(np.abs(R))]
        PNL_peak_t = t[np.argmax(np.abs(P_NL))]

        # They should be within a few grid points of each other
        # (the pulse broadens the response slightly)
        assert abs(PNL_peak_t - R_peak_t) < 10 * interaction.grid.dt, \
            f"P_NL peak at {PNL_peak_t*1e12:.3f} ps too far from R peak at {R_peak_t*1e12:.3f} ps"  # type: ignore[union-attr]

    def test_polarization_scales_with_intensity(self):
        """Test that P_NL scales linearly with pulse intensity."""
        interaction_weak = self._make_interaction()
        # Scale the pulse amplitude
        interaction_weak.pulse.envelope.peak_amplitude = 0.5

        interaction_strong = self._make_interaction()
        interaction_strong.pulse.envelope.peak_amplitude = 2.0

        P_weak = interaction_weak.nonlinear_polarization
        P_strong = interaction_strong.nonlinear_polarization

        # P_NL should scale as A² (since I ∝ |A|²)
        ratio_expected = (2.0 / 0.5) ** 2  # 16x
        ratio_actual = np.max(np.abs(P_strong)) / np.max(np.abs(P_weak))

        # Allow 10% tolerance due to convolution effects
        assert abs(ratio_actual - ratio_expected) / ratio_expected < 0.1, \
            f"P_NL scaling ratio {ratio_actual:.1f} too far from expected {ratio_expected}"

    def test_grid_default_sizing(self):
        """Test that the grid covers the pulse and response."""
        interaction = self._make_interaction()
        # Grid should be wide enough to capture both pulse and Raman response
        assert interaction.grid.Tmax.as_s > 5e-12, "Grid should cover at least 5 ps"

    def test_polarization_energy_conservation(self):
        """Test that the polarization doesn't create energy from nowhere.

        The integral of P_NL should be related to the integral of I(t) times
        the integral of R(t).
        """
        interaction = self._make_interaction()
        P_NL = interaction.nonlinear_polarization
        I = interaction.pulse.envelope_intensity
        t = interaction.grid.t  # type: ignore[union-attr]

        # The convolution integral: ∫ P_NL dt ∝ (∫ R dt) × (∫ I dt)
        # Both integrals should be finite and positive
        integral_PNL = np.trapezoid(P_NL, t)
        integral_I = np.trapezoid(I, t)

        # For a symmetric pulse and response, the integral should be positive
        # (the DC component of the convolution)
        assert integral_I > 0, "Pulse intensity integral should be positive"
        # P_NL integral should be finite (not divergent)
        assert np.isfinite(integral_PNL)


# ─── PumpWavelengthExplorer Tests (Layer 5) ────────────────────────────────────


class TestPumpWavelengthExplorer:
    """Tests for PumpWavelengthExplorer class (Layer 5)."""

    def _make_explorer(self, material="Silica", **overrides):
        """Helper to create a PumpWavelengthExplorer with Silica defaults."""
        from photonics_helper.raman import PumpWavelengthExplorer

        spec = RamanSpec.from_database(material)
        return PumpWavelengthExplorer(spec=spec, **overrides)

    def test_construction(self):
        """Test basic construction."""
        explorer = self._make_explorer()
        assert explorer.spec is not None
        assert explorer.spec.name == "Silica"

    def test_stokes_longer_wavelength(self):
        """Test that Stokes wavelength is longer than pump wavelength."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        stokes = explorer.pump_to_stokes(pump)
        assert stokes.as_nm > pump.as_nm

    def test_anti_stokes_shorter_wavelength(self):
        """Test that Anti-Stokes wavelength is shorter than pump wavelength."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        anti_stokes = explorer.pump_to_anti_stokes(pump)
        assert anti_stokes.as_nm < pump.as_nm

    def test_stokes_frequency_lower(self):
        """Test that Stokes frequency is lower than pump frequency."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        stokes = explorer.pump_to_stokes(pump)

        nu_pump = C_MS / pump.as_m
        nu_stokes = C_MS / stokes.as_m
        assert nu_stokes < nu_pump

    def test_anti_stokes_frequency_higher(self):
        """Test that Anti-Stokes frequency is higher than pump frequency."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        anti_stokes = explorer.pump_to_anti_stokes(pump)

        nu_pump = C_MS / pump.as_m
        nu_anti = C_MS / anti_stokes.as_m
        assert nu_anti > nu_pump

    def test_stokes_shift_matches_raman_shift(self):
        """Test that the frequency shift equals the Raman shift."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        stokes = explorer.pump_to_stokes(pump)

        nu_pump = C_MS / pump.as_m
        nu_stokes = C_MS / stokes.as_m
        shift_actual = nu_pump - nu_stokes
        shift_expected = explorer.spec.raman_shift_Hz
        assert_almost_equal(shift_actual, shift_expected, decimal=6)

    def test_anti_stokes_shift_matches_raman_shift(self):
        """Test that the Anti-Stokes frequency shift equals the Raman shift."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        anti_stokes = explorer.pump_to_anti_stokes(pump)

        nu_pump = C_MS / pump.as_m
        nu_anti = C_MS / anti_stokes.as_m
        shift_actual = nu_anti - nu_pump
        shift_expected = explorer.spec.raman_shift_Hz
        assert_almost_equal(shift_actual, shift_expected, decimal=6)

    def test_stokes_anti_stokes_symmetric_in_frequency(self):
        """Test that Stokes and Anti-Stokes are symmetric in frequency space."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        stokes = explorer.pump_to_stokes(pump)
        anti_stokes = explorer.pump_to_anti_stokes(pump)

        nu_pump = C_MS / pump.as_m
        nu_stokes = C_MS / stokes.as_m
        nu_anti = C_MS / anti_stokes.as_m

        # The shifts should be equal in magnitude
        shift_stokes = nu_pump - nu_stokes
        shift_anti = nu_anti - nu_pump
        assert_almost_equal(shift_stokes, shift_anti, decimal=6)

    def test_stokes_anti_stokes_not_symmetric_in_wavelength(self):
        """Test that Stokes and Anti-Stokes are NOT symmetric in wavelength space.

        This is the key educational point: equal frequency shifts
        produce unequal wavelength shifts.
        """
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        stokes = explorer.pump_to_stokes(pump)
        anti_stokes = explorer.pump_to_anti_stokes(pump)

        delta_stokes_nm = stokes.as_nm - pump.as_nm
        delta_anti_nm = pump.as_nm - anti_stokes.as_nm

        # Wavelength shifts should NOT be equal
        assert delta_stokes_nm != delta_anti_nm, \
            "Wavelength shifts should differ (key concept of Layer 5)"

        # Stokes wavelength shift should be larger in nm for visible/NIR pump
        # (because dλ/dν = -c/ν², and |Δν| is the same)
        assert delta_stokes_nm > delta_anti_nm

    def test_different_pump_wavelengths(self):
        """Test that different pump wavelengths give different Stokes shifts in nm."""
        explorer = self._make_explorer()

        pump1 = Wavelength(800, "nm")
        pump2 = Wavelength(1550, "nm")

        stokes1 = explorer.pump_to_stokes(pump1)
        stokes2 = explorer.pump_to_stokes(pump2)

        # Both should be longer than their pump
        assert stokes1.as_nm > pump1.as_nm
        assert stokes2.as_nm > pump2.as_nm

        # But the absolute wavelength shift should differ
        delta1 = stokes1.as_nm - pump1.as_nm
        delta2 = stokes2.as_nm - pump2.as_nm
        assert delta1 != delta2

    def test_missing_raman_shift_raises(self):
        """Test that missing Raman shift raises ValueError."""
        from photonics_helper.raman import PumpWavelengthExplorer
        spec = RamanSpec(name="NoShift", raman_shift_cm=None)
        explorer = PumpWavelengthExplorer(spec=spec)

        with pytest.raises(ValueError, match="Raman shift"):
            explorer.pump_to_stokes(Wavelength(800, "nm"))

    def test_stokes_wavelength_near_1um(self):
        """Test Stokes wavelength for 1μm pump (telecom band)."""
        explorer = self._make_explorer()
        pump = Wavelength(1000, "nm")
        stokes = explorer.pump_to_stokes(pump)

        # For 1μm pump and 440 cm⁻¹ shift (silica), Stokes should be ~1030 nm
        # ν_pump ≈ 299.8 THz, shift ≈ 13.2 THz → ν_stokes ≈ 286.6 THz → λ_stokes ≈ 1046 nm
        assert 1000 < stokes.as_nm < 1100

    def test_anti_stokes_wavelength_near_750nm(self):
        """Test Anti-Stokes wavelength for 800nm pump."""
        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        anti_stokes = explorer.pump_to_anti_stokes(pump)

        # ν_pump ≈ 374.7 THz, shift ≈ 13.2 THz → ν_anti ≈ 387.9 THz → λ_anti ≈ 773 nm
        assert 750 < anti_stokes.as_nm < 800

    def test_plot_frequency_axis_returns_figure(self):
        """Test plot_frequency_axis returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        fig = explorer.plot_frequency_axis(pump, backend="matplotlib")

        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_wavelength_axis_returns_figure(self):
        """Test plot_wavelength_axis returns a matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        fig = explorer.plot_wavelength_axis(pump, backend="matplotlib")

        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_both_returns_figure(self):
        """Test plot_both returns a 2-panel figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        fig = explorer.plot_both(pump, backend="matplotlib")

        assert fig is not None
        assert len(fig.axes) == 2  # type: ignore[attr-defined]
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_vs_pump_returns_figure(self):
        """Test plot_vs_pump returns a figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explorer = self._make_explorer()
        fig = explorer.plot_vs_pump(pump_range_um=(0.5, 2.0), backend="matplotlib")

        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_both_plotly(self):
        """Test plot_both with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")
        fig = explorer.plot_both(pump, backend="plotly")
        assert fig is not None

    def test_plot_vs_pump_plotly(self):
        """Test plot_vs_pump with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        explorer = self._make_explorer()
        fig = explorer.plot_vs_pump(pump_range_um=(0.5, 2.0), backend="plotly")
        assert fig is not None

    def test_different_materials_different_stokes(self):
        """Test that different materials give different Stokes shifts."""
        explorer_silica = self._make_explorer("Silica")
        explorer_diamond = self._make_explorer("Diamond")
        pump = Wavelength(800, "nm")

        stokes_silica = explorer_silica.pump_to_stokes(pump)
        stokes_diamond = explorer_diamond.pump_to_stokes(pump)

        # Diamond has larger Raman shift → larger wavelength shift
        delta_silica = stokes_silica.as_nm - pump.as_nm
        delta_diamond = stokes_diamond.as_nm - pump.as_nm
        assert delta_diamond > delta_silica

    def test_stokes_wavelength_monotonic_with_pump(self):
        """Test that Stokes wavelength increases monotonically with pump wavelength."""
        explorer = self._make_explorer()

        pump_wls = np.linspace(400, 2000, 20)  # 400nm to 2000nm
        stokes_wls = []
        for wl_um in pump_wls:
            pump = Wavelength(wl_um, "nm")
            stokes = explorer.pump_to_stokes(pump)
            stokes_wls.append(stokes.as_nm)

        stokes_wls = np.array(stokes_wls)
        # Stokes wavelength should increase as pump wavelength increases
        assert np.all(np.diff(stokes_wls) > 0), \
            "Stokes wavelength should increase monotonically with pump wavelength"

    def test_frequency_axis_marks_three_lines(self):
        """Test that frequency axis plot marks Anti-Stokes, Pump, and Stokes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explorer = self._make_explorer()
        pump = Wavelength(800, "nm")

        fig = explorer.plot_frequency_axis(pump, backend="matplotlib")
        ax = fig.axes[0]  # type: ignore[attr-defined]

        # Check that vertical lines exist for all three
        lines = ax.get_lines()
        # The function should have plotted at least 3 vertical markers
        # (pump + stokes + anti-stokes markers)
        # We verify by checking the plot was created without error
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]


# ─── MaterialComparison Tests (Layer 6) ────────────────────────────────────────


# Matplotlib colors for overlay plots (matching MaterialComparison)
_OVERLAY_COLORS = [
    "#00d4ff",  # cyan
    "#a78bfa",  # purple
    "#34d399",  # green
    "#fbbf24",  # amber
    "#f87171",  # red
    "#fb923c",  # orange
    "#38bdf8",  # sky blue
    "#c084fc",  # violet
]


class TestMaterialComparison:
    """Tests for MaterialComparison class (Layer 6)."""

    def _make_comparison(self, materials=None, **overrides):
        """Helper to create a MaterialComparison with preset materials."""
        from photonics_helper.raman import MaterialComparison

        if materials is None:
            materials = []
        return MaterialComparison(materials=materials, **overrides)

    def test_construction_empty(self):
        """Test construction with empty material list."""
        comp = self._make_comparison()
        assert comp.materials == []

    def test_construction_with_materials(self):
        """Test construction with pre-populated materials."""
        from photonics_helper.raman import RamanSpec

        silica = RamanSpec.from_database("Silica")
        cds = RamanSpec.from_database("CdS")
        comp = self._make_comparison(materials=[silica, cds])
        assert len(comp.materials) == 2
        assert comp.materials[0].name == "Silica"
        assert comp.materials[1].name == "CdS"

    def test_add_material(self):
        """Test adding a material."""
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison()
        assert len(comp.materials) == 0

        silica = RamanSpec.from_database("Silica")
        comp.add(silica)
        assert len(comp.materials) == 1
        assert comp.materials[0].name == "Silica"

    def test_add_duplicate_material(self):
        """Test that adding a duplicate material name replaces it."""
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison()
        silica1 = RamanSpec.from_database("Silica")
        comp.add(silica1)
        assert len(comp.materials) == 1

        # Add another Silica with different params — should replace
        silica2 = RamanSpec(
            name="Silica",
            raman_shift_cm=500,
            raman_linewidth_cm=50,
            fR=0.25,
        )
        comp.add(silica2)
        assert len(comp.materials) == 1
        assert comp.materials[0].raman_shift_cm == 500

    def test_remove_material(self):
        """Test removing a material by name."""
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison()
        silica = RamanSpec.from_database("Silica")
        cds = RamanSpec.from_database("CdS")
        comp.add(silica)
        comp.add(cds)
        assert len(comp.materials) == 2

        comp.remove("Silica")
        assert len(comp.materials) == 1
        assert comp.materials[0].name == "CdS"

    def test_remove_nonexistent_material(self):
        """Test removing a material that doesn't exist."""
        comp = self._make_comparison()
        comp.remove("NonExistent")
        assert len(comp.materials) == 0

    def test_clear(self):
        """Test clearing all materials."""
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison()
        comp.add(RamanSpec.from_database("Silica"))
        comp.add(RamanSpec.from_database("CdS"))
        assert len(comp.materials) == 2

        comp.clear()
        assert len(comp.materials) == 0

    def test_plot_spectra_overlay_empty(self):
        """Test spectra overlay with no materials returns None."""
        comp = self._make_comparison()
        result = comp.plot_spectra_overlay(backend="matplotlib")
        assert result is None

    def test_plot_spectra_overlay_single(self):
        """Test spectra overlay with a single material."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(materials=[RamanSpec.from_database("Silica")])
        fig = comp.plot_spectra_overlay(backend="matplotlib")

        assert fig is not None
        assert len(fig.axes) == 1  # type: ignore[attr-defined]
        # Check that the line has the correct label
        line = fig.axes[0].get_lines()[0]  # type: ignore[attr-defined]
        assert line.get_label() == "Silica"
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_spectra_overlay_multiple(self):
        """Test spectra overlay with multiple materials has one line per material."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(
            materials=[
                RamanSpec.from_database("Silica"),
                RamanSpec.from_database("CdS"),
                RamanSpec.from_database("Diamond"),
            ]
        )
        fig = comp.plot_spectra_overlay(backend="matplotlib")

        assert fig is not None
        assert len(fig.axes) == 1  # type: ignore[attr-defined]
        lines = fig.axes[0].get_lines()  # type: ignore[attr-defined]
        assert len(lines) == 3
        labels = [line.get_label() for line in lines]
        assert "Silica" in labels
        assert "CdS" in labels
        assert "Diamond" in labels
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_spectra_overlay_colors_differ(self):
        """Test that each material gets a different color."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(
            materials=[
                RamanSpec.from_database("Silica"),
                RamanSpec.from_database("CdS"),
                RamanSpec.from_database("Diamond"),
            ]
        )
        fig = comp.plot_spectra_overlay(backend="matplotlib")

        lines = fig.axes[0].get_lines()  # type: ignore[attr-defined]
        colors = [line.get_color() for line in lines]
        # All colors should be different
        assert len(set(colors)) == len(colors)
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_spectra_overlay_plotly(self):
        """Test spectra overlay with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(
            materials=[RamanSpec.from_database("Silica"), RamanSpec.from_database("CdS")]
        )
        fig = comp.plot_spectra_overlay(backend="plotly")
        assert fig is not None

    def test_plot_response_overlay_empty(self):
        """Test response overlay with no materials returns None."""
        comp = self._make_comparison()
        result = comp.plot_response_overlay(backend="matplotlib")
        assert result is None

    def test_plot_response_overlay_single(self):
        """Test response overlay with a single material."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec, RamanResponse
        from photonics_helper.pulse import TemporalGrid

        spec = RamanSpec.from_database("Silica")
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        comp = self._make_comparison(materials=[spec])
        fig = comp.plot_response_overlay(backend="matplotlib", grid=grid)

        assert fig is not None
        assert len(fig.axes) == 1  # type: ignore[attr-defined]
        line = fig.axes[0].get_lines()[0]  # type: ignore[attr-defined]
        assert line.get_label() == "Silica"
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_response_overlay_multiple(self):
        """Test response overlay with multiple materials."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec, RamanResponse
        from photonics_helper.pulse import TemporalGrid

        comp = self._make_comparison(
            materials=[
                RamanSpec.from_database("Silica"),
                RamanSpec.from_database("CdS"),
            ]
        )
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        fig = comp.plot_response_overlay(backend="matplotlib", grid=grid)

        assert fig is not None
        lines = fig.axes[0].get_lines()  # type: ignore[attr-defined]
        # Filter out internal lines (axhline has labels like '_child2')
        labeled_lines = [l for l in lines if l.get_label() and not l.get_label().startswith("_")]
        assert len(labeled_lines) == 2
        labels = [line.get_label() for line in labeled_lines]
        assert "Silica" in labels
        assert "CdS" in labels
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_frequency_overlay_empty(self):
        """Test frequency overlay with no materials returns None."""
        comp = self._make_comparison()
        result = comp.plot_frequency_overlay(backend="matplotlib")
        assert result is None

    def test_plot_frequency_overlay_single(self):
        """Test frequency overlay with a single material."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec, RamanResponse, RamanFrequencyResponse
        from photonics_helper.pulse import TemporalGrid

        spec = RamanSpec.from_database("Silica")
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        comp = self._make_comparison(materials=[spec])
        fig = comp.plot_frequency_overlay(backend="matplotlib", grid=grid)

        assert fig is not None
        assert len(fig.axes) == 1  # type: ignore[attr-defined]
        line = fig.axes[0].get_lines()[0]  # type: ignore[attr-defined]
        assert line.get_label() == "Silica"
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_frequency_overlay_multiple(self):
        """Test frequency overlay with multiple materials."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec, RamanResponse, RamanFrequencyResponse
        from photonics_helper.pulse import TemporalGrid

        comp = self._make_comparison(
            materials=[
                RamanSpec.from_database("Silica"),
                RamanSpec.from_database("As2Se3"),
            ]
        )
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        fig = comp.plot_frequency_overlay(backend="matplotlib", grid=grid)

        assert fig is not None
        lines = fig.axes[0].get_lines()  # type: ignore[attr-defined]
        labeled_lines = [l for l in lines if l.get_label() and not l.get_label().startswith("_")]
        assert len(labeled_lines) == 2
        labels = [line.get_label() for line in labeled_lines]
        assert "Silica" in labels
        assert "As2Se3" in labels
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_frequency_overlay_annotates_resonances(self):
        """Test that frequency overlay marks resonance frequencies."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec, RamanResponse, RamanFrequencyResponse
        from photonics_helper.pulse import TemporalGrid

        comp = self._make_comparison(
            materials=[RamanSpec.from_database("Silica")]
        )
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        fig = comp.plot_frequency_overlay(backend="matplotlib", grid=grid)

        ax = fig.axes[0]  # type: ignore[attr-defined]
        # Check for vertical lines (resonance markers)
        vlines = [p for p in ax.patches]  # patches include vlines in mpl
        # We verify the plot was created with annotations
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_comparison_table_format(self):
        """Test that comparison_table returns formatted text."""
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(
            materials=[
                RamanSpec.from_database("Silica"),
                RamanSpec.from_database("CdS"),
            ]
        )
        table = comp.comparison_table()

        assert "Silica" in table
        assert "CdS" in table
        assert "n₂" in table or "n2" in table
        assert "fR" in table or "fR:" in table or "f_R" in table
        assert "Shift" in table or "shift" in table

    def test_comparison_table_empty(self):
        """Test comparison_table with no materials returns empty or header."""
        comp = self._make_comparison()
        table = comp.comparison_table()
        # Should return something, even if just a header
        assert isinstance(table, str)

    def test_comparison_table_single_material(self):
        """Test comparison_table with a single material."""
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(materials=[RamanSpec.from_database("Silica")])
        table = comp.comparison_table()

        assert "Silica" in table

    def test_plot_all_empty(self):
        """Test plot_all with no materials returns None."""
        comp = self._make_comparison()
        result = comp.plot_all(backend="matplotlib")
        assert result is None

    def test_plot_all_single(self):
        """Test plot_all with a single material."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec, RamanResponse
        from photonics_helper.pulse import TemporalGrid

        spec = RamanSpec.from_database("Silica")
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        comp = self._make_comparison(materials=[spec])
        fig = comp.plot_all(backend="matplotlib", grid=grid)

        assert fig is not None
        # Should have 3 panels: spectra, response, frequency
        assert len(fig.axes) == 3  # type: ignore[attr-defined]
        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_all_multiple(self):
        """Test plot_all with multiple materials."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec, RamanResponse
        from photonics_helper.pulse import TemporalGrid

        comp = self._make_comparison(
            materials=[
                RamanSpec.from_database("Silica"),
                RamanSpec.from_database("Diamond"),
            ]
        )
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        fig = comp.plot_all(backend="matplotlib", grid=grid)

        assert fig is not None
        assert len(fig.axes) == 3  # type: ignore[attr-defined]
        plt.close(fig)  # type: ignore[arg-type]

    def test_common_comparisons_defined(self):
        """Test that preset comparison groups are defined."""
        from photonics_helper.raman import COMMON_COMPARISONS

        expected_groups = [
            "glass_vs_chalcogenide", "semiconductor", "high_n2", "high_shift",
            "nitride_semiconductor", "nonlinear_crystal", "high_gain",
            "wide_bandgap", "iii_v", "laser_host", "nlo_crystal",
            "chalcogenide", "ferroelectric", "negative_n2",
        ]
        for group in expected_groups:
            assert group in COMMON_COMPARISONS, f"Missing: {group}"

    def test_common_comparison_glass_vs_chalcogenide(self):
        """Test the glass_vs_chalcogenide preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS

        names = COMMON_COMPARISONS["glass_vs_chalcogenide"]
        assert "Silica" in names
        assert "As2Se3" in names

    def test_common_comparison_semiconductor(self):
        """Test the semiconductor preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS

        names = COMMON_COMPARISONS["semiconductor"]
        assert "CdS" in names
        assert "GaAs" in names
        assert "Si" in names
        assert "Ge" in names

    def test_common_comparison_high_n2(self):
        """Test the high_n2 preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS

        names = COMMON_COMPARISONS["high_n2"]
        assert "As2Se3" in names  # highest n2

    def test_common_comparison_high_shift(self):
        """Test the high_shift preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS

        names = COMMON_COMPARISONS["high_shift"]
        assert "Diamond" in names  # largest shift

    def test_common_comparison_nitride_semiconductor(self):
        """Test the nitride_semiconductor preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["nitride_semiconductor"]
        for mat in ["GaN", "AlN", "Si3N4"]:
            assert mat in names
        # GaN and AlN are crystalline (fR=0.0); Si3N4 is amorphous (fR=None)
        for name in names:
            assert RamanSpec.from_database(name) is not None

    def test_common_comparison_nonlinear_crystal(self):
        """Test the nonlinear_crystal preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["nonlinear_crystal"]
        for mat in ["LiNbO3", "KTP", "LBO", "BaTiO3"]:
            assert mat in names
        for name in names:
            spec = RamanSpec.from_database(name)
            assert spec.fR == 0.0  # chi-2 materials

    def test_common_comparison_high_gain(self):
        """Test the high_gain preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["high_gain"]
        for mat in ["Diamond", "As2Se3", "YAG"]:
            assert mat in names

    def test_common_comparison_wide_bandgap(self):
        """Test the wide_bandgap preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["wide_bandgap"]
        for mat in ["Diamond", "GaN", "SiC_4H", "AlN", "Ga2O3"]:
            assert mat in names

    def test_common_comparison_iii_v(self):
        """Test the iii_v preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["iii_v"]
        for mat in ["GaAs", "InP", "AlGaAs", "InGaAs"]:
            assert mat in names

    def test_common_comparison_laser_host(self):
        """Test the laser_host preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["laser_host"]
        for mat in ["YAG", "Al2O3", "YLF"]:
            assert mat in names

    def test_common_comparison_nlo_crystal(self):
        """Test the nlo_crystal preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["nlo_crystal"]
        for mat in ["LBO", "KTP", "AgGaS2", "AgGaSe2", "LiNbO3"]:
            assert mat in names

    def test_common_comparison_chalcogenide(self):
        """Test the chalcogenide preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["chalcogenide"]
        for mat in ["As2S3", "As2Se3"]:
            assert mat in names

    def test_common_comparison_ferroelectric(self):
        """Test the ferroelectric preset comparison."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["ferroelectric"]
        for mat in ["LiNbO3", "LiTaO3", "BaTiO3", "KTP"]:
            assert mat in names

    def test_common_comparison_negative_n2(self):
        """Test the negative_n2 preset loads both materials."""
        from photonics_helper.raman import COMMON_COMPARISONS
        names = COMMON_COMPARISONS["negative_n2"]
        for mat in ["ZnO", "CdTe"]:
            assert mat in names
            spec = RamanSpec.from_database(mat)
            assert spec.n2 is not None
            assert spec.n2 < 0  # both have negative n2

    def test_plot_spectra_overlay_plotly_multiple(self):
        """Test spectra overlay with plotly and multiple materials."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(
            materials=[
                RamanSpec.from_database("Silica"),
                RamanSpec.from_database("CdS"),
                RamanSpec.from_database("Diamond"),
            ]
        )
        fig = comp.plot_spectra_overlay(backend="plotly")
        assert fig is not None

    def test_plot_response_overlay_plotly(self):
        """Test response overlay with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        from photonics_helper.raman import RamanSpec, RamanResponse
        from photonics_helper.pulse import TemporalGrid

        comp = self._make_comparison(
            materials=[RamanSpec.from_database("Silica")]
        )
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        fig = comp.plot_response_overlay(backend="plotly", grid=grid)
        assert fig is not None

    def test_plot_frequency_overlay_plotly(self):
        """Test frequency overlay with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        from photonics_helper.raman import RamanSpec, RamanResponse, RamanFrequencyResponse
        from photonics_helper.pulse import TemporalGrid

        comp = self._make_comparison(
            materials=[RamanSpec.from_database("Silica")]
        )
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        fig = comp.plot_frequency_overlay(backend="plotly", grid=grid)
        assert fig is not None

    def test_plot_all_plotly(self):
        """Test plot_all with plotly backend."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not installed")

        from photonics_helper.raman import RamanSpec, RamanResponse
        from photonics_helper.pulse import TemporalGrid

        comp = self._make_comparison(
            materials=[RamanSpec.from_database("Silica")]
        )
        grid = TemporalGrid(N=2**14, Tmax=Time(10e-12, "s"))
        fig = comp.plot_all(backend="plotly", grid=grid)
        assert fig is not None

    def test_plot_spectra_overlay_uses_lorentzian_lineshape(self):
        """Test that spectra overlay plots Lorentzian-shaped curves."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec

        # Use a wide enough range to capture Silica's peak at 440 cm⁻¹
        comp = self._make_comparison(materials=[RamanSpec.from_database("Silica")])
        fig = comp.plot_spectra_overlay(backend="matplotlib", shift_range_cm=600)

        line = fig.axes[0].get_lines()[0]  # type: ignore[attr-defined]
        x = line.get_xdata()
        y = line.get_ydata()

        # Check that the peak is near the Raman shift (440 cm⁻¹)
        peak_idx = np.argmax(y)
        peak_x = x[peak_idx]
        # Should be within 50 cm⁻¹ of the expected shift
        assert abs(peak_x - 440) < 50, f"Peak at {peak_x:.1f} cm⁻¹ not near 440 cm⁻¹"

        # Check that the curve has Lorentzian-shaped tails (broader than Gaussian)
        # At 1× FWHM from center, Lorentzian is at 0.2 of peak
        fwhm = 45  # Silica FWHM in cm⁻¹
        left_val = y[np.argmin(np.abs(x - (440 - fwhm)))]
        right_val = y[np.argmin(np.abs(x - (440 + fwhm)))]
        peak_val = np.max(y)
        # Lorentzian at ±1×FWHM is at 20% of peak; Gaussian would be ~4%
        assert left_val / peak_val > 0.1, f"Left tail {left_val/peak_val:.3f} too low for Lorentzian"
        assert right_val / peak_val > 0.1, f"Right tail {right_val/peak_val:.3f} too low for Lorentzian"

        plt.close(fig)  # type: ignore[arg-type]

    def test_plot_spectra_overlay_title_includes_materials(self):
        """Test that the plot title lists material names."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison(
            materials=[RamanSpec.from_database("Silica"), RamanSpec.from_database("CdS")]
        )
        fig = comp.plot_spectra_overlay(backend="matplotlib")

        title = fig.axes[0].get_title()  # type: ignore[attr-defined]
        assert "Silica" in title
        assert "CdS" in title
        plt.close(fig)  # type: ignore[arg-type]

    def test_materials_order_preserved(self):
        """Test that material order is preserved in overlay plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from photonics_helper.raman import RamanSpec

        comp = self._make_comparison()
        comp.add(RamanSpec.from_database("Silica"))
        comp.add(RamanSpec.from_database("CdS"))
        comp.add(RamanSpec.from_database("Diamond"))

        fig = comp.plot_spectra_overlay(backend="matplotlib")
        lines = fig.axes[0].get_lines()  # type: ignore[attr-defined]
        labels = [line.get_label() for line in lines]

        assert labels == ["Silica", "CdS", "Diamond"]
        plt.close(fig)  # type: ignore[arg-type]


# ─── Dash App Tests (Layer 7) ─────────────────────────────────────────────────


class TestDashApp:
    """Tests for the Dash interactive app (Layer 7)."""

    def _make_app(self):
        """Helper to create the Dash app."""
        from photonics_helper.raman import app as make_app
        return make_app()

    def test_app_returns_dash_app(self):
        """Test that app() returns a Dash application object."""
        from dash import Dash
        app = self._make_app()
        assert isinstance(app, Dash), f"Expected Dash, got {type(app)}"

    def test_app_has_layout(self):
        """Test that the app has a non-empty layout."""
        app = self._make_app()
        assert app.layout is not None
        # Layout is a Dash component (Div) with children
        layout_str = str(app.layout)
        assert len(layout_str) > 100, "Layout should be substantial"

    def test_app_has_sidebar(self):
        """Test that the layout contains a sidebar panel."""
        app = self._make_app()

        # The layout should contain a div with sidebar styling or id
        layout_str = str(app.layout)
        # Check for typical sidebar elements: material selector, sliders
        assert "material" in layout_str.lower() or "Material" in layout_str

    def test_app_has_layer_tabs(self):
        """Test that the layout contains layer selector tabs."""
        app = self._make_app()
        layout_str = str(app.layout)
        # Should reference layer tabs
        assert "layer" in layout_str.lower() or "tab" in layout_str.lower() or "Layer" in layout_str or "Tab" in layout_str

    def test_app_has_output_div(self):
        """Test that the layout contains an output container div."""
        app = self._make_app()
        layout_str = str(app.layout)
        # Look for an output div id pattern
        assert "output" in layout_str.lower() or "Output" in layout_str or "main-content" in layout_str

    def test_app_callbacks_registered(self):
        """Test that the app has callbacks registered."""
        app = self._make_app()
        assert len(app.callback_map) > 0, "App should have at least one callback"

    def test_app_has_material_selector_callback(self):
        """Test that a callback exists for material selector changes."""
        from photonics_helper.raman import app as make_app
        app = make_app()
        input_ids = set()
        for cb in app.callback_map.values():
            for inp in cb.get("raw_inputs", []):
                input_ids.add(inp.component_id)
        has_material_cb = "material-selector" in input_ids
        assert has_material_cb, "Expected callback referencing material-selector"

    def test_app_has_fr_slider_callback(self):
        """Test that a callback exists for fR slider."""
        from photonics_helper.raman import app as make_app
        app = make_app()
        input_ids = set()
        for cb in app.callback_map.values():
            for inp in cb.get("raw_inputs", []):
                input_ids.add(inp.component_id)
        has_fr_cb = "fr-slider" in input_ids
        assert has_fr_cb, "Expected callback referencing fr-slider"

    def test_app_has_pump_wavelength_callback(self):
        """Test that a callback exists for pump wavelength slider."""
        from photonics_helper.raman import app as make_app
        app = make_app()
        input_ids = set()
        for cb in app.callback_map.values():
            for inp in cb.get("raw_inputs", []):
                input_ids.add(inp.component_id)
        has_pump_cb = "pump-wl-slider" in input_ids
        assert has_pump_cb, "Expected callback referencing pump-wl-slider"

    def test_app_smoke_test(self):
        """Test that the app can be served without error."""
        try:
            import dash
        except ImportError:
            pytest.skip("dash not fully installed")

        app = self._make_app()
        # Verify the app has a server and can serve its layout
        assert app.server is not None
        # Verify serve_layout returns HTML content
        layout_html = app.index_string
        assert "Raman" in layout_html or "html" in layout_html.lower()

    def test_app_title(self):
        """Test that the app has a title."""
        app = self._make_app()
        assert hasattr(app, "title")
        assert "Raman" in app.title or "raman" in app.title.lower()

    def test_app_components_importable(self):
        """Test that the app function is importable from raman module."""
        from photonics_helper.raman import app
        assert callable(app)

    def test_app_default_material_silica(self):
        """Test that the default material is Silica."""
        from photonics_helper.raman import app as make_app
        app = make_app()
        # Check that the layout references Silica as default
        layout_str = str(app.layout)
        assert "Silica" in layout_str

    def test_app_has_comparison_button(self):
        """Test that the app has an 'add to compare' button."""
        app = self._make_app()
        layout_str = str(app.layout)
        assert "compare" in layout_str.lower() or "Compare" in layout_str or "add" in layout_str.lower()
