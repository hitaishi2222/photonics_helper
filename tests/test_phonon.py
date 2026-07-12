"""Tests for the phonon module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from photonics_helper.phonon import PhononMode, PhononResponse, PHONON_MATERIALS
from photonics_helper.raman import RamanSpec, RamanDatabase
from photonics_helper.base import Wavelength


class TestPhononMode:
    """Tests for PhononMode dataclass."""

    def test_creation(self):
        mode = PhononMode(shift_cm=254, linewidth_cm=14, symmetry="A₁g")
        assert mode.shift_cm == 254
        assert mode.linewidth_cm == 14
        assert mode.symmetry == "A₁g"

    def test_defaults(self):
        mode = PhononMode(shift_cm=254, linewidth_cm=14)
        assert mode.relative_strength == 1.0
        assert mode.lo_phonon_cm is None
        assert mode.to_phonon_cm is None
        assert mode.note is None

    def test_all_fields(self):
        mode = PhononMode(
            shift_cm=254, linewidth_cm=14, symmetry="A₁g",
            relative_strength=0.8, lo_phonon_cm=260, to_phonon_cm=250,
            note="Test mode"
        )
        assert mode.relative_strength == 0.8
        assert mode.lo_phonon_cm == 260
        assert mode.to_phonon_cm == 250
        assert mode.note == "Test mode"


class TestPhononResponse:
    """Tests for PhononResponse class."""

    def test_empty_modes(self):
        response = PhononResponse([])
        w = np.linspace(-100, 100, 100)
        result = response.frequency_domain(w)
        assert np.allclose(result, 0)

    def test_single_mode_peak(self):
        mode = PhononMode(shift_cm=254, linewidth_cm=14, relative_strength=1.0)
        response = PhononResponse([mode])
        w = np.linspace(200, 300, 1000)
        result = response.frequency_domain(w)
        peak_idx = np.argmax(result)
        peak_shift = w[peak_idx]
        assert abs(peak_shift - 254) < 5  # Peak near 254 cm⁻¹

    def test_multi_mode_peaks(self):
        modes = [
            PhononMode(shift_cm=200, linewidth_cm=10, relative_strength=1.0),
            PhononMode(shift_cm=400, linewidth_cm=20, relative_strength=0.5),
        ]
        response = PhononResponse(modes)
        w = np.linspace(-100, 500, 10000)
        result = response.frequency_domain(w)

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(result, distance=50)
        assert len(peaks) == 2

    def test_normalization(self):
        modes = [
            PhononMode(shift_cm=200, linewidth_cm=10, relative_strength=1.0),
            PhononMode(shift_cm=400, linewidth_cm=20, relative_strength=0.5),
        ]
        response = PhononResponse(modes)
        w = np.linspace(-100, 500, 10000)
        result = response.frequency_domain(w)
        assert abs(np.max(result) - 1.0) < 0.01

    def test_fR_default(self):
        modes = [
            PhononMode(shift_cm=200, linewidth_cm=10, relative_strength=0.6),
            PhononMode(shift_cm=400, linewidth_cm=20, relative_strength=0.4),
        ]
        response = PhononResponse(modes)
        assert response.fR == pytest.approx(1.0)

    def test_fR_custom(self):
        modes = [PhononMode(shift_cm=200, linewidth_cm=10)]
        response = PhononResponse(modes, fR=0.5)
        assert response.fR == 0.5

    def test_time_domain(self):
        mode = PhononMode(shift_cm=254, linewidth_cm=14)
        response = PhononResponse([mode])
        t = np.linspace(0, 1e-12, 1000)
        result = response.time_domain(t)
        assert len(result) == len(t)
        # Time domain should not be all zeros
        assert not np.allclose(result, 0)

    def test_demo(self):
        response = PhononResponse([
            PhononMode(shift_cm=200, linewidth_cm=10),
            PhononMode(shift_cm=400, linewidth_cm=20),
        ])
        results = response.demo()
        assert results['empty_freq']
        assert results['single_peak']
        assert results['multi_peaks']
        assert results['normalized']


class TestPHONON_MATERIALS:
    """Tests for seeded phonon data."""

    def test_all_materials_present(self):
        expected = ["LiNbO3", "LiTaO3", "BaTiO3", "YAG", "Al2O3", "KTP", "GaN", "AlN", "SiC_4H", "YLF"]
        for mat in expected:
            assert mat in PHONON_MATERIALS, f"Missing material: {mat}"

    def test_modes_have_valid_params(self):
        for mat, modes in PHONON_MATERIALS.items():
            for mode in modes:
                assert mode.shift_cm > 0, f"{mat}: shift_cm must be positive"
                assert mode.linewidth_cm > 0, f"{mat}: linewidth_cm must be positive"

    def test_li_nbo3_has_7_modes(self):
        assert len(PHONON_MATERIALS["LiNbO3"]) == 7

    def test_ktp_stores_subset(self):
        # KTP has 189 modes in literature, we store top ~5
        assert len(PHONON_MATERIALS["KTP"]) <= 15

    def test_yag_has_modes(self):
        assert len(PHONON_MATERIALS["YAG"]) >= 5


class TestRamanSpecPhononIntegration:
    """Tests for RamanSpec phonon integration."""

    def test_phonon_modes_default_none(self):
        spec = RamanSpec(name="Test", raman_shift_cm=254, raman_linewidth_cm=14)
        assert spec.phonon_modes is None

    def test_phonon_response_none_without_modes(self):
        spec = RamanSpec(name="Test", raman_shift_cm=254, raman_linewidth_cm=14)
        assert spec.phonon_response is None

    def test_phonon_response_with_modes(self):
        modes = [PhononMode(shift_cm=254, linewidth_cm=14)]
        spec = RamanSpec(name="Test", raman_shift_cm=254, raman_linewidth_cm=14, phonon_modes=modes)
        response = spec.phonon_response
        assert response is not None
        assert isinstance(response, PhononResponse)

    def test_multi_stokes_wavelengths(self):
        modes = [
            PhononMode(shift_cm=200, linewidth_cm=10),
            PhononMode(shift_cm=400, linewidth_cm=20),
        ]
        spec = RamanSpec(name="Test", raman_shift_cm=254, raman_linewidth_cm=14, phonon_modes=modes)
        pump = Wavelength(1064, "nm")
        stokes = spec.multi_stokes_wavelengths(pump)
        assert len(stokes) == 2
        # Stokes wavelengths should be longer than pump
        for wl in stokes:
            assert wl.as_nm > 1064

    def test_multi_stokes_empty_without_modes(self):
        spec = RamanSpec(name="Test", raman_shift_cm=254, raman_linewidth_cm=14)
        pump = Wavelength(1064, "nm")
        stokes = spec.multi_stokes_wavelengths(pump)
        assert stokes == []

    def test_summary_mentions_multimode(self):
        modes = [PhononMode(shift_cm=254, linewidth_cm=14)]
        spec = RamanSpec(name="Test", raman_shift_cm=254, raman_linewidth_cm=14, phonon_modes=modes)
        summary = spec.summary()
        assert "Multi-mode: 1 phonon modes" in summary

    def test_summary_without_multimode(self):
        spec = RamanSpec(name="Test", raman_shift_cm=254, raman_linewidth_cm=14)
        summary = spec.summary()
        assert "Multi-mode" not in summary


class TestRamanDatabasePhonon:
    """Tests for RamanDatabase phonon CRUD."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.db = RamanDatabase(db_path=self.db_path)

    def test_phonon_modes_table_created(self):
        # Table should exist after init
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='phonon_modes'")
        table = cursor.fetchone()
        conn.close()
        assert table is not None

    def test_add_phonon_mode(self):
        mode = PhononMode(shift_cm=254, linewidth_cm=14, symmetry="A₁g")
        self.db.add_phonon_mode("LiNbO3", mode)

        modes = self.db.get_phonon_modes("LiNbO3")
        assert len(modes) == 1
        assert modes[0].shift_cm == 254
        assert modes[0].symmetry == "A₁g"

    def test_get_phonon_modes_multiple(self):
        modes_data = [
            PhononMode(shift_cm=200, linewidth_cm=10, symmetry="A"),
            PhononMode(shift_cm=400, linewidth_cm=20, symmetry="B"),
        ]
        for mode in modes_data:
            self.db.add_phonon_mode("YAG", mode)

        modes = self.db.get_phonon_modes("YAG")
        assert len(modes) == 2

    def test_list_phonon_materials(self):
        self.db.add_phonon_mode("LiNbO3", PhononMode(shift_cm=254, linewidth_cm=14))
        self.db.add_phonon_mode("YAG", PhononMode(shift_cm=784, linewidth_cm=8))

        materials = self.db.list_phonon_materials()
        assert "LiNbO3" in materials
        assert "YAG" in materials

    def test_seed_phonon_data(self):
        count = self.db.seed_phonon_data()
        assert count > 0
        # Check that at least one material has modes
        materials = self.db.list_phonon_materials()
        assert len(materials) > 0

    def test_type_check(self):
        with pytest.raises(TypeError):
            self.db.add_phonon_mode("Test", {"shift_cm": 254})  # Not a PhononMode


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_single_mode_still_works(self):
        """Single-mode RamanSpec should work without phonon_modes."""
        spec = RamanSpec(
            name="Silica",
            raman_shift_cm=440,
            raman_linewidth_cm=45,
            fR=0.18,
        )
        assert spec.phonon_modes is None
        assert spec.phonon_response is None
        assert spec.raman_shift_cm == 440
        assert spec.raman_linewidth_cm == 45

    def test_from_database_without_phonon_modes(self):
        """from_database should work without phonon_modes."""
        spec = RamanSpec.from_database("Silica")
        # Should not have phonon_modes (unless seeded)
        assert spec.name == "Silica"
