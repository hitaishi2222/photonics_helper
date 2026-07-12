"""Tests for the physical quantity classes in photonics_helper.base."""

import pytest
from pydantic import ValidationError

from photonics_helper.base import (
    C_MS,
    H_PLANCK,
    HBAR,
    PI,
    Area,
    Energy,
    Frequency,
    Length,
    Power,
    Time,
    Wavelength,
    Wavenumber,
    AngularFrequency,
)


class TestLength:
    def test_construction_m(self):
        L = Length(1.0, "m")
        assert L.as_m == 1.0

    def test_construction_km(self):
        L = Length(1.0, "km")
        assert L.as_m == 1000.0
        assert L.as_km == 1.0

    def test_construction_cm(self):
        L = Length(1.0, "cm")
        assert L.as_m == 0.01
        assert L.as_cm == 1.0

    def test_construction_mm(self):
        L = Length(1.0, "mm")
        assert L.as_m == 0.001
        assert L.as_mm == 1.0

    def test_construction_um(self):
        L = Length(1.0, "um")
        assert L.as_m == 1e-6
        assert L.as_um == 1.0

    def test_construction_nm(self):
        L = Length(1.0, "nm")
        assert L.as_m == 1e-9
        assert L.as_nm == 1.0

    def test_construction_pm(self):
        L = Length(1.0, "pm")
        assert L.as_m == 1e-12
        assert L.as_pm == 1.0

    def test_invalid_unit(self):
        with pytest.raises((ValidationError, ValueError)):
            Length(1.0, "angstrom")

    def test_repr(self):
        L = Length(1.55, "um")
        assert "Length" in repr(L)

    def test_str(self):
        L = Length(1.55, "um")
        s = str(L)
        assert "um" in s or "nm" in s or "m" in s

    def test_to_wl(self):
        L = Length(1550, "nm")
        wl = L.to_wl()
        assert isinstance(wl, Wavelength)
        assert wl.as_nm == pytest.approx(1550.0)

    def test_from_wl(self):
        wl = Wavelength(1550, "nm")
        L = Length.from_wl(wl)
        assert isinstance(L, Length)
        assert L.as_nm == pytest.approx(1550.0)

    def test_roundtrip_length_wavelength(self):
        L = Length(800, "nm")
        L2 = Length.from_wl(L.to_wl())
        assert L2.as_nm == pytest.approx(800.0)

    def test_meep_roundtrip(self):
        L = Length(2.0, "um")
        meep_val = L.as_meep
        L2 = Length.from_meep(meep_val)
        assert L2.as_um == pytest.approx(2.0)


class TestTime:
    def test_construction_s(self):
        t = Time(1.0, "s")
        assert t.as_s == 1.0

    def test_construction_ms(self):
        t = Time(1.0, "ms")
        assert t.as_s == 1e-3
        assert t.as_ms == 1.0

    def test_construction_us(self):
        t = Time(1.0, "us")
        assert t.as_s == 1e-6
        assert t.as_us == 1.0

    def test_construction_ns(self):
        t = Time(1.0, "ns")
        assert t.as_s == 1e-9
        assert t.as_ns == 1.0

    def test_construction_ps(self):
        t = Time(1.0, "ps")
        assert t.as_s == 1e-12
        assert t.as_ps == 1.0

    def test_construction_fs(self):
        t = Time(50.0, "fs")
        assert t.as_s == 50e-15
        assert t.as_fs == 50.0

    def test_construction_as(self):
        t = Time(1.0, "as")
        assert t.as_s == 1e-18
        assert t.as_as == 1.0

    def test_invalid_unit(self):
        with pytest.raises((ValidationError, ValueError)):
            Time(1.0, "minutes")

    def test_to_freq(self):
        t = Time(1.0, "s")
        f = t.to_freq()
        assert isinstance(f, Frequency)
        assert f.as_Hz == pytest.approx(1.0)

    def test_to_omega(self):
        t = Time(1.0, "s")
        w = t.to_omega()
        assert isinstance(w, AngularFrequency)
        assert w.as_rad_s == pytest.approx(2 * PI)

    def test_roundtrip_time_frequency(self):
        t = Time(100, "fs")
        t2 = Time.from_freq(t.to_freq())
        assert t2.as_fs == pytest.approx(100.0)

    def test_roundtrip_time_omega(self):
        t = Time(50, "fs")
        t2 = Time.from_omega(t.to_omega())
        assert t2.as_fs == pytest.approx(50.0)

    def test_meep_roundtrip(self):
        t = Time(1.0, "ps")
        meep_val = t.as_meep
        t2 = Time.from_meep(meep_val)
        assert t2.as_ps == pytest.approx(1.0)


class TestEnergy:
    def test_construction_J(self):
        E = Energy(1.0, "J")
        assert E.as_J == 1.0

    def test_construction_mJ(self):
        E = Energy(1.0, "mJ")
        assert E.as_J == 1e-3
        assert E.as_mJ == 1.0

    def test_construction_uJ(self):
        E = Energy(1.0, "uJ")
        assert E.as_J == 1e-6
        assert E.as_uJ == 1.0

    def test_construction_nJ(self):
        E = Energy(1.0, "nJ")
        assert E.as_J == 1e-9
        assert E.as_nJ == 1.0

    def test_construction_pJ(self):
        E = Energy(1.0, "pJ")
        assert E.as_J == 1e-12
        assert E.as_pJ == 1.0

    def test_construction_eV(self):
        E = Energy(1.0, "eV")
        assert E.as_eV == pytest.approx(1.0)
        assert E.as_J == pytest.approx(1.602176634e-19)

    def test_construction_meV(self):
        E = Energy(1.0, "meV")
        assert E.as_meV == pytest.approx(1.0)
        assert E.as_eV == pytest.approx(1e-3)

    def test_invalid_unit(self):
        with pytest.raises((ValidationError, ValueError)):
            Energy(1.0, "keV")

    def test_to_freq(self):
        E = Energy(1.0, "J")
        f = E.to_freq()
        assert isinstance(f, Frequency)
        assert f.as_Hz == pytest.approx(1.0 / H_PLANCK)

    def test_to_omega(self):
        E = Energy(1.0, "J")
        w = E.to_omega()
        assert isinstance(w, AngularFrequency)
        assert w.as_rad_s == pytest.approx(1.0 / HBAR)

    def test_to_wl(self):
        E = Energy(1.0, "J")
        wl = E.to_wl()
        assert isinstance(wl, Wavelength)
        assert wl.as_m == pytest.approx(H_PLANCK * C_MS)

    def test_to_wn(self):
        E = Energy(1.0, "J")
        wn = E.to_wn()
        assert isinstance(wn, Wavenumber)
        assert wn.as_1_m == pytest.approx(1.0 / (H_PLANCK * C_MS))

    def test_roundtrip_energy_freq(self):
        E = Energy(2.0, "eV")
        E2 = Energy.from_freq(E.to_freq())
        assert E2.as_eV == pytest.approx(2.0)

    def test_roundtrip_energy_omega(self):
        E = Energy(2.0, "eV")
        E2 = Energy.from_omega(E.to_omega())
        assert E2.as_eV == pytest.approx(2.0)

    def test_roundtrip_energy_wavelength(self):
        E = Energy(2.0, "eV")
        E2 = Energy.from_wl(E.to_wl())
        assert E2.as_eV == pytest.approx(2.0)

    def test_roundtrip_energy_wavenumber(self):
        E = Energy(2.0, "eV")
        E2 = Energy.from_wn(E.to_wn())
        assert E2.as_eV == pytest.approx(2.0)

    def test_meep_roundtrip(self):
        E = Energy(1.5, "eV")
        meep_val = E.as_meep
        E2 = Energy.from_meep(meep_val)
        assert E2.as_eV == pytest.approx(1.5)


class TestPower:
    def test_construction_W(self):
        P = Power(1.0, "W")
        assert P.as_W == 1.0

    def test_construction_kW(self):
        P = Power(1.0, "kW")
        assert P.as_W == 1000.0
        assert P.as_kW == 1.0

    def test_construction_mW(self):
        P = Power(1.0, "mW")
        assert P.as_W == 1e-3
        assert P.as_mW == 1.0

    def test_construction_uW(self):
        P = Power(1.0, "uW")
        assert P.as_W == 1e-6
        assert P.as_uW == 1.0

    def test_construction_nW(self):
        P = Power(1.0, "nW")
        assert P.as_W == 1e-9
        assert P.as_nW == 1.0

    def test_invalid_unit(self):
        with pytest.raises((ValidationError, ValueError)):
            Power(1.0, "MW")

    def test_repr(self):
        P = Power(100, "W")
        assert "Power" in repr(P)

    def test_str(self):
        P = Power(10, "kW")
        s = str(P)
        assert "kW" in s

    def test_soliton_peak_power(self):
        P = Power(416.8, "W")
        assert P.as_W == pytest.approx(416.8)
        assert P.as_mW == pytest.approx(416800.0)


class TestArea:
    def test_construction_m2(self):
        A = Area(1.0, "m^2")
        assert A.as_m2 == 1.0

    def test_construction_cm2(self):
        A = Area(1.0, "cm^2")
        assert A.as_m2 == 1e-4
        assert A.as_cm2 == 1.0

    def test_construction_mm2(self):
        A = Area(1.0, "mm^2")
        assert A.as_m2 == 1e-6
        assert A.as_mm2 == 1.0

    def test_construction_um2(self):
        A = Area(80.0, "um^2")
        assert A.as_m2 == 80e-12
        assert A.as_um2 == pytest.approx(80.0)

    def test_construction_nm2(self):
        A = Area(1.0, "nm^2")
        assert A.as_m2 == 1e-18
        assert A.as_nm2 == 1.0

    def test_invalid_unit(self):
        with pytest.raises((ValidationError, ValueError)):
            Area(1.0, "km^2")

    def test_repr(self):
        A = Area(80, "um^2")
        assert "Area" in repr(A)

    def test_str(self):
        A = Area(80, "um^2")
        s = str(A)
        assert "um" in s

    def test_fiber_eff_area(self):
        A = Area(80, "um^2")
        assert A.as_m2 == pytest.approx(80e-12)


class TestCrossTypeConversions:
    def test_wavelength_to_energy_roundtrip(self):
        wl = Wavelength(1064, "nm")
        E = wl.to_energy()
        wl2 = E.to_wl()
        assert wl2.as_nm == pytest.approx(1064.0)

    def test_frequency_to_time_roundtrip(self):
        f = Frequency(1e12, "Hz")
        t = f.to_time()
        f2 = t.to_freq()
        assert f2.as_Hz == pytest.approx(1e12)

    def test_angular_freq_to_time_roundtrip(self):
        w = AngularFrequency(1e15, "rad/s")
        t = w.to_time()
        assert t.as_s == pytest.approx(2 * PI / 1e15)

    def test_wavenumber_to_energy_roundtrip(self):
        wn = Wavenumber(10000, "1/cm")
        E = wn.to_energy()
        wn2 = E.to_wn()
        assert wn2.as_1_cm == pytest.approx(10000.0)

    def test_photon_energy_1064nm(self):
        wl = Wavelength(1064, "nm")
        E = wl.to_energy()
        assert E.as_eV == pytest.approx(1.165, rel=1e-3)

    def test_time_50fs_to_freq(self):
        t = Time(50, "fs")
        f = t.to_freq()
        assert f.as_THz == pytest.approx(20.0)

    def test_energy_1eV_to_wavelength(self):
        E = Energy(1.0, "eV")
        wl = E.to_wl()
        assert wl.as_nm == pytest.approx(1240.0, rel=1e-3)
