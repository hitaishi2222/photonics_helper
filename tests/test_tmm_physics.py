"""Physics regression tests for the DBR/TMM implementation.

These tests use independent reference methods that do not share code with
``photonics_helper.dbr``:

* recursive Airy / multiple-reflection formula, and
* (E+, E-) scattering-matrix propagation,

both validated against the analytic single-layer closed form
``r = (r01 + r12 e^{2iδ}) / (1 + r01 r12 e^{2iδ})``.

The point is to catch the historical reversed-product-order bug, which is
invisible at normal incidence on a lossless stack (reciprocity) but wrong for
absorbing and oblique stacks.
"""

import numpy as np
import pytest

from photonics_helper.base import Length, Wavelength, WavelengthArray
from photonics_helper.dbr import Block, Material, Pattern, TMM


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------

def _make_material(n: float, k: float = 0.0, name: str = "m") -> Material:
    wl = WavelengthArray(np.linspace(1000, 2000, 51), "nm")
    return Material(name=name, n=np.full(51, n), k=np.full(51, k), wl=wl)


def _admittance(n: complex, theta0: float, pol: str) -> complex:
    s = np.sin(theta0) / n
    cos_t = np.sqrt(1 - s**2)
    return n * cos_t if pol == "TE" else n / cos_t


def _airy_reflection(
    layers, lam: float, theta0: float, pol: str, n0=1.0 + 0j, ns=1.0 + 0j
) -> complex:
    """Recursive multiple-reflection (Airy) amplitude reflection coefficient."""
    y0 = _admittance(n0, theta0, pol)
    y_last = _admittance(layers[-1][0], theta0, pol)
    # The exit medium is a semi-infinite medium at the Snell-refracted exit
    # angle, so its admittance uses theta0 (n_incident = n0) not normal incidence.
    ys = _admittance(ns, theta0, pol)
    r_down = (y_last - ys) / (y_last + ys)
    for idx in range(len(layers) - 1, -1, -1):
        n, d = layers[idx]
        yj = _admittance(n, theta0, pol)
        s = np.sin(theta0) / n
        cos_t = np.sqrt(1 - s**2)
        delta = 2 * np.pi * n * d * cos_t / lam
        y_above = y0 if idx == 0 else _admittance(layers[idx - 1][0], theta0, pol)
        r_top = (y_above - yj) / (y_above + yj)
        r_down = (r_top + r_down * np.exp(2j * delta)) / (
            1 + r_top * r_down * np.exp(2j * delta)
        )
    return r_down


def _build_pattern(mats, lengths, style="ABC") -> Pattern:
    mapping = {
        c: Block(length=Length(lengths[c], "m"), material=_make_material(*mats[c]))
        for c in "ABC"
    }
    return Pattern(
        style=style, mapping=mapping, central_wavelength=Wavelength(1550, "nm")
    )


LOSSY = {"A": (2.0, 0.05), "B": (1.5, 0.02), "C": (3.0, 0.1)}
LOSSLESS = {"A": (2.0, 0.0), "B": (1.5, 0.0), "C": (3.0, 0.0)}
LENGTHS = {"A": 180e-9, "B": 250e-9, "C": 120e-9}
LAM = 1550e-9
WL1 = WavelengthArray(np.array([1550.0]), "nm")


# ---------------------------------------------------------------------------
# N1 — transfer-matrix ordering
# ---------------------------------------------------------------------------

def test_lossy_asymmetric_stack_matches_reference():
    """Absorbing asymmetric stack must not alias its reversed counterpart."""
    pattern = _build_pattern(LOSSY, LENGTHS, style="ABC")
    tmm = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    R, _ = tmm.spectrum(WL1)
    layers = [(LOSSY[c][0] + 1j * LOSSY[c][1], LENGTHS[c]) for c in "ABC"]
    ref = abs(_airy_reflection(layers, LAM, 0.0, "TE")) ** 2
    assert np.isclose(R[0], ref, atol=1e-9)


def test_reversed_stack_gives_different_reflectance():
    """A lossy asymmetric stack and its reverse must differ, and match the ref."""
    fwd = _build_pattern(LOSSY, LENGTHS, style="ABC")
    rev = _build_pattern(LOSSY, LENGTHS, style="CBA")
    R_fwd, _ = TMM(fwd, 0.0, "TE").spectrum(WL1)
    R_rev, _ = TMM(rev, 0.0, "TE").spectrum(WL1)
    assert not np.isclose(R_fwd[0], R_rev[0], atol=1e-6)
    layers_rev = [(LOSSY[c][0] + 1j * LOSSY[c][1], LENGTHS[c]) for c in "CBA"]
    ref_rev = abs(_airy_reflection(layers_rev, LAM, 0.0, "TE")) ** 2
    assert np.isclose(R_rev[0], ref_rev, atol=1e-9)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_lossless_oblique_matches_reference(pol):
    """Lossless oblique incidence exposes ordering errors (no reciprocity shortcut)."""
    pattern = _build_pattern(LOSSLESS, LENGTHS, style="ABC")
    theta = np.deg2rad(40.0)
    tmm = TMM(pattern=pattern, angle_of_incidence=theta, polarisation=pol)
    R, _ = tmm.spectrum(WL1)
    layers = [(LOSSLESS[c][0], LENGTHS[c]) for c in "ABC"]
    ref = abs(_airy_reflection(layers, LAM, theta, pol)) ** 2
    assert np.isclose(R[0], ref, atol=1e-9)


def test_lossless_normal_is_reciprocal():
    """At normal incidence a lossless stack reflects identically from both sides."""
    fwd = _build_pattern(LOSSLESS, LENGTHS, style="ABC")
    rev = _build_pattern(LOSSLESS, LENGTHS, style="CBA")
    R_fwd, _ = TMM(fwd, 0.0, "TE").spectrum(WL1)
    R_rev, _ = TMM(rev, 0.0, "TE").spectrum(WL1)
    assert np.isclose(R_fwd[0], R_rev[0], atol=1e-10)


# ---------------------------------------------------------------------------
# N2 — field-profile forward propagation
# ---------------------------------------------------------------------------

def test_field_profile_matches_analytic_single_layer():
    """Non-quarter-wave layer must reproduce the analytic transmitted field."""
    n0, n1, n2 = 1.0, 2.0, 1.0
    d = LAM / 16  # delta = pi/4
    mapping = {"A": Block(length=Length(d, "m"), material=_make_material(n1))}
    pattern = Pattern(
        style="A", mapping=mapping, central_wavelength=Wavelength(1550, "nm")
    )
    tmm = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    field = tmm.field_profile(Wavelength(LAM, "m"))
    n0, n1, n2 = 1.0, 2.0, 1.0
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    delta = 2 * np.pi * n1 * d / LAM
    r_full = (r01 + r12 * np.exp(2j * delta)) / (1 + r01 * r12 * np.exp(2j * delta))
    assert np.isclose(field[0], abs(1 + r_full), atol=1e-9)


def test_field_profile_two_layer_matches_analytic():
    """Interface field after a layer equals the analytic transmitted amplitude."""
    n1 = 2.0
    d = LAM / 16
    mapping = {
        "A": Block(length=Length(d, "m"), material=_make_material(n1)),
        "B": Block(length=Length(LAM / 4, "m"), material=_make_material(1.0)),
    }
    pattern = Pattern(
        style="AB", mapping=mapping, central_wavelength=Wavelength(1550, "nm")
    )
    tmm = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    positions, field = tmm.field_profile(Wavelength(LAM, "m"), return_positions=True)
    n0, n2 = 1.0, 1.0
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    t01 = 2 * n0 / (n0 + n1)
    t12 = 2 * n1 / (n1 + n2)
    delta = 2 * np.pi * n1 * d / LAM
    t_full = t01 * t12 * np.exp(1j * delta) / (1 + r01 * r12 * np.exp(2j * delta))
    assert field.shape[0] == 2
    assert len(positions) == 2
    assert np.isclose(positions[1].as_m, d, rtol=1e-12)
    assert np.isclose(field[1], abs(t_full), rtol=1e-9)


def test_absorbing_stack_energy_deficit():
    """Absorption means 1 - R - T > 0 for a lossy stack."""
    pattern = _build_pattern(LOSSY, LENGTHS, style="ABC")
    tmm = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    R, T = tmm.spectrum(WL1)
    assert R[0] + T[0] < 1.0
    assert 1.0 - R[0] - T[0] > 0.0


# ---------------------------------------------------------------------------
# Configurable media + continuous field profile
# ---------------------------------------------------------------------------

def test_default_media_match_explicit_air():
    """Omitting n_incident/n_substrate equals explicitly setting air."""
    pattern = _build_pattern(LOSSLESS, LENGTHS, style="ABC")
    default = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    explicit = TMM(
        pattern=pattern,
        angle_of_incidence=0.0,
        polarisation="TE",
        n_incident=1.0 + 0j,
        n_substrate=1.0 + 0j,
    )
    R_d, T_d = default.spectrum(WL1)
    R_e, T_e = explicit.spectrum(WL1)
    assert np.isclose(R_d[0], R_e[0], atol=1e-14)
    assert np.isclose(T_d[0], T_e[0], atol=1e-14)


def test_quarter_wave_stack_on_substrate_matches_reference():
    """A quarter-wave stack on glass matches the recursive Airy reference."""
    n_H, n_L, n_s = 2.0, 1.5, 1.5
    d_H = LAM / (4 * n_H)
    d_L = LAM / (4 * n_L)
    mapping = {
        "A": Block(length=Length(d_H, "m"), material=_make_material(n_H)),
        "B": Block(length=Length(d_L, "m"), material=_make_material(n_L)),
    }
    pattern = Pattern(
        style="ABAB", mapping=mapping, central_wavelength=Wavelength(1550, "nm")
    )
    tmm = TMM(
        pattern=pattern,
        angle_of_incidence=0.0,
        polarisation="TE",
        n_substrate=n_s,
    )
    R, T = tmm.spectrum(WL1)
    layers = [(n_H, d_H), (n_L, d_L), (n_H, d_H), (n_L, d_L)]
    ref = abs(_airy_reflection(layers, LAM, 0.0, "TE", ns=n_s)) ** 2
    assert np.isclose(R[0], ref, atol=1e-9)
    assert R[0] + T[0] == pytest.approx(1.0, abs=1e-9)


def test_oblique_substrate_matches_reference():
    """Oblique incidence on a substrate matches the reference for TE and TM."""
    n_s = 1.5
    theta = np.deg2rad(35.0)
    pattern = _build_pattern(LOSSLESS, LENGTHS, style="ABC")
    layers = [(LOSSLESS[c][0], LENGTHS[c]) for c in "ABC"]
    for pol in ("TE", "TM"):
        tmm = TMM(
            pattern=pattern,
            angle_of_incidence=theta,
            polarisation=pol,
            n_substrate=n_s,
        )
        R, T = tmm.spectrum(WL1)
        ref = abs(_airy_reflection(layers, LAM, theta, pol, ns=n_s)) ** 2
        assert np.isclose(R[0], ref, atol=1e-9)
        assert R[0] + T[0] == pytest.approx(1.0, abs=1e-9)


def test_field_profile_z_spans_stack_and_matches_interfaces():
    """Continuous profile spans [0, L] and matches interface values."""
    pattern = _build_pattern(LOSSLESS, LENGTHS, style="ABC")
    tmm = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    wl = Wavelength(LAM, "m")
    z, E = tmm.field_profile_z(wl, n_points_per_layer=15)

    total = pattern.length
    assert z[0] == 0.0
    assert np.isclose(z[-1], total, rtol=1e-12)
    assert np.all(np.diff(z) > 0)

    interfaces = tmm.field_profile(wl)
    for idx, pos in enumerate(pattern._get_positions()):
        z0 = pos[0]
        j = int(np.argmin(np.abs(z - z0)))
        assert np.isclose(z[j], z0, atol=1e-15)
        assert np.isclose(E[j], interfaces[idx], rtol=1e-9, atol=1e-12)


def test_field_profile_z_decays_in_strong_absorber():
    """A strongly absorbing layer attenuates |E(z)| from entrance to exit."""
    d = 1.5e-6
    mapping = {"A": Block(length=Length(d, "m"), material=_make_material(2.0, 0.5))}
    pattern = Pattern(
        style="A", mapping=mapping, central_wavelength=Wavelength(1550, "nm")
    )
    tmm = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    z, E = tmm.field_profile_z(Wavelength(LAM, "m"), n_points_per_layer=41)
    assert np.isclose(z[-1], d, rtol=1e-12)
    assert E[-1] < 0.2 * E[0]
    assert E[-1] < 0.1


def test_field_profile_z_rejects_too_few_points():
    pattern = _build_pattern(LOSSLESS, LENGTHS, style="A")
    tmm = TMM(pattern=pattern, angle_of_incidence=0.0, polarisation="TE")
    with pytest.raises(ValueError):
        tmm.field_profile_z(Wavelength(LAM, "m"), n_points_per_layer=1)
