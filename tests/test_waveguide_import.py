"""Tests for FEM ``n_eff(λ)`` import via :class:`WaveguideMode`.

Covers the CSV / NPZ conventions, the spline validation rules, and the
end-to-end check that a loaded table yields a ``PropagationConstant`` whose
``beta2`` matches the analytic second derivative.
"""

from pathlib import Path

import numpy as np
import pytest

from photonics_helper import WaveguideMode
from photonics_helper.base import Wavelength, WavelengthArray
from photonics_helper.fiber import Dispersion, PropagationConstant

N0, A_SLOPE, B_CURV, LAM0_UM = 2.42, -0.1, 0.05, 1.55


def _neff_um(wl_um: np.ndarray) -> np.ndarray:
    """Analytic effective index n_eff(λ) used as ground truth."""
    d = np.asarray(wl_um, dtype=float) - LAM0_UM
    return N0 + A_SLOPE * d + B_CURV * d**2


def _write_csv(path, header: str | None = None, n_points: int = 41, columns: int = 2):
    wl = np.linspace(1.45, 1.65, n_points)
    neff = _neff_um(wl)
    with open(path, "w") as fh:
        if header is not None:
            fh.write(header + "\n")
        for i, (x, y) in enumerate(zip(wl, neff)):
            if columns == 2:
                fh.write(f"{x:.6f},{y:.10f}\n")
            else:
                fh.write(f"{x:.6f},{y:.10f},{N0:.10f}\n")
    return wl, neff


def test_from_csv_with_header(tmp_path):
    csv = tmp_path / "mode.csv"
    _write_csv(csv, header="wavelength_um, neff")
    mode = WaveguideMode.from_csv(csv)
    assert len(mode.neff) == 41
    assert mode.ng is None
    assert mode.wavelengths.as_um.min() == pytest.approx(1.45)
    # default central wavelength is the mid-point
    assert mode.central_wavelength is not None
    assert mode.central_wavelength.as_nm == pytest.approx(1550.0)


def test_from_csv_without_header_and_ng_column(tmp_path):
    csv = tmp_path / "mode.csv"
    _write_csv(csv, header=None, columns=3)
    mode = WaveguideMode.from_csv(csv)
    assert mode.ng is not None
    assert mode.ng.shape == mode.neff.shape
    np.testing.assert_allclose(mode.ng, N0)


def test_from_csv_explicit_central_wavelength(tmp_path):
    csv = tmp_path / "mode.csv"
    _write_csv(csv, header="wavelength_um, neff")
    mode = WaveguideMode.from_csv(csv, central_wavelength=Wavelength(1550, "nm"))
    assert mode.central_wavelength is not None
    assert mode.central_wavelength.as_nm == pytest.approx(1550.0)


def test_from_csv_too_few_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("1.5\n1.6\n")
    with pytest.raises(ValueError):
        WaveguideMode.from_csv(csv)


def test_from_npz_round_trip(tmp_path):
    wl = np.linspace(1.45, 1.65, 31)
    neff = _neff_um(wl)
    npz = tmp_path / "mode.npz"
    np.savez(
        npz,
        wavelength_um=wl,
        neff=neff,
        ng=np.full_like(wl, N0),
        central_wavelength_nm=1550.0,
    )
    mode = WaveguideMode.from_npz(npz)
    np.testing.assert_allclose(mode.neff, neff)
    assert mode.ng is not None
    assert mode.central_wavelength is not None
    assert mode.central_wavelength.as_nm == pytest.approx(1550.0)


def test_from_npz_central_wavelength_um(tmp_path):
    wl = np.linspace(1.45, 1.65, 11)
    npz = tmp_path / "mode.npz"
    np.savez(npz, wavelength_um=wl, neff=_neff_um(wl), central_wavelength_um=1.55)
    mode = WaveguideMode.from_npz(npz)
    assert mode.central_wavelength is not None
    assert mode.central_wavelength.as_nm == pytest.approx(1550.0)


def test_from_npz_missing_keys(tmp_path):
    npz = tmp_path / "bad.npz"
    np.savez(npz, wavelength_um=np.linspace(1.45, 1.65, 5))
    with pytest.raises(ValueError):
        WaveguideMode.from_npz(npz)


def test_validation_requires_four_points():
    with pytest.raises(ValueError):
        WaveguideMode(
            neff=np.array([1.0, 1.1, 1.2]),
            wavelengths=WavelengthArray(np.array([1.5, 1.55, 1.6]), "um"),
        )


def test_validation_requires_monotonic_wavelengths(tmp_path):
    csv = tmp_path / "nonmono.csv"
    csv.write_text(
        "wavelength_um, neff\n"
        "1.50, 2.42\n"
        "1.55, 2.41\n"
        "1.52, 2.415\n"
        "1.60, 2.40\n"
    )
    with pytest.raises(ValueError):
        WaveguideMode.from_csv(csv)


def test_validation_length_mismatch(tmp_path):
    csv = tmp_path / "mismatch.csv"
    csv.write_text(
        "wavelength_um, neff\n"
        "1.50, 2.42\n"
        "1.55, 2.41\n"
        "1.60, 2.40\n"
        "1.65, 2.39\n"
    )
    mode = WaveguideMode.from_csv(csv)
    assert len(mode.neff) == 4
    with pytest.raises(ValueError):
        WaveguideMode(
            neff=np.array([1.0, 1.1, 1.2, 1.3]),
            wavelengths=WavelengthArray(
                np.array([1.50, 1.55, 1.60, 1.65, 1.70]), "um"
            ),
        )


def test_validation_rejects_non_finite(tmp_path):
    csv = tmp_path / "nan.csv"
    csv.write_text(
        "wavelength_um, neff\n"
        "1.50, 2.42\n"
        "1.55, nan\n"
        "1.60, 2.40\n"
        "1.65, 2.39\n"
    )
    with pytest.raises(ValueError):
        WaveguideMode.from_csv(csv)


def test_neff_at_and_range(tmp_path):
    csv = tmp_path / "mode.csv"
    _write_csv(csv, header="wavelength_um, neff")
    mode = WaveguideMode.from_csv(csv)
    assert mode.neff_at(Wavelength(1550, "nm")) == pytest.approx(N0, abs=1e-6)
    with pytest.raises(ValueError):
        mode.neff_at(Wavelength(2000, "nm"))


def test_to_propagation_constant_beta2_matches_analytic(tmp_path):
    """Spec scenario: beta2 from the imported table matches the analytic value <2%."""
    csv = tmp_path / "mode.csv"
    _write_csv(csv, header="wavelength_um, neff")
    pc = WaveguideMode.from_csv(csv).to_propagation_constant()
    assert isinstance(pc, PropagationConstant)

    # Analytic beta2 via central differences of beta(omega) on a fine grid.
    wl_fine_um = np.linspace(1.45, 1.65, 20001)
    omega = 2 * np.pi * 2.99792458e8 / (wl_fine_um * 1e-6)
    beta = _neff_um(wl_fine_um) * omega / 2.99792458e8
    beta2_fine = np.gradient(np.gradient(beta, omega), omega)
    idx = int(np.argmin(np.abs(wl_fine_um - LAM0_UM)))
    analytic = float(beta2_fine[idx])

    measured = pc.beta2(Wavelength(1550, "nm"))
    assert abs(measured - analytic) / abs(analytic) < 0.02


def test_beta2_requires_four_points():
    pc = PropagationConstant.beta_from_neff(
        neff=np.array([1.0, 1.1, 1.2]),
        x_values=WavelengthArray(np.array([1.5, 1.55, 1.6]), "um"),
    )
    with pytest.raises(ValueError):
        pc.beta2(Wavelength(1550, "nm"))


def test_to_dispersion(tmp_path):
    csv = tmp_path / "mode.csv"
    _write_csv(csv, header="wavelength_um, neff")
    disp = WaveguideMode.from_csv(csv).to_dispersion()
    assert isinstance(disp, Dispersion)
    assert np.all(np.isfinite(disp.as_ps_nm_km))


def test_waveguide_mode_is_exported():
    import photonics_helper

    assert "WaveguideMode" in photonics_helper.__all__


def test_committed_solver_exports_agree():
    """The committed femwell / Tidy3D exports load and agree within 2%."""
    data = Path(__file__).resolve().parents[1] / "examples" / "data"
    fw = WaveguideMode.from_csv(data / "si_strip_femwell.csv")
    td = WaveguideMode.from_npz(data / "si_strip_tidy3d.npz")
    assert fw.neff.shape == td.neff.shape
    np.testing.assert_allclose(fw.wavelengths.as_um, td.wavelengths.as_um)
    rel = np.abs(fw.neff - td.neff) / td.neff
    assert rel.max() < 0.02
    assert fw.neff_at(Wavelength(1550, "nm")) > td.neff_at(Wavelength(1550, "nm"))
