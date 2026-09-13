"""Paper-reproduction regression tests.

Each reproduction lives under ``reproductions/<name>/`` with a
``parameters.json`` and a ``reproduce.py::validate`` that asserts its result
against an analytic/closed-form reference.
"""

from reproductions.macleod_quarter_wave_dbr.reproduce import validate as validate_dbr
from reproductions.stolen_lin_1978_spm.reproduce import validate as validate_spm
from reproductions.gordon_1986_ssfs.reproduce import validate as validate_gordon
from reproductions.dudley_2006_cherenkov_dw.reproduce import validate as validate_dw


def test_stolen_lin_1978_spm():
    """SPM spectrum matches the closed-form Fourier integral and the phi/pi peak rule."""
    result = validate_spm(make_plot=False)
    assert result["max_abs_spectrum_diff"] < 1e-6
    for case in result["cases"]:
        assert case["n_peaks"] == case["n_peaks_expected"]


def test_macleod_quarter_wave_dbr():
    """Quarter-wave DBR reflectance matches the exact characteristic-matrix closed form."""
    result = validate_dbr(make_plot=False)
    assert result["max_peak_error"] < 1e-6
    assert result["R_lambda0"] > 0.999
    assert result["width_rel_error"] < 0.15


def test_gordon_1986_ssfs():
    """Raman soliton self-frequency shift matches the Gordon analytic rate."""
    result = validate_gordon(make_plot=False)
    assert result["measured_nm"] > 0.0
    assert 0.5 <= result["ratio"] <= 2.0


def test_dudley_2006_cherenkov_dw():
    """Dispersive-wave root finder and GNLSE DW peak match the analytic -3β₂/β₃ value."""
    result = validate_dw(make_plot=False)
    assert abs(result["lambda_root_nm"] - result["lambda_analytic_nm"]) / result["lambda_analytic_nm"] < 0.02
    assert result["rel_err"] < 0.05
