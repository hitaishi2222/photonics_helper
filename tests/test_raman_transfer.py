"""Regression tests for the Raman closed forms (Keita 2006 / Zhu 2007).

These pin the analytic layer in ``photonics_helper.raman_transfer`` that the
retired solver-vs-closed benchmark used to check: the Eq. 4.5 → Eq. 4.6 limit,
the Eq. 4.7 gated point, the Zhu Eq. 7 corner/DC relations, the dB-convention
guards (the ``/10`` vs ``/20`` class of past mistakes), and the benchmark
fixture loading itself.
"""

import numpy as np
import pytest

from photonics_helper.base import Length
from photonics_helper.raman_transfer import (
    Geometry,
    ProvenanceError,
    db_from_linear,
    effective_length,
    group_index,
    load_benchmark,
    rin_transfer_dispersion,
    rin_transfer_from_net_gain,
    rin_transfer_low_frequency,
    rin_transfer_monochromatic_pump,
    single_pump_corner_frequency,
    single_pump_transfer,
    walk_off_parameter,
)


def test_eq45_reduces_to_eq46_at_zero_walkoff():
    """Eq. 4.5 at Δk = 0 is exactly Eq. 4.6 (the (1 − e^{−αL})² form)."""
    rho_45 = rin_transfer_monochromatic_pump(4.74e-3, 0.2e-3, 0.0, 1000.0)
    rho_46 = rin_transfer_low_frequency(4.74e-3, 0.2e-3, 1000.0)
    assert rho_45 == pytest.approx(rho_46)


def test_eq45_near_zero_walkoff_matches_eq46():
    """Eq. 4.5 tends to Eq. 4.6 as Δk → 0 (the documented exact limit)."""
    rho_45 = rin_transfer_monochromatic_pump(4.74e-3, 0.2e-3, 1e-12, 1000.0)
    rho_46 = rin_transfer_low_frequency(4.74e-3, 0.2e-3, 1000.0)
    assert rho_45 == pytest.approx(rho_46, rel=1e-9)


def test_eq47_gated_point():
    """Eq. 4.7 at the paper's Fig. 1 point (g_net = 40 dB).

    The 20/ln10 prefactor with the 10 log10 power convention is the
    historically fragile part (amplitude-vs-power slips cost 6 dB); pin it.
    """
    assert rin_transfer_from_net_gain(40.0, 0.0002, 1000.0) == pytest.approx(
        13.451506729543226
    )


def test_eq47_rejects_nonpositive_bracket():
    """Eq. 4.7 raises when the log bracket is not positive."""
    with pytest.raises(ValueError, match="bracket"):
        rin_transfer_from_net_gain(-100.0, 0.0002, 1000.0)


def test_walk_off_geometry_signs():
    """Co-propagating walk-off cancels for equal velocities; counter adds."""
    assert walk_off_parameter(5e-9, 5e-9, 1e6, Geometry.CO_PROPAGATING) == 0.0
    assert walk_off_parameter(5e-9, 5e-9, 1e6, Geometry.COUNTER_PROPAGATING) == (
        pytest.approx(10e-9 * 1e6)
    )


def test_db_conventions():
    """Amplitude convention is 20 log10, power (Keita Eq. 4.7) is 10 log10."""
    assert db_from_linear(10.0, amplitude=True) == pytest.approx(20.0)
    assert db_from_linear(10.0, amplitude=False) == pytest.approx(10.0)


def test_effective_length_limits():
    """Lossless effective length is the physical length; lossy is exact."""
    assert effective_length(0.0, Length(1000.0, "m")).as_m == pytest.approx(1000.0)
    alpha, length = 0.2e-3, 1000.0
    assert effective_length(alpha, Length(length, "m")).as_m == pytest.approx(
        (1.0 - np.exp(-alpha * length)) / alpha
    )


def test_zhu_dc_and_corner_relation():
    """Zhu Eq. 7: DC level is (ln G_R/(α L_eff))² and the corner halves it."""
    alpha_p, v_s, l_eff, g_db = 6.43e-5, 2.05e8, 18000.0, 9.1
    corner = single_pump_corner_frequency(alpha_p, v_s)
    assert corner == pytest.approx(alpha_p * v_s / (4.0 * np.pi))
    transfer = single_pump_transfer(np.array([0.0, corner]), g_db, alpha_p, v_s, l_eff)
    g_linear = 10.0 ** (g_db / 10.0)
    assert transfer[0] == pytest.approx((np.log(g_linear) / (alpha_p * l_eff)) ** 2)
    assert transfer[1] / transfer[0] == pytest.approx(0.5)


def test_dispersion_branch_is_finite_nonnegative():
    """Eq. 4.9/4.10 smoke: finite, non-negative transfer for paper-ish numbers."""
    for geometry in (Geometry.CO_PROPAGATING, Geometry.COUNTER_PROPAGATING):
        rho = rin_transfer_dispersion(4.74e-3, 0.2e-3, 1e-3, 1000.0, geometry)
        assert np.isfinite(rho) and rho >= 0.0


def test_group_index_zero_dispersion_point():
    """Group index equals its zero-dispersion value there, quadratic away."""
    assert group_index(1300.0, 1.47, 0.09, 1300.0) == pytest.approx(1.47)
    up10 = group_index(1310.0, 1.47, 0.09, 1300.0) - 1.47
    up20 = group_index(1320.0, 1.47, 0.09, 1300.0) - 1.47
    assert up10 / up20 == pytest.approx(0.25)


def test_mermelstein_fixture_loads_and_pins():
    """The restored Mermelstein fixture loads with validated provenance."""
    fx = load_benchmark("mermelstein2003")
    assert fx.value("parameters", "second_order_pump_power") == pytest.approx(0.78)


def test_missing_fixture_raises_filenotfound():
    """Unknown fixture keys fail loudly (no silent empty benchmark)."""
    with pytest.raises(FileNotFoundError):
        load_benchmark("definitely_not_a_paper1999")


def test_fixture_rejects_missing_provenance(tmp_path):
    """A fixture without DOI metadata fails validation, not silently."""
    import json

    paper_dir = tmp_path / "benchmarks" / "raman_noise" / "bogus2026"
    paper_dir.mkdir(parents=True)
    (paper_dir / "fixture.json").write_text(
        json.dumps(
            {
                "schema": "photonics-helper/raman-noise-fixture/1",
                "paper": {"title": "Bogus"},
                "parameters": {},
                "targets": {},
            }
        )
    )
    with pytest.raises(ProvenanceError):
        load_benchmark("bogus2026", root=tmp_path)
