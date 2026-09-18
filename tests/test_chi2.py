"""Unit tests for :mod:`photonics_helper.chi2`.

Covers the phase-matching / QPM helpers, the coupling coefficient, and the
coupled SHG / SFG / DFG solver against the analytic ``tanh²(κL)`` result.
"""

import numpy as np
import pytest

from photonics_helper.base import Area, Wavelength
from photonics_helper.chi2 import (
    Chi2Result,
    Lambda_qpm,
    delta_k_shg,
    qpm_grating,
    shg_coupling,
    solve_dfg,
    solve_sfg,
    solve_shg,
    solve_three_wave,
)

WAVELENGTH = Wavelength(1550, "nm")
A_EFF = Area(1.0, "um^2")
D_EFF = 10e-12  # 10 pm/V
N_INDEX = 2.0


def _sigma() -> float:
    return shg_coupling(WAVELENGTH, D_EFF, n=N_INDEX, A_eff=A_EFF)


def test_delta_k_shg_analytic():
    """Δk = β(2ω) − 2β(ω) for a quadratic β(ω) = ω²."""
    assert delta_k_shg(lambda w: w**2, 3.0) == pytest.approx(2 * 3.0**2)


def test_lambda_qpm_round_trip():
    """Λ = 2π/|Δk| and the QPM order scales the period."""
    assert Lambda_qpm(2 * np.pi / 1e-4) == pytest.approx(1e-4)
    assert Lambda_qpm(2 * np.pi / 1e-4, order=3) == pytest.approx(3e-4)


def test_lambda_qpm_validation():
    with pytest.raises(ValueError):
        Lambda_qpm(0.0)
    with pytest.raises(ValueError):
        Lambda_qpm(1.0, order=0)


def test_qpm_grating_square_wave():
    """A 50 % duty-cycle grating alternates sign every half period."""
    period = 2e-5
    z = period * np.array([0.0, 0.25, 0.75, 1.25, 1.75])
    g = qpm_grating(z, period)
    np.testing.assert_array_equal(g, [1.0, 1.0, -1.0, 1.0, -1.0])


def test_qpm_grating_validation():
    with pytest.raises(ValueError):
        qpm_grating(0.0, period=-1e-6)
    with pytest.raises(ValueError):
        qpm_grating(0.0, period=1e-6, duty_cycle=0.0)


def test_shg_coupling_positive_and_scaling():
    sigma = _sigma()
    assert sigma > 0.0
    # σ ∝ d_eff and ∝ 1/√A_eff.
    assert shg_coupling(WAVELENGTH, 2 * D_EFF, n=N_INDEX, A_eff=A_EFF) == pytest.approx(
        2 * sigma
    )
    assert shg_coupling(
        WAVELENGTH, D_EFF, n=N_INDEX, A_eff=Area(4.0, "um^2")
    ) == pytest.approx(sigma / 2.0)


def test_shg_coupling_validation():
    with pytest.raises(ValueError):
        shg_coupling(WAVELENGTH, D_EFF, n=0.0, A_eff=A_EFF)
    with pytest.raises(ValueError):
        shg_coupling(WAVELENGTH, 0.0, n=N_INDEX, A_eff=A_EFF)


@pytest.mark.parametrize("kL", [0.25, 0.5, 1.0, 1.5, 2.0])
def test_shg_matches_tanh_squared(kL):
    """η = tanh²(κL) at perfect phase matching."""
    sigma = _sigma()
    P0 = 0.1
    kappa = sigma * np.sqrt(P0)
    result = solve_shg(length=kL / kappa, P0=P0, sigma=sigma, n_steps=4000)
    assert result.efficiency()[-1] == pytest.approx(np.tanh(kL) ** 2, rel=1e-6)


def test_shg_power_conservation():
    """Fundamental + second-harmonic power is conserved (lossless)."""
    sigma = _sigma()
    P0 = 0.1
    kappa = sigma * np.sqrt(P0)
    result = solve_shg(length=2.0 / kappa, P0=P0, sigma=sigma, n_steps=4000)
    total = result.power("fundamental") + result.power("sh")
    np.testing.assert_allclose(total, P0, rtol=1e-9)


def test_shg_labels_and_accessors():
    sigma = _sigma()
    result = solve_shg(length=1e-3, P0=0.1, sigma=sigma, n_steps=200)
    assert result.labels == ("fundamental", "sh")
    assert result.field("sh").shape == result.z.shape
    assert result.power("sh").shape == result.z.shape
    assert result.powers.shape == (2, len(result.z))
    assert result.kappa == pytest.approx(sigma * np.sqrt(0.1))


def test_shg_validation():
    sigma = _sigma()
    with pytest.raises(ValueError):
        solve_shg(length=1e-3, P0=0.0, sigma=sigma)
    with pytest.raises(ValueError):
        solve_shg(length=-1e-3, P0=0.1, sigma=sigma)
    with pytest.raises(ValueError):
        solve_shg(length=1e-3, P0=0.1, sigma=sigma, n_steps=0)


def test_qpm_recovers_mismatched_shg():
    """First-order QPM recovers tanh²((2/π)κL); off-QPM stays negligible."""
    sigma = _sigma()
    P0 = 0.1
    kappa = sigma * np.sqrt(P0)
    length = 1.0 / kappa
    period = 100e-6
    delta_k = 2 * np.pi / period

    off = solve_shg(length=length, P0=P0, sigma=sigma, delta_k=delta_k, n_steps=4000)
    on = solve_shg(
        length=length,
        P0=P0,
        sigma=sigma,
        delta_k=delta_k,
        qpm_period=period,
        n_steps=4000,
    )
    eta_off = off.efficiency()[-1]
    eta_on = on.efficiency()[-1]
    assert eta_off < 0.01
    assert eta_on == pytest.approx(np.tanh(2.0 / np.pi * kappa * length) ** 2, rel=0.01)


def test_solve_sfg_transfers_power_to_sum():
    sigma = _sigma()
    result = solve_sfg(length=5e-3, P1=0.1, P2=0.1, sigma=sigma, n_steps=1000)
    assert result.labels == ("signal", "pump", "sum")
    assert result.power("sum")[-1] > result.power("sum")[0]
    # Manley-Rowe invariants for the symmetric three-wave system.
    for pump in ("signal", "pump"):
        conserved = result.power(pump) + result.power("sum")
        np.testing.assert_allclose(conserved, conserved[0], rtol=1e-6)


def test_solve_dfg_generates_idler():
    sigma = _sigma()
    result = solve_dfg(length=5e-3, Ppump=0.1, Psignal=1e-3, sigma=sigma, n_steps=1000)
    assert result.labels == ("pump", "signal", "idler")
    assert result.power("idler")[-1] > 0.0
    assert result.power("pump")[-1] < result.power("pump")[0]


def test_solve_sfg_dfg_validation():
    sigma = _sigma()
    with pytest.raises(ValueError):
        solve_sfg(length=1e-3, P1=0.0, P2=0.1, sigma=sigma)
    with pytest.raises(ValueError):
        solve_dfg(length=1e-3, Ppump=0.1, Psignal=0.0, sigma=sigma)


def test_solve_three_wave_validation():
    sigma = _sigma()
    with pytest.raises(ValueError):
        solve_three_wave(
            length=1e-3, sigma=sigma, A1_0=1.0 + 0j, A2_0=1.0 + 0j, n_steps=0
        )


def test_chi2_result_validation():
    with pytest.raises(ValueError):
        Chi2Result(
            z=np.linspace(0, 1, 5),
            A=np.zeros((2, 4), dtype=complex),
            labels=("a", "b"),
            sigma=1.0,
            delta_k=0.0,
        )
    with pytest.raises(ValueError):
        Chi2Result(
            z=np.linspace(0, 1, 4),
            A=np.zeros((2, 4), dtype=complex),
            labels=("a",),
            sigma=1.0,
            delta_k=0.0,
        )


def test_chi2_result_efficiency_zero_pump():
    result = Chi2Result(
        z=np.linspace(0, 1, 3),
        A=np.zeros((2, 3), dtype=complex),
        labels=("fundamental", "sh"),
        sigma=1.0,
        delta_k=0.0,
    )
    with pytest.raises(ValueError):
        result.efficiency()
