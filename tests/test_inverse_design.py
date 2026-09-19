"""Tests for the inverse-design layer (Phase 4 item 5, v1).

Contracts:
- `fit_two_wave` recovers the identifiable invariants (κ = σ√P₀ and
  |Δk|) from synthetic η(z) data exactly; the degenerate (σ, P₀) direction
  is documented and not asserted.
- `design_efficiency` recovers the analytic tanh²(κL) design length exactly
  and fails loudly on unreachable targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.chi2 import solve_shg
from photonics_helper.inverse_design import design_efficiency, fit_two_wave


def _synthetic_eta(L=0.5, sigma=0.5, P0=2.0, delta_k=30.0, n_steps=4000):
    res = solve_shg(length=L, P0=P0, sigma=sigma, n_steps=n_steps,
                    delta_k=delta_k)
    curve = np.abs(res.field("sh")) ** 2 / np.maximum(
        np.abs(res.field("fundamental")) ** 2, 1e-30
    )
    zs = np.linspace(0.05, L, 8)
    return res, zs, np.interp(zs, res.z, curve)


class TestFitTwoWave:
    def test_recovers_kappa_and_delta_k(self):
        _res, zs, data = _synthetic_eta()
        fit = fit_two_wave(z_samples=zs, ratios=data, n_steps=400)
        assert fit.values["kappa"] == pytest.approx(0.5 * np.sqrt(2.0), abs=1e-3)
        assert abs(fit.values["delta_k"]) == pytest.approx(30.0, abs=1e-2)
        assert fit.converged
        assert fit.residual_norm < 1e-4

    def test_kappa_identifiability_contract(self):
        """η(z) identifies κ but not the (σ, P₀) pair inside it."""
        _res, zs, data = _synthetic_eta(sigma=0.2, P0=5.0, delta_k=10.0)
        fit = fit_two_wave(z_samples=zs, ratios=data, n_steps=400)
        assert fit.values["kappa"] == pytest.approx(0.2 * np.sqrt(5.0), abs=1e-3)
        assert abs(fit.values["delta_k"]) == pytest.approx(10.0, abs=1e-2)


class TestDesignEfficiency:
    def test_matches_analytic_tanh_length(self):
        """κL = atanh(√η) ⇒ L = atanh(√target)/κ for a lossless PM design."""
        target = 0.5
        P0, sigma = 2.0, 0.2
        kappa = sigma * np.sqrt(P0)
        d = design_efficiency(target=target, P0=P0, sigma=sigma, n_steps=800)
        expected_L = np.arctanh(np.sqrt(target)) / kappa
        assert d.values["length"] == pytest.approx(expected_L, rel=1e-3)
        assert d.values["efficiency"] == pytest.approx(target, abs=1e-4)

    def test_unreachable_target_fails_loudly(self):
        with pytest.raises(ValueError, match="unreachable"):
            design_efficiency(
                target=0.95, P0=0.01, sigma=1e-4, n_steps=200,
            )

    def test_target_range_enforced(self):
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            design_efficiency(target=1.5, P0=2.0, sigma=0.2)
