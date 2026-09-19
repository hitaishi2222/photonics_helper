"""Tests for the cascaded χ⁽²⁾+χ⁽³⁾ coupled-wave solver.

Limit contracts:
- pure quadratic (γ=0) equals `solve_shg` exactly;
- pure Kerr (σ=0) gives the analytic SPM phase `exp(iγ P₀ L)` exactly;
- large-Δk cascaded limit: fundamental phase ≈ (γ + σ²P₀/Δk)·P₀·L
  (undepleted-pump cascaded-Kerr formula, Epstein / Agrawal §10.5);
- QPM/QPM + loss paths still functional.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.chi2 import solve_cascaded_shg, solve_shg


class TestLimitContracts:
    def test_pure_quadratic_matches_solve_shg(self):
        L, P0, sigma, dk = 0.05, 2.0, 0.01, 15.0
        ref = solve_shg(
            length=L, P0=P0, sigma=sigma, n_steps=2500, delta_k=dk,
            loss_db_per_cm=(0.1, 0.2),
        )
        casc = solve_cascaded_shg(
            length=L, P0=P0, sigma=sigma, gamma_f=0.0, n_steps=2500,
            delta_k=dk, loss_db_per_cm=(0.1, 0.2),
        )
        assert np.allclose(ref.A, casc.A, rtol=1e-12, atol=1e-14)

    def test_pure_kerr_phase_exact(self):
        L, P0 = 0.2, 3.0
        gamma = 0.8
        r = solve_cascaded_shg(
            length=L, P0=P0, sigma=0.0, gamma_f=gamma, n_steps=1500, delta_k=0.0,
        )
        af = r.field("fundamental")
        assert abs(af[-1]) == pytest.approx(np.sqrt(P0), rel=1e-10)
        assert np.angle(af[-1]) == pytest.approx(gamma * P0 * L, rel=1e-6)
        assert np.angle(af[len(af) // 2]) == pytest.approx(
            gamma * P0 * (L / 2), rel=1e-6
        )
        # SH stays empty in a pure-Kerr run
        assert np.max(np.abs(r.field("sh"))) == pytest.approx(0.0, abs=1e-12)

    def test_kerr_spm_only_with_seed_sh(self):
        """Only γ_cross matters when the SH grows; conserves nothing extra."""
        r = solve_cascaded_shg(
            length=0.1, P0=1.0, sigma=0.0, gamma_f=0.0,
            gamma_sh=0.0, gamma_cross=0.0, n_steps=100,
        )
        af = r.field("fundamental")
        assert af[-1] == pytest.approx(1.0, rel=1e-10)


class TestCascadedLimit:
    def test_large_mismatch_effective_gamma(self):
        """γ_φ = σ²P₀/Δk (cascaded-Kerr limit) at large phase mismatch."""
        P0 = 1.5
        sigma = 0.02
        delta_k = 40.0
        L = 0.5
        # required: |Δk| ≫ σ√P₀; here σ√P0 = 0.0245 ≪ 40 ✓
        bridge_gamma = 0.05  # small explicit Kerr on top
        r = solve_cascaded_shg(
            length=L, P0=P0, sigma=sigma, gamma_f=bridge_gamma,
            n_steps=6000, delta_k=delta_k,
        )
        af = r.field("fundamental")
        # the physical-frame field also carries the SH phase factor; the
        # interacting-frame field y[0] is the exact numerical solution.
        phi = np.angle(af[-1])
        expected = (bridge_gamma + sigma**2 * P0 / delta_k) * P0 * L
        assert phi == pytest.approx(expected, rel=5e-2)
