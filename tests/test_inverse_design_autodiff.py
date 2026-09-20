"""Tests for the torch autodiff inverse-design extension (fit_shg_autodiff).

Contract: with the SH-power observable supplied (which breaks the
``kappa = sigma*sqrt(P0)`` degeneracy of the eta-only inverse problem),
the batched multi-start Adam fit recovers (sigma, P0, dk) of a synthetic
chi(2) run within ~10%. The eta-only variant is NOT asserted: the
(sigma, P0) degenerate direction makes the recovery ill-posed there (see
fit_two_wave's documented identifiability contract).
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.chi2 import solve_shg
from photonics_helper.inverse_design import fit_shg_autodiff

torch = pytest.importorskip("torch", reason="torch ([pinns] extra) not installed")


Z = np.linspace(0.0, 1.0, 11)
SIGMA_TRUE = 2.0
P0_TRUE = 4.0
DK_TRUE = -10.0


@pytest.fixture(scope="module")
def synthetic():
    res = solve_shg(
        length=1.0, P0=P0_TRUE, sigma=SIGMA_TRUE, n_steps=800, delta_k=DK_TRUE
    )
    eta = np.abs(res.field("sh")) ** 2 / np.abs(res.field("fundamental")) ** 2
    psh = np.abs(res.field("sh")) ** 2
    return np.interp(Z, res.z, eta), np.interp(Z, res.z, psh)


def test_autodiff_fit_recovers_truth(synthetic):
    """Full triple recovery with the SH-power observable (within ~10%)."""
    eta_z, psh_z = synthetic
    fit = fit_shg_autodiff(
        z_samples=Z,
        ratios=eta_z,
        sh_power=psh_z,
        n_steps=60,
        n_iter=800,
    )
    assert fit.values["delta_k"] == pytest.approx(DK_TRUE, abs=1.5)
    assert fit.values["sigma"] == pytest.approx(SIGMA_TRUE, rel=0.15)
    assert fit.values["P0"] == pytest.approx(P0_TRUE, rel=0.15)


def test_validation_errors():
    with pytest.raises(ValueError):
        fit_shg_autodiff(z_samples=[0.0, 1.0], ratios=[1.0, 2.0])
    with pytest.raises(ValueError):
        fit_shg_autodiff(
            z_samples=[0.0, 1.0, 2.0], ratios=[0.0, 0.0, 0.0], sigma0=-1.0
        )
