"""Unit tests for :mod:`photonics_helper.chi2`.

Covers the phase-matching / QPM helpers, the coupling coefficient, and the
coupled SHG / SFG / DFG solver against the analytic ``tanh²(κL)`` result.
"""

import numpy as np
import pytest
from scipy.special import jn

from photonics_helper.base import C_MS, EPS_0, Area, Wavelength
from photonics_helper.chi2 import (
    Chi2Result,
    Lambda_qpm,
    delta_k_shg,
    pgln_overlap,
    qpm_grating,
    shg_coupling,
    shg_coupling_overlap,
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


def test_shg_coupling_boyd_anchor():
    """Anchor σ against the article parameters of Wang et al. (2017).

    1550 nm, d_eff = d33 = 27 pm/V, n = 2.14, A_eff = 0.52 µm² →
    σ = 1330.8 (√W·m)⁻¹ (Wang et al. plane-wave / Boyd normalisation).
    """
    sigma = shg_coupling(Wavelength(1550, "nm"), 27e-12, n=2.14, A_eff=Area(0.52, "um^2"))
    assert sigma == pytest.approx(1330.8217, rel=1e-3)


def test_shg_coupling_independent_maxwell_derivation():
    """Cross-check σ via a direct coupled-Maxwell-equation derivation.

    Independent path: start from Boyd's coupled equations for the physical
    (V/m) plane waves

        dE₂/dz = i (ω₂ d_eff / (c n₂)) E₁² e^{iΔkz},

    rewrite with power-normalised envelopes A (P = ½ n c ε₀ |E|² A_eff, so
    |E_ω|² = 2|A_ω|²/(n c ε₀ A_eff)) and combine factors:

        dA₂/dz = i (ω d_eff/(n c)) · sqrt(2 Z₀/(n³ A_eff))·A₁²  (n₁ = n₂ = n)

    hence κ = ω d_eff/c · sqrt(2 Z₀/(n³ A_eff)).  The shipped helper must
    equal this for arbitrary arguments (ratio exactly 1).
    """
    for lam, d, n, a_um in [
        (1550.0, 27e-12, 2.14, 0.52),
        (1064.0, 2e-12, 1.0, 1.0),
        (1310.0, 40e-12, 3.5, 7.5),
    ]:
        Z0 = 1.0 / (EPS_0 * C_MS)
        w = 2 * np.pi * C_MS / (lam * 1e-9)
        kappa_expected = (w * d / C_MS) * np.sqrt(2.0 * Z0 / (n**3 * (a_um * 1e-12)))  # = (w*d/(n*C_MS))*sqrt(2*Z0/(n*A))
        kappa_helper = shg_coupling(
            Wavelength(lam, "nm"), d, n=n, A_eff=Area(a_um, "um^2")
        )
        assert kappa_helper == pytest.approx(kappa_expected, rel=1e-12)


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


# ========================== lossy SHG (v0.1.1) ==========================


def test_solve_shg_zero_loss_bit_identical():
    """Loss-free argument must produce the exact pre-loss solver output."""
    sigma = _sigma()
    kwargs = dict(length=2e-3, P0=0.05, sigma=sigma, n_steps=800,
                  delta_k=1e3, qpm_period=6.283e-6)
    r_old = solve_shg(**kwargs)
    r_new = solve_shg(**kwargs, loss_db_per_cm=(0.0, 0.0))
    np.testing.assert_array_equal(r_old.A, r_new.A)
    assert r_new.loss_alpha == (0.0, 0.0)


def test_solve_shg_loss_validation():
    with pytest.raises(ValueError, match=r"\(pump, sh\)"):
        solve_shg(length=1e-3, P0=0.1, sigma=1.0, loss_db_per_cm=(1.0,))


def test_solve_shg_pump_loss_analytic():
    """Undepleted pump, finite pump loss, zero SH loss (Wang Eq. 10 limit).

    A_f(z) = √P₀·e^{−α_f z/2}, A_f² = P₀·e^{−α_f z}, so
    A_sh(z) = iσ∫₀ᶻ A_f² dt = iσP₀(1 − e^{−α_f z})/α_f and
    P_sh = σ²·P₀²·(1 − e^{−α_f L})²/α_f², η = P_sh/P₀.
    """
    sigma = _sigma()
    P0 = 1e-5  # tiny → undepleted
    alpha_f = np.log(10.0) / 10.0 * 100.0 * 3.0  # 3.0 dB/cm → 1/m
    L = 0.02
    res = solve_shg(
        length=L, P0=P0, sigma=sigma, n_steps=4000,
        loss_db_per_cm=(3.0, 0.0),
    )
    expected = sigma**2 * P0**2 * (1.0 - np.exp(-alpha_f * L)) ** 2 / alpha_f**2
    expected_eff = sigma**2 * P0 * (1.0 - np.exp(-alpha_f * L)) ** 2 / alpha_f**2
    assert res.power("sh")[-1] == pytest.approx(expected, rel=1e-3)
    assert res.efficiency()[-1] == pytest.approx(expected_eff, rel=1e-3)
    assert res.efficiency()[-1] < 0.1  # undepleted regime sanity


def test_solve_shg_loss_both_fields_analytic():
    """Exact two-loss solution of the undepleted linear system:

    A_sh(z) = iσP₀ e^{−α_s z/2} · (e^{(α_s/2 − α_f)z} − 1)/(α_s/2 − α_f).
    """
    sigma = _sigma()
    P0 = 1e-3
    cm = np.log(10.0) / 10.0 * 100.0
    a_f, a_s = cm * 3.0, cm * 63.5  # article losses @80 nm grooves
    L = 0.5e-3
    degenerate = abs(a_s / 2.0 - a_f)
    if degenerate < 1e-12:
        return
    res = solve_shg(
        length=L, P0=P0, sigma=sigma, n_steps=8000,
        loss_db_per_cm=(3.0, 63.5),
    )
    expected_abs = (
        sigma
        * P0
        * abs(
            np.exp(-a_s * L / 2.0)
            * (np.exp((a_s / 2.0 - a_f) * L) - 1.0)
            / (a_s / 2.0 - a_f)
        )
    )
    actual = abs(complex(res.field("sh")[-1]))
    assert actual == pytest.approx(expected_abs, rel=0.02)
    assert res.loss_alpha == (a_f, a_s)


def test_solve_shg_loss_heavy_stability():
    """Worst-case loss from the article (13.5 dB/cm pump) stays finite."""
    sigma = _sigma()
    res = solve_shg(length=4e-3, P0=0.1, sigma=sigma, n_steps=2000,
                    loss_db_per_cm=(13.5, 63.5))
    assert np.all(np.isfinite(res.powers))
    assert np.min(res.powers) >= 0.0


# ======================= mode-overlap coupling (v0.1.1) ====================


def _modes(N: int = 160, dx: float = 4e-9, preset: str = "uniform"):
    """Synthetic transverse mode field pair on an (N, N) grid (m units)."""
    coords = np.linspace(-N * dx / 2, N * dx / 2, N)
    X, Z = np.meshgrid(coords, coords, indexing="ij")
    if preset == "uniform":
        return np.ones((N, N), dtype=complex), np.ones((N, N), dtype=complex)
    if preset == "gauss":  # pump & SH same shape (plane-wave reduction)
        w = 1.2e-9
        return np.exp(-(X**2 + Z**2) / w**2), np.exp(-(X**2 + Z**2) / w**2)
    if preset == "trilobe":  # TE₃-like SH: three lobes along x
        pump = np.exp(-(X**2 + Z**2) / 0.8**2)
        sh = np.cos(3.0 * np.pi * X / (2.0 * N * dx))
        sh *= np.exp(-(Z**2) / 0.8**2)
        return pump, sh
    raise ValueError(preset)


def test_overlap_uniform_matches_plane_wave():
    """A uniform mode on area A must reproduce the Boyd plane-wave σ."""
    from photonics_helper.base import Area

    N = 160
    dx = 2e-9
    E_p = np.ones((N, N), dtype=complex)
    E_s = np.ones((N, N), dtype=complex)
    a_m = (N * dx) ** 2
    g = shg_coupling_overlap(
        E_p, E_s, dx, dx, wavelength=WAVELENGTH, d=27e-12,
        n_pump=2.14, n_sh=2.14,
    )
    sigma = shg_coupling(
        WAVELENGTH, 27e-12, n=2.14, A_eff=Area(a_m * 1e12, "um^2")
    )
    assert g == pytest.approx(sigma, rel=1e-6)


def test_overlap_scale_invariance():
    """Any absolute field scaling (and common phase) collapses to one g."""
    p, s = _modes(preset="gauss")
    g1 = shg_coupling_overlap(p, s, 2e-9, 2e-9, wavelength=WAVELENGTH)
    g2 = shg_coupling_overlap(
        p * 3.2j, s * 0.17, 2e-9, 2e-9, wavelength=WAVELENGTH
    )
    assert g1 == pytest.approx(g2, rel=1e-12)


def test_overlap_trilobe_suppression():
    """Sign-alternating (3-lobe) SH mode strongly suppresses g vs TE0-like."""
    p0, s0 = _modes(preset="gauss")
    g_aligned = shg_coupling_overlap(p0, s0, 2e-9, 2e-9, wavelength=WAVELENGTH)
    p3, s3 = _modes(preset="trilobe")
    g_trilobe = shg_coupling_overlap(p3, s3, 2e-9, 2e-9, wavelength=WAVELENGTH)
    assert abs(g_trilobe) < 0.04 * abs(g_aligned)


def test_overlap_validation():
    p, s = _modes(preset="gauss")
    with pytest.raises(ValueError):
        shg_coupling_overlap(np.zeros((4, 4), dtype=complex), s,
                             1e-9, 1e-9, wavelength=WAVELENGTH)
    with pytest.raises(ValueError):
        shg_coupling_overlap(p, s, -1e-9, 1e-9, wavelength=WAVELENGTH)
    with pytest.raises(ValueError):
        shg_coupling_overlap(p, s[::8], 1e-9, 1e-9, wavelength=WAVELENGTH)


def test_pgln_zero_modulation_vanishes():
    """Δε₁ = 0 and d^(1) = 0 → all PGLN correction terms vanish."""
    p, s = _modes(preset="gauss")
    N = p.shape[0]
    r = pgln_overlap(p, s, 2e-9, 2e-9, wavelength=WAVELENGTH,
                     d0=27e-12, d1=np.zeros((N, N)),
                     delta_eps1_pump=0.0, delta_eps1_sh=0.0, delta_k=1e5)
    assert r["g_L_w"] == 0.0
    assert r["g_L_2w"] == 0.0
    assert r["g_eff"] == 0.0


def test_pgln_gnl1_first_order_matches_uniform_shape():
    """d^(1) = d^(0) (full half-period alternation, duty 50 % limit) →
    |g_NL^(1)| equals the uniform-guide overlap factor·(1/π, 2 form)."""
    p, s = _modes(preset="gauss")
    N = p.shape[0]
    r0 = pgln_overlap(p, s, 2e-9, 2e-9, wavelength=WAVELENGTH,
                      d0=27e-12, d1=np.full((N, N), 27e-12),
                      delta_eps1_pump=0.0, delta_eps1_sh=0.0, delta_k=1e5)
    # d^(1) = d^(0) everywhere → D1 == D0 → g_NL^(1) == g_NL^(0)
    assert r0["g_nl_1"] == pytest.approx(r0["g_nl_0"], rel=1e-12)


def test_pgln_bessel_identity():
    """J₀(x)+J₂(x) = 2 J₁(x)/x — the identity used in Eq. (5)."""
    x = 0.7
    assert np.isclose(jn(0, x) + jn(2, x), 2 * jn(1, x) / x)

