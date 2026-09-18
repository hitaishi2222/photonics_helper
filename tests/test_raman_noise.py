"""Step2: spontaneous-Raman noise source (Dudley RMP 2006 Eq. 5 Γ_R).

Ground truth: Dudley–Genty–Coen RMP 78, 1135 (2006), Eq. 5 / Sec. V.B.
"""

import numpy as np

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.noise import coherence_g12, raman_noise_field
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


def _grid(n: int = 2**10) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(5e-12, "s"))


def _wave(grid=None) -> Wave:
    grid = grid or _grid()
    env = Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(10e3),
        pulse_width=Time(28.4e-15, "s"),
    )
    return Wave(envelope=env, central_wavelength=Wavelength(835, "nm"), grid=grid)


def _fiber(grid) -> FiberProfile:
    from photonics_helper.raman import RamanResponse, RamanSpec

    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    raman = RamanResponse(spec=spec, grid=grid)
    return FiberProfile(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(5e-11, "m^2"),
        length=Length(2e-3, "m"),
        raman_response=raman,
    )


def _solver(**kw) -> GNLSESolver:
    grid = _grid()
    return GNLSESolver(
        pulse=_wave(grid), fiber=_fiber(grid), betas=np.array([0.0]), **kw
    )


def test_raman_noise_seed_reproducible():
    a = _solver(include_raman=True)
    a.propagate(20, nsaves=5, raman_noise=True, noise_seed=42)
    b = _solver(include_raman=True)
    b.propagate(20, nsaves=5, raman_noise=True, noise_seed=42)
    assert np.array_equal(
        a.evolution[-1].envelope_field, b.evolution[-1].envelope_field
    )


def test_raman_noise_different_seeds_differ():
    a = _solver(include_raman=True)
    a.propagate(20, nsaves=5, raman_noise=True, noise_seed=1)
    b = _solver(include_raman=True)
    b.propagate(20, nsaves=5, raman_noise=True, noise_seed=2)
    assert not np.array_equal(
        a.evolution[-1].envelope_field, b.evolution[-1].envelope_field
    )


def test_off_by_default_bit_identical():
    a = _solver(include_raman=True)
    a.propagate(20, nsaves=5)
    b = _solver(include_raman=True)
    b.propagate(20, nsaves=5)
    assert np.array_equal(
        a.evolution[-1].envelope_field, b.evolution[-1].envelope_field
    )


def test_noise_changes_single_shot_and_decoheres():
    sols = []
    for seed in range(6):
        s = _solver(include_raman=True)
        s.propagate(30, nsaves=5, raman_noise=True, noise_seed=seed)
        sols.append(s)
    ref = _solver(include_raman=True)
    ref.propagate(30, nsaves=5)
    # Single-shot spectrum differs from the noiseless run.
    assert not np.array_equal(
        sols[0].evolution[-1].envelope_field, ref.evolution[-1].envelope_field
    )
    # Ensemble partially decoheres: mean g12 over bins < 1.
    spectra = np.array(
        [np.abs(s.pulse.grid.fft(s.evolution[-1].envelope_field)) for s in sols]
    )
    g12 = coherence_g12(spectra)
    assert g12.shape == spectra.shape[1:]
    assert float(np.mean(g12)) < 1.0


def test_coherence_g12_identical_runs_is_one():
    rng = np.random.default_rng(0)
    runs = np.abs(rng.standard_normal((4, 64))) + 1j * 0.0
    np.testing.assert_allclose(coherence_g12(np.repeat(runs[:1], 4, axis=0)), 1.0)


def test_raman_noise_field_shaped_by_im_hR():
    grid = _grid()
    h_R = np.zeros(grid.N, dtype=complex)
    h_R[grid.N // 2 + 10] = 1.0j  # gain only in one bin
    n = raman_noise_field(grid, h_R, seed=0, omega0=2 * np.pi * 3e8 / 835e-9)
    assert n.shape == (grid.N,)
    assert np.all(np.isfinite(n))
    assert float(np.max(np.abs(n))) > 0.0


def test_coherence_g12_uses_modulus_of_complex_pair_average():
    """g12 must follow |⟨E*ₘEₙ⟩|: a pure global-phase change in one run
    decoheres the ensemble even though every |E_m|² is unchanged (the old
    real-part convention returned g12 = 1 here)."""
    rng = np.random.default_rng(3)
    runs = rng.standard_normal((6, 128)) + 1j * rng.standard_normal((6, 128))
    runs[2] *= np.exp(1j * 0.4)  # global phase on one realization
    assert float(np.mean(coherence_g12(runs))) < 1.0


def test_raman_noise_thermal_factor():
    """Thermal occupancy: identical at n_th→0, louder for large n_th."""
    grid = _grid()
    h_R = np.zeros(grid.N, dtype=complex)
    h_R[grid.N // 2 + 10] = 1.0j
    omega0 = 2 * np.pi * 3e8 / 835e-9
    cold = raman_noise_field(grid, h_R, seed=0, omega0=omega0, temperature=1e-9)
    cold2 = raman_noise_field(grid, h_R, seed=0, omega0=omega0, temperature=1e-6)
    hot = raman_noise_field(grid, h_R, seed=0, omega0=omega0, temperature=1e6)
    # n_th → 0 well before 1 mK for a 40 THz-shift bin: identical draws
    np.testing.assert_array_equal(cold, cold2)
    # high temperature → stimulated contribution raises the variance
    assert float(np.var(hot)) > float(np.var(cold))
