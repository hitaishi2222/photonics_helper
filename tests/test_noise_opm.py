"""Tests for the one-photon-per-mode (OPM) seed in photonics_helper.noise.

Ground truth: one quantum per frequency bin, ``|q_n|² = ℏω_n/Δν`` with
``Δν = 1/Tmax``.  Because ``TemporalGrid.fft``/``ifft`` are a scaled pair,
``∫|A|²dt = Σ_n |q_n|²·Δν``, so the recorded energy is ``Σ_n ℏω_n`` over the
seeded bins — the check below does not assume the internal scaling, it measures
the injected energy.
"""

import numpy as np
import pytest
from scipy.constants import hbar

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.noise import (
    add_opm_noise,
    coherence_g12,
    opm_noise_field,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL0 = Wavelength(1550, "nm")


def _grid(N=1024, tmax_s=20e-12):
    return TemporalGrid(N=N, Tmax=Time(tmax_s, "s"))


def _mode_energies(grid, wavelength=WL0):
    """``ℏω_n`` per bin, DC bin (index ``N//2``) excluded, shape ``(N,)``."""
    omega = float(wavelength.to_omega().as_rad_s) + np.asarray(grid.w, dtype=float)
    e = hbar * omega
    e[grid.N // 2] = 0.0
    return e


def _zero_wave(grid, wavelength=WL0):
    env = Envelope(
        shape="custom",
        peak_amplitude=0.0,
        pulse_width=Time(1e-12, "s"),
        func=lambda t, T0, A0: np.zeros_like(t, dtype=complex),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=wavelength)


# ─── 1.2 seed statistics ────────────────────────────────────────────────────


def test_injected_energy_matches_mode_sum():
    grid = _grid()
    field = opm_noise_field(grid, WL0, seed=7)
    measured = float(np.sum(np.abs(field) ** 2) * grid.dt)
    expected = float(np.sum(_mode_energies(grid)))
    # The spec allows 1 %; the construction is exact to float round-off, so the
    # gate is tight enough to catch a wrong per-mode energy (measured 3e-14).
    assert measured == pytest.approx(expected, rel=1e-10)


def test_spectral_intensity_is_one_photon_per_bin():
    grid = _grid(N=512)
    dnu = float(grid.dw) / (2.0 * np.pi)
    expected = (
        hbar * (float(WL0.to_omega().as_rad_s) + np.asarray(grid.w, dtype=float)) / dnu
    )
    runs = np.array(
        [np.asarray(grid.fft(opm_noise_field(grid, WL0, seed=s))) for s in range(64)]
    )
    mean_intensity = np.mean(np.abs(runs) ** 2, axis=0)
    keep = np.ones(grid.N, dtype=bool)
    keep[grid.N // 2] = False
    # The spectral amplitude is deterministic (only the phase is random), so
    # the per-bin intensity is exact and the gate can be tight.
    np.testing.assert_allclose(mean_intensity[keep], expected[keep], rtol=1e-12)
    # DC is zeroed in the spectrum; the round trip leaves only round-off there.
    assert mean_intensity[grid.N // 2] < 1e-12 * np.max(expected)
    # ...and the test is discriminating: a constant ħω₀ would differ by >3 %.
    constant = np.full(grid.N, hbar * float(WL0.to_omega().as_rad_s) / dnu)
    assert np.max(np.abs(constant[keep] - expected[keep]) / expected[keep]) > 0.03


def test_ensemble_mean_field_is_zero():
    grid = _grid(N=256)
    runs = np.array([opm_noise_field(grid, WL0, seed=s) for s in range(200)])
    per_run_rms = float(np.sqrt(np.mean(np.abs(runs[0]) ** 2)))
    assert abs(complex(runs.mean())) < 0.05 * per_run_rms
    # No phase is preferred: average the spectral phasor over every run and bin
    # (51000 samples, so the random-walk scale is 1/sqrt(n) = 0.0044 and the
    # bound below is >5 sigma), rather than over one realization.
    spec = np.array([np.asarray(grid.fft(r)) for r in runs])
    keep = np.ones(grid.N, dtype=bool)
    keep[grid.N // 2] = False
    seeded = spec[:, keep]
    mean_phasor = complex(np.mean(seeded / np.abs(seeded)))
    assert abs(mean_phasor) < 0.01, abs(mean_phasor)


def test_seed_is_reproducible():
    grid = _grid(N=256)
    a = opm_noise_field(grid, WL0, seed=42)
    b = opm_noise_field(grid, WL0, seed=42)
    c = opm_noise_field(grid, WL0, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rng_and_seed_conflict():
    grid = _grid(N=256)
    with pytest.raises(ValueError, match="rng and seed"):
        opm_noise_field(grid, WL0, rng=np.random.default_rng(0), seed=0)


def test_negative_absolute_frequency_is_refused():
    # Tmax = 0.5 ps on 256 samples gives Ω_max ≈ 1.6e15 rad/s > ω₀ (1.22e15).
    grid = _grid(N=256, tmax_s=0.5e-12)
    with pytest.raises(ValueError, match="positive absolute frequency"):
        opm_noise_field(grid, WL0, seed=0)


def test_add_opm_noise_takes_the_carrier_from_the_wave():
    grid = _grid(N=256)
    wave = _zero_wave(grid)
    out = add_opm_noise(wave, seed=11)
    np.testing.assert_allclose(
        out.envelope_field, opm_noise_field(grid, wave.central_wavelength, seed=11)
    )
    # The input wave is untouched.
    assert np.all(np.asarray(wave.envelope_field) == 0.0)


# ─── 1.3 end-to-end: OPM ensemble → engine → coherence_g12 ───────────────────


def _propagate_opm(seed, n=256, steps=4):
    """One OPM-seeded run through the scalar engine; returns the final spectrum."""
    grid = _grid(N=n)
    wave = _zero_wave(grid)
    wave.with_field(opm_noise_field(grid, WL0, seed=seed))
    fiber = Fiber(
        n2=1e-19,
        alpha=0.0,
        A_eff=Area(5e-11, "m^2"),
        length=Length(1e-3, "m"),
    )
    engine = SplitStepEngine(pulse=wave, fiber=fiber, betas=np.array([0.02]))
    engine.propagate(num_steps=steps, nsaves=None)
    return np.asarray(grid.fft(engine.A), dtype=complex)


def test_coherence_identical_seeds_is_one():
    spectrum = _propagate_opm(seed=3)
    g12 = coherence_g12(np.array([spectrum] * 8))
    populated = np.abs(spectrum) > 0.0
    assert populated.any(), "no populated bin: the assertion below would be vacuous"
    np.testing.assert_allclose(g12[populated], 1.0, atol=1e-12)


def test_coherence_independent_seeds_approach_the_floor():
    n_runs = 16
    runs = np.array([_propagate_opm(seed=s) for s in range(n_runs)])
    g12 = coherence_g12(runs)
    populated = np.mean(np.abs(runs) ** 2, axis=0) > 0.0
    mean_g12 = float(np.mean(g12[populated]))
    # Independent phases: |<E_m* E_n>| is the modulus of a random-walk pair sum,
    # so it scales as 1/√(N_pairs) = 1/√(n(n−1)), i.e. ~1/n for a large ensemble.
    # The Rayleigh scale of that ratio is 0.886/√N_pairs and the measured value on
    # this deck is 0.69/√N_pairs (both constants verified at n = 16, 32, 64), so
    # the band brackets both: a wrong floor by 40 % or more fails.
    floor = 1.0 / np.sqrt(n_runs * (n_runs - 1))
    assert mean_g12 < 0.5
    assert 0.5 * floor <= mean_g12 <= 0.95 * floor, (mean_g12, floor)


# ─── 1.4 the docs page names the seed the module exports ─────────────────────


def test_noise_docs_name_only_importable_sources():
    import re
    from pathlib import Path

    import photonics_helper.noise as noise_module

    page = Path(__file__).resolve().parents[1] / "docs" / "api" / "noise.md"
    text = page.read_text()
    # Every backticked snake_case identifier on the page must be a real member.
    named = set(re.findall(r"`([a-z][a-z0-9_]+)`", text))
    assert named, "docs/api/noise.md names no source"
    missing = sorted(n for n in named if not hasattr(noise_module, n))
    assert not missing, f"docs name unimportable sources: {missing}"
    # The two one-photon-per-mode seeds are named differently, and the page
    # states the factor that distinguishes them.
    assert {"opm_noise_field", "add_opm_noise"} <= named
    assert "raman_noise_field" in named
    assert "ℏω/2" in text
