"""Pin the two `photonics_helper` defects recorded in the naskar-scg project's GAPS.md.

A1 — the z-dependent (tapered) linear step must live in the retarded frame:
     φ(Ω) = [β(ω₀+Ω,z) − β(ω₀,z) − β₁(ω₀,z)Ω]·Δz, the same subtraction
     `_full_beta_phase` already does for the static full-β path.
A2 — `_validate_shock_grid` must *say* when the symbol actually applied,
     1 + Ω·τ_shock, changes sign on a sampled bin.  Testing `Ω_max < ω₀` alone
     is correct only for τ = 1/ω₀.  It warns rather than raises because a
     negative factor on the innermost red bins is an extrapolation the caller
     may make on purpose (Dudley's 0.56 fs > 1/ω₀ in test_gnlse_unitarity.py);
     the defect was the silence, not the permission.

Both failures were reproduced on 2026-11-05 before the fix:
  A1: a uniform 3 mm section expressed as a z-table drifted the centroid by
      β₁L = 28.78 ps, which *wraps* a 32 ps window to −3.22 ps (periodic FFT,
      so the pulse re-enters and interacts with itself). Spectra were identical
      to 4e-11 dB, i.e. a spectral comparison cannot see the bug.
  A2: the paper's τ_sh = 1.5 fs on N=8192/T=32 ps gave τ·Ω_max = 1.206 and
      701 of 8192 bins with 1 + Ωτ < 0, accepted with only a bandwidth warning.
"""

from __future__ import annotations

import warnings

import numpy as np

from photonics_helper import (
    Fiber,
    GNLSESolver,
    TaperedGNLSESolver,
    TemporalGrid,
    Wavelength,
    ZDependentDispersion,
)
from photonics_helper.base import Area, Length, Time
from photonics_helper.pulse import Envelope, Wave

C = 299792458.0
WL0 = 2.0e-6
OM0 = 2 * np.pi * C / WL0


def _n(wl_um: float) -> float:
    """Smooth chalcogenide-like index (As₂Se₃, Thompson 2008): only n_g matters here."""
    A = (4.994871876241, 0.120715248481, 1.712368530625)
    B = (0.0583898896, 361.0, 0.2335595584)
    l2 = wl_um**2
    return float(np.sqrt(1.0 + sum(a * (l2 / (l2 - b)) for a, b in zip(A, B))))


def _beta_abs(omega):
    """β(ω) = n(ω)·ω/c, SI."""
    wl_um = 2 * np.pi * C / np.asarray(omega, dtype=float) * 1e6
    return np.array([_n(float(w)) for w in np.atleast_1d(wl_um)]) * np.atleast_1d(
        omega
    ) / C


def _pulse(grid, power=50.0, width_fs=50.0):
    env = Envelope(
        shape="gaussian",
        peak_amplitude=np.sqrt(power),
        pulse_width=Time(width_fs, "fs"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))


def _fiber(length_m: float) -> Fiber:
    return Fiber(
        n2=2.3e-17,
        alpha=0.0,
        A_eff=Area(0.9808e-12, "m^2"),
        length=Length(length_m, "m"),
        gamma=71.56,
    )


def _centroid_ps(grid, field) -> float:
    power = np.abs(field) ** 2
    t = np.asarray(grid.t, dtype=float)
    return float((power * t).sum() / power.sum()) * 1e12


class TestRetardedFrameInZDependentStep:
    """A1: a z-table that is constant in z must behave exactly like the static β(ω)."""

    def test_uniform_z_table_matches_static_full_beta(self):
        # 0.5 mm keeps β₁L ≈ 4.8 ps well inside the 32 ps window, so a missing
        # β₁Ω term shows up as a clean ~4.8 ps centroid offset, not an alias.
        length_m = 5e-4
        grid = TemporalGrid(N=1024, Tmax=Time(32e-12, "s"))
        w = np.asarray(grid.w, dtype=float)

        beta_grid = _beta_abs(OM0 + w)
        zdep = ZDependentDispersion(
            omegas=OM0 + w,
            z_positions=np.array([0.0, length_m]),
            beta=np.column_stack([beta_grid, beta_grid]),  # constant in z
            central_wavelength=WL0,
        )

        tapered = TaperedGNLSESolver(
            pulse=_pulse(grid),
            fiber=_fiber(length_m),
            dispersion_profile=zdep,
            include_raman=False,
            include_self_steepening=False,
        )
        tapered.propagate(num_steps=100)
        a_z = np.asarray(tapered.evolution[-1].envelope_field, dtype=complex)

        static = GNLSESolver(
            pulse=_pulse(grid),
            fiber=_fiber(length_m),
            betas=None,
            full_beta=_beta_abs,
            include_raman=False,
            include_self_steepening=False,
        )
        static.propagate(num_steps=100)
        a_s = np.asarray(static.evolution[-1].envelope_field, dtype=complex)

        drift_ps = abs(_centroid_ps(grid, a_z) - _centroid_ps(grid, a_s))
        assert drift_ps < 0.2, (
            "z-dependent step is not in the retarded frame: centroid differs "
            f"from the static full-β path by {drift_ps:.2f} ps over "
            f"{length_m * 1e3:.1f} mm"
        )

        # Both paths must also agree in the field itself, not just its centroid.
        scale = np.max(np.abs(a_s) ** 2)
        rel_err = np.max(np.abs(a_z - a_s) ** 2) / scale
        assert rel_err < 1e-2, f"fields differ by {rel_err:.3e} of peak power"

    def test_centroid_stays_put_in_the_retarded_frame(self):
        """Absolute check: no walk at all, independent of the static path."""
        length_m = 5e-4
        grid = TemporalGrid(N=1024, Tmax=Time(32e-12, "s"))
        w = np.asarray(grid.w, dtype=float)
        beta_grid = _beta_abs(OM0 + w)
        zdep = ZDependentDispersion(
            omegas=OM0 + w,
            z_positions=np.array([0.0, length_m]),
            beta=np.column_stack([beta_grid, beta_grid]),
            central_wavelength=WL0,
        )
        solver = TaperedGNLSESolver(
            pulse=_pulse(grid),
            fiber=_fiber(length_m),
            dispersion_profile=zdep,
            include_raman=False,
            include_self_steepening=False,
        )
        solver.propagate(num_steps=100)
        c0 = _centroid_ps(grid, np.asarray(solver.evolution[0].envelope_field))
        c1 = _centroid_ps(grid, np.asarray(solver.evolution[-1].envelope_field))
        assert abs(c1 - c0) < 0.2, (
            f"pulse walked {c1 - c0:.2f} ps over {length_m * 1e3:.1f} mm of "
            "uniform waveguide; the linear step is not retarded-frame"
        )


class TestShockGuardTestsTheOperatorItApplies:
    """A2: reject a (τ, grid) pair whose operator 1 + Ωτ changes sign."""

    def _grid(self, n=8192, tmax_ps=32.0):
        return TemporalGrid(N=n, Tmax=Time(tmax_ps * 1e-12, "s"))

    def test_paper_tau_is_flagged_on_the_project_grid(self):
        # N = 8192, T = 32 ps: Ω_max/ω₀ = 0.854 passes the Ω_max < ω₀ test,
        # while τ = 1.5 fs makes 701 bins negative.
        grid = self._grid()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = GNLSESolver(
                pulse=_pulse(grid),
                fiber=_fiber(1e-3),
                betas=np.array([1.98e-25]),
                include_raman=False,
                include_self_steepening=True,
                tau_shock=1.5e-15,
            )
            solver.propagate(num_steps=2)  # the engine, and its guard, build here
        flip = [str(w.message) for w in caught if "flips the sign" in str(w.message)]
        assert flip, "no sign-flip warning was emitted"
        assert "701" in flip[0]

    def test_default_tau_is_still_accepted(self):
        # τ = 1/ω₀, the Blow & Wood operator: Ω_max < ω₀ is the exact condition,
        # so no sign-flip warning may fire (the bandwidth one may).
        grid = self._grid(n=2048, tmax_ps=32.0)
        solver = GNLSESolver(
            pulse=_pulse(grid),
            fiber=_fiber(1e-3),
            betas=np.array([1.98e-25]),
            include_raman=False,
            include_self_steepening=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver.propagate(num_steps=2)
        assert not [w for w in caught if "flips the sign" in str(w.message)]
        assert np.all(np.isfinite(solver.evolution[-1].envelope_field))

    def test_short_tau_accepted_when_no_bin_flips(self):
        # Ω_max = 3.93e14 rad/s -> τ must stay under 2.5 fs to keep 1 + Ωτ > 0.
        grid = self._grid(n=8192, tmax_ps=64.0)
        solver = GNLSESolver(
            pulse=_pulse(grid),
            fiber=_fiber(1e-3),
            betas=np.array([1.98e-25]),
            include_raman=False,
            include_self_steepening=True,
            tau_shock=1.0e-15,
        )
        solver.propagate(num_steps=2)
        assert np.all(np.isfinite(solver.evolution[-1].envelope_field))
