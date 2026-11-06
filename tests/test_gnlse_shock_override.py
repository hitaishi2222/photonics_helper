"""The two shock-grid knobs: Naskar et al. *Phys. Scr.* **101** 385502 (2026) G1/G2.

G1 — the **hard** bound ``Ω_max < ω₀`` raised unconditionally, so a published
     time grid (64 000 points over 32 ps, i.e. Ω_max/ω₀ = 6.7) could not be
     reproduced at all. ``allow_out_of_band_shock=True`` is the opt-in that
     turns that refusal into a loud ``UserWarning``. The default still raises.
G2 — the **soft** bound ``τ_shock·Ω_max ≤ 0.2`` (the small-bandwidth regime
     Blow & Wood derive the GNLS under) was a module constant, so a deck that
     lives near ``Ω ~ ω₀/3`` had no honest way to acknowledge it.
     ``shock_bandwidth_limit=`` makes it per run; the default is unchanged.

Both knobs are pure policy: the operator is untouched. What the G1 override
*would* hide is the (A5) photon number, which has no meaning on a grid that
samples ``ω = ω₀ + Ω ≤ 0`` — so it reports ``nan`` and
:meth:`SplitStepEngine.shock_domain_ok` says why, rather than fabricating a
number that a drift gate would read as "converged".
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from photonics_helper import Fiber, GNLSESolver, TemporalGrid, Wavelength
from photonics_helper.base import Area, Length, Time
from photonics_helper.gnlse import _SHOCK_TAYLOR_LIMIT, SplitStepEngine
from photonics_helper.pulse import Envelope, Wave

C = 299792458.0
WL0 = 2.0e-6
OM0 = 2 * np.pi * C / WL0  # 9.42e14 rad/s

#: The paper's coherence grid's half-bandwidth ratio (0.5 fs sample over 32 ps).
PAPER_RATIO = 6.671

#: Ω_max/ω₀ ≈ 3.4: trips the hard bound, and is 26× cheaper than 64 000 points.
NARROW_N, NARROW_TMAX = 8192, Time(8e-12, "s")
#: Ω_max/ω₀ ≈ 0.85: inside the hard bound, far outside the soft one.
SOFT_N, SOFT_TMAX = 8192, Time(32e-12, "s")


def _pulse(grid, power=50.0, width_fs=50.0):
    env = Envelope(
        shape="gaussian",
        peak_amplitude=np.sqrt(power),
        pulse_width=Time(width_fs, "fs"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))


def _fiber(length_m=1e-3) -> Fiber:
    return Fiber(
        n2=2.3e-17,
        alpha=0.0,
        A_eff=Area(1.0e-12, "m^2"),
        length=Length(length_m, "m"),
        gamma=20.0,
    )


def _engine(grid, **kwargs) -> SplitStepEngine:
    return SplitStepEngine(
        pulse=_pulse(grid),
        fiber=_fiber(),
        betas=np.array([-0.56e-27, 1e-40]),
        include_self_steepening=True,
        **kwargs,
    )


class TestG1OutOfBandOptIn:
    """G1: default raises, opt-in warns loudly and runs."""

    def test_default_still_raises(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        assert float(grid.omega_max) > OM0  # the grid really is out of band
        with pytest.raises(ValueError, match=r"Ω_max < ω₀"):
            _engine(grid)

    def test_opt_in_warns_loudly_instead_of_raising(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        with pytest.warns(UserWarning, match="CONTROLLED OVERRIDE"):
            engine = _engine(grid, allow_out_of_band_shock=True)
        # The override is recorded, not silently assumed away.
        assert engine.allow_out_of_band_shock is True
        assert engine.shock_domain_ok is False
        assert engine.photon_number(np.asarray(engine.A)) != engine.photon_number(
            np.asarray(engine.A)
        )  # nan

    def test_soft_bandwidth_warning_survives_the_opt_in(self):
        """One opt-in must not silence the other, unrelated diagnostic."""
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _engine(grid, allow_out_of_band_shock=True)
        messages = [str(w.message) for w in caught]
        assert any("CONTROLLED OVERRIDE" in m for m in messages)
        assert any("GNLS derivation regime" in m for m in messages)

    def test_opt_in_run_completes_and_admits_the_invariant_is_undefined(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        with pytest.warns(UserWarning, match="CONTROLLED OVERRIDE"):
            engine = _engine(grid, allow_out_of_band_shock=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine.propagate(num_steps=8, nsaves=5)
        # The field is finite and the energy moved; only the (A5) invariant is
        # meaningless, and it says so instead of returning a plausible number.
        assert np.all(np.isfinite(np.asarray(engine.A)))
        assert np.all(np.isfinite(engine.energy_vs_z))
        assert np.all(np.isnan(engine.photon_number_vs_z))

    def test_gnlse_solver_forwards_the_opt_in_and_its_default_raises(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        # The wrapper builds its engine at propagate(), so that is where the
        # guard fires — same place, same message.
        solver = GNLSESolver(
            _pulse(grid),
            _fiber(),
            betas=np.array([-0.56e-27]),
            include_self_steepening=True,
        )
        with pytest.raises(ValueError, match=r"Ω_max < ω₀"):
            solver.propagate(num_steps=2)
        with pytest.warns(UserWarning, match="CONTROLLED OVERRIDE"):
            solver = GNLSESolver(
                _pulse(grid),
                _fiber(),
                betas=np.array([-0.56e-27]),
                include_self_steepening=True,
                allow_out_of_band_shock=True,
            )
            solver.propagate(num_steps=2)
        assert solver.allow_out_of_band_shock is True
        assert solver.shock_bandwidth_limit == _SHOCK_TAYLOR_LIMIT

    def test_verify_convergence_reports_the_gate_as_not_applicable(self):
        """A nan drift must not read as ``inf`` and fake a failed run."""
        grid = TemporalGrid(N=1024, Tmax=Time(2e-12, "s"))  # Ω_max/ω₀ ≈ 1.7
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = GNLSESolver(
                _pulse(grid),
                _fiber(2e-4),
                betas=np.array([-0.56e-27]),
                include_self_steepening=True,
                allow_out_of_band_shock=True,
                step_size=Length(2e-5, "m"),
            )
            solver.propagate(num_steps=6)
            report = solver.verify_convergence(probe_factor=2)
        assert report.invariant.startswith("n/a")
        assert np.isfinite(report.drift_pct)
        assert any("(A5) photon number is undefined" in n for n in report.notes)


class TestG2ConfigurableSoftBound:
    """G2: the soft bound is a keyword, and the default is unchanged."""

    def test_default_bound_is_the_module_constant(self):
        grid = TemporalGrid(N=SOFT_N, Tmax=SOFT_TMAX)
        with pytest.warns(UserWarning, match="GNLS derivation regime"):
            engine = _engine(grid)
        assert engine.shock_bandwidth_limit == _SHOCK_TAYLOR_LIMIT
        # ... and the hard bound is not what fired.
        assert engine.shock_domain_ok is True

    def test_raised_bound_silences_only_the_soft_warning(self):
        grid = TemporalGrid(N=SOFT_N, Tmax=SOFT_TMAX)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _engine(grid, shock_bandwidth_limit=1.0)
        assert not [w for w in caught if "derivation regime" in str(w.message)]

    def test_the_warning_names_the_configured_bound(self):
        grid = TemporalGrid(N=SOFT_N, Tmax=SOFT_TMAX)
        with pytest.warns(UserWarning, match=r"shock_bandwidth_limit = 0.3"):
            _engine(grid, shock_bandwidth_limit=0.3)

    def test_non_positive_bound_is_refused(self):
        grid = TemporalGrid(N=SOFT_N, Tmax=SOFT_TMAX)
        for bad in (0.0, -0.2):
            with pytest.raises(ValueError, match="shock_bandwidth_limit"):
                _engine(grid, shock_bandwidth_limit=bad)

    def test_solver_and_engine_agree_on_the_bound(self):
        grid = TemporalGrid(N=SOFT_N, Tmax=SOFT_TMAX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = GNLSESolver(
                _pulse(grid),
                _fiber(),
                betas=np.array([-0.56e-27]),
                include_self_steepening=True,
                shock_bandwidth_limit=0.9,
            )
            engine = solver._make_engine(None)
        assert solver.shock_bandwidth_limit == engine.shock_bandwidth_limit == 0.9
        with pytest.raises(ValueError, match="shock_bandwidth_limit"):
            GNLSESolver(
                _pulse(grid),
                _fiber(),
                betas=np.array([-0.56e-27]),
                shock_bandwidth_limit=0.0,
            )

    def test_the_paper_grid_is_reachable_with_both_knobs(self):
        """The G1 row of the paper's own table, as a construction smoke test."""
        grid = TemporalGrid(N=64000, Tmax=Time(32e-12, "s"))
        ratio = float(grid.omega_max) / OM0
        assert ratio == pytest.approx(PAPER_RATIO, rel=1e-3)
        with pytest.warns(UserWarning, match="CONTROLLED OVERRIDE"):
            engine = _engine(
                grid,
                allow_out_of_band_shock=True,
                shock_bandwidth_limit=2 * ratio,  # the paper's regime, declared
            )
        assert engine.shock_domain_ok is False


# ─── The same two knobs on the engines a real replication runs ───────────────


def _taper_table(grid, length_m: float, central_wavelength=WL0):
    """A constant-in-z β(ω) table spanning the grid (SI), for a taper deck."""
    from photonics_helper import ZDependentDispersion

    w = np.asarray(grid.w, dtype=float)
    beta = (2.6 * (OM0 + w) / C).astype(float)
    return ZDependentDispersion(
        omegas=OM0 + w,
        z_positions=np.array([0.0, length_m]),
        beta=np.column_stack([beta, beta]),
        central_wavelength=central_wavelength,
    )


class TestTaperedShockOptIn:
    """`TaperedGNLSESolver` is what a taper replication runs: same two knobs."""

    def _solver(self, grid, *, length_m=2e-4, **kwargs):
        from photonics_helper import TaperedGNLSESolver

        return TaperedGNLSESolver(
            pulse=_pulse(grid),
            fiber=_fiber(length_m),
            dispersion_profile=_taper_table(grid, length_m),
            include_raman=False,
            include_self_steepening=True,
            step_size=Length(2e-5, "m"),
            **kwargs,
        )

    def test_default_still_raises(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        with pytest.raises(ValueError, match=r"Ω_max < ω₀"):
            self._solver(grid).propagate(num_steps=2)

    def test_opt_in_warns_and_runs_end_to_end(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        solver = self._solver(grid, allow_out_of_band_shock=True)
        assert solver.shock_domain_ok is False
        with pytest.warns(UserWarning, match="CONTROLLED OVERRIDE"):
            solver.propagate(num_steps=4, nsaves=3)
        assert np.all(np.isfinite(np.asarray(solver.evolution[-1].envelope_field)))
        assert np.all(np.isnan(solver.photon_number_vs_z))

    def test_in_band_grid_keeps_its_domain_flag(self):
        grid = TemporalGrid(N=1024, Tmax=Time(64e-12, "s"))  # Ω_max/ω₀ ≈ 0.053
        solver = self._solver(grid)
        solver.propagate(num_steps=2)
        assert solver.shock_domain_ok is True

    def test_verify_convergence_calls_the_gate_not_applicable(self):
        grid = TemporalGrid(N=1024, Tmax=Time(2e-12, "s"))  # Ω_max/ω₀ ≈ 1.7
        solver = self._solver(grid, allow_out_of_band_shock=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.propagate(num_steps=4)
            report = solver.verify_convergence(probe_factor=2)
        assert report.invariant.startswith("n/a")
        assert report.invariant_undefined is True
        assert report.verdict == "CONVERGED (conservation n/a)"
        assert "CONVERGED (conservation n/a)" in report.summary()

    def test_bandwidth_limit_is_validated_and_forwarded(self):
        grid = TemporalGrid(N=SOFT_N, Tmax=SOFT_TMAX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = self._solver(grid, shock_bandwidth_limit=0.95)
            solver.propagate(num_steps=2)
        assert solver.shock_bandwidth_limit == 0.95
        assert solver._engine.shock_bandwidth_limit == 0.95
        with pytest.raises(ValueError, match="shock_bandwidth_limit"):
            self._solver(grid, shock_bandwidth_limit=0.0)


class TestMultimodeParity:
    """One policy, two engines: the multimode guard must not be a second copy."""

    def _scalar(self, grid, **kwargs):
        return SplitStepEngine(
            pulse=_pulse(grid),
            fiber=_fiber(),
            betas=np.array([-0.56e-27]),
            include_self_steepening=True,
            **kwargs,
        )

    def _multimode(self, grid, **kwargs):
        from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine

        return MultimodeSplitStepEngine(
            [_pulse(grid)],
            _fiber(),
            np.array([-0.56e-27]),
            include_shock=True,
            **kwargs,
        )

    def test_both_engines_raise_the_same_message_by_default(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        messages = []
        for build in (self._scalar, self._multimode):
            with pytest.raises(ValueError) as excinfo:
                build(grid)
            messages.append(str(excinfo.value))
        assert messages[0] == messages[1]
        assert "Ω_max < ω₀" in messages[0]

    def test_both_engines_warn_the_same_way_under_the_opt_in(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        messages = []
        for build in (self._scalar, self._multimode):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                engine = build(grid, allow_out_of_band_shock=True)
            assert engine.shock_domain_ok is False
            messages.append(
                [
                    str(w.message)
                    for w in caught
                    if "CONTROLLED OVERRIDE" in str(w.message)
                ]
            )
        assert len(messages[0]) == len(messages[1]) == 1
        assert messages[0][0] == messages[1][0]

    def test_both_engines_share_the_soft_bound(self):
        grid = TemporalGrid(N=SOFT_N, Tmax=SOFT_TMAX)
        for build in (self._scalar, self._multimode):
            with pytest.warns(UserWarning, match="GNLS derivation regime"):
                build(grid)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                build(grid, shock_bandwidth_limit=1.0)
            assert not [w for w in caught if "derivation regime" in str(w.message)]
            with pytest.raises(ValueError, match="shock_bandwidth_limit"):
                build(grid, shock_bandwidth_limit=0.0)

    def test_multimode_photon_number_is_nan_on_an_overridden_grid(self):
        grid = TemporalGrid(N=NARROW_N, Tmax=NARROW_TMAX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mm = self._multimode(grid, allow_out_of_band_shock=True)
            mm.propagate(4, nsaves=3)
        assert np.all(np.isnan(np.asarray(mm._photon_vs_z, dtype=float)))


class TestPropagateAutoRespectsTheOptIn:
    """`propagate_auto` widens an out-of-band grid by default; not on request."""

    def _solver(self, grid, **kwargs):
        return GNLSESolver(
            _pulse(grid),
            _fiber(2e-4),
            betas=np.array([-0.56e-27]),
            include_self_steepening=True,
            step_size=Length(2e-5, "m"),
            **kwargs,
        )

    def test_default_still_widens_the_grid(self):
        grid = TemporalGrid(N=1024, Tmax=Time(2e-12, "s"))
        solver = self._solver(grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = solver.propagate_auto(max_rounds=1, nsaves=3)
        assert report["widened"] is True
        assert solver.pulse.grid.omega_max < solver.pulse.central_frequency

    def test_opt_in_keeps_the_grid_and_says_the_invariant_is_gone(self):
        grid = TemporalGrid(N=1024, Tmax=Time(2e-12, "s"))
        solver = self._solver(grid, allow_out_of_band_shock=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = solver.propagate_auto(max_rounds=1, nsaves=3)
        assert report["widened"] is False
        assert report["N"] == 1024 and report["invariant"].startswith("n/a")
        # The grid the caller asked for is still there, unresolved.
        assert report["Tmax_s"] == pytest.approx(Time(2e-12, "s").as_s)
        assert any("not widened" in str(w.message) for w in caught)


class TestVerdictCarriesTheConservationState:
    """The verdict word alone must not read as "the invariant passed"."""

    @staticmethod
    def _diag(**kwargs):
        from photonics_helper.gnlse import ConvergenceDiagnostics

        base = dict(
            scheme="strang",
            num_steps=100,
            dz_m=1e-5,
            probe_steps=200,
            invariant="photon number (Blow & Wood A5)",
            drift_pct=1e-6,
            drift_tol_pct=1.0,
            rms_db=1e-3,
            peak_shift_nm=1e-3,
            overlap=1.0,
            rms_tol_db=0.5,
            peak_tol_nm=2.0,
            overlap_tol=0.99,
        )
        base.update(kwargs)
        return ConvergenceDiagnostics(**base)

    def test_default_verdicts_are_unchanged(self):
        assert self._diag().verdict == "CONVERGED"
        assert self._diag(rms_db=99.0).verdict == "NOT CONVERGED"
        # The long-standing loss/gain "n/a" case stays a plain CONVERGED: the
        # run is lossy by design, not undefined.
        lossy = self._diag(invariant="n/a (loss/gain/free-carrier run)", drift_pct=0.0)
        assert lossy.verdict == "CONVERGED"

    def test_undefined_invariant_says_so_in_the_verdict(self):
        diag = self._diag(
            invariant="n/a (shock samples ω ≤ 0)",
            drift_pct=0.0,
            invariant_undefined=True,
        )
        assert diag.converged is True
        assert diag.verdict == "CONVERGED (conservation n/a)"
        assert "verdict: CONVERGED (conservation n/a)" in diag.summary()

    def test_a_failed_run_is_still_not_converged_whatever_the_gate(self):
        diag = self._diag(
            invariant="n/a (shock samples ω ≤ 0)", invariant_undefined=True, overlap=0.5
        )
        assert diag.verdict == "NOT CONVERGED"
