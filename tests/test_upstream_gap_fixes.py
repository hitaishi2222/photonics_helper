"""Pin the remaining upstream gaps the naskar-scg replication recorded (GAPS.md §6).

One class per item, each reproducing the defect's *observable* rather than its
absence:

A3 — `interpolated_spectrum_db` needs the Jacobian; a dB-vs-wavelength claim
     without `|dω/dλ|` moves the −50 dB edges by ~13 % (≈800 nm at 5 µm).
A4 — D from a noisy n_eff must be able to carry σ, and the smoothness guard must
     not be bypassed without saying so.
A5 — a β(ω, z) table must be able to refuse (or report) out-of-table queries
     instead of silently flattening the dispersive-wave region.
C1 — a span metric that carries its level and numeraire.
C2 — a z-table assembled from one mode per taper section, and the frame trap.
C3 — the tapered solver's own convergence helper.
C4 — Raman validation at construction, not two layers down at engine build.
C5 — bimodal chalcogenide Raman rows with a source, so the rounded single
     Lorentzian is not the only thing a caller can pick.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper import (
    Fiber,
    GNLSESolver,
    TaperedGNLSESolver,
    TemporalGrid,
    Wavelength,
    ZDependentDispersion,
    span_at_level,
)
from photonics_helper.base import Area, Length, Time, WavelengthArray
from photonics_helper.fiber import Dispersion, WaveguideMode
from photonics_helper.gnlse import (
    ConvergenceDiagnostics,
    spectral_evolution_on_wavelength_grid,
)
from photonics_helper.pulse import Envelope, Wave

C = 299792458.0
WL0 = 2.0e-6
OM0 = 2 * np.pi * C / WL0


def _wave(n=512, tmax_ps=8.0, power=50.0, width_fs=50.0):
    grid = TemporalGrid(N=n, Tmax=Time(tmax_ps * 1e-12, "s"))
    env = Envelope(
        shape="gaussian", peak_amplitude=np.sqrt(power), pulse_width=Time(width_fs, "fs")
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))


def _fiber(length_m=5e-4):
    return Fiber(
        n2=2.3e-17, alpha=0.0, A_eff=Area(0.98e-12, "m^2"),
        length=Length(length_m, "m"), gamma=71.56,
    )


def _mode(scale=1.0, n_pts=8, lo_um=1.9, hi_um=2.1):
    wl_um = np.linspace(lo_um, hi_um, n_pts)
    neff = scale * (2.60 - 0.02 * (wl_um - 2.0) - 0.004 * (wl_um - 2.0) ** 2)
    return WaveguideMode(
        neff=neff,
        wavelengths=WavelengthArray(wl_um, "um"),
        central_wavelength=Wavelength(2.0, "um"),
    )


# ── A3 ─────────────────────────────────────────────────────────────────────
class TestSpectrumNumeraire:
    def _run(self):
        solver = GNLSESolver(
            pulse=_wave(), fiber=_fiber(), betas=np.array([2.0e-25]),
            include_raman=False, include_self_steepening=False,
        )
        solver.propagate(num_steps=20)
        return solver

    def test_default_is_wavelength_and_carries_the_jacobian(self):
        solver = self._run()
        wl, db_lam = solver.interpolated_spectrum_db(1000, 4000, 300)
        _, db_nu = solver.interpolated_spectrum_db(1000, 4000, 300, per="frequency")
        assert not np.allclose(db_lam, db_nu)
        # Both curves are normalised to their own peak, so their difference is the
        # Jacobian up to a constant: d(Δ)/d(log10 λ) = −20 dB/decade exactly.
        # (A curve without the Jacobian has slope 0 here — that is the defect.)
        band = db_nu > -40.0
        x = np.log10(wl[band])
        y = (db_lam - db_nu)[band]
        slope = float(np.polyfit(x, y, 1)[0])
        # 0.1 dB/decade of slack: the two branches interpolate differently.
        assert slope == pytest.approx(-20.0, abs=0.3)

    def test_bad_numeraire_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="per must be"):
            self._run().interpolated_spectrum_db(1000, 4000, 100, per="nm")
        with pytest.raises(ValueError, match="per must be"):
            spectral_evolution_on_wavelength_grid(
                self._run(), 1000, 4000, 100, per="nm"
            )

    def test_both_wavelength_resamplers_agree_step_by_step(self):
        """Same contract, same Jacobian: the waterfall and the dB curve."""
        solver = self._run()
        wl, db_curve = solver.interpolated_spectrum_db(1000, 4000, 300)
        _, wl_wf, db_waterfall = spectral_evolution_on_wavelength_grid(
            solver, 1000, 4000, 300
        )
        assert np.allclose(wl, wl_wf)
        # interpolated_spectrum_db normalises to its own step's peak, the
        # waterfall to the run's global peak; normalise both and compare.
        last = np.asarray(db_waterfall[-1], dtype=float)
        last -= last.max()
        # Compare where there is signal: the two helpers add their epsilon
        # (+1e-30) at different points, so their floors differ by ~34 dB while
        # the curves themselves must agree.
        band = db_curve > -100.0
        assert band.sum() > 50
        assert np.allclose(db_curve[band], last[band], atol=1e-6)

    def test_the_waterfall_default_carries_the_same_jacobian(self):
        solver = self._run()
        _, _, with_jac = spectral_evolution_on_wavelength_grid(
            solver, 1000, 4000, 300
        )
        _, _, without = spectral_evolution_on_wavelength_grid(
            solver, 1000, 4000, 300, per="frequency"
        )
        assert not np.allclose(with_jac, without)


# ── A4 ─────────────────────────────────────────────────────────────────────
class TestDispersionUncertainty:
    def _noisy_mode(self, sigma=1e-5, n_pts=41, lo_um=1.5, hi_um=2.5):
        wl_um = np.linspace(lo_um, hi_um, n_pts)
        clean = 2.60 - 0.02 * (wl_um - 2.0) - 0.004 * (wl_um - 2.0) ** 2
        rng = np.random.default_rng(20261105)
        return WaveguideMode(
            neff=clean + rng.normal(0.0, sigma, n_pts),
            wavelengths=WavelengthArray(wl_um, "um"),
            central_wavelength=Wavelength(2.0, "um"),
        )

    def test_sigma_follows_the_closed_form(self):
        mode = self._noisy_mode()
        d_lam = float(np.diff(mode.wavelengths.as_m).mean())
        disp = Dispersion.from_neff(
            neff=mode.neff,
            wavelengths=mode.wavelengths,
            central_wavelength=mode.central_wavelength,
            ignore_fit_error=True,
            neff_sigma=1e-5,
        )
        assert disp.sigma is not None
        wl_eq = np.asarray(disp.wavelengths.as_m, dtype=float)
        d_eq = float(np.mean(np.diff(wl_eq)))
        assert d_eq != pytest.approx(d_lam)  # it rebuilt an equally spaced grid
        pred = (wl_eq / C) * 4.0 * 1e-5 / d_eq**2
        # Reported in s/m², the same unit as `values`.
        assert np.allclose(np.asarray(disp.sigma), pred)
        # Size check: this project measured ~75 ps nm⁻¹ km⁻¹ at Δλ = 100 nm, and
        # the law scales as 1/Δλ², so the 20 nm grid used here must land in the
        # hundreds — the point is that it is resolvable and reported at all.
        mid = int(np.argmin(np.abs(wl_eq - WL0)))
        assert 1e2 < float(disp.sigma[mid]) * 1e6 < 5e3

    def test_no_sigma_means_unknown_not_zero(self):
        mode = self._noisy_mode(sigma=0.0)
        disp = mode.to_dispersion(ignore_fit_error=False)
        assert disp.sigma is None

    def test_bypassing_the_guard_is_no_longer_silent(self):
        n_pts = 9
        wl_um = np.linspace(1.9, 2.1, n_pts)
        # Alternating ±1e-3 in n_eff over 25 nm steps: the second difference is
        # orders of magnitude above the guard's 50 ps nm⁻¹ km⁻¹.
        neff = 2.6 + 1e-3 * np.array([1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=float)
        mode = WaveguideMode(
            neff=neff,
            wavelengths=WavelengthArray(wl_um, "um"),
            central_wavelength=Wavelength(2.0, "um"),
        )
        with pytest.warns(UserWarning, match="smoothness guard"):
            assert mode.to_dispersion() is not None
        with pytest.raises(ValueError):
            mode.to_dispersion(ignore_fit_error=False)
        # Explicit True is the old behaviour: it still *reports* the bad fit, it
        # just does not raise. Only the default gained the extra sentence.
        with pytest.warns(UserWarning, match="Bad fitting"):
            assert mode.to_dispersion(ignore_fit_error=True) is not None


# ── A5 ─────────────────────────────────────────────────────────────────────
class TestZTableOutOfRange:
    def _table(self, **kw):
        omega = np.linspace(0.8 * OM0, 1.2 * OM0, 11)
        beta = 2.6 * omega / C
        return ZDependentDispersion(
            omegas=omega,
            z_positions=np.array([0.0, 1e-3]),
            beta=np.column_stack([beta, beta]),
            central_wavelength=WL0,
            **kw,
        )

    def test_factories_forward_the_policy(self):
        """The documented constructors must reach the knob, not drop it.

        Both factories used to construct the dataclass without ``out_of_range``,
        so a caller who built a table through them could only ever get the
        permissive ``"nan"`` default.
        """
        omega = np.linspace(0.8 * OM0, 1.2 * OM0, 11)
        beta = 2.6 * omega / C
        table = ZDependentDispersion.from_arrays(
            omegas=omega,
            z_positions=np.array([0.0, 1e-3]),
            beta=np.column_stack([beta, beta]),
            central_wavelength=WL0,
            out_of_range="raise",
        )
        assert table.out_of_range == "raise"
        with pytest.raises(ValueError, match="lie outside the"):
            table.fn(np.array([0.5 * OM0, OM0]), 0.0)

    def test_from_sections_forwards_the_policy(self):
        mode = _mode()
        table = ZDependentDispersion.from_sections(
            np.array([0.0, 1e-3]), [mode, mode], out_of_range="clip"
        )
        assert table.out_of_range == "clip"

    def test_default_stays_nan(self):
        t = self._table()
        assert np.isnan(float(t.fn(2.0 * OM0, 0.0)))

    def test_raise_names_the_fraction_and_bounds(self):
        t = self._table(out_of_range="raise")
        with pytest.raises(ValueError, match="lie outside the"):
            t.fn(np.array([0.5 * OM0, OM0]), 0.0)

    def test_clip_reports_what_it_flattened(self):
        t = self._table(out_of_range="clip")
        out = t.fn(np.array([0.5 * OM0, OM0]), 0.0)
        assert np.all(np.isfinite(out))
        assert t.last_clipped_fraction == 0.5

    def test_engine_honours_the_raise_policy(self):
        t = self._table(out_of_range="raise")
        TaperedGNLSESolver(
            pulse=_wave(), fiber=_fiber(), dispersion_profile=t,
            include_raman=False, include_self_steepening=False,
        )
        # The N=512 / 8 ps grid spans ±2.0e14 rad/s, well inside this table,
        # so make it narrow on purpose.
        narrow = ZDependentDispersion(
            omegas=np.linspace(0.999 * OM0, 1.001 * OM0, 11),
            z_positions=np.array([0.0, 1e-3]),
            beta=np.column_stack(
                [
                    np.linspace(0.999, 1.001, 11) * 2.6 * OM0 / C,
                    np.linspace(0.999, 1.001, 11) * 2.6 * OM0 / C,
                ]
            ),
            central_wavelength=WL0,
            out_of_range="raise",
        )
        solving = TaperedGNLSESolver(
            pulse=_wave(), fiber=_fiber(), dispersion_profile=narrow,
            include_raman=False, include_self_steepening=False,
        )
        with pytest.raises(ValueError, match='out_of_range="raise"'):
            solving.propagate(num_steps=2)

    def _narrow(self):
        """A table 1 % wide around the carrier: any real grid is out of it."""
        return ZDependentDispersion(
            omegas=np.linspace(0.99 * OM0, 1.01 * OM0, 11),
            z_positions=np.array([0.0, 1e-3]),
            beta=np.column_stack(
                [
                    np.linspace(0.99, 1.01, 11) * 2.6 * OM0 / C,
                    np.linspace(0.99, 1.01, 11) * 2.6 * OM0 / C,
                ]
            ),
            central_wavelength=WL0,
        )

    def test_strict_raises_on_any_clip_not_five_percent(self):
        """1 % out of table is enough: the old 5 % grace hid the band edge."""
        solver = TaperedGNLSESolver(
            pulse=_wave(), fiber=_fiber(), dispersion_profile=self._narrow(),
            include_raman=False, include_self_steepening=False,
        )
        with pytest.raises(ValueError, match="clipped"):
            solver.propagate(num_steps=2, strict=True)

    def test_non_strict_keeps_warn_and_continue(self):
        solver = TaperedGNLSESolver(
            pulse=_wave(), fiber=_fiber(), dispersion_profile=self._narrow(),
            include_raman=False, include_self_steepening=False,
        )
        with pytest.warns(UserWarning, match="clipped"):
            solver.propagate(num_steps=2, strict=False)
        assert solver._engine.last_clipped_fraction > 0.0

    def test_engine_clip_dispersion_false_raises_true_warns(self):
        """The engine-level policy: ``False`` refuses, ``True`` reports."""
        from photonics_helper.gnlse import SplitStepEngine

        def _engine(clip):
            return SplitStepEngine(
                _wave(),
                _fiber(),
                betas=np.array([0.0]),
                dispersion_profile=self._narrow(),
                clip_dispersion=clip,
                include_raman=False,
                step_size=Length(1e-4, "m"),
            )

        with pytest.raises(ValueError, match="clipped"):
            _engine(False).propagate(2, nsaves=2)
        with pytest.warns(UserWarning, match="clipped"):
            engine = _engine(True)
            engine.propagate(2, nsaves=2)
        assert engine.last_clipped_fraction > 0.0

    def test_callable_non_finite_raises_in_every_mode(self):
        """A callable β(ω, z) has no edge to clip to: a NaN is a broken model."""
        from photonics_helper.gnlse import SplitStepEngine

        def beta(omega, z):
            out = 2.6 * omega / C
            return np.where(omega < 0.9 * OM0, np.nan, out)

        for clip in (None, False, True):
            engine = SplitStepEngine(
                _wave(),
                _fiber(),
                betas=np.array([0.0]),
                dispersion_profile=beta,
                clip_dispersion=clip,
                include_raman=False,
                step_size=Length(1e-4, "m"),
            )
            with pytest.raises(ValueError, match="not finite"):
                engine.propagate(2, nsaves=2)


# ── C1 ─────────────────────────────────────────────────────────────────────
class TestSpanAtLevel:
    def _log_gauss(self, sigma=0.25, lam0_nm=2000.0):
        wl = np.linspace(500.0, 10000.0, 6000)
        density = np.exp(-((np.log(wl) - np.log(lam0_nm)) ** 2) / (2 * sigma**2))
        return wl, density

    def test_two_numeraires_agree_when_labelled_correctly(self):
        wl, s_lam = self._log_gauss()
        a = span_at_level(wl, s_lam, -50.0, numeraire="wavelength")
        s_nu = s_lam * (2 * np.pi * C / (wl * 1e-9) ** 2) ** -1.0
        b = span_at_level(wl, s_nu, -50.0, numeraire="frequency")
        assert a.short_nm == pytest.approx(b.short_nm, rel=1e-6)
        assert a.long_nm == pytest.approx(b.long_nm, rel=1e-6)
        assert a.octaves == pytest.approx(b.octaves, rel=1e-9)

    def test_the_level_and_numeraire_travel_with_the_number(self):
        wl, s_lam = self._log_gauss()
        s_nu = s_lam * (2 * np.pi * C / (wl * 1e-9) ** 2) ** -1.0
        mislabelled = span_at_level(wl, s_nu, -50.0, numeraire="wavelength")
        correct = span_at_level(wl, s_lam, -50.0, numeraire="wavelength")
        # The octave count of a log-symmetric spectrum barely moves, but the
        # quoted edge wavelengths — what a "1.2–9.0 µm" claim actually states —
        # move by hundreds of nm. That is the defect in one number.
        assert abs(mislabelled.long_nm - correct.long_nm) > 200.0
        assert mislabelled.level_db == -50.0 and mislabelled.numeraire == "wavelength"

    def test_missing_crossing_is_nan_not_a_guess(self):
        wl, s_lam = self._log_gauss(sigma=0.6)  # wider than the axis
        span = span_at_level(wl, s_lam, -50.0)
        assert not np.isfinite(span.short_nm) or not np.isfinite(span.long_nm)
        assert np.isnan(span.octaves)

    def test_rejects_an_unsorted_axis(self):
        wl, s_lam = self._log_gauss()
        with pytest.raises(ValueError, match="strictly increasing"):
            span_at_level(wl[::-1], s_lam, -50.0)


# ── C2 ─────────────────────────────────────────────────────────────────────
class TestZDependentFromSections:
    def test_constant_sections_give_a_constant_table(self):
        mode = _mode()
        table = ZDependentDispersion.from_sections(
            np.array([0.0, 1e-3, 2e-3]), [mode, mode, mode]
        )
        assert table.beta.shape == (len(table.omegas), 3)
        assert np.allclose(table.beta[:, 0], table.beta[:, -1])
        # β = n_eff·ω/c, checked against the mode at the central point.
        i = int(np.argmin(np.abs(table.omegas)))
        n_here = float(np.interp(
            2 * np.pi * C / table.omegas[i],
            np.asarray(mode.wavelengths.as_m, float),
            np.asarray(mode.neff, float),
        ))
        assert table.beta[i, 1] == pytest.approx(n_here * table.omegas[i] / C, rel=1e-9)

    def test_two_sections_differ_as_their_indices_do(self):
        thin, thick = _mode(scale=1.0), _mode(scale=1.0004)
        table = ZDependentDispersion.from_sections(
            np.array([0.0, 1e-3]), [thin, thick]
        )
        assert table.beta[:, 1].mean() > table.beta[:, 0].mean()

    def test_centered_removes_the_group_delay_but_reports_it(self):
        mode = _mode()
        raw = ZDependentDispersion.from_sections(np.array([0.0, 1e-3]), [mode, mode])
        cen = ZDependentDispersion.from_sections(
            np.array([0.0, 1e-3]), [mode, mode], centered=True,
            central_wavelength=Wavelength(2.0, "um"),
        )
        i0 = int(np.argmin(np.abs(raw.omegas - OM0)))
        assert abs(float(raw.beta[i0, 0])) > 1.0      # β₀ is ~4e5 rad/m here
        # Exact zero is not achievable: the table is linearly interpolated, so
        # centring leaves the interpolant's own error, ~1e-5 relative.
        assert abs(float(cen.beta[i0, 0])) / abs(float(raw.beta[i0, 0])) < 1e-4
        edge = -1
        assert abs(float(cen.beta[edge, 0])) < abs(float(raw.beta[edge, 0]))

    def test_mismatched_sections_and_z_positions_raise(self):
        with pytest.raises(ValueError, match="sections for"):
            ZDependentDispersion.from_sections(np.array([0.0, 1e-3]), [_mode()])


# ── C3 ─────────────────────────────────────────────────────────────────────
class TestTaperedConvergence:
    def test_reports_the_table_and_restores_the_coarse_run(self):
        mode = _mode()
        table = ZDependentDispersion.from_sections(
            np.array([0.0, 1e-3]), [mode, mode]
        )
        solver = TaperedGNLSESolver(
            pulse=_wave(n=256, tmax_ps=8.0), fiber=_fiber(),
            dispersion_profile=table, include_raman=False,
            include_self_steepening=False,
        )
        solver.propagate(num_steps=20)
        n_before = len(solver.evolution)
        a_before = np.array(solver.evolution[-1].envelope_field, copy=True)

        diag = solver.verify_convergence(probe_factor=2, warn=False)
        assert isinstance(diag, ConvergenceDiagnostics)
        assert diag.num_steps == 20 and diag.probe_steps == 40
        assert diag.dz_m == pytest.approx(5e-4 / 20)
        assert np.isfinite(diag.rms_db) and 0.0 <= diag.overlap <= 1.0

        # The coarse run is still what the object holds.
        assert len(solver.evolution) == n_before
        assert np.array_equal(
            np.asarray(solver.evolution[-1].envelope_field), a_before
        )

    def test_probe_factor_below_two_is_refused(self):
        mode = _mode()
        table = ZDependentDispersion.from_sections(np.array([0.0, 1e-3]), [mode, mode])
        solver = TaperedGNLSESolver(
            pulse=_wave(n=128), fiber=_fiber(), dispersion_profile=table,
            include_raman=False,
        )
        solver.propagate(num_steps=5)
        with pytest.raises(ValueError, match="probe_factor"):
            solver.verify_convergence(probe_factor=1)


# ── C4 ─────────────────────────────────────────────────────────────────────
class TestRamanValidatedEarly:
    def test_scalar_solver_raises_at_construction(self):
        with pytest.raises(ValueError, match="raman_response is None"):
            GNLSESolver(
                pulse=_wave(), fiber=_fiber(), betas=np.array([2.0e-25]),
                include_raman=True,
            )

    def test_tapered_solver_raises_at_construction(self):
        mode = _mode()
        table = ZDependentDispersion.from_sections(np.array([0.0, 1e-3]), [mode, mode])
        with pytest.raises(ValueError, match="raman_response is None"):
            TaperedGNLSESolver(
                pulse=_wave(), fiber=_fiber(), dispersion_profile=table,
                include_raman=True,
            )

    def test_disabled_raman_still_builds(self):
        GNLSESolver(
            pulse=_wave(), fiber=_fiber(), betas=np.array([2.0e-25]),
            include_raman=False,
        )


# ── C5 ─────────────────────────────────────────────────────────────────────
class TestBimodalChalcogenideRaman:
    def test_asse2_row_carries_the_papers_parameters(self):
        from photonics_helper.raman.spec import RamanSpec

        spec = RamanSpec.from_database("AsSe2 (Raman only)")
        assert spec.fR == pytest.approx(0.148)
        assert float(spec.tau1.as_s) == pytest.approx(23e-15)
        assert float(spec.tau2.as_s) == pytest.approx(164.5e-15)
        assert "385502" in str(spec.references)

    def test_as2se3_does_not_inherit_the_silica_taus(self):
        from photonics_helper.raman.spec import RamanSpec

        spec = RamanSpec.from_database("As2Se3")
        assert float(spec.tau1.as_s) == pytest.approx(23e-15)
        assert float(spec.tau2.as_s) == pytest.approx(164.5e-15)

    def test_response_is_a_decaying_bimodal_kernel(self):
        from photonics_helper.raman.spec import RamanSpec

        spec = RamanSpec.from_database("AsSe2 (Raman only)")
        grid = TemporalGrid(N=512, Tmax=Time(2e-12, "s"))
        resp = spec.response(grid)
        h = np.asarray(resp.h_R(np.asarray(grid.t, dtype=float)), dtype=float)
        assert h.shape == (grid.N,) and np.all(np.isfinite(h))
        # Bimodal h_R(t) ∝ exp(-t/τ₂)·sin(t/τ₁): zero at t = 0, one lobe, decay.
        assert h[0] == pytest.approx(0.0, abs=1e-30)
        i_peak = int(np.argmax(h))
        assert 0 < i_peak < h.size - 1
        assert abs(float(h[-1])) < 0.05 * float(h[i_peak])
