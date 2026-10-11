"""Composition-scheme tests: ``strang`` (2nd order) vs ``yoshida4`` (4th order).

Rationale (``formal/PLAN.md`` §17): the outer splitting ``exp(L dz/2) exp(N dz)
exp(L dz/2)`` is Blow & Wood (1989) Eq. (17), which its authors state is
"globally second-order accurate". Composing three such steps with the Suzuki
triple-jump weights (Yoshida 1990, Eqs. 4.1/4.5/4.6) cancels the leading
``dz**3`` term and gives a fourth-order method for the same physics code.

These tests pin the *algebra* of the composition (exact), and the
*conservation* properties (exact), without paying for a full order-measurement
sweep. The order itself is measured by
:func:`photonics_helper.gnlse_validation.order_study`; see ``PLAN.md`` §17.6
Stage S0/S1 for those numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import (
    YOSHIDA4_W0,
    YOSHIDA4_W1,
    YOSHIDA4_WEIGHTS,
    FiberProfile,
    SplitStepEngine,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL0 = 1550e-9
OMEGA0 = 2 * np.pi * C_MS / WL0
BETA2 = -21e-27
GAMMA = 2.0
T0 = 50e-15 / 1.763
NGRID = 4096
LENGTH = 2.0 * (np.pi / 2) * T0**2 / abs(BETA2)


# ---------------------------------------------------------------------------
# Composition algebra (Yoshida 1990, Eq. 4.5) — exact
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    """``2*w0 + w1 = 1``: the three sub-steps advance exactly ``dz``."""
    assert 2 * YOSHIDA4_W0 + YOSHIDA4_W1 == pytest.approx(1.0, abs=1e-14)


def test_cubic_error_term_cancels():
    """``2*w0**3 + w1**3 = 0``: the leading ``dz**3`` error is removed."""
    assert 2 * YOSHIDA4_W0**3 + YOSHIDA4_W1**3 == pytest.approx(0.0, abs=1e-14)


def test_weight_order_is_outer_middle_outer():
    assert YOSHIDA4_WEIGHTS == (YOSHIDA4_W0, YOSHIDA4_W1, YOSHIDA4_W0)
    assert sum(YOSHIDA4_WEIGHTS) == pytest.approx(1.0, abs=1e-14)


def test_middle_weight_is_negative():
    """One sub-step runs backwards — the extrapolation half of the scheme."""
    assert YOSHIDA4_W1 < 0 < YOSHIDA4_W0


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------


def _engine(scheme: str, num_steps: int = 200) -> SplitStepEngine:
    grid = TemporalGrid(N=NGRID, Tmax=Time(8e-12, "s"))
    peak = 9.0 * abs(BETA2) / (GAMMA * T0**2)  # N = 3 soliton
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(peak), pulse_width=Time(T0, "s")
    )
    pulse = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m")
    )
    fiber = FiberProfile.from_gamma(
        gamma=GAMMA, n2=2.6e-20, omega0=OMEGA0, length=Length(LENGTH, "m")
    )
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),
        step_size=Length(LENGTH / num_steps, "m"),
        scheme=scheme,
    )
    engine.propagate(num_steps=num_steps)
    return engine


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError, match="Unknown scheme"):
        _engine("rk45")


def test_strang_is_the_default():
    grid = TemporalGrid(N=NGRID, Tmax=Time(8e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(T0, "s"))
    pulse = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m")
    )
    fiber = FiberProfile.from_gamma(
        gamma=GAMMA, n2=2.6e-20, omega0=OMEGA0, length=Length(LENGTH, "m")
    )
    engine = SplitStepEngine(
        pulse=pulse, fiber=fiber, betas=np.array([BETA2 * 1e24])
    )
    assert engine.scheme == "strang"


@pytest.mark.parametrize("scheme", ["strang", "yoshida4"])
def test_energy_conserved_for_pure_nlse(scheme: str):
    """With no loss, TPA or shock, both schemes conserve energy to round-off.

    Blow & Wood (1989) note that "energy is conserved exactly by this split step
    scheme for the pure NLS equation"; a symmetric composition of
    energy-conserving steps inherits that, so this must hold for ``yoshida4``
    too. Any drift here would mean the backward sub-step broke symmetry.
    """
    energy = _engine(scheme).energy_vs_z
    drift = abs(float(energy[-1]) - float(energy[0])) / float(energy[0])
    assert drift < 1e-12, f"{scheme} drifted by {drift:.3e} on a lossless run"


def test_yoshida4_advances_the_same_distance_as_strang():
    """Both schemes end at ``z = L``; only their error differs."""
    a = _engine("strang")
    b = _engine("yoshida4")
    assert float(a._z_positions[-1]) == pytest.approx(float(b._z_positions[-1]))
    assert float(b._z_positions[-1]) == pytest.approx(LENGTH, rel=1e-9)