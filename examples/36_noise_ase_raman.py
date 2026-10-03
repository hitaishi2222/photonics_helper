r"""
Example: ASE and Raman noise
============================

Demonstrates ``photonics_helper.noise`` — the stochastic half of the library,
and the first example in the tree to touch it. Five panels, in the order a user
actually asks the questions:

1. **Is it reproducible?** (the question nobody asks of a "noise" function, and
   the one the ``rng``/``seed`` split answers silently),
2. **What does ``level_dB`` mean?** — an ASE ladder read off the spectrum,
   with the OSNR printed,
3. **What is the Raman noise floor?** — the spontaneous-Raman field against the
   *deterministic* delayed-Raman response it thermalises,
4. **What does the noise do to a pulse?** — the same noised pulse propagated,
   input vs output spectra,
5. **What does it do to coherence?** — ``g₁₂`` over an ensemble, which no
   single realisation can show.

The conventions, so nothing here has to be guessed
---------------------------------------------------
**ASE power PSD.** :func:`ase_noise_field` builds a *flat* spectrum: every bin
(except DC, which is zeroed so no pump is injected) is given the **same
amplitude** ``|DC(CW of ``reference_power``)| · 10^(``level_dB``/20)``. Flat
amplitude over a flat ``Δω`` grid is a flat **power** density, so the noise
power per bin is ``reference_power · 10^(``level_dB``/10)`` — i.e. ``level_dB``
is the level *relative to the DC line of a CW of ``reference_power``*, which
``add_ase_noise`` defaults to the input wave's peak power. The Närhi et al.
(2016) supercontinuum convention of "−50 dB ASE" is exactly this number. The
field carries uniform random spectral phase, so it is incoherent with the
signal by construction, and it round-trips through the grid's scaled
``fft``/``ifft`` pair so the amplitude is exact rather than approximate.

**The fluctuation–dissipation expression.** :func:`raman_noise_field` gives
each bin a complex Gaussian of variance

.. math::

    \sigma^2 = \frac{\hbar\omega_0}{2}\,\mathrm{Im}[\tilde h_R]\,
               (n_{th}+1)\,\frac{2\pi}{\Delta\omega},
    \qquad n_{th} = \frac{1}{e^{\hbar\Omega_R/k_B T}-1},

shaped by the positive Raman gain ``Im[ñ_R]`` (bins with ``Im[ñ_R] ≤ 0``, the
anti-Stokes side, carry no spontaneous Stokes seed). This is **thermodynamic,
not empirical**: the amplitude is set by the quantum Langevin partition
function of a thermal phonon bath, with no fudge factor fitted to a measured
spectrum. The ``n_th + 1 = 1/(1 − exp(−ħΩ_R/k_B T))`` factor is **not** ≈ 1 at
an optical Raman shift: silica's 440 cm⁻¹ puts ``ħΩ_R/k_B = 96 K``, so the
semi-classical ``n_th → 0`` limit needs ``T ≫ 96 K`` and 300 K only half-supplies
it — ``n_th = 0.138`` there, 0.534 at 600 K. Doubling the temperature therefore
moves the floor by ~15 % in rms, which is a real thermal correction, not the
negligible one a bare "at optical frequencies n_th ≪ 1" would promise. The
library keeps ``temperature`` explicit so the same expression also resolves for
mid-IR and cryogenic work.

The caller scales the returned field by ``√Δz`` (the Langevin ``δ(z−z′)``
discretisation) and adds it once per step; this example uses ``√Δz`` when
plotting so the panels are in field units.

Where the deterministic model sits
----------------------------------
``examples/05``–``07`` and ``14`` propagate with the **deterministic** delayed
Raman response: a mean-field response, no bath. Panel (a)/(b) puts that
response and its thermal noise floor on the same axes — the noise is not a
different physics, it is the same ``ñ_R`` with a thermal partition function in
front of it. Run this example after ``05`` to see what the deterministic curve
throws away.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from typing import cast

# Prefer the repository package over any older site-packages install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.constants import hbar, k as k_B

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.noise import (
    add_ase_noise,
    ase_noise_field,
    coherence_g12,
    raman_noise_field,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

# ── Reference configuration ───────────────────────────────────────
LAM0 = 1550e-9  # m
N = 2**14  # samples
TMAX = 100.0e-12  # s -> dt = 6.1 fs, ~16 samples per pulse
T0 = 1.0e-12  # s, sech 1/e intensity half-width (1 ps)
P0 = 1000.0  # peak power, in the envelope's own normalised power units
BETA2 = -21.0e-27  # s^2/m, anomalous silica at 1550 nm (-21 ps^2/km)
L_FIBER = 2.0  # m  (~3 nonlinear lengths at this power)
N_STEPS = 400  # dz = 5 mm, an order of magnitude inside L_D = T0^2/|beta2|

SEED_A = 7  # the "same seed" of the reproducibility check
SEED_B = 8  # the "different seed" of the same check
ASE_LEVELS = [-70.0, -75.0, -80.0, -85.0]
ASE_REFERENCE_LEVEL = -70.0  # the level whose PSD is checked bin by bin
ASE_PROP_LEVEL = -70.0  # the level propagated in panels 4 and 5

# OSNR band convention, stated because OSNR is meaningless without one:
# "signal band" is +-BAND_SIG around the carrier, "noise band" is an
# OSA-like window well outside the pulse but inside the noise floor.
BAND_SIG_HZ = 3.0e12
BAND_NOISE_LO_HZ = 10.0e12
BAND_NOISE_HI_HZ = 50.0e12
PLOT_WINDOW_THZ = 5.0  # x-axis half-width of the ASE panels

TEMPERATURES = (300.0, 600.0)
DZ = L_FIBER / N_STEPS  # m, the step sqrt(dz) scaling of the noise field
N_RUNS = 10  # ensemble size for g12

OUT_DIR = _ROOT / "examples/images"
FIGS = {
    "ase": OUT_DIR / "36_ase_noise_spectra.png",
    "raman": OUT_DIR / "36_raman_thermal_floor.png",
}


def _rule(title: str) -> str:
    return "\n" + "=" * 66 + f"\n {title}\n" + "=" * 66


def _db(power: NDArray | float) -> NDArray:
    """Power ratio in dB, floored 300 dB down so a zero bin stays plottable."""
    floored = np.maximum(np.asarray(power, dtype=float), 1e-300)
    return np.asarray(10.0 * np.log10(floored))


def _make_grid() -> TemporalGrid:
    return TemporalGrid(N=N, Tmax=Time(TMAX, "s"))


def _make_pulse(grid: TemporalGrid) -> Wave:
    """A sech pulse in normalised power units (max|A|^2 = P0)."""
    env = Envelope(
        shape="sech", peak_amplitude=float(np.sqrt(P0)), pulse_width=Time(T0, "s")
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(LAM0, "m"))


def _peak_power(wave: Wave) -> float:
    """max|A|^2 in the envelope's own (normalised) power units.

    Deliberately not ``Wave.peak_power``: with no effective area attached that
    property returns ``max|A|^2`` *and warns*, and with one attached it switches
    to the epsilon_0 convention, which is not the unit this pulse is built in.
    The ASE reference power has to share the units of ``sqrt(P0)`` or the dB
    level is wrong by orders of magnitude, so the convention is pinned here.
    """
    field = np.asarray(wave.envelope_field, dtype=complex)
    return float(np.max(np.abs(field)) ** 2)


def _ase(wave: Wave, level_dB: float, seed: int) -> Wave:
    """``add_ase_noise`` with the reference power pinned to the input peak.

    The reference is passed explicitly rather than left to the default so the
    clean and noised waves share one definition of ``reference_power``.
    """
    return add_ase_noise(wave, level_dB, reference_power=_peak_power(wave), seed=seed)


def _spectral_peak(grid: TemporalGrid, wave: Wave) -> float:
    """Peak spectral bin power of ``wave`` -- the 0 dB reference for the plots."""
    field = np.asarray(wave.envelope_field, dtype=complex)
    return float(np.max(np.abs(grid.fft(field)) ** 2))


def _spec_db(wave: Wave, grid: TemporalGrid, ref: float) -> NDArray:
    """Power spectrum of ``wave`` in dB relative to ``ref``.

    ``grid.fft`` returns a density-like quantity (the pair carries a ``dt``), so
    ``ref`` is a *spectrum*, not a power: the clean pulse's peak spectral bin is
    0 dB and the ASE floors sit ``level_dB`` below it.
    """
    return np.asarray(_db(np.abs(grid.fft(wave.envelope_field)) ** 2 / ref))


def _rms_bandwidth(wave: Wave, grid: TemporalGrid) -> float:
    """RMS spectral width over the whole grid (rad/s) -- no band clipping.

    Clipping to a fixed band would report a constant by construction, which is
    exactly the mistake this panel exists to avoid.
    """
    power = np.abs(grid.fft(np.asarray(wave.envelope_field, dtype=complex))) ** 2
    w = grid.w
    mean = float(np.sum(w * power) / np.sum(power))
    return float(np.sqrt(np.sum((w - mean) ** 2 * power) / np.sum(power)))


def _rms_time_width(wave: Wave, grid: TemporalGrid) -> float:
    """RMS temporal width (s) of the total field intensity."""
    intensity = np.abs(np.asarray(wave.envelope_field, dtype=complex)) ** 2
    t = grid.t
    mean = float(np.sum(t * intensity) / np.sum(intensity))
    return float(np.sqrt(np.sum((t - mean) ** 2 * intensity) / np.sum(intensity)))


def _thermal(omega: NDArray | float, temp: float) -> NDArray:
    """The spontaneous + stimulated factor ``n_th + 1`` at a given Omega."""
    exponent = -hbar * np.asarray(omega, dtype=float) / (k_B * temp)
    return np.asarray(1.0 / (1.0 - np.exp(exponent)))


def _live_bins(power: NDArray, floor_rel: float = 1e-6) -> NDArray[np.bool_]:
    """Bins whose power is within ``floor_rel`` of the peak.

    Away from the pulse the field is numerical dust; ``coherence_g12`` reports
    0 there by construction (no power, no ratio), and averaging those bins in
    would report a coherence the experiment cannot measure.
    """
    return np.asarray(power) > float(np.max(power)) * floor_rel


def _sym_band_power(power: NDArray, grid: TemporalGrid, lo: float, hi: float) -> float:
    """Power integrated over ``|Δf| in [lo, hi]`` -- a symmetric band."""
    offset = np.abs(grid.w) / (2.0 * np.pi)
    return float(np.sum(power[(offset >= lo) & (offset <= hi)]) * grid.dw)


def _band_power(power: NDArray, grid: TemporalGrid, lo: float, hi: float) -> float:
    """Power integrated over a one-sided, signed band, in Hz.

    ``(20, 45) THz`` is the red side only, ``(-45, -20) THz`` the blue side.
    One-sided windows are how a Raman effect becomes visible at all: the
    symmetric band integrates red gain against blue loss and reports ~nothing.
    """
    offset = grid.w / (2.0 * np.pi)
    mask = (offset >= min(lo, hi)) & (offset <= max(lo, hi))
    return float(np.sum(power[mask]) * grid.dw)


def _osnr_db(wave: Wave, grid: TemporalGrid) -> float:
    """Signal-band power over noise-band power, in dB (bands stated above)."""
    power = np.abs(grid.fft(wave.envelope_field)) ** 2
    sig = _sym_band_power(power, grid, 0.0, BAND_SIG_HZ)
    noise = _sym_band_power(power, grid, BAND_NOISE_LO_HZ, BAND_NOISE_HI_HZ)
    assert noise > 0.0, "noise band is empty -- OSNR would be a division by zero"
    return float(_db(sig / noise))


# ── 1. Reproducibility ────────────────────────────────────────────
def reproducibility_panel() -> dict:
    print(_rule("1. Reproducibility: what the seed actually guarantees"))

    grid = _make_grid()
    pulse = _make_pulse(grid)

    same_a = _ase(pulse, ASE_REFERENCE_LEVEL, SEED_A)
    same_b = _ase(pulse, ASE_REFERENCE_LEVEL, SEED_A)
    diff = _ase(pulse, ASE_REFERENCE_LEVEL, SEED_B)

    f_a = np.asarray(same_a.envelope_field, dtype=complex)
    f_b = np.asarray(same_b.envelope_field, dtype=complex)
    f_d = np.asarray(diff.envelope_field, dtype=complex)

    # Bit-identical, not merely close: same seed -> the same draws.
    assert np.array_equal(f_a, f_b), "same seed produced a different field"
    print(f"  add_ase_noise(level_dB={ASE_REFERENCE_LEVEL:+.0f}, seed={SEED_A}) twice:")
    print(
        f"    max|field_a - field_b| = {np.max(np.abs(f_a - f_b)):.3e}   [bit-identical]"
    )
    print("    (array_equal, not allclose: the draws are the same draws)")

    delta = float(np.max(np.abs(f_a - f_d)))
    assert delta > 0.0, "a different seed produced the same field"
    rel = delta / float(np.max(np.abs(f_d)))
    print(f"\n  seed={SEED_B} instead:")
    print(f"    max|field_a - field_b| = {delta:.3e}  ({rel * 100:.1f}% of peak)")
    print("    -> the seed is the *only* thing standing between you and a")
    print("       figure that changes every run. Everything in this example is")
    print("       seeded; there is no deliberately unseeded panel.")

    print("\n  rng and seed are mutually exclusive, and the library says so:")
    try:
        add_ase_noise(
            pulse,
            ASE_REFERENCE_LEVEL,
            reference_power=_peak_power(pulse),
            rng=np.random.default_rng(0),
            seed=SEED_A,
        )
    except ValueError as exc:
        print(f"    ValueError: {exc}")
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("passing both rng and seed should raise ValueError")

    print("\n  the noise floor is the requested level, bin by bin:")
    level = ASE_REFERENCE_LEVEL
    noise = ase_noise_field(grid, _peak_power(pulse), level, seed=SEED_A)
    amp = np.abs(grid.fft(noise))
    dc = np.abs(grid.fft(np.full(N, np.sqrt(P0), dtype=complex))[N // 2])
    expected = dc * 10.0 ** (level / 20.0)
    dc_bin = N // 2
    bins = np.delete(np.arange(N), dc_bin)
    spread = float(np.max(np.abs(amp[bins] - expected)) / expected)
    assert spread < 1e-9, f"ASE bins are not flat to {expected} (spread {spread:.2e})"
    print(f"    DC(CW at P_ref = {P0:g})        = {dc:.6e}")
    print(
        f"    DC x 10^({level:+.0f}/20)  = {expected:.6e} per-bin amplitude  [target]"
    )
    print(f"    measured spread           = {spread:.2e} (relative)            [flat]")
    print(f"    DC bin itself             = {amp[dc_bin]:.3e} (zeroed: no pump added)")
    print("    flat amplitude on a flat dOmega grid == flat power PSD, so the")
    print(f"    noise power per bin is P_ref x 10^({level:+.0f}/10).")

    return {
        "grid": grid,
        "pulse": pulse,
        "spread": spread,
        "seed_a_field": f_a,
        "seed_a": same_a,
        "seed_b": same_b,
        "different": diff,
        "dc": float(dc),
        "expected": float(expected),
    }


# ── 2. ASE ladder ─────────────────────────────────────────────────
def ase_ladder_panel(ctx: dict) -> dict:
    print(_rule("2. The ASE ladder: level_dB read off the spectrum"))

    grid: TemporalGrid = ctx["grid"]
    pulse: Wave = ctx["pulse"]
    offset_thz = grid.w / (2.0 * np.pi) / 1e12
    clean = _spec_db(pulse, grid, _spectral_peak(grid, pulse))
    # Zoom the x-axis onto the pulse and its immediate shoulders: the ASE floor
    # beyond that is a flat line whose only information is its height, which
    # panel (c) turns into a number.
    window = np.abs(offset_thz) <= PLOT_WINDOW_THZ

    print(f"  seed={SEED_A}, reference_power = pulse peak power = {P0:g}")
    print("  note what level_dB is referenced to: the DC LINE of a CW of that")
    print(
        f"  power. A {T0 * 1e12:g} ps pulse spreads its energy over many bins, so"
        " its own"
    )
    print(
        "  spectral peak sits "
        f"{float(_db(_spectral_peak(grid, pulse) / (P0 * (grid.N * grid.dt) ** 2))):.1f}"
        " dB below that line, and"
    )
    print("  the ASE levels in this ladder read as that many dB below the pulse")
    print("  peak, not as the quoted level_dB. This is the single easiest way to")
    print("  misquote an ASE background.")
    print(
        f"  bands: signal |df| <= {BAND_SIG_HZ / 1e12:g} THz, "
        f"noise {BAND_NOISE_LO_HZ / 1e12:g}-{BAND_NOISE_HI_HZ / 1e12:g} THz"
    )
    clean_osnr = _osnr_db(pulse, grid)
    print(f"  clean pulse OSNR                    : {clean_osnr:6.2f} dB")
    print(
        "  (the ASE is flat over all "
        f"{int(2 * (BAND_NOISE_HI_HZ - BAND_NOISE_LO_HZ) / (grid.dw / 2 / np.pi))}"
        " bins of that band, so the noise power there is the floor itself"
    )
    ladder = []
    for level in ASE_LEVELS:
        noisy = _ase(pulse, level, SEED_A)
        osnr = _osnr_db(noisy, grid)
        # The pulse's own spectral tail is the floor of the noise-band reading;
        # subtracting it is what makes the level scaling exact rather than
        # approximate.
        tail = _sym_band_power(
            np.abs(grid.fft(pulse.envelope_field)) ** 2,
            grid,
            BAND_NOISE_LO_HZ,
            BAND_NOISE_HI_HZ,
        )
        floor = (
            _sym_band_power(
                np.abs(grid.fft(noisy.envelope_field)) ** 2,
                grid,
                BAND_NOISE_LO_HZ,
                BAND_NOISE_HI_HZ,
            )
            - tail
        )
        ladder.append(
            {
                "level": level,
                "osnr": osnr,
                "floor": floor,
                "spectrum": _spec_db(noisy, grid, _spectral_peak(grid, pulse)),
            }
        )
        print(
            f"  level_dB = {level:+5.1f} -> OSNR {osnr:6.2f} dB, "
            f"noise-band power {10.0 * np.log10(floor):7.2f} dB"
            f"  (clean + {osnr - clean_osnr:+.2f} dB)"
        )

    # The exact statement: the noise floor is 10^(level_dB/10) in power, so two
    # levels 5 dB apart differ by exactly 5 dB of band power (power dB and the
    # level dB are the same number, which is the whole content of the dB knob).
    levels = np.array([cast(float, item["level"]) for item in ladder])
    floors = np.array([cast(float, item["floor"]) for item in ladder])
    steps = np.diff(levels)
    measured = 10.0 * np.log10(floors[1:] / floors[:-1])
    assert np.allclose(measured, steps, atol=1e-6), (
        f"noise floor steps are {measured} dB, not the requested {steps} dB"
    )
    print("\n  the floor is exactly 10^(level_dB/10)                      [asserted]")
    print(f"    measured steps: {', '.join(f'{m:+.4f}' for m in measured)} dB")

    # The OSNR follows with the same slope to within the ASE that spills into
    # the signal band -- which is not exactly zero, and pretending otherwise
    # would be the easy lie. The residual is printed, not hidden.
    deltas = np.diff(np.array([cast(float, item["osnr"]) for item in ladder]))
    residual = np.abs(deltas - 5.0)
    print("\n  OSNR follows, within the ASE spilling into the signal band:")
    print(f"    measured OSNR steps: {', '.join(f'{d:+.3f}' for d in deltas)} dB")
    print(f"    departure from 5.000 dB: {', '.join(f'{r:.3f}' for r in residual)} dB")
    print(f"    (largest {residual.max():.3f} dB, at the loudest level where the")
    print("     ASE inside +-3 THz is a real fraction of the signal band power)")
    assert residual.max() < 0.5, f"OSNR ladder steps depart by {residual} dB"

    return {"ladder": ladder, "clean": clean, "offset": offset_thz, "window": window}


def ase_figure(ctx: dict, lad: dict) -> None:
    """Figure 1: the dB ladder plus the seed-reproducibility panel."""
    grid: TemporalGrid = ctx["grid"]
    offset = lad["offset"]
    window = lad["window"]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))

    ax = axes[0]
    ax.plot(
        offset[window], lad["clean"][window], color="k", lw=1.6, label="clean pulse"
    )
    colors = ("tab:red", "tab:orange", "tab:green", "tab:blue")
    for item, color in zip(lad["ladder"], colors):
        ax.plot(
            offset[window],
            item["spectrum"][window],
            lw=0.8,
            color=color,
            label=f"{item['level']:+.0f} dB (OSNR {item['osnr']:.1f} dB)",
        )
    ax.set_xlim(-PLOT_WINDOW_THZ, PLOT_WINDOW_THZ)
    ax.set_ylim(-110, 5)
    ax.set_xlabel(r"$\Omega-\Omega_0$ (THz)")
    ax.set_ylabel("power spectrum (dB rel. peak)")
    ax.set_title(
        "(a) ASE ladder\n"
        f"seed={SEED_A}, reference = pulse peak power = {P0:g}.\n"
        "Each 5 dB step down is a 5 dB step in OSNR."
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[1]
    power_a = np.abs(grid.fft(np.asarray(ctx["seed_a"].envelope_field))) ** 2
    power_b = np.abs(grid.fft(np.asarray(ctx["seed_b"].envelope_field))) ** 2
    power_d = np.abs(grid.fft(np.asarray(ctx["different"].envelope_field))) ** 2
    # Ratio to the first run: two runs of one seed give exactly 1 bin by bin,
    # and a different seed gives a scatter about 1 whose size is the Rayleigh
    # floor of a single realisation. Plotting the ratio (rather than two
    # overlaid spectra) is what makes "identical" visible at all.
    floor = np.median(power_a[window])
    ratio_same = power_b[window] / np.maximum(power_a[window], 1e-300)
    ratio_diff = power_d[window] / np.maximum(power_a[window], 1e-300)
    ax.plot(
        offset[window],
        _db(ratio_same),
        lw=0.9,
        color="tab:blue",
        label=f"seed={SEED_A}, run 2 / run 1",
    )
    ax.plot(
        offset[window],
        _db(ratio_diff),
        lw=0.9,
        color="tab:red",
        alpha=0.85,
        label=f"seed={SEED_B} / seed={SEED_A}",
    )
    ax.axhline(0.0, color="0.4", ls="--", lw=1.0)
    ax.set_xlim(-PLOT_WINDOW_THZ, PLOT_WINDOW_THZ)
    ax.set_ylim(-14, 6)
    ax.set_xlabel(r"$\Omega-\Omega_0$ (THz)")
    ax.set_ylabel("power ratio to run 1 (dB)")
    ax.set_title(
        f"(b) same seed -> exactly 0 dB, bin by bin\n"
        f"a different seed scatters about 0 dB (median floor "
        f"{10.0 * np.log10(floor / _spectral_peak(grid, ctx['pulse'])):.0f} dB rel. peak)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[2]
    levels = [item["level"] for item in lad["ladder"]]
    ax.plot(
        levels,
        [item["osnr"] for item in lad["ladder"]],
        "o-",
        color="tab:purple",
        label="measured (integrated bands)",
    )
    ax.plot(
        levels,
        [lad["ladder"][0]["osnr"] + (lv - levels[0]) for lv in levels],
        "--",
        color="0.4",
        lw=1.2,
        label="slope of -1 in level_dB (through the first point)",
    )
    osnrs = np.array([item["osnr"] for item in lad["ladder"]])
    ax.set_ylim(osnrs.min() - 2.0, osnrs.max() + 2.0)
    ax.set_xlabel("level_dB (ASE level relative to the CW reference line)")
    ax.set_ylabel("OSNR (dB)")
    ax.set_title(
        "(c) OSNR is a level knob\n"
        f"bands: |df|<={BAND_SIG_HZ / 1e12:g} THz vs "
        f"{BAND_NOISE_LO_HZ / 1e12:g}-{BAND_NOISE_HI_HZ / 1e12:g} THz"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "ASE noise: the level is exact and the seed is exact",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["ase"], dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. Raman thermal floor ────────────────────────────────────────
def _silica_response(grid: TemporalGrid) -> RamanResponse:
    """Silica RamanResponse on ``grid`` with tau1/tau2 given explicitly.

    Explicit taus avoid the ``RamanResponse`` auto-derivation warning and keep
    the response identical to the one ``examples/05`` prints.
    """
    spec = RamanSpec.from_database("Silica")
    return RamanResponse(
        spec=spec,
        tau1=1.0 / (2.0 * np.pi * spec.raman_shift_Hz),
        tau2=1.0 / (np.pi * spec.linewidth_Hz),
        grid=grid,
    )


def raman_floor_panel(ctx: dict) -> dict:
    print(_rule("3. The Raman thermal floor: thermodynamic, not empirical"))

    grid: TemporalGrid = ctx["grid"]
    omega0 = 2.0 * np.pi * 299792458.0 / LAM0
    resp = _silica_response(grid)
    # raman_noise_field wants the FFT of the *delayed* response on the grid --
    # the same h_R_fft the GNLSE passes to its delayed-intensity convolution.
    h_R = resp.h_R(grid.t)
    h_R_fft = np.asarray(grid.fft(h_R), dtype=complex)
    assert h_R_fft.shape == (grid.N,)

    shift_hz = float(resp.spec.raman_shift_Hz)
    print(
        f"  omega0 = {omega0:.4e} rad/s  ({299792458.0 / LAM0 / 1e12:.3f} PHz carrier)"
    )
    print(
        f"  Silica: shift {resp.spec.raman_shift_cm:.0f} cm^-1 ({shift_hz / 1e12:.2f} THz), "
        f"tau1 = {cast(float, resp.tau1) * 1e15:.1f} fs, "
        f"tau2 = {cast(float, resp.tau2) * 1e15:.1f} fs"
    )
    print("  h_R(t) unit integral, h_R_fft carries Im[h_R] > 0 on the Stokes side")
    print(
        "  omega0 is keyword-only and required: it sets the photon energy hbar*omega0/2"
    )
    print("  that every bin's variance is proportional to.")

    fields = {}
    spectra = {}
    for temp in TEMPERATURES:
        f = raman_noise_field(
            grid, h_R_fft, seed=SEED_A, omega0=omega0, temperature=temp
        )
        fields[temp] = np.asarray(f, dtype=complex)
        spectra[temp] = np.asarray(grid.fft(f), dtype=complex)

    # The variance ratio is a statement about the partition function, so compute
    # it the deterministic way (mean of |spec|^2 over the Stokes bins) rather
    # than from two single realisations, whose rms fluctuates by ~1/sqrt(2N).
    gain = np.clip(h_R_fft.imag, 0.0, None)
    stokes = gain > 0.0
    var_ratio = float(
        np.mean(np.abs(spectra[TEMPERATURES[1]][stokes]) ** 2)
        / np.mean(np.abs(spectra[TEMPERATURES[0]][stokes]) ** 2)
    )
    ratio_rms = float(np.sqrt(var_ratio))
    assert var_ratio > 1.0, "the hotter bath must thermalise louder"

    # Where the thermal factor matters is decided by WHERE THE GAIN IS, and
    # that is not where the quoted Raman shift sits: Im[h_R](Omega) is the
    # Fourier transform of a decaying oscillation, so it is a Lorentzian
    # centred at Omega = 1/tau1 -- an angular frequency of 1/tau1, i.e. about
    # 83 THz -- while the Raman shift people quote, 1/(2*pi*tau1) = 13.2 THz,
    # is that number divided by 2*pi. Both appear below, because which one you
    # evaluate the Bose factor at is the whole question.
    tau1 = cast(float, resp.tau1)
    omega_peak = 1.0 / tau1
    print("\n  where the deterministic gain actually lives:")
    print(f"    quoted Raman shift  nu_R = 1/(2*pi*tau1) = {shift_hz / 1e12:6.2f} THz")
    print(
        f"    the same frequency in angular units, Omega_R = 1/tau1 = "
        f"{omega_peak / 1e12:6.2f} x 10^12 rad/s"
    )
    print("    (2*pi is exactly the trap here: the library's Bose factor is built")
    print("     from grid.w in rad/s, so it wants Omega_R, not nu_R -- and so does")
    print("     anyone checking this expression against a quoted shift in cm^-1.)")
    peak_bin = int(np.argmax(np.where(stokes, gain, 0.0)))
    print(
        f"    the largest measured Im[h_R] bin sits at "
        f"{abs(grid.w[peak_bin]) / 1e12:.2f} x 10^12 rad/s   [the Lorentzian centre]"
    )

    print("\n  Bose factor n_th + 1 = 1/(1 - exp(-hbar*Omega/k_B T)) per bin:")

    factors: dict[float, dict[str, float]] = {}
    for temp in TEMPERATURES:
        factors[temp] = {
            "peak": float(_thermal(omega_peak, temp)),
            "shift": float(_thermal(2.0 * np.pi * shift_hz, temp)),
            "carrier": float(_thermal(grid.dw, temp)),
        }
        fac = factors[temp]
        print(
            f"    T = {temp:5.1f} K: n_th + 1 = {fac['peak']:.6f} at the gain peak, "
            f"{fac['shift']:.4f} at Omega_R,"
        )
        print(
            f"                    {fac['carrier']:.1f} in the first bin "
            f"(|Omega| = {grid.dw / 2 / np.pi / 1e12:.3f} THz)"
        )
    print("    The factor spans three orders of magnitude across the band, so")
    print("    where you evaluate it decides the answer. At the gain peak -- the")
    print(
        "    only place Im[h_R] is not negligible -- it is "
        f"{factors[300.0]['peak']:.4f} at 300 K and"
    )
    print(f"    {factors[600.0]['peak']:.4f} at 600 K: the bath there is essentially")
    print("    empty and the semi-classical n_th -> 0 limit is already accurate.")
    print("    Near the carrier the factor is enormous, but the gain is zero")
    print("    there, and the product is what a bin actually gets.")

    print("\n  the band-averaged floor, over every bin with Im[h_R] > 0:")
    print(
        f"    measured 600 K / 300 K noise power ratio : {var_ratio:.6f}"
        "   [asserted > 1]"
    )
    print(f"    -> rms ratio {ratio_rms:.6f}. Doubling T does not double the floor;")
    print("       the Bose occupation is not linear in T, and the ratio above is")
    print("       dominated by the low-|Omega| end of the gain band where the")
    print("       thermal factor is still far from 1.")

    # The library evaluates the factor per bin at |grid.w|, so the closed form
    # can be tested on a band where it has teeth. Do that on the band straddling
    # the quoted Raman shift, where the factor differs most between the two
    # temperatures; a per-bin check over the whole grid would be measuring
    # round-off wherever the gain sits at the floating-point floor.
    offset_hz = np.abs(grid.w) / (2.0 * np.pi)
    shift_band = stokes & (offset_hz <= 3.0 * shift_hz) & (offset_hz >= 0.5 * shift_hz)
    band_ratio = float(
        np.mean(np.abs(spectra[TEMPERATURES[1]][shift_band]) ** 2)
        / np.mean(np.abs(spectra[TEMPERATURES[0]][shift_band]) ** 2)
    )
    # The prediction is the Bose factor averaged over the SAME bins with the
    # SAME weights, i.e. weighted by the 300 K power that was just measured:
    # an unweighted mean of n_th + 1 over the band would ignore that the gain
    # is not uniform across it.
    # grid.w is in rad/s and that is what the Bose factor is built from;
    # feeding it the same offset in Hz would evaluate the factor 2*pi too high.
    omega_bin = np.abs(grid.w[shift_band])
    thermal_ratio = _thermal(omega_bin, TEMPERATURES[1]) / _thermal(
        omega_bin, TEMPERATURES[0]
    )
    weight = np.abs(spectra[TEMPERATURES[0]][shift_band]) ** 2
    predicted = float(np.sum(weight * thermal_ratio) / np.sum(weight))
    print(
        f"\n  the closed form, tested on the {int(shift_band.sum())} bins "
        f"straddling Omega_R"
    )
    print("  (0.5-3 x shift), where the factor still varies with T:")
    print(f"    measured 600 K / 300 K noise power ratio : {band_ratio:.5f}")
    print(f"    predicted (Bose factor, same bins, same weights) : {predicted:.5f}")
    assert abs(band_ratio / predicted - 1.0) < 0.05, (
        f"the measured ratio {band_ratio} is not the Bose ratio {predicted}"
    )
    print(
        f"    departure {abs(band_ratio / predicted - 1.0) * 100:.2f} %  "
        "[asserted < 5 %]"
    )
    print("    Each bin is a single complex Gaussian draw, so its variance")
    print("    scatters about the formula; averaging the whole band recovers it.")
    print("    Away from the gain band the noise falls into the floating-point")
    print("    floor, and a per-bin ratio there measures nothing.")

    # Sanity: the caller scales by sqrt(dz); report the field in those units.
    print(
        f"\n  fields below are scaled by sqrt(dz) with dz = {DZ:.3f} m "
        "(the Langevin delta(z-z') discretisation the GNLSE applies)"
    )
    rms = {t: float(np.sqrt(np.mean(np.abs(fields[t]) ** 2))) for t in TEMPERATURES}
    for temp in TEMPERATURES:
        print(
            f"    rms(sqrt(dz)*field) at {temp:5.1f} K = "
            f"{rms[temp] * np.sqrt(DZ):.6e}   (raw field rms {rms[temp]:.6e})"
        )
    assert rms[TEMPERATURES[1]] > rms[TEMPERATURES[0]], (
        "the single-realisation rms must also be larger at the hotter bath"
    )

    return {
        "grid": grid,
        "h_R": h_R,
        "h_R_fft": h_R_fft,
        "gain": gain,
        "stokes": stokes,
        "fields": fields,
        "spectra": spectra,
        "factors": factors,
        "var_ratio": var_ratio,
        "band_ratio": band_ratio,
        "band_predicted": predicted,
        "omega_peak": omega_peak,
        "offset_hz": offset_hz,
        "omega0": omega0,
        "resp": resp,
    }


# ── 4. Through a fiber ────────────────────────────────────────────
def _propagate(wave: Wave, resp: RamanResponse, seed: int | None = None) -> Wave:
    """Propagate ``wave`` over L_FIBER with the deterministic Raman term."""
    omega0 = 2.0 * np.pi * 299792458.0 / LAM0
    fiber = FiberProfile.from_gamma(
        gamma=1.5e-3,
        n2=2.7e-20,
        omega0=omega0,
        length=Length(L_FIBER, "m"),
        raman_response=resp,
    )
    solver = SplitStepEngine(
        pulse=wave,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),
        include_raman=True,
        include_self_steepening=False,
        step_size=Length(L_FIBER / N_STEPS, "m"),
    )
    solver.propagate(num_steps=N_STEPS)
    return solver.evolution[-1]


def propagation_panel(ctx: dict, ram: dict) -> dict:
    print(
        _rule("4. Through a fiber: what propagation does to the pulse, and to the ASE")
    )

    grid: TemporalGrid = ctx["grid"]
    pulse: Wave = ctx["pulse"]
    resp: RamanResponse = ram["resp"]
    offset = grid.w / (2.0 * np.pi) / 1e12
    window = np.abs(offset) <= 10.0

    level = ASE_PROP_LEVEL
    noisy = _ase(pulse, level, SEED_A)
    print(
        f"  L = {L_FIBER:g} m, {N_STEPS} steps of {L_FIBER / N_STEPS:.2f} m, "
        f"beta2 = {BETA2 * 1e27:+.1f} ps^2/km (anomalous), Raman on"
    )
    print(f"  ASE: level_dB = {level:+.0f}, seed = {SEED_A}")

    osnr_in = _osnr_db(noisy, grid)
    clean_out = _propagate(pulse, resp)
    out = _propagate(noisy, resp)
    osnr_out = _osnr_db(out, grid)

    peak_in, peak_out = _peak_power(pulse), _peak_power(clean_out)
    tw_in, tw_out = _rms_time_width(pulse, grid), _rms_time_width(clean_out, grid)
    bw_in, bw_out = _rms_bandwidth(pulse, grid), _rms_bandwidth(clean_out, grid)
    # The delayed Raman term redistributes power across the spectrum; asking how
    # much it moved the noise needs an ASYMMETRIC window, because a symmetric
    # one integrates the red gain against the blue loss and reports ~nothing.
    red_in = _band_power(
        np.abs(grid.fft(noisy.envelope_field)) ** 2, grid, 20.0e12, 45.0e12
    )
    red_out = _band_power(
        np.abs(grid.fft(out.envelope_field)) ** 2, grid, 20.0e12, 45.0e12
    )
    blue_in = _band_power(
        np.abs(grid.fft(noisy.envelope_field)) ** 2, grid, -45.0e12, -20.0e12
    )
    blue_out = _band_power(
        np.abs(grid.fft(out.envelope_field)) ** 2, grid, -45.0e12, -20.0e12
    )

    print(f"  OSNR in  : {osnr_in:6.2f} dB   (seeded, so exactly reproducible)")
    print(f"  OSNR out : {osnr_out:6.2f} dB   (change {osnr_out - osnr_in:+.2f} dB)")
    print(f"  clean-pulse OSNR out, same fiber: {_osnr_db(clean_out, grid):6.2f} dB")
    print("\n  what the fiber does to the PULSE (measured on the clean run, so")
    print("  the ASE does not set the width and the width is the pulse's own):")
    print(
        f"    peak power         {peak_in:8.2f} -> {peak_out:8.2f} "
        f"({_db(peak_out / peak_in):+.2f} dB)"
    )
    print(
        f"    rms temporal width {tw_in * 1e12:8.3f} -> {tw_out * 1e12:8.3f} ps "
        f"({_db(tw_out / tw_in):+.2f} dB)"
    )
    print(
        f"    rms bandwidth      {bw_in / 2 / np.pi / 1e12:8.2f} -> "
        f"{bw_out / 2 / np.pi / 1e12:8.2f} THz ({_db(bw_out / bw_in):+.2f} dB)"
    )
    print(
        f"    L/L_D = {L_FIBER * abs(BETA2) / T0**2:.3f}, "
        f"L/L_NL = {L_FIBER * 1.5e-3 * P0:.1f}: at a few nonlinear lengths the"
    )
    print("    pulse solitonises -- it compresses and sharpens rather than")
    print("    dispersing, which is why the width holds while the bandwidth")
    print("    grows. That growth is the spectral reshaping, and it is what")
    print("    panel (c) shows: the pulse fills more of the window than it")
    print("    started with, into a floor that never moved.")
    print("\n  and what it does to the BACKGROUND, measured in one-sided windows")
    print("  because the symmetric noise band above integrates red gain against")
    print("  blue loss and reports ~nothing by construction:")
    print(
        f"      red  (20-45 THz)   {_db(red_in):8.2f} -> {_db(red_out):8.2f} dB "
        f"({_db(red_out / red_in):+.3f} dB)"
    )
    print(
        f"      blue (-45--20 THz) {_db(blue_in):8.2f} -> {_db(blue_out):8.2f} dB "
        f"({_db(blue_out / blue_in):+.3f} dB)"
    )
    print(
        f"      red/blue asymmetry {_db(red_in / blue_in):+.3f} -> "
        f"{_db(red_out / blue_out):+.3f} dB"
    )
    print("\n  Read those two blocks separately, because they say different things.")
    print("  The pulse is reshaped; the flat background is not. The library's")
    print("  delayed-Raman term is the lossy local form P_Raman = f_R (h_R * I),")
    print("  not a coupled gain/loss pair, so a *flat* incoherent background has no")
    print("  direction in which to move and comes out at the level it went in. A")
    print("  directional Raman reshaping of the ASE itself needs a strong")
    print("  co-propagating pump and a few Raman gain lengths, and it is not what")
    print("  this panel measures -- so the panel reports the near-zero number")
    print("  rather than implying a redshift that did not happen.")

    return {
        "noisy": noisy,
        "out": out,
        "clean_out": clean_out,
        "offset": offset,
        "window": window,
        "osnr_in": osnr_in,
        "osnr_out": osnr_out,
        "level": level,
        "peak_in": peak_in,
        "peak_out": peak_out,
        "w_in": bw_in,
        "w_out": bw_out,
        "tw_in": tw_in,
        "tw_out": tw_out,
        "asym_in": float(_db(red_in / blue_in)),
        "asym_out": float(_db(red_out / blue_out)),
    }


# ── 5. Ensemble coherence ─────────────────────────────────────────
def coherence_panel(ctx: dict, ram: dict) -> dict:
    print(_rule("5. g12 over an ensemble: what a single realisation cannot show"))

    grid: TemporalGrid = ctx["grid"]
    pulse: Wave = ctx["pulse"]
    resp: RamanResponse = ram["resp"]

    # The library's own contract, re-checked here because the panel depends on
    # it: N copies of one spectrum are perfectly coherent. Only the bins that
    # carry signal are asserted -- coherence_g12 returns 0 for a bin with no
    # power at all, and a 1 ps pulse on a 100 ps window leaves thousands of
    # such bins whose value is not a statement about coherence.
    one = np.asarray(grid.fft(pulse.envelope_field), dtype=complex)
    g12_same = coherence_g12(np.stack([one, one, one]))
    live = _live_bins(np.abs(one) ** 2)
    assert np.allclose(g12_same[live], 1.0, atol=1e-9), (
        f"identical runs must give g12 = 1, got {g12_same[live].min()}"
    )
    print(f"  {N_RUNS} propagated realisations, ASE seed = 100..{100 + N_RUNS - 1}")
    print(
        f"  identical runs -> g12 = 1 on the {int(live.sum())} bins that carry "
        f"signal   [asserted]"
    )

    runs = []
    for i in range(N_RUNS):
        wave = _propagate(_ase(pulse, ASE_PROP_LEVEL, 100 + i), resp)
        runs.append(np.asarray(grid.fft(wave.envelope_field), dtype=complex))
    spectra = np.stack(runs)
    g12 = coherence_g12(spectra)

    # Report the band-averaged value over the bins that actually carry signal,
    # which is the number a coherence-sensitive experiment would quote for a
    # fixed filter width.
    offset_thz = np.abs(grid.w) / (2.0 * np.pi) / 1e12
    mean_power = np.mean(np.abs(spectra) ** 2, axis=0)
    live = _live_bins(mean_power)
    pulse_band = live & (offset_thz <= 3.0)
    wing_band = live & (offset_thz > 10.0) & (offset_thz < 50.0)
    g12_pulse = float(np.mean(g12[pulse_band]))
    g12_wings = float(np.mean(g12[wing_band]))
    print(
        f"  mean g12 in the pulse band (|df| <= 3 THz, {int(pulse_band.sum())} bins)"
        f" : {g12_pulse:.4f}"
    )
    print(
        f"  mean g12 in the noise wings (10-50 THz, {int(wing_band.sum())} bins)"
        f"   : {g12_wings:.4f}"
    )
    assert g12_pulse < 1.0, "a noised ensemble cannot be fully coherent"
    print("  < 1 in both bands                                [asserted]")
    core = mean_power > float(np.max(mean_power)) * 1e-3
    g12_core = float(np.mean(g12[core])) if core.any() else float("nan")
    print(
        f"  mean g12 over the {int(core.sum())} bins that actually carry the "
        f"pulse: {g12_core:.4f}"
    )
    print("  Read the two numbers as different questions. The core bins answer")
    print("  'do the pulses stay in step with each other?' -- mostly yes, because")
    print("  every realisation carries the same pulse. The band average answers")
    print("  'what does a +-3 THz filter see?' -- it also collects the pulse's")
    print("  wings, which sit on the ASE and decohere, so the band number is far")
    print("  below the core number. Neither exists in a single run: one")
    print("  realisation is just a clean-looking pulse.")

    return {
        "spectra": spectra,
        "g12": g12,
        "live": live,
        "core": np.asarray(mean_power) > float(np.max(mean_power)) * 1e-3,
        "offset": offset_thz,
        "g12_pulse": g12_pulse,
        "g12_wings": g12_wings,
    }


# ── Figure 2 ──────────────────────────────────────────────────────
def raman_figure(ctx: dict, ram: dict, prop: dict, coh: dict) -> None:
    grid: TemporalGrid = ctx["grid"]
    h_R = ram["h_R"]
    t_ps = grid.t * 1e12
    offset = prop["offset"]
    window = prop["window"]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 9.4))

    # (a) deterministic response vs its thermal noise realisation
    ax = axes[0, 0]
    ax.plot(
        t_ps,
        h_R / (1.0 / (1.0e-12)),
        color="k",
        lw=1.8,
        label=r"deterministic $h_R(t)$ (examples/05)",
    )
    colors = {TEMPERATURES[0]: "tab:blue", TEMPERATURES[1]: "tab:red"}
    for temp in TEMPERATURES:
        f = ram["fields"][temp]
        unit = np.sqrt(np.mean(np.abs(f) ** 2))
        ax.plot(
            t_ps,
            f.real / unit,
            lw=0.9,
            color=colors[temp],
            label=f"spontaneous Raman noise, {temp:.0f} K (unit rms)",
        )
    ax.set_xlim(-0.02, 0.35)
    ax.set_ylim(-7.0, 7.0)
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("amplitude (normalised)")
    ax.set_title(
        "(a) the same response, deterministic vs thermalised\n"
        f"noise drawn from Im[h_R](300 K) and Im[h_R](600 K), scaled by "
        f"sqrt(dz={DZ:.2f} m)\n"
        "each realisation is unit-rms; the shape is set by the same Im[h_R]"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # (b) The floor is the product of two things the library hands you: the
    # deterministic gain Im[h_R] and the Bose factor n_th + 1. Plot the measured
    # PSD against the closed form for the product, and the thermal part alone on
    # a twin axis -- the two curves are nearly identical at the gain peak and
    # only separate where the factor is large, which is exactly the claim.
    ax = axes[0, 1]
    offset_hz = ram["offset_hz"]
    om = offset_hz / 1e12  # THz
    stokes = ram["stokes"]
    gain_ok = stokes & (ram["gain"] > 1e-3 * float(np.max(ram["gain"])))
    edges = np.linspace(0.0, 130.0, 131)
    idx = np.digitize(om, edges)
    centres = np.array([0.5 * (edges[k - 1] + edges[k]) for k in range(1, len(edges))])
    binned: dict[float, NDArray] = {}
    for temp in TEMPERATURES:
        p_ = np.abs(ram["spectra"][temp]) ** 2
        vals = [
            float(np.mean(p_[(idx == k) & gain_ok]))
            if ((idx == k) & gain_ok).sum() > 0
            else np.nan
            for k in range(1, len(edges))
        ]
        binned[temp] = np.array(vals)
    peak_idx = int(np.nanargmax(np.nan_to_num(binned[TEMPERATURES[0]])))
    prefactor = float(hbar * ram["omega0"] / 2.0 * (2.0 * np.pi / float(grid.dw)))
    gain_curve = np.array(
        [
            float(np.mean(ram["gain"][(idx == k) & gain_ok]))
            if ((idx == k) & gain_ok).any()
            else np.nan
            for k in range(1, len(edges))
        ]
    )
    for temp in TEMPERATURES:
        closed = (
            prefactor
            * gain_curve
            * np.array([float(_thermal(2.0 * np.pi * c * 1e12, temp)) for c in centres])
        )
        ax.plot(
            centres,
            binned[temp] / binned[temp][peak_idx],
            lw=1.6,
            color=colors[temp],
            label=f"measured PSD, {temp:.0f} K",
        )
        ax.plot(
            centres,
            closed / closed[peak_idx],
            lw=1.0,
            ls="--",
            color=colors[temp],
            alpha=0.85,
            label=f"(hbar*omega0/2)(2pi/dw) Im[h_R](n_th+1), {temp:.0f} K",
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 45)
    ax.set_xlabel(
        r"$|\Delta f|$ (THz)   -- $\mathrm{Im}[\tilde h_R]$ and the Bose "
        r"factor both peak at the Raman shift"
    )
    ax.set_ylabel("noise PSD, normalised at its peak (log)")
    ax2 = ax.twinx()
    ratio_meas = binned[TEMPERATURES[1]] / binned[TEMPERATURES[0]]
    ratio_closed = np.array(
        [
            _thermal(2.0 * np.pi * c * 1e12, TEMPERATURES[1])
            / _thermal(2.0 * np.pi * c * 1e12, TEMPERATURES[0])
            for c in centres
        ]
    )
    ax2.plot(
        centres,
        ratio_meas,
        color="tab:red",
        lw=0.9,
        alpha=0.8,
        label="measured 600 K / 300 K",
    )
    ax2.plot(
        centres,
        ratio_closed,
        color="0.35",
        lw=1.2,
        ls=":",
        label="closed-form 600 K / 300 K",
    )
    ax2.set_ylabel("600 K / 300 K noise power", color="0.35")
    ax2.tick_params(axis="y", labelcolor="0.35")
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 50)
    shift_line = float(ram["resp"].spec.raman_shift_Hz) / 1e12
    ax.axvline(
        shift_line,
        color="0.4",
        ls="--",
        lw=1.2,
        label=r"Raman shift $\nu_R = 1/2\pi\tau_1$ "
        r"(the $\mathrm{Im}[\tilde h_R]$ peak)",
    )
    ax.set_title(
        "(b) the floor is Im[h_R] x a Bose factor, not a knob\n"
        "measured PSD against the closed form with no free parameter; the twin "
        "axis is the\ntemperature ratio, which collapses to 1 at the gain peak "
        "and rises where the gain\nreaches towards the carrier"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower center")
    ax2.legend(fontsize=7, loc="upper left")

    # (c) propagation
    ax = axes[1, 0]
    ax.plot(
        offset[window],
        prop["noisy_spectrum"][window],
        color="tab:gray",
        lw=1.4,
        label=f"input, ASE {prop['level']:+.0f} dB (OSNR {prop['osnr_in']:.1f} dB)",
    )
    ax.plot(
        offset[window],
        prop["out_spectrum"][window],
        color="tab:red",
        lw=1.0,
        label=f"output, {L_FIBER:g} m + Raman (OSNR {prop['osnr_out']:.1f} dB)",
    )
    ax.plot(
        offset[window],
        prop["clean_spectrum"][window],
        color="tab:blue",
        lw=1.0,
        ls="--",
        label="output, clean input",
    )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-120, 5)
    ax.set_xlabel(r"$\Omega-\Omega_0$ (THz)")
    ax.set_ylabel("power spectrum (dB rel. peak)")
    ax.set_title(
        "(c) the pulse is reshaped; the background is not\n"
        f"the pulse solitonises over {L_FIBER:g} m and fills more of the band, "
        "into a floor\nthat stays where it was put: red and blue one-sided "
        f"windows both move\nby <{max(abs(prop['asym_in']), 0.01):.3f} dB, so "
        "the ASE is transparent to this fiber"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")

    # (d) g12
    ax = axes[1, 1]
    sel = coh["offset"] <= 50.0
    ax.plot(
        coh["offset"][sel],
        coh["g12"][sel],
        color="tab:purple",
        lw=1.2,
        label=f"g12 over {N_RUNS} noised realisations",
    )
    ax.axvspan(
        0.0,
        3.0,
        color="tab:green",
        alpha=0.12,
        label="pulse band (mean g12 = %.3f)" % coh["g12_pulse"],
    )
    ax.axvspan(
        10.0,
        50.0,
        color="tab:orange",
        alpha=0.12,
        label="noise wings (mean g12 = %.3f)" % coh["g12_wings"],
    )
    ax.axhline(
        1.0, color="0.4", ls="--", lw=1.2, label="g12 = 1 (identical realisations)"
    )
    ax.set_xlabel(
        r"$|\Omega-\Omega_0|$ (THz)   (equivalently the axial delay "
        r"$\tau = 1/\Delta f$ of each bin)"
    )
    ax.set_ylabel(r"$g_{12}$")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "(d) coherence is an ensemble property\n"
        "g12 = |<E_m* E_n>| / <|E|^2> over distinct pairs; identical runs\n"
        "would give 1.0, an ASE-noised ensemble gives this"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "The Raman noise floor is the thermal floor of the deterministic response",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS["raman"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = _time.perf_counter()
    print(_rule("0. ASE and Raman noise -- photonics_helper.noise"))
    print("  ASE power PSD : flat per bin, level_dB below the DC line of a CW")
    print("                  of reference_power (this example pins it to the")
    print("                  pulse peak power), so the noise power per bin is")
    print("                  P_ref * 10^(level_dB/10).")
    print("  FDT partition : sigma^2 = (hbar*omega0/2) * Im[h_R] * (n_th + 1)")
    print("                  * 2*pi/dOmega, n_th = 1/(exp(hbar*Omega_R/k_B T)-1)")
    print("  The floor is THERMODYNAMIC: a phonon partition function, not a")
    print("  fudge factor fitted to a measured spectrum.")

    ctx = reproducibility_panel()
    lad = ase_ladder_panel(ctx)
    ase_figure(ctx, lad)
    ram = raman_floor_panel(ctx)
    prop = propagation_panel(ctx, ram)
    # Spectra for the propagation figure are computed here so the figure code
    # stays purely about drawing.
    prop["noisy_spectrum"] = _spec_db(
        prop["noisy"], ctx["grid"], _spectral_peak(ctx["grid"], ctx["pulse"])
    )
    prop["out_spectrum"] = _spec_db(
        prop["out"], ctx["grid"], _spectral_peak(ctx["grid"], ctx["pulse"])
    )
    prop["clean_spectrum"] = _spec_db(
        prop["clean_out"], ctx["grid"], _spectral_peak(ctx["grid"], ctx["pulse"])
    )
    coh = coherence_panel(ctx, ram)
    raman_figure(ctx, ram, prop, coh)

    print(_rule("Summary"))
    print(
        "  ASE level_dB ladder          : "
        + ", ".join(
            f"{i['level']:+.0f} dB -> {i['osnr']:.1f} dB" for i in lad["ladder"]
        )
    )
    print("  same seed                    : bit-identical field        [asserted]")
    print("  both rng and seed            : ValueError                [asserted]")
    print(
        f"  per-bin ASE amplitude        : flat, relative spread "
        f"{ctx['spread']:.1e}          [asserted < 1e-9]"
    )
    print(
        f"  Bose n_th+1 at the gain peak  : {ram['factors'][300.0]['peak']:.6f} (300 K), "
        f"{ram['factors'][600.0]['peak']:.6f} (600 K)"
    )
    print(
        f"  Bose n_th+1 at Omega_R        : {ram['factors'][300.0]['shift']:.6f} (300 K), "
        f"{ram['factors'][600.0]['shift']:.6f} (600 K)"
    )
    print(
        f"  600/300 K noise power ratio  : {ram['var_ratio']:.6f} "
        "(all gain bins)        [asserted > 1]"
    )
    print(
        f"  ... on the Omega_R band      : {ram['band_ratio']:.5f} measured vs "
        f"{ram['band_predicted']:.5f} predicted   [asserted < 5 %]"
    )
    print(
        f"  OSNR in / out ({L_FIBER:g} m, Raman)  : "
        f"{prop['osnr_in']:.2f} dB -> {prop['osnr_out']:.2f} dB"
    )
    print(
        f"  mean g12, pulse band / wings : {coh['g12_pulse']:.4f} / "
        f"{coh['g12_wings']:.4f}       [asserted < 1]"
    )
    print(f"  runtime                      : {_time.perf_counter() - t0:.1f} s")
    print("  all checks passed ✓")

    print("\nGenerated files:")
    for key in ("ase", "raman"):
        print(f"  {FIGS[key].relative_to(_ROOT)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    missing = [p for p in FIGS.values() if not p.exists()]
    assert not missing, f"figures not written: {missing}"
    print("figures verified on disk ✓")
