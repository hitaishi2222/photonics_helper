"""Example: RIN transfer in a dual-order CW Raman cascade
==========================================================

Mermelstein, Brar and Headley 2003, *J. Lightwave Technol.* **21**(6), 1518,
doi 10.1109/JLT.2003.812461, report the relative-intensity-noise transfer from
two pump lasers to the signal of a dual-order Raman fiber amplifier, for both
propagation geometries:

==================  =============  =====================  =====================
configuration       band           DC (2nd / 1st order)    6 dB corner (2nd / 1st)
==================  =============  =====================  =====================
counter-propagating  30 Hz–10 kHz  15.6 / 0.04 dB         1.33 / 1.59 kHz
co-propagating      100 kHz–40 MHz  15.4 / 0.7 dB         11.2 / 18.5 MHz
==================  =============  =====================  =====================

This example builds the Mermelstein Table I configuration from the benchmark
fixture, solves the steady-state cascade, linearizes it into six complex
modulation indices, and plots the model response beside the published targets.

What reproduces, and what does not
----------------------------------
The **counter-propagating 6 dB corners reproduce**: 1.30 and 1.31 kHz against
1.33 and 1.59 kHz, within 2.5 and 18 percent. Since the corner is set by the
ratio of the net gain coefficient to the walk-off, ``d = 1/v_s + 1/v_p``, this
is a real check on the walk-off sign, on the group indices, and on the
gain-per-unit-length normalization all at once.

The **DC levels do not reproduce**. This example prints 17.8 dB and 18.6 dB for
the counter-propagating case against 15.6 dB and 0.04 dB published: the
second-order value lands within about 2 dB, but the two pumps come out nearly
equal where the paper reports the first-order 15 dB below the second-order. The
**co-propagating corners** come out at 7.8 and 8.2 MHz against 11.2 and 18.5
MHz. Both misses are recorded in ``docs/raman-cascade.md`` and guarded by
``xfail`` tests that name the discrepancy rather than hiding it.

Units
-----
Every public function in :mod:`photonics_helper.raman_transfer` takes documented
SI units and converts nothing implicitly. The one convention worth repeating:
the Raman gains on ``RamanChannel`` are in ``(W km)^-1``, the literature
convention, while the fiber losses are in the usual ``dB/km`` and are converted
to inverse metres internally.

Run with ``python examples/44_cw_cascade_rin_transfer.py``. Pass
``--no-plot`` to skip the figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from matplotlib.figure import Figure

_ROOT = Path(__file__).resolve().parents[1]
FIGS = {
    "counter": _ROOT / "docs" / "img" / "raman_transfer_counter.png",
    "co": _ROOT / "docs" / "img" / "raman_transfer_co.png",
}

COUNTER_BAND = (30.0, 10e3)
CO_BAND = (100e3, 40e6)


def log_band(band: tuple[float, float], n: int = 120) -> np.ndarray:
    """Logarithmically spaced frequencies spanning ``band``."""
    return np.logspace(np.log10(band[0]), np.log10(band[1]), n)


def mermelstein_cascade(geometry: int):
    """Build and solve the Mermelstein 2003 Table I cascade from the fixture."""
    from photonics_helper.base import Length, Wavelength
    from photonics_helper.raman_transfer import CWWCascade, RamanChannel, build_cascade, load_benchmark

    fx = load_benchmark("mermelstein2003", root=_ROOT)
    w = fx.data["wavelengths_nm"]
    v = lambda key: fx.value("parameters", key)  # noqa: E731

    channels = build_cascade(
        [
            RamanChannel(
                1,
                "second_order_pump",
                Wavelength(w["second_order_pump"], "nm"),
                v("second_order_pump_power"),
                v("loss_1375nm"),
                (0.0, v("gain_1375_to_1465"), v("gain_1375_to_1560")),
            ),
            RamanChannel(
                2,
                "first_order_pump",
                Wavelength(w["first_order_pump"], "nm"),
                v("first_order_pump_power"),
                v("loss_1465nm"),
                (v("gain_1375_to_1465"), 0.0, v("gain_1465_to_1560")),
            ),
            RamanChannel(
                3,
                "signal",
                Wavelength(w["signal"], "nm"),
                v("signal_power"),
                v("loss_1560nm"),
                (v("gain_1375_to_1560"), v("gain_1465_to_1560"), 0.0),
            ),
        ],
        group_index_zero=v("group_index_at_lambda0"),
        slope_ps_per_km_nm2=v("dispersion_slope_at_lambda0"),
        lambda0_nm=v("zero_dispersion_wavelength"),
    )
    length = Length(fx.value("fiber", "span_length"), "km")
    cascade = CWWCascade(channels, length, geometry=geometry)
    cascade.solve()
    return cascade, fx


def transfer_panel(ax, cascade, fx, geometry_key: str, band, title: str) -> None:
    """Plot both pump responses with the published targets overlaid."""
    from photonics_helper.raman_transfer import double_pole_fit

    targets = fx.data["targets"][geometry_key]
    f = log_band(band)
    for order, source in (("second_order", 1), ("first_order", 2)):
        response = cascade.noise_response(f, source=source)
        ax.semilogx(f, response.db, label=f"{order.replace('_', ' ')} pump, model")
        dc, corner = double_pole_fit(f, response.db)
        expected_dc = targets["dc_db"][order]["value"]
        expected_corner = targets["corner_6db_hz"][order]["value"]
        ax.semilogx(
            [band[0], expected_corner],
            [expected_dc - 6.0, expected_dc],
            linestyle="--",
            linewidth=1.0,
            color=f"C{source - 1}",
            label=f"{order.replace('_', ' ')} pump, Fig. target",
        )
        ax.plot([], [], " ", label=f"  DC {dc:.2f} dB (target {expected_dc:g}), "
                                f"corner {corner:.3g} Hz (target {expected_corner:.3g})")

    ax.set_title(title)
    ax.set_xlabel("offset frequency (Hz)")
    ax.set_ylabel("RIN transfer (dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def make_figure(cascade, fx, geometry_key: str, band, title: str) -> Figure:
    """One figure with the steady state on top and both transfer functions below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), constrained_layout=True)
    result = cascade.result
    z_km = result.z_km
    for column, label in enumerate(result.labels):
        axes[0].semilogy(z_km, result.powers_w[:, column], label=label)
    axes[0].set_xlabel("distance (km)")
    axes[0].set_ylabel("power (W)")
    axes[0].set_title("Mermelstein 2003 Table I steady state, 60 km")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    transfer_panel(axes[1], cascade, fx, geometry_key, band, title)
    return fig


def main() -> None:
    from photonics_helper.raman_transfer import Geometry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    print("Mermelstein, Brar, Headley 2003, doi 10.1109/JLT.2003.812461")
    print("RIN transfer in a dual-order CW Raman cascade\n")

    for geometry, geometry_key, band, key in (
        (Geometry.COUNTER_PROPAGATING, "counter_propagating", COUNTER_BAND, "counter"),
        (Geometry.CO_PROPAGATING, "co_propagating", CO_BAND, "co"),
    ):
        cascade, fx = mermelstein_cascade(geometry)
        targets = fx.data["targets"][geometry_key]
        result = cascade.result
        on_off = result.on_off_gain_db(cascade.loss_per_m)
        print(f"--- {geometry_key.replace('_', '-')} ---")
        print(f"  on-off gain, signal          : {on_off[-1]:.2f} dB "
              f"(Mermelstein reports about 13 dB)")
        print(f"  power balance drift          : {result.power_balance_drift:.2e}")
        for order, source in (("second_order", 1), ("first_order", 2)):
            response = cascade.noise_response(log_band(band), source=source)
            dc, corner = response.double_pole()
            print(
                f"  {order.replace('_', ' '):>13} pump     : "
                f"DC {dc:6.2f} dB (target {targets['dc_db'][order]['value']:5.2f}), "
                f"corner {corner:9.4g} Hz (target {targets['corner_6db_hz'][order]['value']:.3g}), "
                f"source {targets['source']}"
            )
        print()

    print("Reproduced: the counter-propagating 6 dB corners, within 2.5 and 18 percent.")
    print("Not reproduced: the four DC levels, the two co-propagating corners, and the")
    print("Fig. 7 interaction lengths. See docs/raman-cascade.md for the analysis.\n")

    if args.no_plot:
        return

    for geometry, geometry_key, band, key in (
        (Geometry.COUNTER_PROPAGATING, "counter_propagating", COUNTER_BAND, "counter"),
        (Geometry.CO_PROPAGATING, "co_propagating", CO_BAND, "co"),
    ):
        cascade, fx = mermelstein_cascade(geometry)
        fig = make_figure(
            cascade,
            fx,
            geometry_key,
            band,
            f"{geometry_key.replace('_', '-')}: model against Fig. "
            f"{'5' if geometry < 0 else '6'} target",
        )
        path = FIGS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        import matplotlib.pyplot as plt

        plt.close(fig)
        print(f"  wrote {path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()