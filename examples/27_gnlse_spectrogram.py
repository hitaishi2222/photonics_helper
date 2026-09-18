"""Example: GNLSE spectrogram (cross-correlation FROG)
=====================================================

Demonstrates :func:`photonics_helper.gnlse.plot_spectrogram`, which computes
the cross-correlation spectrogram of a propagated supercontinuum field,

    Sigma(Omega, tau) = |int E(t) g(t - tau) exp(-i Omega t) dt|^2 ,

using the input pulse as the gate ``g`` (Dudley, Genty & Coen, *Rev. Mod.
Phys.* **78**, 1135 (2006), Eq. 4 and Fig. 10).  Unlike a conventional STFT
(:meth:`photonics_helper.pulse.Envelope.visualize_3d`), this trace is phase
sensitive, uses the real pulse as the gate, and is returned on a wavelength
axis.

The example propagates a 50 fs, 10 kW sech pulse in 8 cm of the Dudley PCF
(835 nm, anomalous GVD) so that soliton fission and a blue dispersive wave
develop, then plots the spectrogram with its temporal and spectral
projections.

Usage::

    python examples/27_gnlse_spectrogram.py
    python examples/27_gnlse_spectrogram.py --num-steps 4000 --output fig.png
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver, plot_spectrogram
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

# ── Dudley et al. (2006) Table I PCF at 835 nm ──────────────────────────────
CENTRAL_WL_NM = 835.0
GAMMA = 0.11  # 1/(W m)
N2 = 2.7e-20  # m^2/W
BETAS = np.array(
    [
        -11.830e-3,
        8.1038e-5,
        -9.5205e-8,
        2.0737e-10,
        -5.3943e-13,
        1.3486e-15,
        -2.5495e-18,
        3.0524e-21,
        -1.7140e-24,
    ]
)


def make_pulse(N: int = 4096, Tmax_ps: float = 10.0) -> Wave:
    grid = TemporalGrid(N=N, Tmax=Time(Tmax_ps * 1e-12, "s"))
    envelope = Envelope.from_fwhm(
        "sech",
        peak_amplitude=np.sqrt(10_000.0),
        fwhm=Time(50.0, "fs"),
    )
    return Wave(
        grid=grid,
        envelope=envelope,
        central_wavelength=Wavelength(CENTRAL_WL_NM, "nm"),
    )


def make_fiber(pulse: Wave, length_m: float = 0.08) -> FiberProfile:
    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    raman = RamanResponse(
        spec=spec, fR=0.18, tau1=0.0122e-12, tau2=0.032e-12, grid=pulse.grid
    )
    return FiberProfile.from_gamma(
        gamma=GAMMA,
        n2=N2,
        omega0=float(pulse.central_frequency),
        length=Length(length_m, "m"),
        raman_response=raman,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=2500)
    parser.add_argument("--n-delays", type=int, default=141)
    parser.add_argument("--length-mm", type=float, default=80.0)
    parser.add_argument(
        "--output",
        type=str,
        default="examples/images/27_gnlse_spectrogram.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pulse = make_pulse()
    fiber = make_fiber(pulse, length_m=args.length_mm * 1e-3)
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=BETAS,
        include_raman=True,
        include_self_steepening=True,
    )
    solver.propagate(num_steps=args.num_steps, nsaves=41)

    fig = plot_spectrogram(
        solver,
        n_delays=args.n_delays,
        wl_bounds=(450.0, 1250.0),
        dynamic_range_db=45.0,
        with_projections=True,
        t_min=-3.0,
        t_max=3.0,
    )
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
