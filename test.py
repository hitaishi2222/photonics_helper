# Example script demonstrating the use of the Pulse library
#
# This script shows how to build a Gaussian (and optionally a Sech) ultrashort
# optical pulse with the classes defined in ``photonics_helper.pulse``.
# It creates a temporal grid, an ``Envelope`` and a ``Wave`` instance,
# prints a few characteristic quantities and finally visualises the pulse.

import matplotlib.pyplot as plt

from photonics_helper.base import Wavelength, Time
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


def example_gaussian_pulse():
    """Create and visualise a transform‑limited Gaussian pulse.

    The pulse has a central wavelength of 800 nm and an intensity full‑width at
    half‑maximum (FWHM) of roughly 50 fs.  All units are SI unless they are
    explicitly scaled in the call to ``Wave.visualize``.
    """
    # ---- parameters -------------------------------------------------
    central_wl = Wavelength(800e-9, "m")  # 800 nm
    fwhm_intensity = Time(50, "fs")

    # Build an Envelope from the desired FWHM.
    env = Envelope.from_fwhm(
        shape="gaussian",
        peak_amplitude=1.0,
        fwhm=fwhm_intensity,
    )

    # ---- temporal grid -----------------------------------------------
    N = 2**12  # power‑of‑2 for FFT efficiency
    Tmax = 10 * env.pulse_width.as_s  # total simulation window (≈10 × T0)
    grid = TemporalGrid(N=N, Tmax=Tmax)

    # ---- wave (the full pulse) ----------------------------------------
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=central_wl,
        refractive_index=1.0,
    )

    # ---- diagnostics ---------------------------------------------------
    print("=== Gaussian pulse ===")
    print(f"Pulse energy      : {pulse.pulse_energy():.3e} J")
    print(f"Peak power        : {pulse.peak_power():.3e} W")
    print(f"TBP (≈0.44)       : {pulse.time_bandwidth_product():.3f}")
    print(f"FWHM (intensity) : {env.fwhm.as_fs:.2f} fs")

    # ---- visualisation ------------------------------------------------
    pulse.visualize(
        t_unit="fs",
        w_unit="THz",
        t_scale=1e15,  # seconds → femtoseconds
        w_scale=1e-12,  # rad/s   → THz
        show_electric_field=True,
        show_phase=True,
        show_spectrogram=False,
        figsize=(12, 8),
    )
    plt.show()


def example_sech_pulse():
    """Create a Sech‑shaped pulse and visualise it.

    This demonstrates that the same workflow works for any of the supported
    envelope shapes (``gaussian``, ``sech``, ``lorentzian`` and ``rectangular``).
    """
    central_wl = Wavelength(1550e-9, "m")  # telecom wavelength
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(60, "fs"), chirp=5e-27)

    N = 2**12
    Tmax = 12 * env.pulse_width.as_s
    grid = TemporalGrid(N=N, Tmax=Tmax)

    pulse = Wave(grid=grid, envelope=env, central_wavelength=central_wl)

    print("=== Sech pulse ===")
    print(f"Pulse energy : {pulse.pulse_energy():.3e} J")
    print(f"Peak power   : {pulse.peak_power():.3e} W")

    pulse.visualize(
        t_unit="fs",
        w_unit="THz",
        t_scale=1e15,
        w_scale=1e-12,
        show_electric_field=False,
        show_phase=True,
        show_spectrogram=False,
        figsize=(10, 6),
    )
    plt.show()


if __name__ == "__main__":
    # Run the Gaussian example by default; uncomment the next line to see the Sech example.
    example_gaussian_pulse()
    # example_sech_pulse()
