#!/usr/bin/env python3
"""Benchmark the FFT backends, including a large-N GNLSE supercontinuum run.

Measures the pure FFT layer and an end-to-end supercontinuum-generation (SCG)
propagation for every available backend. The cupy (GPU) backend is included
only when ``cupy`` is importable; on CPU-only machines the script compares
FFTW3 / scipy / numpy and reports the per-transform cost.

Usage
-----
    python benchmarks/fft_backend_benchmark.py
    python benchmarks/fft_backend_benchmark.py --N 65536 --steps 1000
    python benchmarks/fft_backend_benchmark.py --reps 200 --skip-scg

The SCG case is a 50 fs, ~10 kW sech pulse at 1550 nm in a 5 cm HNLF
(β₂ = −20 ps²/km, γ ≈ 10 /W/km, Raman on) — a soliton-fission supercontinuum
that exercises both the linear and the Raman FFTs on every split step.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# The sys.path bootstrap above intentionally precedes the library imports.

import numpy as np

from photonics_helper import _fftw
from photonics_helper._fftw import backend_name, set_backend


def _candidate_backends() -> list[str]:
    names = list(_fftw._available_backends())  # fftw, scipy, numpy (best first)
    if _fftw._HAS_CUPY:
        names.insert(0, "cupy")
    return names


def bench_fft(n: int, reps: int) -> float:
    """Seconds per forward+inverse FFT pair at length ``n``."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    _fftw.fft(A)  # warm up (planning / CUDA context)
    _fftw.ifft(A)
    t0 = time.perf_counter()
    for _ in range(reps):
        _fftw.fft(A)
        _fftw.ifft(A)
    return (time.perf_counter() - t0) / reps


def _make_scg_solver(N: int, length_m: float):
    from photonics_helper.base import Area, Length, Time, Wavelength
    from photonics_helper.gnlse import FiberProfile, GNLSESolver
    from photonics_helper.pulse import Envelope, TemporalGrid, Wave
    from photonics_helper.raman import RamanResponse, RamanSpec

    grid = TemporalGrid(N=N, Tmax=Time(8e-12, "s"))
    envelope = Envelope(
        shape="sech", peak_amplitude=np.sqrt(10_000.0), pulse_width=Time(50, "fs")
    )
    pulse = Wave(
        grid=grid, envelope=envelope, central_wavelength=Wavelength(1550, "nm")
    )
    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    raman = RamanResponse(spec=spec, grid=grid)
    fiber = FiberProfile(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(10e-12, "m^2"),
        length=Length(length_m, "m"),
        raman_response=raman,
    )
    betas = np.array([-20e-3, 1e-4])  # β₂, β₃ in ps²/m, ps³/m (anomalous)
    return GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=True,
        include_self_steepening=True,
    )


def bench_scg(N: int, steps: int, length_m: float):
    """Run the SCG case and return (seconds, output_field)."""
    solver = _make_scg_solver(N, length_m)
    t0 = time.perf_counter()
    solver.propagate(steps, nsaves=10)
    elapsed = time.perf_counter() - t0
    return elapsed, np.asarray(solver.evolution[-1].envelope_field)


def main() -> int:
    # Keep the timing table readable: these two warnings fire for every run.
    warnings.filterwarnings("ignore", message=".*normalized envelope units.*")
    warnings.filterwarnings("ignore", message=".*Auto-derived.*")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=2**14, help="grid points (default 2^14)")
    parser.add_argument("--reps", type=int, default=50, help="FFT pairs per timing")
    parser.add_argument("--steps", type=int, default=500, help="GNLSE steps")
    parser.add_argument("--length-mm", type=float, default=50.0, help="fiber length (mm)")
    parser.add_argument("--skip-scg", action="store_true", help="FFT layer only")
    args = parser.parse_args()

    backends = _candidate_backends()
    print(f"FFT backend benchmark — N={args.N}, reps={args.reps}, steps={args.steps}")
    print(f"candidates: {', '.join(backends)}\n")

    fft_times: dict[str, float] = {}
    scg_times: dict[str, float] = {}
    reference: np.ndarray | None = None

    for name in backends:
        set_backend(name)
        active = _fftw._active_name
        label = backend_name()
        try:
            fft_times[active] = bench_fft(args.N, args.reps)
            line = f"  {active:6s} {label:34s} FFT pair: {fft_times[active] * 1e6:9.2f} µs"
            if not args.skip_scg:
                elapsed, field = bench_scg(args.N, args.steps, args.length_mm * 1e-3)
                scg_times[active] = elapsed
                if reference is None:
                    reference = field
                    err = 0.0
                else:
                    err = float(
                        np.linalg.norm(field - reference)
                        / max(np.linalg.norm(reference), 1e-300)
                    )
                line += f"   SCG: {elapsed:6.2f} s   ΔL2 vs numpy: {err:.2e}"
            print(line)
        finally:
            set_backend(None)

    if fft_times:
        fastest = min(fft_times, key=fft_times.get)
        print(f"\nfastest FFT pair: {fastest} ({fft_times[fastest] * 1e6:.2f} µs)")
    if scg_times:
        fastest = min(scg_times, key=scg_times.get)
        print(f"fastest SCG run:  {fastest} ({scg_times[fastest]:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
