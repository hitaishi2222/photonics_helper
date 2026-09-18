"""Shared model for the femtosecond supercontinuum scenario of

    J. M. Dudley, G. Genty, S. Coen,
    "Supercontinuum generation in photonic crystal fiber,"
    Rev. Mod. Phys. 78, 1135 (2006).

The scenario is the paper's Sec. V-VI reference case: 15 cm of highly
nonlinear PCF (hole diameter d = 1.4 um, pitch Lambda = 1.6 um, ZDW around
780 nm) pumped at 835 nm in the anomalous-GVD regime.  The Taylor series
coefficients are Table I of the paper; the input is a sech pulse with
T0 = 28.4 fs (50 fs FWHM at 10 kW), the Raman fraction is fR = 0.18 and the
corrected shock time is tau_shock = 0.56 fs.

Everything that several figure scripts need lives here so that each
``figNN_*.py`` file is only the physics and the assertion for its figure.

Note on self-steepening
-----------------------
The library implements the shock term as ``(1 + i/omega0 d/dt)`` with
``omega0`` the carrier frequency, i.e. ``tau_shock = 1/omega0 = 0.443 fs`` at
835 nm.  The paper uses ``tau_shock = 0.56 fs``, which includes the frequency
dependence of the effective mode area (their Eq. 3).  This ~26% difference is
noted in the analysis; the script-level reproductions therefore validate
against the paper's *physics* (soliton scales, fission, DW phase matching)
and treat the shock-time refinement as a documented limitation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from math import factorial, pi
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

HERE = Path(__file__).resolve().parent

with open(HERE / "parameters.json", encoding="utf-8") as _fh:
    PARAMS: dict = json.load(_fh)

# ── Shared reference parameters (Table I and Sec. V.A) ──────────────────────

#: β₂ … β₁₀ at 835 nm in ps^k / m (Table I), used by the solver.
BETAS: np.ndarray = np.asarray(PARAMS["taylor_betas_psN_per_m"], dtype=float)
BETA2_PS2_M: float = float(PARAMS["beta2_ps2_per_m"])
BETA3_PS3_M: float = float(PARAMS["beta3_ps3_per_m"])
GAMMA: float = float(PARAMS["gamma_per_Wm"])
N2: float = float(PARAMS["n2_m2_per_W"])
LENGTH_M: float = float(PARAMS["fiber_length_m"])
WL0_NM: float = float(PARAMS["central_wavelength_nm"])
T0_FS: float = float(PARAMS["T0_fs"])
FWHM_FS: float = 1.763 * T0_FS  # 50 fs
SHOCK_FS: float = float(PARAMS["shock_time_fs"])
FR: float = float(PARAMS["raman_fR"])
TAU1_S: float = float(PARAMS["raman_tau1_fs"]) * 1e-15
TAU2_S: float = float(PARAMS["raman_tau2_fs"]) * 1e-15


def beta2_si() -> float:
    """β₂ in SI units (s²/m) for the analytic soliton length scales."""
    return BETA2_PS2_M * 1e-24


def beta3_si() -> float:
    """β₃ in SI units (s³/m)."""
    return BETA3_PS3_M * 1e-36


# ── Scalar soliton scales ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SolitonScales:
    """Characteristic length scales of the input pulse (paper Sec. V.B)."""

    T0_s: float
    beta2_si: float
    gamma: float
    P0: float

    @property
    def L_D(self) -> float:
        """Dispersion length L_D = T0²/|β₂| (m)."""
        return self.T0_s**2 / abs(self.beta2_si)

    @property
    def L_NL(self) -> float:
        """Nonlinear length L_NL = 1/(γP0) (m)."""
        return 1.0 / (self.gamma * self.P0)

    @property
    def N(self) -> float:
        """Soliton order N = sqrt(L_D/L_NL)."""
        return float(np.sqrt(self.L_D / self.L_NL))

    @property
    def z_sol(self) -> float:
        """Soliton period z_sol = (π/2) L_D (m)."""
        return 0.5 * pi * self.L_D

    @property
    def L_fiss(self) -> float:
        """Fission length estimate L_fiss ~ L_D/N (m)."""
        return self.L_D / self.N


def soliton_scales(
    T0_fs: float = T0_FS,
    gamma: float = GAMMA,
    P0: float = PARAMS["peak_power_W"],
    beta2_si_value: float | None = None,
) -> SolitonScales:
    """Return the soliton length scales for the given pulse/fiber parameters."""
    return SolitonScales(
        T0_s=T0_fs * 1e-15,
        beta2_si=beta2_si() if beta2_si_value is None else beta2_si_value,
        gamma=gamma,
        P0=P0,
    )


def kodama_hasegawa(N_order: float, P0: float, T0_fs: float = T0_FS) -> list[dict]:
    """Kodama-Hasegawa fundamental solitons ejected by an ideal N-soliton.

    From the paper (Sec. V.B.1, quoting Kodama and Hasegawa 1987)::

        P_j = P0 (2N - 2j + 1)² / N²
        T_j = T0 / (2N - 2j + 1)

    for j = 1 … N.  ``T_j`` is the sech parameter, so the intensity FWHM is
    ``1.763 * T_j``.
    """
    out: list[dict] = []
    for j in range(1, int(round(N_order)) + 1):
        m = 2 * N_order - 2 * j + 1
        out.append(
            {
                "j": j,
                "P_W": P0 * m**2 / N_order**2,
                "T_fs": T0_fs / m,
                "fwhm_fs": 1.763 * T0_fs / m,
            }
        )
    return out


# ── Pulse / fiber / solver builders ──────────────────────────────────────────


def build_pulse(
    peak_power_W: float,
    *,
    T0_fs: float = T0_FS,
    N_points: int = 4096,
    Tmax_ps: float = 12.5,
    wl_nm: float = WL0_NM,
    fwhm: bool = True,
) -> Wave:
    """Build a sech input pulse.

    Parameters
    ----------
    peak_power_W : float
        Peak power P0 in watts; the envelope peak amplitude is sqrt(P0).
    T0_fs : float
        sech parameter T0 in fs.  If ``fwhm=True`` (default) the envelope is
        constructed from the intensity FWHM = 1.763 T0.
    N_points, Tmax_ps : int, float
        Temporal grid size and half-window; Tmax is the full window.
    """
    grid = TemporalGrid(N=N_points, Tmax=Time(Tmax_ps * 1e-12, "s"))
    if fwhm:
        envelope = Envelope.from_fwhm(
            "sech",
            peak_amplitude=float(np.sqrt(peak_power_W)),
            fwhm=Time(1.763 * T0_fs, "fs"),
        )
    else:
        envelope = Envelope(
            shape="sech",
            peak_amplitude=float(np.sqrt(peak_power_W)),
            pulse_width=Time(T0_fs, "fs"),
        )
    return Wave(
        grid=grid,
        envelope=envelope,
        central_wavelength=Wavelength(wl_nm, "nm"),
    )


def build_raman(grid: TemporalGrid, fR: float = FR) -> RamanResponse:
    """Silica Raman response with the paper's explicit τ₁ = 12.2 fs, τ₂ = 32 fs."""
    spec = RamanSpec(
        name="Silica",
        raman_shift_cm=440.0,
        raman_linewidth_cm=45.0,
        fR=fR,
    )
    return RamanResponse(
        spec=spec,
        fR=fR,
        tau1=TAU1_S,
        tau2=TAU2_S,
        grid=grid,
    )


def build_fiber(
    pulse: Wave,
    *,
    length_m: float = LENGTH_M,
    raman: bool = False,
) -> FiberProfile:
    """PCF profile with γ = 0.11 /W/m (no loss, as in the paper's simulations)."""
    return FiberProfile.from_gamma(
        gamma=GAMMA,
        n2=N2,
        omega0=float(pulse.central_frequency),
        length=Length(length_m, "m"),
        raman_response=build_raman(pulse.grid) if raman else None,
    )


def make_solver(
    pulse: Wave,
    fiber: FiberProfile,
    betas: np.ndarray,
    *,
    raman: bool = False,
    shock: bool = False,
    tau_shock: float | None = None,
) -> GNLSESolver:
    """Construct a ``GNLSESolver`` for the Dudley model.

    ``tau_shock`` is the shock timescale in seconds (SI); ``None`` uses the
    library default ``1/ω₀``. Pass ``SHOCK_FS * 1e-15`` (0.56 fs) for the
    paper's effective-area-corrected value (RMP 2006 Eq. (3), Sec. V.B).
    """
    return GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.asarray(betas, dtype=float),
        include_raman=raman,
        include_self_steepening=shock,
        include_tpa=False,
        tau_shock=tau_shock,
    )


# ── Field-evolution container and analysis helpers ───────────────────────────


@dataclass
class Evolution:
    """Stored field snapshots from a GNLSE run, plus spectral helpers."""

    z: np.ndarray  # (nz,) propagation distance (m)
    fields: np.ndarray  # (nz, N) complex envelope
    t: np.ndarray  # (N,) time grid (s)
    omega: np.ndarray  # (N,) angular-frequency offset grid (rad/s)
    omega0: float  # carrier angular frequency (rad/s)
    metadata: dict = field(default_factory=dict)

    # -- intensity / spectra -------------------------------------------------

    @property
    def intensity(self) -> NDArray:
        """Temporal intensity |A(z,T)|² in W, shape (nz, N)."""
        return np.asarray(np.abs(self.fields) ** 2)

    @property
    def spectra(self) -> np.ndarray:
        """Spectral power |Ã(z,Ω)|² (arb. units), shape (nz, N).

        Uses the fftshifted convention of :attr:`TemporalGrid.fft` so that the
        spectral axis is aligned with :attr:`omega`.
        """
        shifted = np.fft.fftshift(np.fft.fft(self.fields, axis=1), axes=1)
        return np.abs(shifted) ** 2

    def wavelength_nm(self) -> np.ndarray:
        """Vacuum wavelength (nm) for each spectral bin, ascending."""
        wl = 2.0 * pi * C_MS / (self.omega0 + self.omega)
        return np.sort(wl * 1e9)

    def spectrum_on_wavelength(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (wavelength_nm ascending, PSD) for a single stored field.

        The spectral index permutation is the same for every z snapshot, so
        this is also used to sort entire evolution arrays.
        """
        wl = 2.0 * pi * C_MS / (self.omega0 + self.omega)
        order = np.argsort(wl)
        return wl[order] * 1e9, self.spectra[:, order]

    def output_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Spectrum at the last stored z (ascending wavelength, normalised)."""
        wl, psd = self.spectrum_on_wavelength()
        last = psd[-1]
        return wl, last / last.max()

    def output_intensity(self) -> tuple[np.ndarray, np.ndarray]:
        """Temporal intensity at the last stored z (time in ps, W)."""
        return self.t * 1e12, self.intensity[-1]

    def at(self, index: int) -> NDArray:
        return np.asarray(self.fields[index])


def run_gnlse(
    pulse: Wave,
    fiber: FiberProfile,
    betas: np.ndarray,
    *,
    num_steps: int,
    nsaves: int,
    raman: bool = False,
    shock: bool = False,
    show_progress: bool = False,
) -> Evolution:
    """Propagate and package the result as an :class:`Evolution`."""
    solver = make_solver(pulse, fiber, betas, raman=raman, shock=shock)
    solver.propagate(num_steps=num_steps, nsaves=nsaves, show_progress=show_progress)

    z = np.asarray(solver._z_positions, dtype=float)[: len(solver.evolution)]
    fields = np.asarray(
        [wave.envelope_field for wave in solver.evolution], dtype=complex
    )
    return Evolution(
        z=z,
        fields=fields,
        t=pulse.grid.t,
        omega=pulse.grid.w,
        omega0=float(pulse.central_frequency),
        metadata={
            "num_steps": num_steps,
            "nsaves": nsaves,
            "raman": raman,
            "shock": shock,
            "betas": np.asarray(betas, dtype=float).tolist(),
        },
    )


def save_evolution(evo: Evolution, path: Path) -> None:
    """Persist an :class:`Evolution` to a compressed ``.npz`` file."""
    np.savez_compressed(
        path,
        z=evo.z,
        fields=evo.fields,
        t=evo.t,
        omega=evo.omega,
        omega0=evo.omega0,
        metadata=json.dumps({k: v for k, v in evo.metadata.items()}),
    )


def load_evolution(path: Path) -> Evolution:
    """Load an :class:`Evolution` written by :func:`save_evolution`."""
    data = np.load(path, allow_pickle=False)
    return Evolution(
        z=data["z"],
        fields=data["fields"],
        t=data["t"],
        omega=data["omega"],
        omega0=float(data["omega0"]),
        metadata=json.loads(str(data["metadata"])),
    )


def spectrogram(
    field: np.ndarray,
    t: np.ndarray,
    gate: np.ndarray,
    omega: np.ndarray,
    *,
    n_delays: int = 161,
    delay_span_ps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-correlation spectrogram (paper Eq. 4).

    ``Σ(ω, τ) = |∫ E(t) g(t − τ) e^{−iωt} dt|²`` with the gate ``g`` sampled on
    the same time grid as ``E`` and shifted by ``τ``.  Returns
    ``(delay_ps, omega, S)`` with ``S`` of shape ``(n_delays, N)``.  This is
    the time-domain equivalent of a cross-correlation FROG trace.
    """
    E = np.asarray(field, dtype=complex)
    g = np.asarray(gate, dtype=complex)
    t = np.asarray(t)
    omega = np.asarray(omega)

    if delay_span_ps is None:
        span = (t.max() - t.min()) * 1e12 * 0.6
    else:
        span = delay_span_ps
    delays = np.linspace(-span, span, n_delays) * 1e-12

    S = np.zeros((n_delays, len(E)), dtype=float)
    for i, tau in enumerate(delays):
        gated = E * np.interp(t - tau, t, g.real)
        S[i] = np.abs(np.fft.fftshift(np.fft.fft(gated))) ** 2
    return delays * 1e12, omega, S


def input_gate(pulse: Wave) -> np.ndarray:
    """Real input-pulse envelope used as the spectrogram gate."""
    return np.real(pulse.envelope.field(pulse.grid.t))


# ── Peak measurement helpers ─────────────────────────────────────────────────
def measure_peak_fwhm(t: np.ndarray, intensity: np.ndarray, idx: int) -> float:
    """Intensity FWHM (same units as *t*) of the peak at *idx*."""
    profile = np.asarray(intensity, dtype=float)
    half = profile[idx] / 2.0
    left = idx
    while left > 0 and profile[left] > half:
        left -= 1
    right = idx
    while right < len(profile) - 1 and profile[right] > half:
        right += 1

    def _cross(i0: int, i1: int) -> float:
        if profile[i1] == profile[i0]:
            return float(t[i1])
        return float(
            t[i0] + (half - profile[i0]) * (t[i1] - t[i0]) / (profile[i1] - profile[i0])
        )

    t_left = _cross(left, left + 1) if left < idx else float(t[left])
    t_right = _cross(right - 1, right) if right > idx else float(t[right])
    return abs(t_right - t_left)


def temporal_peaks(
    intensity: np.ndarray,
    *,
    rel_height: float = 0.02,
    min_distance: int = 40,
) -> np.ndarray:
    """Indices of temporal intensity peaks above ``rel_height`` of the max."""
    from scipy.signal import find_peaks

    profile = np.asarray(intensity, dtype=float)
    peaks, _ = find_peaks(
        profile,
        height=rel_height * profile.max(),
        prominence=rel_height * profile.max(),
        distance=min_distance,
    )
    return np.asarray(peaks)


def taylor_beta_fn(
    omega0: float | None = None, betas: NDArray = BETAS
) -> Callable[[NDArray], NDArray]:
    """Return β(ω) (1/m) reconstructed from the Table I Taylor series.

    The returned callable takes an absolute angular frequency in rad/s and
    evaluates ``Σ_{k≥2} β_k (ω − ω_ref)^k / k!`` with the β_k in ps^k/m, i.e.
    everything is kept in ``rad/ps`` internally.  β₀ and β₁ are irrelevant for
    the MI phase mismatch and are dropped.  The reconstruction is accurate
    near 835 nm and gives a ZDW of ≈ 780 nm, matching the paper's Fig. 2.
    """
    if omega0 is None:
        omega0 = 2.0 * pi * C_MS / (WL0_NM * 1e-9)
    coefficients = np.asarray(betas, dtype=float)

    def _beta(omega):
        omega_ps_offset = (np.asarray(omega, dtype=float) - omega0) * 1e-12
        return sum(
            float(b) * omega_ps_offset**k / factorial(k)
            for k, b in enumerate(coefficients, start=2)
        )

    return _beta


def beta2_at_wavelength(wl_nm: float) -> float:
    """β₂(λ) in ps²/m from the Taylor reconstruction (for the MI check)."""
    omega0 = 2.0 * pi * C_MS / (WL0_NM * 1e-9)
    omega = 2.0 * pi * C_MS / (wl_nm * 1e-9)
    offset = (omega - omega0) * 1e-12
    return sum(
        float(b) * offset ** (k - 2) / factorial(k - 2)
        for k, b in enumerate(BETAS, start=2)
    )


def dispersive_wave_wavelength_nm(
    Ps: float,
    *,
    fR: float = FR,
    gamma: float = GAMMA,
    betas: np.ndarray = BETAS,
    omega0: float | None = None,
    wl0_nm: float = WL0_NM,
    nonlinear: bool = True,
) -> float:
    """Dispersive-wave wavelength for a soliton (paper Fig. 8 / Sec. V.B.2).

    Solves the phase-matching condition quoted in the paper,

        β(ω_DW) − β(ω_s) − β₁(ω_s)(ω_DW − ω_s) = (1 − fR) γ Ps .

    With ω_s = ω0 this is the root of the Taylor linear mismatch
    ``Σ_{k≥2} β_k Ω^k/k! = (1−fR) γ Ps`` (``Ω = ω_DW − ω0``).  Setting
    ``nonlinear=False`` recovers the purely linear Cherenkov root.
    Returns the vacuum wavelength in nm for the blue-side (Ω > 0) root.
    """
    from scipy.optimize import brentq

    if omega0 is None:
        omega0 = 2.0 * pi * C_MS / (wl0_nm * 1e-9)
    rhs = (1.0 - fR) * gamma * Ps if nonlinear else 0.0

    def _mismatch(Omega_ps: float) -> float:
        return (
            sum(
                float(b) * Omega_ps**k / factorial(k)
                for k, b in enumerate(np.asarray(betas, dtype=float), start=2)
            )
            - rhs
        )

    grid = np.linspace(100.0, 1600.0, 60000)
    values = np.array([_mismatch(x) for x in grid])
    roots = [
        brentq(_mismatch, grid[i], grid[i + 1])
        for i in range(len(grid) - 1)
        if values[i] * values[i + 1] < 0
    ]
    blue = [r for r in roots if r > 0]
    if not blue:
        raise ValueError("no blue-side dispersive-wave root found")
    Omega_s = omega0 + blue[0] * 1e12  # rad/ps -> rad/s
    return float(2.0 * pi * C_MS / Omega_s * 1e9)


def strongest_peak_in_band(
    wavelength_nm: np.ndarray,
    psd: np.ndarray,
    wl_min: float,
    wl_max: float,
    *,
    rel_height: float = 0.01,
    min_prominence: float = 0.01,
) -> tuple[float, float]:
    """Return (wavelength, relative power) of the strongest peak in a band."""
    from scipy.signal import find_peaks

    psd = np.asarray(psd, dtype=float)
    normalized = psd / psd.max() if psd.max() > 0 else psd
    mask = (wavelength_nm >= wl_min) & (wavelength_nm <= wl_max)
    band = np.asarray(wavelength_nm)[mask]
    values = normalized[mask]
    if band.size == 0:
        return float("nan"), 0.0
    peaks, _ = find_peaks(
        values,
        height=rel_height,
        prominence=min_prominence,
    )
    if peaks.size == 0:
        idx = int(np.argmax(values))
    else:
        idx = int(peaks[np.argmax(values[peaks])])
    return float(band[idx]), float(values[idx])


def mean_spectral_wavelength_nm(evo: Evolution, index: int = -1) -> float:
    """Energy-weighted mean wavelength (nm) of snapshot ``index``."""
    wl, psd = evo.spectrum_on_wavelength()
    power = psd[index]
    return float((wl * power).sum() / power.sum())


# ── Plotting helpers ─────────────────────────────────────────────────────────


def interpolate_on_wavelength(
    wl_sorted: NDArray,
    spectra: NDArray,
    wl_min: float,
    wl_max: float,
    n_points: int = 500,
) -> tuple[NDArray, NDArray]:
    """Resample spectra from the native (non-uniform) wavelength bins to a
    uniform wavelength grid suitable for ``imshow``.

    ``spectrum_on_wavelength`` returns the PSD sorted by ascending wavelength,
    but because ``λ = 2πc/(ω₀+Ω)`` the bin spacing is strongly non-uniform.
    Plotting the sorted array against a linear wavelength extent misplaces
    every feature, so interpolate onto a uniform grid first.
    """
    grid = np.linspace(wl_min, wl_max, n_points)
    rows = np.asarray(spectra, dtype=float)
    if rows.ndim == 1:
        return grid, np.interp(grid, wl_sorted, rows)
    resampled = np.empty((rows.shape[0], n_points), dtype=float)
    for i, row in enumerate(rows):
        resampled[i] = np.interp(grid, wl_sorted, row)
    return grid, resampled


def density_clip(psd: np.ndarray, dynamic_range_db: float = 40.0) -> np.ndarray:
    """Log-normalised power in [0, 1] relative to the global maximum."""
    psd = np.asarray(psd, dtype=float)
    peak = psd.max() if psd.size else 1.0
    if peak <= 0:
        peak = 1.0
    db = 10.0 * np.log10(np.maximum(psd / peak, 1e-12))
    lo = -abs(dynamic_range_db)
    return np.asarray(np.clip((db - lo) / (-lo), 0.0, 1.0))


#: Default wavelength plotting window, as fractions of the carrier wavelength.
WL_BOUNDS_FRACTION: tuple[float, float] = (0.5, 1.6)

#: Delay (ps) beyond which a cell is labelled DW-like / soliton-like on hover.
TEMPORAL_FEATURE_DELAY_PS: float = 0.5

#: Spectral offset (nm) beyond which a cell is labelled DW-like / soliton-like.
SPECTRAL_FEATURE_OFFSET_NM: float = 80.0


def default_wl_bounds(
    wl0_nm: float = WL0_NM,
    fraction: tuple[float, float] = WL_BOUNDS_FRACTION,
) -> tuple[float, float]:
    """Default spectral plotting window around the carrier wavelength.

    Returns ``(fraction[0]·λ0, fraction[1]·λ0)`` in nm.  The default
    ``(0.5, 1.6)`` gives roughly 418-1336 nm for the 835 nm Dudley pump; it
    leans to the red so that Raman-shifted solitons stay in frame, and the
    caller can pass an explicit ``wl_bounds`` or different ``fraction``.
    """
    lo, hi = float(fraction[0]), float(fraction[1])
    if not 0.0 < lo < hi:
        raise ValueError(
            f"fraction must be an increasing positive pair, got {fraction!r}"
        )
    return float(wl0_nm) * lo, float(wl0_nm) * hi


def carrier_wavelength_nm(evo: Evolution) -> float:
    """Carrier wavelength of the simulation (nm), from ``omega0``."""
    return 2.0 * pi * C_MS / evo.omega0 * 1e9


def _resolve_wl_bounds(
    evo: Evolution,
    wl_bounds: tuple[float, float] | None,
    wl_min: float | None,
    wl_max: float | None,
) -> tuple[float, float]:
    """Resolve the plotted wavelength window from explicit bounds or limits."""
    if wl_bounds is not None:
        bounds = (float(wl_bounds[0]), float(wl_bounds[1]))
    else:
        default_lo, default_hi = default_wl_bounds(carrier_wavelength_nm(evo))
        bounds = (
            default_lo if wl_min is None else float(wl_min),
            default_hi if wl_max is None else float(wl_max),
        )
    if bounds[1] <= bounds[0]:
        raise ValueError(f"wavelength bounds must be increasing, got {bounds!r}")
    return bounds


def _z_axis(evo: Evolution, z_scale: str) -> tuple[NDArray, str]:
    """Return propagation distance in the requested unit and its axis label."""
    try:
        factor, label = {
            "m": (1.0, "Distance (m)"),
            "cm": (1e2, "Distance (cm)"),
            "mm": (1e3, "Distance (mm)"),
        }[z_scale]
    except KeyError as exc:
        raise ValueError(f"z_scale must be 'm', 'cm' or 'mm', got {z_scale!r}") from exc
    return np.asarray(evo.z, dtype=float) * factor, label


def classify_spectral_features(
    wavelength_nm: NDArray,
    pump_nm: float = WL0_NM,
    *,
    offset_nm: float = SPECTRAL_FEATURE_OFFSET_NM,
) -> NDArray:
    """Label each wavelength bin for interactive hover text.

    Blue-shifted normal-GVD components become *dispersive wave*, the pump
    region is *SPM / pump*, and red-shifted anomalous-GVD components are
    *Raman soliton*.  This is a guide for exploration, not a substitute for
    checking the dispersion sign at the pump wavelength.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    offset = wl - float(pump_nm)
    labels = np.full(wl.shape, "SPM / pump", dtype=object)
    labels[offset < -abs(offset_nm)] = "Dispersive wave (blue)"
    labels[offset > abs(offset_nm)] = "Raman soliton (red)"
    return labels


def classify_temporal_features(
    t_ps: NDArray,
    *,
    delay_ps: float = TEMPORAL_FEATURE_DELAY_PS,
) -> NDArray:
    """Label each time bin for interactive hover text.

    Once the axis uses the literature comoving convention (Raman solitons at
    positive delay), leading/normal-GVD energy is labelled *dispersive wave*,
    the centre is *SPM / pump*, and delayed energy is *Raman soliton*.
    """
    t = np.asarray(t_ps, dtype=float)
    labels = np.full(t.shape, "SPM / pump", dtype=object)
    labels[t < -abs(delay_ps)] = "Dispersive wave (blue, leading)"
    labels[t > abs(delay_ps)] = "Raman soliton (red, delayed)"
    return labels


def spectral_evolution_data(
    evo: Evolution,
    wl_bounds: tuple[float, float],
    n_points: int = 500,
) -> tuple[NDArray, NDArray]:
    """Resample stored spectra onto a uniform wavelength grid (nm)."""
    wl, psd = evo.spectrum_on_wavelength()
    return interpolate_on_wavelength(wl, psd, wl_bounds[0], wl_bounds[1], n_points)


def temporal_evolution_data(
    evo: Evolution,
    *,
    time_reversal: bool = True,
) -> tuple[NDArray, NDArray]:
    """Return (time_ps, intensity) in the literature comoving convention.

    The library's internal time grid runs opposite to the standard Agrawal /
    Dudley convention, so Raman-red-shifted solitons appear at *negative*
    internal time (see :meth:`Evolution.output_intensity`).  With
    ``time_reversal=True`` the axis is flipped so solitons appear at positive
    delay, matching Dudley et al. (2006) Fig. 3(b).  Spectra are unaffected.
    """
    t_ps = np.asarray(evo.t, dtype=float) * 1e12
    intensity = np.asarray(evo.intensity, dtype=float)
    if time_reversal:
        t_ps = -t_ps
        order = np.argsort(t_ps)
        t_ps = t_ps[order]
        intensity = intensity[:, order]
    return t_ps, intensity


def temporal_feature_labels(
    evo: Evolution,
    *,
    time_reversal: bool = True,
    n_bands: int = 24,
    wl_range: tuple[float, float] = (400.0, 1400.0),
    floor_db: float = 40.0,
) -> NDArray:
    """Per-cell feature labels for the temporal-evolution hover text.

    A single time bin can carry several spectrally distinct features (e.g. a
    far-red Raman soliton and the blue dispersive wave can share a delay), so
    the label is decided by a coarse spectral filter bank on each stored
    field: the band with the most local energy wins, and cells below
    ``floor_db`` of the global peak are labelled *low-level background*.

    Returns an object array of shape ``(n_z, n_time)`` aligned with
    :func:`temporal_evolution_data` for the same ``time_reversal`` flag.
    """
    omega = np.asarray(evo.omega, dtype=float)
    lam = 2.0 * pi * C_MS / (evo.omega0 + omega) * 1e9
    edges = np.geomspace(float(wl_range[0]), float(wl_range[1]), n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    band_of_bin = np.digitize(lam, edges) - 1
    band_masks = [(band_of_bin == b) for b in range(int(n_bands))]

    pump_nm = carrier_wavelength_nm(evo)
    n_z, n_t = evo.fields.shape
    labels = np.empty((n_z, n_t), dtype=object)
    for iz, snapshot in enumerate(evo.fields):
        spectrum = np.fft.fftshift(np.fft.fft(snapshot))
        best = np.full(n_t, -np.inf)
        best_band = np.zeros(n_t, dtype=int)
        for b, mask in enumerate(band_masks):
            if not mask.any():
                continue
            profile = np.abs(np.fft.ifft(np.fft.ifftshift(spectrum * mask))) ** 2
            update = profile > best
            best[update] = profile[update]
            best_band[update] = b
        row = np.asarray(
            classify_spectral_features(centers[best_band], pump_nm), dtype=object
        )
        peak = float(best.max()) if np.isfinite(best).any() else 0.0
        if peak > 0.0:
            row[best < peak * 10.0 ** (-abs(floor_db) / 10.0)] = "low-level background"
        labels[iz] = row

    if time_reversal:
        labels = labels[:, ::-1]
    return labels


def _db_clipped(psd: NDArray, dynamic_range_db: float) -> NDArray:
    """Power in dB, normalised to the peak and clipped to the colour range."""
    psd = np.asarray(psd, dtype=float)
    peak = max(float(psd.max()) if psd.size else 0.0, 1e-30)
    db = 10.0 * np.log10(psd / peak + 1e-30)
    return np.asarray(np.clip(db, -abs(dynamic_range_db), 0.0))


def _plotly_density(
    *,
    x: NDArray,
    y: NDArray,
    z_db: NDArray,
    labels: NDArray,
    x_title: str,
    y_title: str,
    title: str,
    dynamic_range_db: float,
    colorscale: str = "Jet",
    annotate: bool = True,
):
    """Interactive Plotly heatmap whose hover names the local SC feature."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotly is required for plotly=True; install with "
            "`pip install plotly` or `pip install photonics-helper[plotting]`"
        ) from exc

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z_db = np.asarray(z_db, dtype=float)
    labels = np.asarray(labels, dtype=object)
    if labels.ndim == 1:
        custom = np.tile(labels, (z_db.shape[0], 1))
    elif labels.shape == z_db.shape:
        custom = labels
    else:
        raise ValueError(
            f"labels must be 1-D or match z {z_db.shape}, got {labels.shape}"
        )

    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=z_db,
            customdata=custom,
            colorscale=colorscale,
            zmin=-abs(dynamic_range_db),
            zmax=0.0,
            colorbar=dict(title="Power (dB)"),
            hovertemplate=(
                f"{x_title}: %{{x:.3g}}"
                f"<br>{y_title}: %{{y:.4g}}"
                f"<br>Power: %{{z:.1f}} dB"
                f"<br>Feature: %{{customdata}}"
                "<extra></extra>"
            ),
        )
    )

    if annotate:
        for name in np.unique(custom):
            mask = custom == name
            if not mask.any():
                continue
            masked = np.where(mask, z_db, -np.inf)
            iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
            fig.add_trace(
                go.Scatter(
                    x=[x[ix]],
                    y=[y[iy]],
                    mode="markers+text",
                    text=[str(name).split(" (")[0]],
                    textposition="middle right",
                    marker=dict(
                        size=9, color="white", line=dict(color="black", width=1)
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
    )
    return fig


def plot_spectral_evolution(
    evo: Evolution,
    ax=None,
    *,
    wl_bounds: tuple[float, float] | None = None,
    wl_min: float | None = None,
    wl_max: float | None = None,
    n_points: int = 500,
    dynamic_range_db: float = 40.0,
    cmap: str = "jet",
    z_scale: str = "cm",
    plotly: bool = False,
    annotate_features: bool = True,
):
    """Spectral-evolution density plot (wavelength vs distance).

    Parameters
    ----------
    evo : Evolution
        Stored propagation result.
    ax : matplotlib Axes, optional
        Axis to draw on.  A new figure is created when omitted.
    wl_bounds : (float, float), optional
        Wavelength window in nm.  Defaults to the carrier wavelength ±
        :data:`WL_BOUNDS_FRACTION`-style margins (approximately 0.5·λ0 to
        1.6·λ0), so it tracks the pump instead of hard-coded limits.
    wl_min, wl_max : float, optional
        Individual overrides used only when *wl_bounds* is None.
    n_points : int
        Samples on the uniform wavelength grid.
    dynamic_range_db : float
        Colour range below the global peak (dB).
    cmap : str
        Matplotlib colormap (ignored by the Plotly backend).
    z_scale : {"m", "cm", "mm"}
        Propagation-distance unit on the y axis.
    plotly : bool
        If True, return an interactive ``plotly.graph_objects.Figure`` whose
        hover labels each ``(λ, z)`` cell as DW / SPM / Raman soliton.
    annotate_features : bool
        Mark the strongest cell of each feature class in the Plotly backend.

    Returns
    -------
    matplotlib Figure or plotly Figure
        The figure, so the caller can keep adjusting it.
    """
    lo, hi = _resolve_wl_bounds(evo, wl_bounds, wl_min, wl_max)
    grid, resampled = spectral_evolution_data(evo, (lo, hi), n_points)
    z, z_label = _z_axis(evo, z_scale)

    if plotly:
        return _plotly_density(
            x=grid,
            y=z,
            z_db=_db_clipped(resampled, dynamic_range_db),
            labels=classify_spectral_features(grid, carrier_wavelength_nm(evo)),
            x_title="Wavelength (nm)",
            y_title=z_label,
            title="Spectral evolution",
            dynamic_range_db=dynamic_range_db,
            annotate=annotate_features,
        )

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.imshow(
        density_clip(resampled, dynamic_range_db),
        origin="lower",
        aspect="auto",
        extent=(lo, hi, z[0], z[-1]),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(z_label)
    return fig


def plot_temporal_evolution(
    evo: Evolution,
    ax=None,
    *,
    t_bounds: tuple[float, float] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    dynamic_range_db: float = 40.0,
    cmap: str = "jet",
    z_scale: str = "cm",
    time_reversal: bool = True,
    plotly: bool = False,
    annotate_features: bool = True,
):
    """Temporal-evolution density plot (time vs distance).

    Parameters
    ----------
    evo : Evolution
        Stored propagation result.
    ax : matplotlib Axes, optional
        Axis to draw on.  A new figure is created when omitted.
    t_bounds : (float, float), optional
        Time window in ps (literature convention).  Defaults to the full grid.
    t_min, t_max : float, optional
        Individual overrides used only when *t_bounds* is None.
    dynamic_range_db : float
        Colour range below the global peak (dB).
    cmap : str
        Matplotlib colormap (ignored by the Plotly backend).
    z_scale : {"m", "cm", "mm"}
        Propagation-distance unit on the y axis.
    time_reversal : bool
        Use the standard literature (Agrawal/Dudley) comoving time, where
        Raman-red-shifted solitons appear at positive delay.  The internal
        grid has the opposite sign; set False for the raw internal time.
    plotly : bool
        If True, return an interactive ``plotly.graph_objects.Figure`` whose
        hover labels each ``(t, z)`` cell as DW / SPM / Raman soliton.
    annotate_features : bool
        Mark the strongest cell of each feature class in the Plotly backend.

    Returns
    -------
    matplotlib Figure or plotly Figure
        The figure, so the caller can keep adjusting it.
    """
    t_ps, intensity = temporal_evolution_data(evo, time_reversal=time_reversal)
    z, z_label = _z_axis(evo, z_scale)

    if t_bounds is not None:
        lo, hi = float(t_bounds[0]), float(t_bounds[1])
    else:
        lo = float(t_ps[0]) if t_min is None else float(t_min)
        hi = float(t_ps[-1]) if t_max is None else float(t_max)
    if hi <= lo:
        raise ValueError(f"time bounds must be increasing, got ({lo}, {hi})")

    if plotly:
        fig = _plotly_density(
            x=t_ps,
            y=z,
            z_db=_db_clipped(intensity, dynamic_range_db),
            labels=temporal_feature_labels(evo, time_reversal=time_reversal),
            x_title="Time (ps)",
            y_title=z_label,
            title="Temporal evolution",
            dynamic_range_db=dynamic_range_db,
            annotate=annotate_features,
        )
        fig.update_xaxes(range=[lo, hi])
        return fig

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.imshow(
        density_clip(intensity, dynamic_range_db),
        origin="lower",
        aspect="auto",
        extent=(t_ps[0], t_ps[-1], z[0], z[-1]),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(z_label)
    return fig
