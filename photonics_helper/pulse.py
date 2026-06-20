from dataclasses import dataclass, field
from math import sqrt, log, acosh, factorial, pi
from typing import Callable, Dict, Literal, Self, Optional
from functools import cached_property
import logging
from matplotlib import gridspec
from photonics_helper.base import Wavelength

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy, hermite as hermite_poly, genlaguerre

logger = logging.getLogger(__name__)

SHAPE_FACTORS: Dict[str, float] = {
    "gaussian": 2 * sqrt(log(2)),
    "sech": 2 * acosh(sqrt(2)),
    "lorentzian": 2 * sqrt(sqrt(2) - 1),
    "rectangular": 2.0,
}


def _super_gaussian_factor(order: int = 2) -> float:
    """FWHM factor for super-gaussian: 2*T0*(log(2)/2)^(1/(2N))."""
    return 2.0 * T0 * (log(2) / 2) ** (1.0 / (2 * order))


def _triangular_factor() -> float:
    """FWHM factor for triangular: 2*T0*(1 - 1/sqrt(2))."""
    return 2.0 * (1.0 - 1.0 / sqrt(2))


def _cosine_factor() -> float:
    """FWHM factor for raised cosine: 2*T0*(1 - 1/sqrt(2))^(1/2)
    raised cos: cos(pi*t/(2*T0)) at half power.
    cos^2(pi*t/(2*T0)) = 0.5 -> cos(pi*t/(2*T0)) = 1/sqrt(2) -> pi*t/(2*T0) = pi/4 -> t = T0/2
    FWHM = 2*t = T0
    Wait let me recalculate.
    """
    # A(t) = cos(pi*t/(2*T0)) for |t| < T0
    # I(t) = cos^2(pi*t/(2*T0))
    # FWHM: cos^2(pi*t/(2*T0)) = 0.5
    # cos(pi*t/(2*T0)) = 1/sqrt(2)
    # pi*t/(2*T0) = pi/4
    # t = T0/2
    # FWHM = 2*t = T0
    return 1.0


def _exponential_factor() -> float:
    """FWHM factor for bi-exponential: 2*T0*log(2)."""
    return 2.0 * log(2)


def _airy_factor() -> float:
    """FWHM factor for Airy: first maximum of Ai function."""
    # The first maximum of Ai(-x) is at x ≈ 2.3381 (first zero of Ai).
    # Actually for the Airy pulse, the peak is at t=0, Ai(0) ≈ 0.3550.
    # FWHM: Ai(-x)/Ai(0) = sqrt(0.5), find x such that Ai(-x) = Ai(0)*sqrt(0.5)
    # Ai(0) ≈ 0.35503, Ai(0)*sqrt(0.5) ≈ 0.25104
    # Ai(-x) = 0.25104 -> x ≈ 1.0 (approximately)
    # Use numerical search
    from scipy.optimize import brentq
    target = 0.35503 * sqrt(0.5)
    f = lambda x: airy(-x)[0] - target
    root = brentq(f, 0.1, 2.0)
    return 2.0 * root


@dataclass
class Envelope:
    """Analytic description of an optical pulse envelope.

    Parameters
    ----------
    shape : The functional shape of the pulse. Supported values:
        ``"gaussian"``, ``"sech"``, ``"lorentzian"``, ``"rectangular"``,
        ``"super-gaussian"``, ``"triangular"``, ``"parabolic"``,
        ``"cosine"``, ``"exponential"``, ``"gauss-hermite"``,
        ``"airy"``, ``"custom"``.
    peak_amplitude : Peak amplitude of the electric-field envelope *A(t)*.
    pulse_width : Characteristic width *T₀* (the "1/e" width for a Gaussian, etc.).
    chirp : Linear chirp coefficient *C* (default ``0.0``). The instantaneous
        phase added to the envelope is ``0.5*C*(t/T₀)**2``.
    """

    shape: Literal[
        "gaussian", "sech", "lorentzian", "rectangular",
        "super-gaussian", "triangular", "parabolic", "cosine",
        "exponential", "gauss-hermite", "airy", "custom",
    ]
    peak_amplitude: float
    pulse_width: float  # T0
    chirp: float = 0.0

    # Shape-specific extra parameters
    super_gaussian_order: int = 2  # for "super-gaussian"
    beam_waist: Optional[float] = None  # for "gauss-hermite"
    hg_mode: int = 0  # Hermite polynomial mode index m
    func: Optional[Callable] = None  # for "custom": func(t, T0, A0)
    phase_func: Optional[Callable] = None  # for "custom": phase_func(t, T0, chirp)

    @property
    def fwhm(self) -> float:
        if self.shape == "super-gaussian":
            return 2.0 * self.pulse_width * (log(2) / 2) ** (1.0 / (2 * self.super_gaussian_order))
        elif self.shape == "triangular":
            return 2.0 * self.pulse_width * (1.0 - 1.0 / sqrt(2))
        elif self.shape == "cosine":
            return self.pulse_width
        elif self.shape == "exponential":
            return 2.0 * self.pulse_width * log(2)
        elif self.shape == "airy":
            from scipy.optimize import brentq
            # Ai(0) ≈ 0.35503, half-max of intensity = sqrt(0.5) * Ai(0) ≈ 0.25104
            target = 0.35503 * sqrt(0.5)
            f = lambda x: airy(-x)[0] - target
            root = brentq(f, 0.1, 2.0)
            return 2.0 * root * self.pulse_width
        elif self.shape in SHAPE_FACTORS:
            return SHAPE_FACTORS[self.shape] * self.pulse_width
        else:
            # custom, parabolic, gauss-hermite — return pulse_width as a lower-bound
            return self.pulse_width

    @classmethod
    def from_fwhm(
        cls,
        shape: Literal[
            "gaussian", "sech", "lorentzian", "rectangular",
            "super-gaussian", "triangular", "cosine", "exponential", "airy",
        ],
        peak_amplitude: float,
        fwhm: float,
    ) -> Self:
        """Construct an Envelope from a desired full-width at half-maximum."""
        if shape == "super-gaussian":
            T0 = fwhm / (2.0 * (log(2) / 2) ** (1.0 / (2 * 2)))  # default order=2
            return cls(
                shape="super-gaussian", peak_amplitude=peak_amplitude,
                pulse_width=T0, super_gaussian_order=2,
            )
        elif shape == "triangular":
            T0 = fwhm / (2.0 * (1.0 - 1.0 / sqrt(2)))
            return cls(shape="triangular", peak_amplitude=peak_amplitude, pulse_width=T0)
        elif shape == "cosine":
            T0 = fwhm  # FWHM = T0
            return cls(shape="cosine", peak_amplitude=peak_amplitude, pulse_width=T0)
        elif shape == "exponential":
            T0 = fwhm / (2.0 * log(2))
            return cls(shape="exponential", peak_amplitude=peak_amplitude, pulse_width=T0)
        elif shape == "airy":
            from scipy.optimize import brentq
            target = 0.35503 * sqrt(0.5)
            f = lambda x: airy(-x)[0] - target
            root = brentq(f, 0.1, 2.0)
            T0 = fwhm / (2.0 * root)
            return cls(shape="airy", peak_amplitude=peak_amplitude, pulse_width=T0)
        else:
            T0 = fwhm / SHAPE_FACTORS[shape]
            return cls(
                shape=shape, peak_amplitude=peak_amplitude, pulse_width=T0,
            )

    def field(self, t: np.ndarray) -> np.ndarray:
        """Returns complex envelope A(t)."""
        T0 = self.pulse_width
        A0 = self.peak_amplitude

        match self.shape:
            case "gaussian":
                amp = A0 * np.exp(-(t**2) / (2 * T0**2))
            case "sech":
                amp = A0 / np.cosh(t / T0)
            case "lorentzian":
                amp = A0 / (1 + (t / T0) ** 2)
            case "rectangular":
                amp = A0 * (np.abs(t) <= T0)
            case "super-gaussian":
                N = self.super_gaussian_order
                amp = A0 * np.exp(-(abs(t / T0) ** (2 * N)))
            case "triangular":
                amp = A0 * np.maximum(0.0, 1.0 - abs(t) / T0)
            case "parabolic":
                x = t / T0
                amp = A0 * np.where(abs(x) <= 1.0, 1.0 - x**2, 0.0)
            case "cosine":
                x = t / T0
                amp = A0 * np.where(abs(x) <= 1.0, np.cos(pi * x / 2), 0.0)
            case "exponential":
                amp = A0 * np.exp(-abs(t) / T0)
            case "gauss-hermite":
                w = self.beam_waist if self.beam_waist is not None else T0
                x = sqrt(2) * t / w
                H_m = hermite_poly(self.hg_mode)
                amp = A0 * H_m(x) * np.exp(-x**2 / 2)
            case "airy":
                # Standard Airy pulse: Ai(-(t-t0)/T0), t0=0, accelerating towards +t
                amp = A0 * airy(t / T0)[0]
            case "custom":
                if self.func is None:
                    raise ValueError("'custom' shape requires 'func' to be set")
                amp = self.func(t, T0, A0)
            case _:
                raise ValueError(f"Unknown shape: {self.shape}")

        # Apply phase
        if self.shape == "parabolic":
            # Parabolic chirp: phase = chirp * (t/T0)^2, active only where amp != 0
            x = t / T0
            phase = np.where(abs(x) <= 1.0, self.chirp * x**2, 0.0)
        elif self.shape == "custom":
            if self.phase_func is not None:
                phase = self.phase_func(t, T0, self.chirp)
            else:
                phase = 0.5 * self.chirp * (t / T0) ** 2
        else:
            phase = 0.5 * self.chirp * (t / T0) ** 2

        return amp * np.exp(1j * phase)

    def intensity(self, t: np.ndarray) -> np.ndarray:
        A = self.field(t)
        return np.abs(A) ** 2

    @classmethod
    def from_parabolic_asymptotic(
        cls,
        peak_amplitude: float,
        pulse_width: float,
        gain: float,
        length: float,
        chirp: float = 0.0,
    ) -> Self:
        """Construct a parabolic pulse with asymptotic amplifier chirp.

        The chirp coefficient follows the asymptotic parabolic solution:
        ``α ≈ 0.2726 × z × gain``.

        Parameters
        ----------
        peak_amplitude : A₀
        pulse_width : T₀
        gain : Small-signal gain coefficient (unitless or per-length as used in the amplifier model)
        length : Propagation length in the amplifier
        chirp : Additional chirp on top of the asymptotic value (default 0)
        """
        alpha_asym = 0.2726 * length * gain + chirp
        logger.info(
            "Using asymptotic parabolic chirp: α ≈ 0.2726 × z × gain = %.4f",
            alpha_asym,
        )
        return cls(
            shape="parabolic",
            peak_amplitude=peak_amplitude,
            pulse_width=pulse_width,
            chirp=alpha_asym,
        )


@dataclass
class TemporalGrid:
    N: int
    Tmax: float  # total time window

    @cached_property
    def dt(self):
        return self.Tmax / self.N

    @cached_property
    def t(self):
        dt = self.dt
        return np.linspace(-self.Tmax / 2, self.Tmax / 2 - dt, self.N)

    @cached_property
    def w(self):
        return np.fft.fftshift(2 * np.pi * np.fft.fftfreq(self.N, d=self.dt))

    @cached_property
    def dw(self):
        w = self.w
        return w[1] - w[0]

    def fft(self, A_t):
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_t))) * self.dt

    def ifft(self, A_w):
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A_w))) / self.dt

    @property
    def omega_max(self):
        return np.max(np.abs(self.w))

    @property
    def time_window(self):
        return self.N * self.dt

    @classmethod
    def for_pulse_train(
        cls,
        repetition_rate: float,
        n_pulses: int,
        pulse_width: float,
        N: int = 2**12,
    ) -> Self:
        """Compute the right Tmax to cover a pulse train.

        Parameters
        ----------
        repetition_rate : Hz — pulse spacing = 1 / repetition_rate
        n_pulses : number of pulses
        pulse_width : T₀ — characteristic width, used to estimate needed padding
        N : number of time points (default 2¹²)
        """
        spacing = 1.0 / repetition_rate
        # Window must cover all pulses + padding for tails
        Tmax = n_pulses * spacing + 10 * pulse_width
        return cls(N=N, Tmax=Tmax)

    def check_aliasing(self, T0):
        if self.time_window < 10 * T0:
            print("Warning: Time window too small")


@dataclass
class Wave:
    grid: TemporalGrid
    envelope: Envelope

    central_wavelength: Wavelength
    refractive_index: float = 1.0
    repetition_rate: float | None = None

    @cached_property
    def central_frequency(self) -> float:
        return self.central_wavelength.to_omega().as_rad_s

    @property
    def envelope_field(self):
        return self.envelope.field(self.grid.t)

    @property
    def electric_field(self):
        A = self.envelope_field
        return np.real(A * np.exp(-1j * self.central_frequency * self.grid.t))

    def instantaneous_intensity(self):
        E = self.electric_field
        return np.abs(E) ** 2

    @property
    def envelope_intensity(self):
        return self.envelope.intensity(self.grid.t)

    def pulse_energy(self):
        return np.sum(self.envelope.intensity(self.grid.t)) * self.grid.dt

    def peak_power(self):
        return np.max(self.envelope.intensity(self.grid.t))

    def average_power(self, repetition_rate: float) -> float:
        """Average power = pulse energy × repetition_rate.

        Parameters
        ----------
        repetition_rate : Hz
        """
        return self.pulse_energy() * repetition_rate

    @classmethod
    def from_pulse_train(
        cls,
        envelope: Envelope,
        central_wavelength: Wavelength,
        grid: TemporalGrid,
        repetition_rate: float,
        n_pulses: int = 10,
        refractive_index: float = 1.0,
    ) -> Self:
        """Construct a pulse train Wave from a single-envelope shape.

        Parameters
        ----------
        envelope : The single-pulse envelope shape to repeat
        central_wavelength : Central wavelength of the carrier
        grid : TemporalGrid covering the full window (all pulses + padding)
        repetition_rate : Hz — spacing between consecutive pulses
        n_pulses : number of pulses (default 10)
        refractive_index : background refractive index (default 1.0)
        """
        # Build the multi-pulse envelope field
        full_field = np.zeros_like(grid.t, dtype=complex)
        spacing = 1.0 / repetition_rate
        for k in range(n_pulses):
            t_centered = grid.t - k * spacing
            full_field += envelope.field(t_centered)

        # Create a temporary "virtual" envelope that stores the train flag
        train_env = Envelope(
            shape=envelope.shape,
            peak_amplitude=envelope.peak_amplitude,
            pulse_width=envelope.pulse_width,
            chirp=envelope.chirp,
            super_gaussian_order=envelope.super_gaussian_order,
            beam_waist=envelope.beam_waist,
            hg_mode=envelope.hg_mode,
            func=envelope.func,
            phase_func=envelope.phase_func,
        )
        train_env._is_pulse_train = True
        train_env._n_pulses = n_pulses
        train_env._repetition_rate = repetition_rate

        wave = cls(
            grid=grid,
            envelope=train_env,
            central_wavelength=central_wavelength,
            refractive_index=refractive_index,
            repetition_rate=repetition_rate,
        )
        # Override the envelope_field property via a wrapper
        wave._pulse_train_field = full_field
        return wave

    @property
    def envelope_field(self):
        if hasattr(self, '_pulse_train_field') and self._pulse_train_field is not None:
            return self._pulse_train_field
        return self.envelope.field(self.grid.t)

    @cached_property
    def spectrum(self) -> np.ndarray:
        return self.grid.fft(self.envelope_field)

    def time_bandwidth_product(self) -> float:
        """Equals ~0.44 for transform-limited Gaussian pulses."""
        intensity = self.envelope_intensity
        spectral_intensity = np.abs(self.spectrum) ** 2

        t, w = self.grid.t, self.grid.w
        dt, dw = self.grid.dt, self.grid.dw

        E_t = self.pulse_energy()
        t_mean = np.sum(t * intensity) * dt / E_t
        rms_t = sqrt(np.sum((t - t_mean) ** 2 * intensity) * dt / E_t)

        E_w = np.sum(spectral_intensity) * dw
        w_mean = np.sum(w * spectral_intensity) * dw / E_w
        rms_w = sqrt(np.sum((w - w_mean) ** 2 * spectral_intensity) * dw / E_w)

        return rms_t * rms_w

    def visualize(
        self,
        t_unit: str = "s",
        w_unit: str = "rad/s",
        t_scale: float = 1.0,
        w_scale: float = 1.0,
        show_electric_field: bool = False,
        show_phase: bool = True,
        show_spectrogram: bool = False,
        figsize: tuple[float, float] | None = None,
        save_path: str | None = None,
    ):
        """
        Plot a comprehensive overview of the pulse.

        Parameters
        ----------
        t_unit : label for the time axis (e.g. "ps", "fs")
        w_unit : label for the frequency axis (e.g. "THz", "rad/ps")
        t_scale : multiply grid.t by this before plotting (e.g. 1e12 for ps)
        w_scale : multiply grid.w by this before plotting
        show_electric_field : add a panel with the real electric field
        show_phase : overlay instantaneous phase on the temporal panel
        show_spectrogram : add a spectrogram (STFT) panel
        figsize : passed to plt.figure()
        save_path : if given, save figure to this path
        """

        t = self.grid.t * t_scale
        w = self.grid.w * w_scale
        intensity = self.envelope_intensity
        spectral_intensity = np.abs(self.spectrum) ** 2

        # Normalise for clean plotting
        norm_t = intensity / np.max(intensity)
        norm_w = spectral_intensity / np.max(spectral_intensity)

        # Instantaneous phase (unwrapped)
        phase = np.unwrap(np.angle(self.envelope_field))

        # ---- figure layout ----------------------------------------
        n_rows = 2
        if show_electric_field:
            n_rows += 1
        if show_spectrogram:
            n_rows += 1

        fig = plt.figure(
            figsize=figsize or (10, 3.5 * n_rows),
            facecolor="#0f0f0f",
        )
        gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.45, left=0.1, right=0.95)

        COLORS = {
            "intensity": "#00d4ff",
            "phase": "#ff6b6b",
            "spectrum": "#a78bfa",
            "efield": "#34d399",
            "grid": "#2a2a2a",
            "text": "#e0e0e0",
            "fwhm": "#fbbf24",
        }

        def _style_ax(ax, xlabel, ylabel, title):
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors=COLORS["text"], labelsize=9)
            ax.xaxis.label.set_color(COLORS["text"])
            ax.yaxis.label.set_color(COLORS["text"])
            ax.title.set_color(COLORS["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
            ax.grid(True, color=COLORS["grid"], linewidth=0.5, linestyle="--")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10, pad=6, fontweight="bold")

        row = 0

        # ---- 1. Temporal intensity (+ optional phase) ---------------
        ax_t = fig.add_subplot(gs[row])
        row += 1

        ax_t.fill_between(t, norm_t, alpha=0.25, color=COLORS["intensity"])
        ax_t.plot(
            t, norm_t, color=COLORS["intensity"], linewidth=1.8, label="Intensity"
        )

        # FWHM marker
        fwhm_val = self.envelope.fwhm * t_scale
        ax_t.axvspan(-fwhm_val / 2, fwhm_val / 2, alpha=0.08, color=COLORS["fwhm"])
        ax_t.axhline(0.5, color=COLORS["fwhm"], linewidth=0.8, linestyle=":")
        ax_t.annotate(
            f"FWHM = {fwhm_val:.3g} {t_unit}",
            xy=(fwhm_val / 2, 0.5),
            xytext=(fwhm_val / 2 + (t[-1] - t[0]) * 0.03, 0.55),
            color=COLORS["fwhm"],
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=COLORS["fwhm"], lw=0.8),
        )

        if show_phase:
            ax_ph = ax_t.twinx()
            ax_ph.plot(
                t,
                phase,
                color=COLORS["phase"],
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label="Phase",
            )
            ax_ph.set_ylabel(f"Phase (rad)", color=COLORS["phase"], fontsize=9)
            ax_ph.tick_params(colors=COLORS["phase"], labelsize=9)
            ax_ph.spines["right"].set_edgecolor(COLORS["phase"])

        _style_ax(
            ax_t,
            f"Time ({t_unit})",
            "Normalised Intensity",
            f"Temporal Profile  [{self.envelope.shape}]",
        )

        tbp = self.time_bandwidth_product()
        ax_t.text(
            0.02,
            0.92,
            f"TBP = {tbp:.3f}   |   Peak power = {self.peak_power():.3g} W",
            transform=ax_t.transAxes,
            color=COLORS["text"],
            fontsize=8,
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333", boxstyle="round,pad=0.3"),
        )

        # ---- 2. Spectral intensity -----------------------------------
        ax_w = fig.add_subplot(gs[row])
        row += 1

        ax_w.fill_between(w, norm_w, alpha=0.25, color=COLORS["spectrum"])
        ax_w.plot(w, norm_w, color=COLORS["spectrum"], linewidth=1.8)

        # Spectral FWHM (numerical)
        half_max_mask = norm_w >= 0.5
        if half_max_mask.any():
            w_fwhm = w[half_max_mask]
            spec_fwhm = w_fwhm[-1] - w_fwhm[0]
            ax_w.axvspan(w_fwhm[0], w_fwhm[-1], alpha=0.08, color=COLORS["fwhm"])
            ax_w.text(
                0.02,
                0.92,
                f"Spectral FWHM ≈ {spec_fwhm:.3g} {w_unit}",
                transform=ax_w.transAxes,
                color=COLORS["text"],
                fontsize=8,
                bbox=dict(
                    facecolor="#1a1a1a", edgecolor="#333", boxstyle="round,pad=0.3"
                ),
            )

        _style_ax(
            ax_w, f"Angular frequency ({w_unit})", "Normalised PSD", "Spectral Profile"
        )

        # ---- 3. Electric field (optional) ---------------------------
        if show_electric_field:
            ax_e = fig.add_subplot(gs[row])
            row += 1
            E = self.electric_field
            E_norm = E / np.max(np.abs(E))
            ax_e.plot(t, E_norm, color=COLORS["efield"], linewidth=0.8, alpha=0.9)
            ax_e.plot(
                t,
                norm_t,
                color=COLORS["intensity"],
                linewidth=1.2,
                linestyle="--",
                alpha=0.6,
                label="Envelope",
            )
            ax_e.plot(
                t,
                -norm_t,
                color=COLORS["intensity"],
                linewidth=1.2,
                linestyle="--",
                alpha=0.6,
            )
            _style_ax(ax_e, f"Time ({t_unit})", "Normalised E-field", "Electric Field")

        # ---- 4. Spectrogram (optional) ------------------------------
        if show_spectrogram:
            ax_sg = fig.add_subplot(gs[row])
            row += 1

            # Use a window ~1/5 of pulse width for STFT
            win_size = max(16, int(self.envelope.pulse_width / self.grid.dt / 5))
            win_size = min(win_size, self.grid.N // 4)
            # Make even
            win_size = win_size if win_size % 2 == 0 else win_size + 1

            from scipy.signal import spectrogram as scipy_spectrogram

            f_sg, t_sg, Sxx = scipy_spectrogram(
                self.envelope_field,
                fs=1.0 / self.grid.dt,
                window="hann",
                nperseg=win_size,
                noverlap=win_size * 3 // 4,
                mode="complex",
            )
            t_sg_scaled = (t_sg - self.grid.Tmax / 2) * t_scale
            f_sg_scaled = np.fft.fftshift(f_sg) * 2 * np.pi * w_scale

            im = ax_sg.pcolormesh(
                t_sg_scaled,
                np.fft.fftshift(f_sg_scaled),
                np.fft.fftshift(np.abs(Sxx) ** 2, axes=0),
                shading="gouraud",
                cmap="inferno",
            )
            fig.colorbar(im, ax=ax_sg, label="Power", fraction=0.03, pad=0.02)
            _style_ax(
                ax_sg, f"Time ({t_unit})", f"Frequency ({w_unit})", "Spectrogram (STFT)"
            )

        fig.suptitle(
            f"Pulse Analysis  —  λ-equivalent  |  chirp = {self.envelope.chirp}",
            color=COLORS["text"],
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        if save_path:
            fig.savefig(
                save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
            )

        return fig
