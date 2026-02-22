from dataclasses import dataclass
from math import sqrt, log, acosh
from typing import Dict, Literal, Self
from functools import cached_property
from matplotlib import gridspec
from photonics_helper.base import Wavelength

import numpy as np
import matplotlib.pyplot as plt

SHAPE_FACTORS: Dict[str, float] = {
    "gaussian": 2 * sqrt(log(2)),
    "sech": 2 * acosh(sqrt(2)),
    "lorentzian": 2 * sqrt(sqrt(2) - 1),
    "rectangular": 2.0,
}


@dataclass
class Envelope:
    shape: Literal["gaussian", "sech", "lorentzian", "rectangular"]
    peak_amplitude: float
    pulse_width: float  # T0
    chirp: float = 0.0

    @property
    def fwhm(self) -> float:
        return SHAPE_FACTORS[self.shape] * self.pulse_width

    @classmethod
    def from_fwhm(
        cls,
        shape: Literal["gaussian", "sech", "lorentzian", "rectangular"],
        peak_amplitude: float,
        fwhm: float,
    ) -> Self:
        T0 = fwhm / SHAPE_FACTORS[shape]
        return cls(
            shape=shape,
            peak_amplitude=peak_amplitude,
            pulse_width=T0,
        )

    def field(self, t: np.ndarray) -> np.ndarray:
        """
        Returns complex envelope A(t)
        """

        T0 = self.pulse_width
        A0 = self.peak_amplitude

        match self.shape:
            case "gaussian":
                amp = A0 * np.exp(-(t**2) / (2 * T0**2))
            case "sech":
                amp = A0 / np.cosh(t / T0)
            case "lorentzian":
                """
                Lorentzian defined at amplitude level:
                """
                amp = A0 / (1 + (t / T0) ** 2)
            case "rectangular":
                amp = A0 * (np.abs(t) <= T0)

        phase = 0.5 * self.chirp * (t / T0) ** 2

        return amp * np.exp(1j * phase)

    def intensity(self, t: np.ndarray) -> np.ndarray:
        A = self.field(t)
        return np.abs(A) ** 2


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
