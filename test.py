import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
t_max = 5e-12  # Total time window (±2.5 ps)
n_points = 2**14  # Number of time points

t = np.linspace(-t_max / 2, t_max / 2, n_points)
dt = t[1] - t[0]  # Time step
# Pulse parameters
tau = 100e-15  # Pulse width (100 fs)
A0 = 1  # Peak amplitude
chirp = 0  # Set to nonzero to include chirp, e.g., chirp = 1e28

# Envelope of the pulse
envelope = A0 * np.exp(-(t**2) / (2 * tau**2)) * np.exp(1j * chirp * t**2)
carrier_freq = 200e12  # 200 THz (optical frequency)
carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)

# Total field
E_t = envelope * carrier
plt.figure(figsize=(10, 4))
plt.plot(t * 1e15, np.real(E_t), label="Re[E(t)]")
plt.plot(t * 1e15, np.abs(envelope), label="|Envelope|", linestyle="--")
plt.xlabel("Time (fs)")
plt.ylabel("Amplitude")
plt.title("Gaussian Optical Pulse")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

E_f = np.fft.fftshift(np.fft.fft(E_t))
freq = np.fft.fftshift(np.fft.fftfreq(n_points, dt))

plt.figure(figsize=(10, 4))
plt.plot(freq * 1e-12, np.abs(E_f) / np.max(np.abs(E_f)))
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Spectrum")
plt.title("Optical Pulse Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()
