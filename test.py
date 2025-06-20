import photonics_helper as ph

pulse = ph.Pulse(50, "fs", 1000)
print(pulse.duration)
print(pulse.period)
print(pulse.rate)

rect_pulse = ph.RectangularPulse(100, "fs", 500)
print(rect_pulse.amplitude)
