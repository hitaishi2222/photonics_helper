# import photonics_helper as ph
#
# pulse = ph.Pulse(50, "fs", 1000)
# print(pulse.duration)
# print(pulse.period)
# print(pulse.rate)
#
# rect_pulse = ph.RectangularPulse(100, "fs", 500)
# print(rect_pulse.amplitude)


from typing import Literal, Union

import numpy
from numpy.typing import NDArray


class Wl:
    def __init__(self, value: Union[float, NDArray], unit: Literal["m", "nm"]) -> None:
        self.value = value
        self.unit = unit


def main():
    wl = Wl(200, "nm")
    print(wl.value, type(wl.value))
    wl2 = Wl(numpy.array([200, 300, 400]), "nm")
    print(wl2.value, type(wl2.value))


if __name__ == "__main__":
    main()
