import numpy as np

from photonics_helper.helper import WavelengthArray
from photonics_helper.materials import RefractiveIndex


def main():
    x = np.arange(0.6e-6, 1.7e-6, 0.1e-6)
    y = np.linspace(1, 10, len(x))

    wl = WavelengthArray(x, "m")
    ri = RefractiveIndex(y, np.zeros(len(x)), wl)
    ri.plot(False)


if __name__ == "__main__":
    main()
