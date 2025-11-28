import warnings
from .base import PI, WavelengthArray

from typing import List, Self, Tuple
from numpy.typing import NDArray
from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_splrep
from rich.traceback import install

install()


class RefractiveIndex:
    def __init__(self, n: NDArray, k: NDArray, wl: WavelengthArray) -> None:
        self._n = n
        self._k = k
        self._wl = wl

    @cached_property
    def n(self) -> NDArray:
        return self._n

    @cached_property
    def k(self) -> NDArray:
        return self._k

    @cached_property
    def wl(self) -> WavelengthArray:
        return self._wl

    @cached_property
    def nk(self) -> NDArray:
        return np.array(complex(self._n, self._k), dtype=np.complex128)

    @classmethod
    def from_complex(cls, nk: NDArray, wl: WavelengthArray) -> Self:
        return cls(n=np.real(nk), k=np.imag(nk), wl=wl)

    def n_func(self, wavelength: float):
        if wavelength < min(self._wl.as_um) or wavelength > max(self._wl.as_um):
            raise AttributeError(
                f"Index can be found only in between ({min(self._wl.as_um)}) and ({max(self._wl.as_um)})"
            )
        fn = make_splrep(self._wl.as_um, self._n)
        return fn(wavelength)

    def k_func(self, wavelength: float):
        if wavelength < min(self._wl.as_um) or wavelength > max(self._wl.as_um):
            raise AttributeError(
                f"Index can be found only in between ({min(self._wl.as_um)}) and ({max(self._wl.as_um)})"
            )
        fn = make_splrep(self._wl.as_um, self._k)
        return fn(wavelength)

    def nk_func(self, wavelength: float):
        if wavelength < min(self._wl.as_um) or wavelength > max(self._wl.as_um):
            raise AttributeError(
                f"Index can be found only in between ({min(self._wl.as_um)}) and ({max(self._wl.as_um)})"
            )
        fn = make_splrep(self._wl.as_um, self.nk)
        return fn(wavelength)

    def plot(self, include_k: bool = True):

        plt.plot(self._wl, self.n, label="n")
        plt.xlabel("wavelength [m]")
        plt.ylabel("n")

        if include_k:
            plt.plot(self._wl, self._k, label="k")
            plt.ylabel("n,k")
            plt.legend()

        plt.show()

    @classmethod
    def from_sellmeier(
        cls,
        A0: int | float,
        A: List[float],
        B: List[float],
        wl_from_to_in_um: Tuple[float, float],
        n_points=200,
    ) -> Self:
        if len(A) != len(B):
            raise ValueError("Length of A and B should be same")
        else:
            n = []
            wl = np.linspace(wl_from_to_in_um[0], wl_from_to_in_um[1], n_points)
            wls = WavelengthArray(wl, "um")
            for wl in wls.as_um:
                sum = 0.0
                for i in range(len(A)):
                    sum += A[i] * wl**2 / (wl**2 - B[i])
                n.append(np.sqrt(A0 + sum))
            k = np.zeros(len(wls))

        return cls(n=np.array(n), k=k, wl=wls)

    @classmethod
    def from_alt_sellmeier(
        cls,
        A0: int | float,
        A: List[float],
        B: List[float],
        wl_from_to_in_um: Tuple[float, float],
        n_points=200,
    ) -> Self:
        if len(A) != len(B):
            raise ValueError("Length of A and B should be same")
        else:
            n = []
            wl = np.linspace(wl_from_to_in_um[0], wl_from_to_in_um[1], n_points)
            wls = WavelengthArray(wl, "m")
            for wl in wls.as_um:
                sum = 0.0
                for i in range(len(A)):
                    sum += A[i] / (wl**2 - B[i] ** 2)
                n.append(np.sqrt(A0 + sum))
            k = np.zeros(len(wls))

        return cls(n=np.array(n), k=k, wl=wls)

    def propagation_loss(self):
        if not np.all(self.k):
            warnings.warn(
                "RefractiveIndex doesn't have imaginary index values. please provide it before loss calculation"
            )
        else:
            return -20 * np.log10(np.exp(-2 * PI * self.k / self.wl.as_m))
