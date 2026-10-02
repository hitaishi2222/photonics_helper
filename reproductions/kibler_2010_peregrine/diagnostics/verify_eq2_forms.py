"""Scratch: verify candidate forms of Kibler-2010 Eq. (2) against the NLSE.

Paper Eq. (1): i psi_xi + 0.5 psi_tautau + |psi|^2 psi = 0  (self-focusing).
a = 0.5*(1 - Om^2/4)  =>  Om = 2*sqrt(1-2a);  b = [8a(1-2a)]^(1/2).

Anchors: peak/background at max compression (xi=0) must be:
  a=0.25 -> (1+sqrt(4a))^2? no: known AB ratio 5.8284 for a=0.25
  a->0.5 -> 9  (Peregrine).
Anchor values from the AB literature: |psi(0,0)|^2 = 4a^2 * ( ) ... test forms.
"""

import numpy as np

L = 20 * np.pi
N = 8192
tau = np.linspace(-L, L, N, endpoint=False)
w = 2 * np.pi * np.fft.fftfreq(N, d=L / N)


def make_psi(vec_xi, tau, a, variant):
    Om = 2 * np.sqrt(1 - 2 * a)
    b = np.sqrt(8 * a * (1 - 2 * a))
    X = vec_xi[:, None]
    cw = np.cosh(b * X)
    sw = np.sinh(b * X)
    c = np.cos(Om * tau)[None, :]
    s = np.sin(Om * tau)[None, :]
    if variant == "printed":
        num = (
            (1 - 4 * a) * cw
            + 1j * np.sqrt(2 * a) * np.cos(2 * Om * tau)[None, :]
            + np.sqrt(2 * a) * c
        )
        den = np.sqrt(2 * a) * c - cw
    elif variant == "sinh_xi":
        num = (1 - 4 * a) * cw + 1j * np.sqrt(2 * a) * sw + np.sqrt(2 * a) * c
        den = np.sqrt(2 * a) * c - cw
    elif variant == "std_aki":
        # Standard Akhmediev breather (e.g. Kibler 2012 Sci.Rep Eq (1) with their
        # parametrisation; equivalently Närhi-repro D):
        #    psi = e^{ixi} * [ cosh(bX) + i*sinh(bX)/? ... ]
        # Implement: psi = e^{ixi}[a cos(Om t) - (1-2a)/(...)...]  -- fill below
        # Known-correct standard AB (Wikipedia form):
        #   psi = e^{ixi} [ b*sqrt(2a)*cosh? ] -- derive instead:
        # AB satisfies psi(0,tau) = e^{ixi}*[1 - 4(1-2a)/(1+4(1-2a) a tau^2) ...]
        # Simplest robust: build AB via MODULATION of CW: known closed form:
        #   psi = e^{ixi} * [ cosh(gamma X)*cos(p t) + i sinh(gamma X)*sin(gamma? ...
        # use: psi = e^{ixi} sqrt(2a)*cos(Om t) + ... no — use the documented one:
        #   psi = e^{ixi} * [ (Om cosh(bX) + i (1-2a)^{1/2} sinh(bX)) / (sqrt(2a) cos(Om t) - cosh(bX)) ] * 2a?
        num = (
            np.sqrt(2 * a) * c * cw
            + 1j * (np.sqrt(1 - 2 * a)) * s_den_helper(vec_xi, a)[0]
        )
        raise NotImplementedError("placeholder")
    return (num / den) * np.exp(1j * X)


def s_den_helper(vec_xi, a):
    return np.sinh(np.sqrt(8 * a * (1 - 2 * a)) * vec_xi[:, None])


def residual(p0, p1, p2, dxi):
    dpsi = (p2 - p0) / (2 * dxi)
    ptt = np.fft.ifft(-(w**2) * np.fft.fft(p1))
    return 1j * dpsi + 0.5 * ptt + np.abs(p1) ** 2 * p1


def r122ron():
    # AB known-correct closed form (Akhmediev & Korneev 1986, Eq. 10):
    #   psi = e^{ixi} * [ (1 - 2a) + sqrt(2a) cos(Om tau) cosh(b x)
    #                     + i sinh(b x) * sqrt(2a(1-2a)) *?  ] / ...
    pass


# Direct approach: construct AB from Darboux / known closed form used in
# Hammani et al. 2011 (the follow-up): psi = e^{ixi} [ A cos(Om t) + i B sinh(bx) ] / [ C cosh(bx) - cos(Omt) ]
# Peak at (0,0): |psi|^2 = (1+4a).  Period tau_per = pi/Om * 2? etc.
# Let psi = e^{ixi} (cosh(B x) + i * K * sinh(B x) ... - D cos(Om t)) / (cosh - E cos)
