# Reproduction — Quarter-wave DBR stopband (Macleod / Born & Wolf)

**Reference:** H. A. Macleod, *Thin-Film Optical Filters* (4th ed.), Ch. 5; see
also M. Born and E. Wolf, *Principles of Optics*.

## Result reproduced

For a quarter-wave stack `(HL)^N` designed at `lambda0`, at `lambda0` every
layer has phase thickness `pi/2`, so one period has the diagonal characteristic
matrix `diag(-eta_L/eta_H, -eta_H/eta_L)`. The exact peak reflectance is

```
r = (eta0 (-etaL/etaH)^N - etas (-etaH/etaL)^N)
    / (eta0 (-etaL/etaH)^N + etas (-etaH/etaL)^N),   R = |r|^2,
```

and the first-order stopband full width is

```
Delta_lambda / lambda0 = (4/pi) arcsin((nH - nL)/(nH + nL)).
```

## Ground truth

The exact quarter-wave closed forms above, evaluated independently of the TMM.

## Outcome

```
N=2: R(lambda0)=0.593729 (closed form 0.593729)
N=3: R(lambda0)=0.829643 (closed form 0.829643)
N=4: R(lambda0)=0.935017 (closed form 0.935017)
N=20: R(lambda0)=1.000000
stopband width (R>0.5) = 533.1 nm, analytic = 498.7 nm (err 6.9%)
```

Figure: `dbr_spectrum.png`.

## Note / limitation

`TMM.spectrum` and `_reflection_coefficient` hard-code **air on the exit side**
(`eta_exit = 1`). A substrate index is not exposed; to model a mirror on glass
the substrate must be added as an explicit layer. The reproduction therefore
uses `n_substrate = 1.0`.
