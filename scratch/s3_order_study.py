"""Stage S3 gate: order study of scheme ∈ {strang, yoshida4, etdrk4} on the
N=3-soliton deck (β₂=−21 ps²/km, γ=2, T₀≈28 fs, 1550 nm), ladder dz = L/{200,...,3200}.
Same methodology as PLAN §17.6 Stage S0/S1 (order_study, fixed grid, step-only ladder)."""
import warnings, json, sys
warnings.filterwarnings("ignore")
import numpy as np
from photonics_helper.base import Length, Time, Wavelength, C_MS, Area
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec
from photonics_helper.gnlse_validation import order_study

WL0=1550e-9; OM=2*np.pi*C_MS/WL0; B2=-21e-27; G=2.0; T0=50e-15/1.763; N=4096
L = 2.0*(np.pi/2)*T0**2/abs(B2)

def build(scheme, raman, shock):
    def factory(step_size=None, **kw):
        g=TemporalGrid(N=N,Tmax=Time(1.2e-11,"s"))
        r=RamanResponse(spec=RamanSpec(name="Silica",raman_shift_cm=440,raman_linewidth_cm=45,fR=0.18),grid=g) if raman else None
        peak=9*abs(B2)/(G*T0**2)
        env=Envelope(shape="sech",peak_amplitude=np.sqrt(peak),pulse_width=Time(T0,"s"))
        pulse=Wave(grid=g,envelope=env,central_wavelength=Wavelength(WL0,"m"))
        fib=FiberProfile(n2=2.6e-20,alpha=0.0,A_eff=Area(80e-12,"m^2"),length=Length(L,"m"),raman_response=r)
        # step ladder given as num_steps-like L/{200..3200}: use fixed step_size entries
        e=SplitStepEngine(pulse=pulse,fiber=fib,betas=np.array([B2*1e24]),
            include_raman=raman,include_self_steepening=shock,
            step_size=Length(step_size,"m"),scheme=scheme)
        nsteps=int(round(L/step_size))
        e.propagate(num_steps=nsteps)
        return e
    return factory

ladder=[L/k for k in (25,50,100,200,400)]
out={}
for label,(raman,shock) in {"A. Kerr only":(False,False),"B. + Raman":(True,False),"C. + Raman + shock":(True,True)}.items():
    for scheme in ("strang","yoshida4","etdrk4"):

        rep=order_study(build(scheme,raman,shock), ladder, shared={})
        out[f"{label}|{scheme}"]={"order":rep.order,"note":str(rep)}
        print(label, scheme, "order:", rep.order, flush=True)
json.dump(out, open("/tmp/s3_order.json","w"))
