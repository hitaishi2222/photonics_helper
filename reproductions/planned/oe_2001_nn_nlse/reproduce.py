"""Reproduction: solving the NLSE with an unsupervised neural network
(Monterola & Saloma 2001).

Reference
---------
C. Monterola & C. Saloma, "Solving the nonlinear Schrödinger equation with
an unsupervised neural network", Opt. Express 9, 72 (2001),
doi:10.1364/OE.9.000072 (author-supplied PDF, gitignored; page refs in
parameters.json).

What is reproduced
------------------
The paper (a pre-PINN ancestor of Raissi 2019) solves the fibre NLSE

    -j dPsi/dz + (beta/2) d2Psi/dt2 - gamma |Psi|^2 Psi = 0      (Eq. 4)

(t in pulse-width units, z in propagation-length units) with an
*unsupervised* two-output MLP: outputs (Psi_R, Psi_I), hidden units tanh
(paper H = 42), energy = |F|^2 + |C1|^2 + |C2|^2 (PDE residual +
initial-condition + far-field decay, Eq. 7).  Gaussian input
Psi0 = exp(-t^2/2); training domain z in [0, 5], t core [-3, 3] plus the
far-field wings [-27, -10] u [10, 27] (paper Sec. 4.1).

Physics mapping (asserted below): with T0 = 1 ps, L = 1 m and
beta2 = beta * T0^2 / L = beta * 1e-24 s^2/m, the engine's convention
A_z = -i(beta2/2) A_TT + i gamma |A|^2 A IS paper Eq. (4); gamma = 1
(normalized) maps to 1 W^-1 m^-1 via A_eff = 2 pi n2 / (lambda0 gamma).
gamma = 0 (case A) uses A_eff = const.

Checks
------
1. ORACLE LAYER (float64, asserted first, ~seconds):
   a. reproduction-local split-step vs the paper's analytic case-A
      solution at z = 5: rel-L2 < 1e-6;
   b. same vs the paper's analytic case-B (SPM) solution: < 1e-6;
   c. photonics_helper engine vs the local oracle on all three cases:
      rel-L2 < 1e-4 (FFT window + step discretization budgeted).
2. PINN case A (beta = 1, gamma = 0): NMSE <= 1e-4 vs the analytic
   dispersive solution (paper: xi = 7.07e-5 after 200 adaptive-gradient
   iterations; the Sec.-4.2 NMSE definition is used verbatim).
3. PINN case B (beta = 0, gamma = 1): NMSE <= 1e-4 (paper: xi = 2.87e-6).
4. PINN case C (beta = 1, gamma = 1): rel-L2 <= 5e-3 vs the engine-oracle
   snapshots (no closed form; the paper compares this case against a
   finite-difference reference).

NMSE = sum |Psi_ref - Psi^q|^2 / sum |Psi_ref|^2 over 2e4 test datapoints,
z sampled uniformly in [0, 5] (cases A, B) or from the engine snapshot
z-values (case C), t uniformly in [-6, 6] (generalization band).

Usage
-----
    python reproductions/planned/oe_2001_nn_nlse/reproduce.py
        [--cases A,B,C] [--device {cpu,cuda,auto}] [--adam-iters N]
        [--fast]

Device guardrails follow the Raissi-2019 reproduction: this rig's ROCm
iGPU is unsafe for float64 training; default cpu (ISSUES.md #7).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from photonics_helper.gnlse import FiberProfile
from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"

T0_S = 1.0e-12  # normalized t = 1  <->  1 ps
L_M = 1.0  # normalized z = 1  <->  1 m

CASES: dict[str, tuple[float, float]] = {
    "A": (1.0, 0.0),
    "B": (0.0, 1.0),
    "C": (1.0, 1.0),
}

T_N_MAX = 27.0  # normalized t extent (paper far-field boundary)
EVAL_T_MAX = 6.0  # NMSE evaluation band (paper Sec. 4.2 band)
Z_N_MAX = 5.0
ENGINE_T_WINDOW_N = 54.0  # engine window: normalized t in [-27, 27)
ENGINE_N_T = 8192
ENGINE_N_Z = 5000
ENGINE_N_SAVES = 601

PAPER_QUOTED = {"A": 7.07e-5, "B": 2.87e-6}

SEED = 20010702


# ---------------------------------------------------------------------------
# analytic solutions of paper Eq. (4)
# ---------------------------------------------------------------------------


def psi_exact_A(t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(1 - j z)^(-1/2) exp[ -t^2 / (2 (1 - j z)) ] (paper Sec. 4.3)."""
    denom = 1.0 - 1j * np.asarray(z, float)
    return np.power(denom, -0.5) * np.exp(-(np.asarray(t, float) ** 2) / (2.0 * denom))


def psi_exact_B(t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """exp(-t^2/2) exp( j z exp(-t^2) ) (paper Sec. 4.3, SPM case)."""
    amp = np.exp(-(np.asarray(t, float) ** 2) / 2.0)
    return amp * np.exp(1j * np.asarray(z, float) * amp**2)


EXACT_FN = {"A": psi_exact_A, "B": psi_exact_B}


# ---------------------------------------------------------------------------
# reproduction-local float64 split-step oracle (paper Eq. 4, norm. units)
# ---------------------------------------------------------------------------


def script_grid(n_t: int = ENGINE_N_T) -> tuple[np.ndarray, np.ndarray]:
    """Normalized t grid on [-T_N_MAX, T_N_MAX), plus matching rad-grid."""
    t = np.linspace(-T_N_MAX, T_N_MAX, n_t, endpoint=False)
    dt_norm = t[1] - t[0]
    omega_norm = 2.0 * np.pi * np.fft.fftfreq(n_t, d=dt_norm)
    return t, omega_norm


def script_oracle(
    beta: float, gamma: float, n_t: int = ENGINE_N_T, n_z: int = 6000
) -> tuple[np.ndarray, np.ndarray]:
    """Split-step of paper Eq. (4); returns (psi_final, t_norm).

    psi_z = -i (beta/2) psi_tt  ->  linear multiplier exp(+i beta Omega^2 dz/2)
    psi_z = +i gamma |psi|^2 psi        ->  exp(+i gamma |psi|^2 dz).
    """
    t, om = script_grid(n_t)
    psi = np.exp(-0.5 * t**2).astype(complex)
    dz = Z_N_MAX / n_z
    expL = np.exp(1j * (beta / 2.0) * om**2 * dz)
    for _ in range(n_z):
        psi = np.fft.ifft(np.fft.fft(psi) * expL)
        psi = psi * np.exp(1j * gamma * np.abs(psi) ** 2 * dz)
    return psi, t


def engine_oracle(beta: float, gamma: float) -> tuple:
    """photonics_helper engine on the mapped (T0, L, beta2) grid.

    Returns (H [n_saves, n_t] complex, t_norm, z_norm).
    """
    from photonics_helper.gnlse import GNLSESolver

    grid = TemporalGrid(N=ENGINE_N_T, Tmax=Time(ENGINE_T_WINDOW_N * T0_S, "s"))
    n2_use = 3.0e-20 if gamma > 0 else 1.0e-30
    a_eff = (
        2.0 * np.pi * n2_use / (1550.0e-9 * gamma) if gamma > 0 else 1.0e-12
    )  # case A: bounded A_eff, gamma_eff ~ 4e-11 /W/m
    fiber = FiberProfile(
        n2=n2_use,
        alpha=0.0,
        A_eff=Area(a_eff, "m^2"),
        length=Length(Z_N_MAX * L_M, "m"),
    )
    t_norm = np.asarray(grid.t, float) / T0_S
    field = np.exp(-0.5 * t_norm**2).astype(complex)
    wave = Wave(
        grid=grid,
        envelope=Envelope(
            shape="gaussian", peak_amplitude=1.0, pulse_width=Time(T0_S, "s")
        ),
        central_wavelength=Wavelength(1550.0, "nm"),
    ).with_field(field)
    solver = GNLSESolver(
        pulse=wave,
        fiber=fiber,
        betas=np.array([beta * 1.0]),  # ps^k/m native, beta normalized
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(ENGINE_N_Z, nsaves=ENGINE_N_SAVES)
    H = np.array([w.envelope_field for w in solver.evolution])
    return H, t_norm, np.asarray(solver.z_array, float) / L_M


def rel_l2(err: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(err) / np.linalg.norm(ref))


def nmse(psi_q: np.ndarray, psi_ref: np.ndarray) -> float:
    """Paper Sec. 4.2 definition."""
    return float(np.sum(np.abs(psi_ref - psi_q) ** 2) / np.sum(np.abs(psi_ref) ** 2))


# ---------------------------------------------------------------------------
# PINN (torch, float64, unsupervised — the paper's energy function)
# ---------------------------------------------------------------------------


def _torch():
    try:
        import torch
    except ImportError:
        raise ImportError("requires torch: pip install torch")
    return torch


def _gpu_guardrails(torch, device: str) -> str:
    """ROCm/amdgpu float64 guardrails (ISSUES.md #7; Raissi-2019 harness).

    Memory cap + event cache flush around training; callers catch OOM and
    fall back to cpu. Returns the effective device.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return "cpu" if not torch.cuda.is_available() else device
    props = torch.cuda.get_device_properties(0)
    cap = float(os.environ.get("PH_GPU_CAP_FRAC", "0.6"))
    try:
        torch.cuda.set_per_process_memory_fraction(cap, 0)
    except Exception:
        pass
    print(
        f"  [guardrail] GPU {getattr(props, 'name', '?')} capped at "
        f"{cap:.2f} of VRAM (shared system RAM device!) — OOM falls "
        "back to cpu"
    )
    return device


def train_pinn(
    case: str,
    *,
    adam_iters: int = 30000,
    seed: int = SEED,
    device: str = "cpu",
    fast: bool = False,
    resume: bool = False,
) -> tuple:
    """Train the unsupervised PINN; return (torch.Module net, curves).

    Net: 4 tanh layers x 42 nodes, two linear outputs merged into
    (u, v) = (Psi_R, Psi_I).
    Trial ansatz: Psi^q = alpha(t) + s(z) * N(z, t)
        alpha(t) = exp(-t^2/2)  (Psi(0, t) = Psi0 EXACT, C1 = 0),
        s(z) = 1 - exp(-3z)     (s(0) = 0),
        N(z, t) -> (u, v) MLP.
    Energy: E = <|F|^2> + <|Psi|^2 + |dPsi/dt|^2>_{t = +-27}   (C2 term)

    Checkpointing: the net + optimizer state + curves are saved every
    ``CKPT_EVERY`` Adam iterations and after the L-BFGS refinement to
    ``oe2001_checkpoint_<case>.pt``; with ``resume=True`` (for crashed
    runs) the Adam phase restarts from the last checkpoint and only the
    remaining iterations run.
    """
    torch = _torch()
    dtype = torch.float64
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _gpu_guardrails(torch, device.strip().lower())
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # the guardrail cap MUST apply to any cuda training, including the
        # auto-resolved path (ISSUES #7: the amdgpu failed once before)
        device = _gpu_guardrails(torch, "cuda")
    if fast:
        adam_iters = min(adam_iters, 3000)

    beta, gamma = CASES[case]
    nf, width, n_far = (20000, 42, 200) if not fast else (6000, 32, 100)
    print(f"  [{case}] PINN on {device}: 4x{width} tanh, Nf = {nf}, Adam {adam_iters}")

    # collocation cloud: core band + far-field wings (paper Sec. 4.1)
    n_core = int(0.7 * nf)
    n_wing = nf - n_core
    t_core = rng.uniform(-3.0, 3.0, n_core)
    t_wing = np.concatenate(
        [
            rng.uniform(-T_N_MAX, -10.0, n_wing // 2),
            rng.uniform(10.0, T_N_MAX, n_wing - n_wing // 2),
        ]
    )
    tf = np.concatenate([t_core, t_wing])
    zf = rng.uniform(0.0, Z_N_MAX, nf)
    z_far = rng.uniform(0.0, Z_N_MAX, n_far)
    t_far = T_N_MAX * np.where(rng.integers(0, 2, n_far) == 0, 1.0, -1.0)

    def tens(a):
        return torch.tensor(np.asarray(a, float), dtype=dtype, device=device).reshape(
            -1, 1
        )

    tf_t, zf_t = tens(tf), tens(zf)
    tfr_t, zfr_t = tens(t_far), tens(z_far)

    mods: list = []
    din = 2
    for _ in range(4):
        lin = torch.nn.Linear(din, width, dtype=dtype)
        torch.nn.init.xavier_normal_(lin.weight)
        mods += [lin, torch.nn.Tanh()]
        din = width
    mods.append(torch.nn.Linear(din, 2, dtype=dtype))
    torch.nn.init.xavier_normal_(mods[-1].weight)
    net = torch.nn.Sequential(*mods).to(device)

    def alpha(t_t):
        return torch.exp(-0.5 * t_t**2)

    def s_of(z_t):
        return 1.0 - torch.exp(-3.0 * z_t)

    def psi_of(t_t, z_t):
        out = net(torch.cat([t_t / T_N_MAX, z_t / Z_N_MAX], dim=1))
        u, v = out[:, 0:1], out[:, 1:2]
        sc = s_of(z_t)
        return alpha(t_t) + sc * u, sc * v

    def residual(t_t, z_t):
        t_t = t_t.clone().requires_grad_(True)
        z_t = z_t.clone().requires_grad_(True)
        u, v = psi_of(t_t, z_t)
        n2v = u * u + v * v
        grads = []
        for y in (u, v):
            g_z = torch.autograd.grad(y.sum(), z_t, create_graph=True)[0]
            g_t = torch.autograd.grad(y.sum(), t_t, create_graph=True)[0]
            g_tt = torch.autograd.grad(g_t.sum(), t_t, create_graph=True)[0]
            grads.append((g_z, g_tt))
        (u_z, u_tt), (v_z, v_tt) = grads
        f_u = v_z + 0.5 * beta * u_tt - gamma * n2v * u
        f_v = -u_z + 0.5 * beta * v_tt - gamma * n2v * v
        return f_u, f_v

    def loss_fn():
        fu, fv = residual(tf_t, zf_t)
        mse_f = (fu**2 + fv**2).mean()
        tc = tfr_t.clone().requires_grad_(True)
        uf, vf = psi_of(tc, zfr_t)
        du = torch.autograd.grad(uf.sum() + vf.sum(), tc, create_graph=True)[0]
        mse_far = (uf**2 + vf**2 + du**2).mean()
        return mse_f + mse_far

    # cheap NMSE probe (analytic cases A/B only); case C compares against
    # the engine snapshots and skips the probe
    rngp = np.random.default_rng(seed + 7)
    zp_g = np.repeat(rngp.uniform(0.0, Z_N_MAX, 250), 250)
    tp_g = np.tile(rngp.uniform(-EVAL_T_MAX, EVAL_T_MAX, 250), 250)
    ref_probe = EXACT_FN.get(case)

    def probe_nmse() -> float:
        if ref_probe is None:
            return float("nan")
        with torch.no_grad():
            u, v = psi_of(tens(tp_g), tens(zp_g))
            psi_q = (u + 1j * v).cpu().numpy().reshape(-1)
        return nmse(psi_q, ref_probe(tp_g, zp_g))

    opt = torch.optim.Adam(list(net.parameters()), lr=1e-3)
    loss_curve: list[float] = []
    nmse_curve: list[tuple[int, float]] = []
    # --- checkpointing (crash-resilient, Raissi-2019 folder pattern) ------
    CKPT_EVERY = 2500 if not fast else 1500
    ckpt_path = HERE / f"oe2001_checkpoint_{case}.pt"

    def state():
        return {
            "net": net.state_dict(),
            "opt": opt.state_dict(),
            "it": it_now,
            "adam_target": adam_iters,
            "loss_curve": loss_curve,
            "nmse_curve": nmse_curve,
        }

    it_now = 0
    start_it = 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        it_now = start_it = int(ck["it"])
        loss_curve = list(ck.get("loss_curve", []))
        nmse_curve = list(ck.get("nmse_curve", []))
        print(
            f"  [{case}] resumed from checkpoint at Adam it {start_it} "
            f"(target {adam_iters})"
        )
    elif resume:
        print(
            f"  [{case}] resume requested but no checkpoint found — "
            "training from scratch"
        )
    pbar = tqdm(
        range(start_it, adam_iters),
        desc=f"pinn[{case}] Adam",
        unit="it",
        initial=start_it,
        total=adam_iters,
        leave=True,
    )
    for it in pbar:
        it_now = it
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if it % 250 == 0:
            for g in opt.param_groups:
                g["lr"] = max(1e-3 * 0.99 ** (it // 250), 3e-5)
            n_probe = probe_nmse()
            loss_curve.append(float(loss))
            nmse_curve.append((it, n_probe))
            pbar.set_postfix(E=f"{float(loss):.3e}", NMSE=f"{n_probe:.3e}")
        if device == "cuda" and it % 200 == 0:
            torch.cuda.empty_cache()
        if (it + 1) % CKPT_EVERY == 0 or it == adam_iters - 1:
            torch.save(state(), ckpt_path)
            pbar.set_postfix_str(f"checkpointed @ it {it}")
    pbar.close()

    # L-BFGS refinement with closure-count-corrected iteration counting
    # (ISSUES.md #6 fix: torch's LBFGS counts closures, not iterations; a
    #   naive max_iter budget exits the strong-Wolfe loop early.  Count
    #   ACCEPTED iterations by tracking the optimizer state steps.)
    lbfgs = torch.optim.LBFGS(
        list(net.parameters()),
        lr=0.8,
        max_iter=50,
        max_eval=100,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        history_size=60,
        line_search_fn="strong_wolfe",
    )
    n_lbfgs_target = 3000
    pbar2 = tqdm(
        range(n_lbfgs_target), desc=f"pinn[{case}] LBFGS", unit="it", leave=True
    )
    accepted = 0

    def n_iter_counter():
        outs = []
        for g in lbfgs.param_groups:
            for p in g["params"]:
                st = lbfgs.state.get(p, {})
                if "n_iter" in st:
                    outs.append(int(st["n_iter"]))
        return max(outs) if outs else 0

    prev_n = n_iter_counter()
    while accepted < n_lbfgs_target:

        def closure():
            lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        lbfgs.step(closure)
        cur_n = n_iter_counter()
        accepted += max(cur_n - prev_n, 1)
        prev_n = cur_n
        # NOTE: no torch.no_grad() here — loss_fn builds autograd graphs
        # internally (clone().requires_grad_ + grad with create_graph);
        # float() then detaches naturally.
        loss_now = float(loss_fn())
        loss_curve.append(loss_now)
        cur = probe_nmse()
        nmse_curve.append((accepted, cur))
        pbar2.update(1)
        pbar2.set_postfix(E=f"{loss_now:.3e}", NMSE=f"{cur:.3e}")
    pbar2.close()
    # final checkpoint (L-BFGS refinement state)
    torch.save(state(), ckpt_path)
    return net, loss_curve, nmse_curve


def pinn_final(
    net, s_of_psi=None, *, seed: int = SEED, eval_points: int = 20000
) -> dict:
    """Paper Sec.-4.2 NMSE: 2e4 test datapoints, z u [0, 5], t u [-6, 6]."""
    torch = _torch()
    rng = np.random.default_rng(seed + 13)
    zt = rng.uniform(0.0, Z_N_MAX, eval_points)
    tt = rng.uniform(-EVAL_T_MAX, EVAL_T_MAX, eval_points)
    dtype, device = torch.float64, next(net.parameters()).device

    def tens(a):
        return torch.tensor(np.asarray(a, float), dtype=dtype, device=device).reshape(
            -1, 1
        )

    with torch.no_grad():
        out = net(torch.cat([tens(tt) / T_N_MAX, tens(zt) / Z_N_MAX], dim=1))
        sc = 1.0 - torch.exp(-3.0 * tens(zt))
        u = torch.exp(-0.5 * tens(tt) ** 2) + sc * out[:, 0:1]
        v = sc * out[:, 1:2]
        psi_q = (u + 1j * v).cpu().numpy().reshape(-1)
    return {"t": tt, "z": zt, "psi_q": psi_q}


# ---------------------------------------------------------------------------
# case metrics
# ---------------------------------------------------------------------------


def case_metrics_A_or_B(net, case: str) -> dict:
    """NMSE of the pin net vs the paper's analytic solution."""
    torch = _torch()
    rng = np.random.default_rng(SEED + 13)
    n = 20000
    zt = rng.uniform(0.0, Z_N_MAX, n)
    tt = rng.uniform(-EVAL_T_MAX, EVAL_T_MAX, n)
    device = next(net.parameters()).device

    def tens(a):
        return torch.tensor(
            np.asarray(a, float), dtype=torch.float64, device=device
        ).reshape(-1, 1)

    with torch.no_grad():
        out = net(torch.cat([tens(tt) / T_N_MAX, tens(zt) / Z_N_MAX], dim=1))
        sc = 1.0 - torch.exp(-3.0 * tens(zt))
        u = torch.exp(-0.5 * tens(tt) ** 2) + sc * out[:, 0:1]
        v = sc * out[:, 1:2]
        psi_q = (u + 1j * v).cpu().numpy().reshape(-1)
    ref = EXACT_FN[case](tt, zt)
    m = nmse(psi_q, ref)
    return {
        "case": case,
        "nmse": m,
        "paper_nmse": PAPER_QUOTED.get(case),
        "rel_l2": rel_l2(psi_q - ref, ref),
    }


def case_metrics_C(net) -> dict:
    """rel-L2 of the PIN net vs the engine's snapshot stack (z, t).

    Evaluated on 2e4 test points: z sampled FROM the engine snapshot
    values (no z-interpolation error), t uniformly in [-6, 6].
    """
    torch = _torch()
    H, t_norm, z_norm = engine_oracle(1.0, 1.0)
    rng = np.random.default_rng(SEED + 13)
    n = 20000
    zt = rng.choice(z_norm, size=n)
    tt = rng.choice(np.linspace(-6.0, 6.0, 201), size=n)  # uniform t in [-6, 6]
    k_idx = np.searchsorted(z_norm, zt)
    k_idx = np.clip(k_idx, 0, len(z_norm) - 1)
    # snap each sample z to the nearest snapshot (choice already picked
    # snapshot z-values, this is exact for those)
    t_idx = np.searchsorted(t_norm, tt)
    t_idx = np.clip(t_idx, 0, len(t_norm) - 1)
    psi_ref = H[k_idx, t_idx]

    device = next(net.parameters()).device

    def tens(a):
        return torch.tensor(
            np.asarray(a, float), dtype=torch.float64, device=device
        ).reshape(-1, 1)

    with torch.no_grad():
        out = net(torch.cat([tens(tt) / T_N_MAX, tens(zt) / Z_N_MAX], dim=1))
        sc = 1.0 - torch.exp(-3.0 * tens(zt))
        u = torch.exp(-0.5 * tens(tt) ** 2) + sc * out[:, 0:1]
        v = sc * out[:, 1:2]
        psi_q = (u + 1j * v).cpu().numpy().reshape(-1)

    rel = rel_l2(psi_q - psi_ref, psi_ref)
    m = nmse(psi_q, psi_ref)
    return {
        "case": "C",
        "nmse": m,
        "rel_l2": rel,
        "oracle": "photonics_helper GNLSESolver snapshots",
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate(
    *,
    fast: bool = False,
    make_plot: bool = True,
    cases: str = "A,B,C",
    device: str = "cpu",
    adam_iters: int | None = None,
    resume: bool = False,
) -> dict:
    params = json.loads(PARAMETERS.read_text())
    results: dict = {"parameters": params}
    iters = adam_iters or 30000

    # --- 1a/1b. local oracle vs analytic (both z = 5 end-lines) -------------
    t_row, _ = script_grid(ENGINE_N_T)
    psi_refA, _ = script_oracle(1.0, 0.0, n_z=30000)
    r_a = rel_l2(psi_refA - psi_exact_A(t_row, 5.0), psi_exact_A(t_row, 5.0))
    results["script_caseA_vs_analytic_rel_l2"] = r_a
    assert r_a < 1e-6, r_a

    psi_refB, _ = script_oracle(0.0, 1.0, n_z=30000)
    r_b = rel_l2(psi_refB - psi_exact_B(t_row, 5.0), psi_exact_B(t_row, 5.0))
    results["script_caseB_vs_analytic_rel_l2"] = r_b
    assert r_b < 1e-6, r_b

    # --- 1c. engine vs local oracle on all three cases ----------------------
    engine_rel = {}
    for case, (beta, gamma) in CASES.items():
        psi_ref, _ = script_oracle(beta, gamma, n_z=30000)
        H, t_norm, _ = engine_oracle(beta, gamma)
        assert np.allclose(t_norm, t_row, atol=1e-9)
        r = rel_l2(H[-1] - psi_ref, psi_ref)
        engine_rel[case] = r
        assert r < 1e-4, (case, r)
    results["engine_vs_oracle_rel_l2"] = engine_rel

    # --- 2-4. PINN cases -----------------------------------------------------
    pins: dict[str, dict] = {}
    torch_mod = _torch()
    for case in cases.split(","):
        try:
            net, loss_curve, nmse_curve = train_pinn(
                case,
                adam_iters=iters,
                seed=SEED,
                device=device,
                fast=fast,
                resume=resume,
            )
        except torch_mod.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
            torch_mod.cuda.empty_cache()
            print(f"  [{case}] GPU OOM — falling back to cpu (per ISSUES #7)")
            net, loss_curve, nmse_curve = train_pinn(
                case,
                adam_iters=iters,
                seed=SEED,
                device="cpu",
                fast=fast,
                resume=resume,
            )
        if case in ("A", "B"):
            m = case_metrics_A_or_B(net, case)
            m["net"] = net
            pins[case] = m
            results[f"loss_curve_{case}"] = loss_curve
            results[f"nmse_curve_{case}"] = nmse_curve
            assert m["nmse"] < 1e-4, m
        elif case == "C":
            m = case_metrics_C(net)
            pins["C"] = m
            assert m["rel_l2"] < 5e-3, m
    results["pinn_metrics"] = pins
    results["adam_iters"] = iters

    if make_plot:
        _plot(results, pins)
    print("Monterola & Saloma (2001) NN-NLSE: validation passed")
    for k, v in pins.items():
        print(
            f"  case {k}: NMSE {v['nmse']:.3e} "
            f"(paper: A {PAPER_QUOTED['A']:.2e} / B {PAPER_QUOTED['B']:.2e})"
        )
    return results


def _plot(results, pins) -> None:
    """Three panels of |Psi_Pinn| maps (cases A/B) + case-C profile, all
    computed against the pinned reference, PLUS a loss/NMSE curve row"""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    tgrid = np.linspace(-EVAL_T_MAX, EVAL_T_MAX, 201)
    zgrid = np.linspace(0.0, Z_N_MAX, 201)
    tg, zg = np.meshgrid(tgrid, zgrid)
    tflat = tg.ravel()
    zflat = zg.ravel()
    torch_mod = _torch()
    for k, (case, m) in enumerate(pins.items()):
        ax = axes.flat[k]
        if case == "C":
            _psi_ref = "engine oracle (snapshot stack)"  # placeholder text
            ax.text(
                0.5,
                0.5,
                "case C: rel-L2 %.3e" % m["rel_l2"],
                transform=ax.transAxes,
                ha="center",
                fontsize=14,
            )
            ax.set_title(f"(C) engine-oracle rel-L2 = {m['rel_l2']:.2e}")
            continue
        net = m.get("net")
        if net is not None:
            with torch_mod.no_grad():
                dev = next(net.parameters()).device

                def tens(a):
                    return torch_mod.tensor(
                        np.asarray(a, float), dtype=torch_mod.float64, device=dev
                    ).reshape(-1, 1)

                out = net(
                    torch_mod.cat([tens(tflat) / T_N_MAX, tens(zflat) / Z_N_MAX], dim=1)
                )
                sc = 1.0 - torch_mod.exp(-3.0 * tens(zflat))
                u = torch_mod.exp(-0.5 * tens(tflat) ** 2) + sc * out[:, 0:1]
                v = sc * out[:, 1:2]
                psi_q = (u + 1j * v).cpu().numpy().reshape(tg.shape)
        else:
            psi_q = np.full_like(tg, np.nan, dtype=complex)
        im = ax.imshow(
            np.abs(psi_q),
            extent=[tgrid[0], tgrid[-1], zgrid[0], zgrid[-1]],
            origin="lower",
            aspect="auto",
            vmax=1.0,
        )
        ax.set_title(f"({case}) |Psi_NN|  NMSE = {m['nmse']:.2e}")
        fig.colorbar(im, ax=ax)
    ax = axes.flat[3]
    if "loss_curve_A" in results:
        ax.plot(results["loss_curve_A"], "k-", lw=0.7, label="Adam loss (case A)")
        ax.set_yscale("log")
        ax.set_xlabel("checkpoint index")
        ax.set_ylabel("E (paper Eq. 7)")
        ax.legend(fontsize=8)
    ax.set_title("(d) training loss (case A)")
    fig.suptitle("Monterola & Saloma 2001 - NN-NLSE")
    fig.tight_layout()
    fig.savefig(HERE / "oe_2001_nn_nlse.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="A,B,C")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--adam-iters", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = validate(
        fast=args.fast,
        cases=args.cases,
        device=args.device,
        adam_iters=args.adam_iters,
        resume=args.resume,
    )
    print(json.dumps(out, indent=2, default=float))
