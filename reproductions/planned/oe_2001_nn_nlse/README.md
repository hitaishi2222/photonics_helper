# Planned — Solving the nonlinear Schrödinger equation with an unsupervised neural network (Monterola & Saloma 2001)

**Reference.** C. Monterola, C. Saloma, *Optics Express* **9**, 72 (2001),
**doi:10.1364/OE.9.000072**. PDF: `OE.9.000072.pdf` (provided by author; the
direct download is captcha-protected); 10 pages under `pages/`.

**Status:** `[~]` PDF present, plan below; `reproduce.py` to be written.
**Target:** the PINN layer — baseline/companion to
`raissi_2019_pinn_nlse` and the upstream ancestor of modern fiber PINNs.
Uses the torch `[pinns]` extra (`inverse_design.fit_shg_autodiff` precedent).

## What to reproduce (exact text-level contract extracted from the PDF text)

- Read the paper's two-variable network ansatz (their unsupervised NN with
  Gaussian radial-spline/transfer function of x, t constructed via a
  "characteristic sigmoid decomposition"?) and its training rule (Marquardt
  second-order? windowed sequential forward pass — extract exact equations
  from pages 2–5 of the PDF).
- Solve the **scalar NLS** (fiber envelope of a 1-ps pulse at 1.55 µm, with
  the SPM + GVD truncated operator), reproducing:
  1. the propagation figure of the NN-predicted soliton against the paper's
     reference (Sec. IV, Fig. N) — figure-level comparison at least,
     and more importantly
  2. quant beats: the network's RMS error vs the paper's quoted benchmark
     (they quote ~1e-10 to exact solution? re-extract from `pages/`).
- The paper is not structured as a PINN (pre-Raissi); reproduction options:
  - **Option A (faithful):** re-implement their unsupervised network (sigmoid
    characteristic + Marquardt training?) at their declared parameters.
  - **Option B (modern):** expose the same problem as a PINN residual training
    (Raissi-style, 2019 machinery) on the same problem setup, showing the two
    agree on |A(z,t)|. Option B is preferred for our library, comparing the
    solution against a *their-claimed* benchmark and our own GNLSE.

## Ties to the photonics_helper stack

- Solve the same NLS analytically with `gnlse` engine for the ground truth —
  so this reproduction validates the *PINN layer as a fiber solver*, with the
  GNLSE engine as the oracle; publishable as "photonics_helper PINN back-end
  matches its own split-step engine + the 2001 exhibition". Reuse of the
  same evaluation harness as `raissi_2019_pinn_nlse` (relative L2, cut profiles).

## Execution checklist

- [x] PDF rendered (`pages/p-01..10.png`).
- [ ] Read pages 2–6: extract the network architecture, training rule, and
  the exact claimed error numbers (the paper predates tensors; document
  design choices in `parameters.json`).
- [ ] `reproduce.py` with torch: PINN training against the scalar-NLSE setup;
  metric: relative L2 ≤ the paper's quoted error band, and matching the SSFM
  engine reference to ≤ 5 %.
- [ ] README: progressive story (2001 unsupervised NN → 2019 PINN → our data).
- [ ] Move to `reproductions/oe_2001_nn_nlse/`, update index.
