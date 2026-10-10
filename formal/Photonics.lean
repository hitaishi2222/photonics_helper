/-
Copyright (c) 2026 photonics_helper contributors.
Released under the repository licence (see LICENSE).

# Photonics — the formal specification

The mathematical specification that `photonics_helper` implements.

Import order is strictly downward: `Conventions` knows nothing, `Constants`
imports `Conventions`, and later modules import both. No cycles.

See `formal/PLAN.md` for the staged roadmap and `formal/CONFORMANCE.md` for
the theorem → Python register.
-/

import Photonics.Conventions
import Photonics.Constants
import Photonics.Grid.Basic
import Photonics.Grid.Spectrum
import Photonics.Dispersion.Beta
import Photonics.Nonlinear.Shock
import Photonics.Nonlinear.Raman
import Photonics.Optics.DBR
import Photonics.Optics.Chi2
import Photonics.Optics.FWM
import Photonics.Optics.Structured
import Photonics.Solvers.RK4
import Photonics.Solvers.Strang
import Photonics.Solvers.Dispersion