"""test_biarc_notebook_smoketest.py — mirrors cells 1..7 of biArcTesting.ipynb
to verify the cells-free, balls-only flow still runs end-to-end. Skips the
long-running FIRE cells."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


# Cell 1: build softBody, init balls only (no cell grid, no outersections)
N, n, seed = 64, 32, 42
kappa, rho = 3.7, 0.001
m = pcp.model(size=N*n, seed=seed)
m.setModelEnum("softBody")
m.generateRandomPolygons(N, n)
m.setBiPerimeters(kappa)
m.setDelta(rho)
m.setStiffness(1)
m.setCompressibility(1)
m.initializeNeighborBall()
m.updateForceEnergy()
print(f"[cell 1] softBody init OK. neighborType={m.getNeighborType()}  "
      f"searchFactor={m.getSearchFactor()}  maxEdgeLength={m.getMaxEdgeLength():.4e}")
print(f"          E0 = {m.getEnergy():.6e}   |F|max = {m.getMaxUnbalancedForce():.6e}")

# Cell 2: minimize softBody, setPhi (short-circuited so the test runs in seconds)
m.minimizeFIRE(maxForceThreshold=1e-10, maxSteps=2000, progressBar=False)
print(f"[cell 2] after softBody FIRE (2k steps):  E={m.getEnergy():.6e}  "
      f"|F|max={m.getMaxUnbalancedForce():.6e}")
m.setPhi(1.1)
viol = m.getConstraintViolation()
print(f"[cell 2] after setPhi(1.1): rmsAreaViol={viol['rmsAreaViolation']:.3e}  "
      f"rmsEdgeViol={viol['rmsEdgeViolation']:.3e}")

# Cell 4: updatePolygonGeometry, getConstraintViolation
m.updatePolygonGeometry()
viol = m.getConstraintViolation()
print(f"[cell 4] post-updatePolygonGeometry violation: rmsArea={viol['rmsAreaViolation']:.3e}  "
      f"rmsEdge={viol['rmsEdgeViolation']:.3e}")

# Cell 5: switch to rounded, balls-only init, FIRE 1 step
m.setModelEnum("rounded")
m.initializeNeighborBall()
m.updatePolygonGeometry()
m.updateNeighborBall()
m.updateForceEnergy()
print(f"[cell 5] rounded init: E={m.getEnergy():.6e}  |F|max={m.getMaxUnbalancedForce():.6e}")
print(f"          neighborType={m.getNeighborType()}  "
      f"ball list mean count={np.mean(m.getNumBallNeighbors()):.1f}  "
      f"rebuilds={m.getBallRebuildCount()}")
res = m.minimizeFIRE(maxSteps=1, progressBar=False, shakeIter=100)
print(f"[cell 5] FIRE 1 step returned: E={res[0]:.6e}  dt={res[1]:.3e}  steps={res[2]}")

# Cell 6: constraint violation after 1 FIRE step
viol = m.getConstraintViolation()
print(f"[cell 6] post-FIRE rounded violation: rmsArea={viol['rmsAreaViolation']:.3e}  "
      f"rmsEdge={viol['rmsEdgeViolation']:.3e}")

# Cell 7: another 1-step FIRE
res = m.minimizeFIRE(dt=1e-3, maxSteps=1, progressBar=False, shakeIter=100)
print(f"[cell 7] FIRE 1 step (dt=1e-3): E={res[0]:.6e}")

print("\nCELLS-FREE NOTEBOOK SMOKE TEST OK (cells 1..7).")
