"""test_biarc_rounded_minimization.py — reproduce the user's biArcTesting flow
and verify that after polygon-scale ball reach, the rounded minimization
does NOT collapse below the physical minimum."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

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

m.minimizeFIRE(maxForceThreshold=1e-14, maxSteps=20000, progressBar=False)
m.setPhi(1.1)
m.updatePolygonGeometry()
print(f"after softBody+phi=1.1: maxEdgeLength={m.getMaxEdgeLength():.4e}  "
      f"polygonSize={m.getNumVertices()//m.getNumPolygons()}")

m.setModelEnum("rounded")
m.initializeNeighborBall()
m.updatePolygonGeometry()
m.updateNeighborBall()
m.updateForceEnergy()
print(f"[rounded init]  E={m.getEnergy():.4e}  "
      f"|F|max={m.getMaxUnbalancedForce():.4e}  "
      f"mean ball={np.mean(m.getNumBallNeighbors()):.1f}  "
      f"rebuilds={m.getBallRebuildCount()}")

# Run the long FIRE that originally collapsed E below 0.1
res = m.minimizeFIRE(dt=1e-8, maxSteps=1000, shakeIter=10000, progressBar=False)
E_final = res[0]
print(f"[after 1000 FIRE steps, dt=1e-8]  E={E_final:.4e}  "
      f"rebuilds={m.getBallRebuildCount()}")
print(f"|F|max = {m.getMaxUnbalancedForce():.4e}")

# At phi=1.1 with rounded polygons, the system can't relax to E=0; areas exceed
# unit-box area by 10%, so substantial overlap energy persists.
assert E_final >= 0.1, f"E={E_final} below 0.1 — kernel is missing overlap pairs"
print("\nROUNDED MINIMIZATION OK (energy stays >= 0.1).")
