"""diag: does projectForce actually modify the force array?"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

N, n, seed, kappa, rho = 64, 32, 42, 3.7, 0.001
m = pcp.model(size=N*n, seed=seed)
m.setModelEnum("softBody")
m.generateRandomPolygons(N, n)
m.setBiPerimeters(kappa)
m.setDelta(rho)
m.setStiffness(1)
m.setCompressibility(1)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateOutersections()
m.updateForceEnergy()
m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=3000, progressBar=False)
m.setPhi(1.0)
m.setModelEnum("rounded")
m.setMaxEdgeLength(0.1)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateOutersections()
m.updateForceEnergy()
m.minimizeFIRE(maxSteps=1, shakeIter=100, progressBar=False)
m.minimizeFIRE(dt=1e-3, maxSteps=1, shakeIter=100, progressBar=False)
m.minimizeFIRE(dt=1e-8, maxSteps=1000, shakeIter=100, progressBar=False)

# After minimization, fresh force computation
m.updateForceEnergy()
F_before = np.array(m.getForces())
m.projectForce()
F_after  = np.array(m.getForces())

print(f"|F_before|_max  = {np.max(np.abs(F_before)):.6e}", flush=True)
print(f"|F_after|_max   = {np.max(np.abs(F_after)):.6e}",  flush=True)
print(f"|F_after - F_before|_max = {np.max(np.abs(F_after - F_before)):.6e}", flush=True)
print(f"|F_before|_2     = {np.linalg.norm(F_before):.6e}", flush=True)
print(f"|F_after|_2      = {np.linalg.norm(F_after):.6e}",  flush=True)
print(f"identical?       = {np.array_equal(F_before, F_after)}", flush=True)
