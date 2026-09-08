"""
test_fire_fix.py — verify the wrapper's early-stop fix.
Before: minimizeFIRE bailed at |F|~1e-9 with "energy is getting stuck".
After:  with stuckEnergyTol=0.0 (new default) it runs to maxSteps or
        maxForceThreshold like FIRE should.
"""

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

print("--- default (stuckEnergyTol=0.0, no early-stop) ---", flush=True)
res = m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=10000,
                     shakeIter=0, progressBar=False)
print(f"  returned steps = {res[2]}", flush=True)
print(f"  E      = {m.getEnergy():.6e}", flush=True)
print(f"  |F|max = {m.getMaxUnbalancedForce():.6e}", flush=True)

# Rebuild fresh and try the opposite — explicit early-stop with old behaviour
m2 = pcp.model(size=N*n, seed=seed)
m2.setModelEnum("softBody")
m2.generateRandomPolygons(N, n)
m2.setBiPerimeters(kappa)
m2.setDelta(rho)
m2.setStiffness(1)
m2.setCompressibility(1)
m2.initializeNeighborCells()
m2.updateNeighborCells()
m2.updateOutersections()
m2.updateForceEnergy()
print("\n--- explicit stuckEnergyTol=1e-15 (old behaviour, now properly counter-reset) ---", flush=True)
res2 = m2.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=10000,
                       shakeIter=0, progressBar=False,
                       stuckEnergyTol=1e-15)
print(f"  returned steps = {res2[2]}", flush=True)
print(f"  E      = {m2.getEnergy():.6e}", flush=True)
print(f"  |F|max = {m2.getMaxUnbalancedForce():.6e}", flush=True)
