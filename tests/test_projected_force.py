"""
test_projected_force.py — at φ=1.0 the system is jammed, so raw |F| has a
constraint-force floor.  The real convergence indicator is the force projected
onto the constraint-tangent manifold (already implemented in projectForceCUDA).
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
print("softBody relax ...", flush=True)
m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=3000, progressBar=False)
m.setPhi(1.0)

m.setModelEnum("rounded")
m.setMaxEdgeLength(0.1)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateOutersections()
m.updateForceEnergy()
print("rounded minimise ...", flush=True)
m.minimizeFIRE(maxSteps=1,    shakeIter=100, progressBar=False)
m.minimizeFIRE(dt=1e-3, maxSteps=1,    shakeIter=100, progressBar=False)
m.minimizeFIRE(dt=1e-8, maxSteps=2000, shakeIter=100, progressBar=False)

# Raw force at the constrained-jammed state
m.updateForceEnergy()
f_raw_max = m.getMaxUnbalancedForce()
E         = m.getEnergy()

# Project onto the constraint-tangent manifold and re-query the max
m.updateForceEnergy()      # ensures force is fresh
m.projectForce()           # in-place: replaces force with its tangent projection
f_tan_max = m.getMaxUnbalancedForce()

# Restore raw force afterwards (so other downstream code sees raw)
m.updateForceEnergy()

print(f"\nat φ = 1.0  jammed state:")
print(f"  energy                          = {E:.6e}")
print(f"  max raw |F|       (constrained) = {f_raw_max:.6e}")
print(f"  max projected |F| (true residual) = {f_tan_max:.6e}")
print(f"  ratio (projected / raw)         = {f_tan_max/f_raw_max:.3e}")
