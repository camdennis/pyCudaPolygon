"""
test_projected_force_convergence.py — run FIRE at φ=1.0 and track BOTH raw
|F|max and projected |F|max. Raw stays bounded above ~constraint-force scale;
projected should drop towards 0 as FIRE actually converges to a constrained
minimum.
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
m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=3000, progressBar=False)
m.setPhi(1.0)

m.setModelEnum("rounded")
m.setMaxEdgeLength(0.1)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateOutersections()
m.updateForceEnergy()

def report(tag):
    m.updateForceEnergy()
    f_raw = m.getMaxUnbalancedForce()
    e     = m.getEnergy()
    m.updateForceEnergy()
    m.projectForce()
    f_tan = m.getMaxUnbalancedForce()
    m.updateForceEnergy()        # restore raw
    print(f"  {tag:<20}  E={e:.4e}  |F|raw={f_raw:.4e}  |F|tan={f_tan:.4e}", flush=True)

print(f"\n{'phase':<22}{'E':>12}{'|F|raw':>14}{'|F|tan':>14}", flush=True)
report("initial")

chunks = [(1e-3, 100), (1e-3, 200), (1e-3, 500), (1e-3, 1000), (1e-3, 2000)]
total = 0
for dt, ns in chunks:
    m.minimizeFIRE(dt=dt, maxSteps=ns, shakeIter=100, progressBar=False)
    total += ns
    report(f"after {total:>7d} steps")
