"""
test_shake_floor.py — hypothesis: softBody's |F|max floor is set by SHAKE
tolerance. Increasing shakeIter should drop both rms edge violation and
|F|max in lockstep.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

N, n, seed, kappa, rho = 64, 32, 42, 3.7, 0.001

for shake_iter in (5, 50, 500, 5000):
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
    m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=3000,
                   shakeIter=shake_iter, progressBar=False)
    cv = m.getConstraintViolation()
    m.updateForceEnergy()
    f_raw = m.getMaxUnbalancedForce()
    m.updateForceEnergy(); m.projectForce()
    f_tan = m.getMaxUnbalancedForce()
    m.updateForceEnergy()
    print(f"shakeIter = {shake_iter:>5d}   rmsEdge = {cv['rmsEdgeViolation']:.3e}   "
          f"|F|raw = {f_raw:.3e}   |F|tan = {f_tan:.3e}", flush=True)
