"""
demo_tangential_force.py — show that the max projected force
(m.projectForce() then m.getMaxUnbalancedForce()) is a valid convergence
criterion that drops to ~0 even at φ=1.0 where raw |F|max stays floored by
the constraint-force scale.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

def build(phi):
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
    m.setPhi(phi)
    m.setModelEnum("rounded")
    m.setMaxEdgeLength(0.1)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    m.updateOutersections()
    m.updateForceEnergy()
    return m

def measure(m):
    m.updateForceEnergy()
    f_raw = m.getMaxUnbalancedForce()
    m.updateForceEnergy()
    m.projectForce()
    f_tan = m.getMaxUnbalancedForce()
    m.updateForceEnergy()
    return m.getEnergy(), f_raw, f_tan

for phi in (0.8, 1.0):
    print(f"\n────────────  φ = {phi}  ────────────", flush=True)
    m = build(phi)
    print(f"{'steps':>8}  {'E':>12}  {'raw |F|max':>12}  {'proj |F|max':>12}", flush=True)

    total = 0
    E, fr, ft = measure(m); print(f"{total:>8d}  {E:>12.4e}  {fr:>12.4e}  {ft:>12.4e}", flush=True)
    for ns in (200, 500, 1000, 2000, 4000, 8000):
        m.minimizeFIRE(dt=1e-3, dtMax=5e-3, maxSteps=ns,
                       shakeIter=200, progressBar=False)
        total += ns
        E, fr, ft = measure(m)
        print(f"{total:>8d}  {E:>12.4e}  {fr:>12.4e}  {ft:>12.4e}", flush=True)
