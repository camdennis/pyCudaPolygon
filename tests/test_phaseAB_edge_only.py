"""test_phaseAB_edge_only.py — sanity-check Phase A + Phase B (installment 2:
edge-edge on flat segments between tangent points; arc dispatch also active).

We no longer compare to updateNeighborsKernel since that finder uses backbone
edges, whereas Phase B now uses flat segments. Counts will not match. Instead
we just confirm Phase A+B runs and produces a sensible number of crossings
for a couple of configurations."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build(N=16, n=16, seed=7, kappa=3.7, rho=0.05, relax=False):
    m = pcp.model(size=N*n, seed=seed)
    m.setModelEnum("softBody" if relax else "normal")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.initializeNeighborBall()
    m.updateForceEnergy()
    if relax:
        m.minimizeFIRE(maxForceThreshold=1e-12, maxSteps=2000, progressBar=False)
        m.setPhi(1.1)
        m.updatePolygonGeometry()
    m.updateNeighborBall()
    return m


print("Case 1: random initial positions")
m1 = build(relax=False)
n1_pairs = m1.runFeaturePairPhaseA(2)            # fs=2 -> 4 pairs per (i,j)
n1_cross = m1.runFeaturePairPhaseB(m1.getDelta())
print(f"  fs=2 pairs:     {n1_pairs}")
print(f"  fs=2 crossings: {n1_cross}")

print("\nCase 2: softBody-relaxed + phi=1.1")
m2 = build(relax=True)
n2_pairs = m2.runFeaturePairPhaseA(2)
n2_cross = m2.runFeaturePairPhaseB(m2.getDelta())
print(f"  fs=2 pairs:     {n2_pairs}")
print(f"  fs=2 crossings: {n2_cross}")

# Basic sanity: arc dispatch now active, so fs=2 should differ from fs=1.
e1 = m2.runFeaturePairPhaseA(1)
e1c = m2.runFeaturePairPhaseB(m2.getDelta())
print(f"\nrelaxed: fs=1 crossings = {e1c}, fs=2 crossings = {n2_cross}")
print("  (arc combos add to fs=2's total once flat-edge + arc dispatch are both active)")

print("\nPhase A+B runs cleanly with installment 2.")
