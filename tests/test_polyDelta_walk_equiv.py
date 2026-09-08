"""test_polyDelta_walk_equiv.py — Phase C force kernel sanity for per-polygon
delta wiring.

Two checks:
  1. scalar mode (no per-poly enable) -> walked area + force exactly matches
     a manually-frozen per-polygon mode where every polygon's delta equals
     the global scalar. Confirms the polyDelta_d -> kernel chain is wired
     correctly and the unequal-radii AA Jacobian reduces to the equal-radii
     case when both polygons share a delta.
  2. After enablePerPolygonDelta with zero perturbation, polyDelta values
     equal the global scalar (the closed-form gives delta(s) = scalar at
     the freeze configuration), and force kernel output is identical.

Note: this test does NOT validate force correctness when polyDelta values
differ across polygons. That requires Phase B to also use per-polygon delta
(deferred — Phase B currently uses the scalar `delta` for crossing detection,
so its geometry diverges from the per-polygon force kernel's assumption when
delta varies).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build(delta):
    m = pcp.model(size=16*16, seed=7)
    m.setModelEnum("rounded")
    m.generateRandomPolygons(16, 16)
    m.setBiPerimeters(3.7)
    m.setDelta(delta)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.initializeNeighborBall()
    m.updateForceEnergy()
    m.updateNeighborBall()
    return m


def run_walk(m):
    m.runFeaturePairPhaseA(2)
    m.runFeaturePairPhaseB(m.getDelta())
    nFA = m.runFeaturePairPhaseC_sort()
    if nFA == 0:
        return 0.0, np.zeros(2 * m.getNumVertices())
    area = np.array(m.runFeaturePairPhaseC_walkAreaAndForce())
    return area.sum(), np.array(m.getPhaseCForce())


DELTA = 0.04

# Run 1: scalar mode (no per-poly enable)
m1 = build(DELTA)
W1, F1 = run_walk(m1)
print(f"scalar mode:        W={W1:.6e}  max|F|={np.abs(F1).max():.3e}")

# Run 2: same config + enablePerPolygonDelta (at zero perturbation,
# delta(s) = scalar for every polygon)
m2 = build(DELTA)
m2.enablePerPolygonDelta()
poly_d = np.array(m2.getPolyDelta())
print(f"per-poly mode:      polyDelta range [{poly_d.min():.6e}, {poly_d.max():.6e}]")
assert np.allclose(poly_d, DELTA, rtol=1e-12), \
    f"polyDelta should equal scalar at freeze: max dev {np.abs(poly_d - DELTA).max():.3e}"

W2, F2 = run_walk(m2)
print(f"per-poly mode walk: W={W2:.6e}  max|F|={np.abs(F2).max():.3e}")

dW = abs(W1 - W2)
dF = np.abs(F1 - F2).max()
print(f"\nscalar vs per-poly: |dW|={dW:.3e}   max|dF|={dF:.3e}")
assert dW < 1e-12, f"W mismatch: {dW:.3e}"
assert dF < 1e-12, f"F mismatch: {dF:.3e}"

print("\nPer-polygon delta wiring OK (scalar == per-poly with frozen delta).")
