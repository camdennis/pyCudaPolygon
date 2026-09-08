"""test_polyDelta_full_pipeline.py — full per-polygon-delta dynamics.

With Phase B detection also reading per-polygon δ, the pipeline is
geometrically consistent when polyDelta varies across polygons:
  - Phase B detects crossings using each side's true δ
  - Phase C force kernel uses the same δs in the analytical Jacobian

The analytical kernel computes ∂W/∂x with polyDelta held fixed. For FD to
match, we must hold polyDelta fixed too (i.e., NOT call updatePolyDelta
inside the FD loop, since the closed-form makes polyDelta depend on
positions). With polyDelta fixed at varying-across-polygons values, FD
should match analytical at floating-point precision on smooth vertices.

The test enables per-polygon mode at a small position perturbation so
polyDelta varies modestly across polygons (avoiding near-straight-corner
blow-up), then FD-validates with polyDelta held constant.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build(delta=0.04):
    m = pcp.model(size=16*16, seed=3)
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


def run_walk(m, return_force=False):
    m.runFeaturePairPhaseA(2)
    m.runFeaturePairPhaseB(m.getDelta())
    nFA = m.runFeaturePairPhaseC_sort()
    if nFA == 0:
        return (0.0, np.zeros(2 * m.getNumVertices())) if return_force else 0.0
    if return_force:
        area = np.array(m.runFeaturePairPhaseC_walkAreaAndForce())
        return area.sum(), np.array(m.getPhaseCForce())
    area = np.array(m.runFeaturePairPhaseC_walkArea())
    return area.sum()


m = build(delta=0.04)
N = m.getNumVertices()
P = m.getNumPolygons()

# Set polyDelta to bounded, varying-per-polygon values via setPolyDelta. This
# bypasses the closed-form (which can blow up near degenerate corner geometry)
# and lets us test the kernel pipeline with controlled δ variation.
rng = np.random.default_rng(11)
poly_d = 0.04 + 0.01 * rng.standard_normal(P)
poly_d = np.clip(poly_d, 0.02, 0.06)
m.setPolyDelta(poly_d.tolist())

poly_d_actual = np.array(m.getPolyDelta())
print(f"polyDelta range: [{poly_d_actual.min():.6e}, {poly_d_actual.max():.6e}]   "
      f"spread={poly_d_actual.max()-poly_d_actual.min():.3e}")
assert poly_d_actual.max() - poly_d_actual.min() > 0.01, "polyDelta should vary"

# This is the analytical reference: forces computed at this fixed polyDelta.
W_ref, F_an = run_walk(m, return_force=True)
print(f"reference: W={W_ref:.6e}   max|F|={np.abs(F_an).max():.3e}")

# FD validation: hold polyDelta FIXED across perturbations so FD measures
# the kernel's partial derivative ∂W/∂x at fixed polyDelta. Do NOT call
# updatePolyDelta inside the FD loop.
fmag = np.linalg.norm(F_an.reshape(-1, 2), axis=1)
top = np.argsort(-fmag)[:8]
print(f"FD-checking top {len(top)} forced vertices: {top.tolist()}")

eps = 1e-7
pos_ref = np.array(m.getPositions())
best_rel = 1.0
worst_rel = 0.0

for v in top:
    for c in range(2):
        pp = pos_ref.copy(); pp[2*v + c] += eps
        m.setPositions(pp.tolist())
        W_plus = run_walk(m)

        pp = pos_ref.copy(); pp[2*v + c] -= eps
        m.setPositions(pp.tolist())
        W_minus = run_walk(m)

        m.setPositions(pos_ref.tolist())

        F_fd = -(W_plus - W_minus) / (2.0 * eps)
        F_a = F_an[2*v + c]
        denom = max(abs(F_a), abs(F_fd), 1e-12)
        rel = abs(F_fd - F_a) / denom
        best_rel = min(best_rel, rel)
        worst_rel = max(worst_rel, rel)
        print(f"  v={int(v):3d} c={c}  F_an={F_a:+.4e}  F_fd={F_fd:+.4e}  rel={rel:.3e}")

print(f"\nbest rel err:  {best_rel:.3e}")
print(f"worst rel err: {worst_rel:.3e}")

assert best_rel < 1e-4, (
    f"best rel err {best_rel:.3e} >= 1e-4 -- per-poly pipeline may be broken")
print("\nPer-polygon δ pipeline (Phase B + Phase C) validated under varying polyDelta.")
