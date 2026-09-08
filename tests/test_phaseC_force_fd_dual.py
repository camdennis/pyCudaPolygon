"""test_phaseC_force_fd_dual.py — FD validation of Phase C edge-edge forces
with delta>0 via the dual-traced walk kernel.

Same FD scheme as test_phaseC_force_fd.py, but uses the new
runFeaturePairPhaseC_walkAreaAndForceDual entry which chains through
tangent-point Jacobians (DPoint duals + getDf) so delta>0 is honored.

A *secondary* check: with delta=0, the dual variant must produce identical
forces to the plain (installment 3) variant -- the tangent points reduce to
the vertex itself when delta=0.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build(N=8, n=10, seed=2, kappa=3.0, delta=0.0):
    m = pcp.model(size=N * n, seed=seed)
    m.setModelEnum("rounded")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(delta)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.initializeNeighborBall()
    m.updateForceEnergy()
    m.updateNeighborBall()
    return m


def run_walk(m, mode):
    """mode in {'area', 'force', 'force_dual'}"""
    m.runFeaturePairPhaseA(1)
    m.runFeaturePairPhaseB(m.getDelta())
    nFA = m.runFeaturePairPhaseC_sort()
    if nFA == 0:
        return 0.0, np.zeros(2 * m.getNumVertices())
    if mode == 'area':
        area_vec = np.array(m.runFeaturePairPhaseC_walkArea())
        return area_vec.sum(), None
    elif mode == 'force':
        area_vec = np.array(m.runFeaturePairPhaseC_walkAreaAndForce())
        return area_vec.sum(), np.array(m.getPhaseCForce())
    elif mode == 'force_dual':
        area_vec = np.array(m.runFeaturePairPhaseC_walkAreaAndForceDual())
        return area_vec.sum(), np.array(m.getPhaseCForce())


# ---- 1) Cross-check: delta=0 -> dual variant matches getDf-only variant ----
print("=== Cross-check at delta=0 (dual vs getDf-only) ===")
m0 = build(delta=0.0)
W0a, F0a = run_walk(m0, 'force')
W0b, F0b = run_walk(m0, 'force_dual')
print(f"  area dev: {abs(W0a - W0b):.3e}   max|F| diff: {np.abs(F0a - F0b).max():.3e}")
assert abs(W0a - W0b) < 1e-12
assert np.abs(F0a - F0b).max() < 1e-12, f"delta=0 force mismatch: {np.abs(F0a - F0b).max():.3e}"
print("  delta=0 dual reduces to plain. OK.")

# ---- 2) FD validation at delta=0.04 ----
delta_run = 0.04
print(f"\n=== FD validation at delta={delta_run} (dual) ===")
m = build(delta=delta_run)
N = m.getNumVertices()
print(f"N vertices = {N}")

W_ref, F_an = run_walk(m, 'force_dual')
print(f"reference walked area W = {W_ref:.6e}")
fmag = np.linalg.norm(F_an.reshape(-1, 2), axis=1)
nz = np.count_nonzero(fmag > 1e-12)
print(f"vertices with nonzero analytical force: {nz} / {N}")
if nz == 0:
    raise SystemExit("no nonzero forces at this delta")

order = np.argsort(-fmag)
top = order[:min(8, nz)]
print(f"FD-checking top {len(top)} forced vertices: {top.tolist()}")

eps = 1e-7
pos_ref = np.array(m.getPositions())
max_err = 0.0
worst = None

for v in top:
    for c in range(2):
        pp = pos_ref.copy(); pp[2*v + c] += eps
        m.setPositions(pp.tolist())
        W_plus, _ = run_walk(m, 'area')

        pp = pos_ref.copy(); pp[2*v + c] -= eps
        m.setPositions(pp.tolist())
        W_minus, _ = run_walk(m, 'area')

        m.setPositions(pos_ref.tolist())

        F_fd = -(W_plus - W_minus) / (2.0 * eps)
        F_a = F_an[2*v + c]
        err = abs(F_fd - F_a)
        if err > max_err:
            max_err = err; worst = (int(v), c, F_a, F_fd, err)
        print(f"  v={int(v):3d} c={c}  F_an={F_a:+.6e}  F_fd={F_fd:+.6e}  abs err={err:.3e}")

print(f"\nmax abs FD error: {max_err:.3e}")
print(f"worst: v={worst[0]} c={worst[1]} F_an={worst[2]:+.6e} F_fd={worst[3]:+.6e}")
assert max_err < 1e-4, f"FD mismatch at delta>0: max abs {max_err:.3e}"
print("\nDual EE forces at delta>0 match FD.")
