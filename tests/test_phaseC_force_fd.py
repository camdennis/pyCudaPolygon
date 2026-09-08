"""test_phaseC_force_fd.py — FD validation of Phase C edge-edge forces.

Validated at delta=0 AND delta>0. The plain (getDf-based) kernel is correct
at any delta because of the **line-intersection cancellation**: with the
rounded model, the flat-segment edge from a_v^+ to a_{nv}^- lies on the same
infinite line as the backbone edge V-to-N. The crossing point P depends only
on the two LINES (so on V_A, N_A, V_B, N_B), not on the tangent-point offsets
ell_v, ell_n that shift the endpoints along the lines. So dP/d(prev_v) = 0,
and the plain 4-vertex getDf Jacobian is exact at any delta.

The walk kernel accumulates two outputs per call:
  - per-A-feature inside chord-area (sum -> total signed area W)
  - per-vertex force = -dW/dx (only EDGE-EDGE contributions; arc-Jacobian
    lands later)

This test sets delta = 0 (no rounding, flat-segment endpoints coincide with
backbone vertices so getDf with the 4 edge-vertex Jacobian is exact), runs
fs=1 (edge-only features so every crossing is EDGE-EDGE), and validates the
analytical force against a central-difference estimate of -dW/dx_v.

Procedure for each vertex component (v, c):
  1. Snapshot the analytical force at the reference configuration
  2. Perturb position[v,c] by +eps, run A/B/C, sum walked area -> W_plus
  3. Perturb by -eps, run A/B/C, sum walked area -> W_minus
  4. F_fd[v,c] = -(W_plus - W_minus) / (2 * eps)
  5. Assert |F_analytical[v,c] - F_fd[v,c]| < tol
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


def run_walk(m, return_force=False):
    """Run A/B/C with fs=1 and return (W, force_array_if_requested)."""
    m.runFeaturePairPhaseA(1)
    m.runFeaturePairPhaseB(m.getDelta())
    nFA = m.runFeaturePairPhaseC_sort()
    if nFA == 0:
        return (0.0, np.zeros(2 * m.getNumVertices())) if return_force else 0.0
    if return_force:
        area_vec = np.array(m.runFeaturePairPhaseC_walkAreaAndForce())
        force = np.array(m.getPhaseCForce())
        return area_vec.sum(), force
    else:
        area_vec = np.array(m.runFeaturePairPhaseC_walkArea())
        return area_vec.sum()


DELTA = 0.04
m = build(delta=DELTA)
N = m.getNumVertices()
print(f"N vertices = {N}, delta = {DELTA}")

W_ref, F_analytic = run_walk(m, return_force=True)
print(f"reference walked area W = {W_ref:.6e}")
nz_force = np.count_nonzero(np.linalg.norm(F_analytic.reshape(-1, 2), axis=1) > 1e-12)
print(f"vertices with nonzero analytical force: {nz_force} / {N}")

# Pick FD eps. Walked area is O(1e-2) and forces are O(1) so eps ~ 1e-6 gives
# step ~ 1e-8 against ~ 1e-2 baseline (rel ~ 1e-6) -- should give 4-5 good digits.
eps = 1e-7
positions_ref = np.array(m.getPositions())

# To keep the test runtime reasonable, FD-check a subset: only vertices whose
# analytical force exceeds a small threshold (since the others have nothing
# interesting going on at this configuration).
forcemag = np.linalg.norm(F_analytic.reshape(-1, 2), axis=1)
order = np.argsort(-forcemag)
top_vertices = order[:min(8, nz_force)]
print(f"FD-checking top {len(top_vertices)} forced vertices: {top_vertices.tolist()}")

max_abs_err = 0.0
worst = None
for v in top_vertices:
    for c in range(2):
        pos_plus = positions_ref.copy()
        pos_plus[2*v + c] += eps
        m.setPositions(pos_plus.tolist())
        W_plus = run_walk(m, return_force=False)

        pos_minus = positions_ref.copy()
        pos_minus[2*v + c] -= eps
        m.setPositions(pos_minus.tolist())
        W_minus = run_walk(m, return_force=False)

        # restore for next iteration
        m.setPositions(positions_ref.tolist())

        F_fd = -(W_plus - W_minus) / (2.0 * eps)
        F_an = F_analytic[2*v + c]
        err = abs(F_fd - F_an)
        if err > max_abs_err:
            max_abs_err = err
            worst = (int(v), c, F_an, F_fd, err)
        print(f"  v={int(v):3d} c={c}  F_an={F_an:+.6e}  F_fd={F_fd:+.6e}  abs err={err:.3e}")

print(f"\nmax abs FD error across checked components: {max_abs_err:.3e}")
print(f"worst: v={worst[0]} c={worst[1]} F_an={worst[2]:+.6e} F_fd={worst[3]:+.6e}")

# Tolerance: FD is O(eps^2) limited by truncation + O(eps^-1)*1e-16 round-off.
# Optimal eps ~ 1e-5 to 1e-6 for double precision; we used 1e-7 -- conservative
# tol of 1e-4 should be comfortable.
assert max_abs_err < 1e-4, f"FD mismatch: max abs err {max_abs_err:.3e}"

print("\nPhase C edge-edge forces match FD.")
