"""test_rounded_ball_parity.py — verify Phase 2 unified candidate list.

After Phase 2, the rounded kernel walks a per-vertex flat candidate list
populated by either updateNeighborCells (cell-stencil expansion) or
updateNeighborBall (Verlet build). When both builders produce the same
candidate set (which they do whenever the ball radius covers the cell
stencil's diagonal reach), the kernel gives bit-identical forces and
energies.

Note: over a FIRE trajectory the two paths drift apart because the cells
path's reach is frozen at setMaxEdgeLength-time (boxSize fixed) while the
ball radius adapts to the live maxEdgeLength. That's an intentional
difference; this test verifies single-shot equivalence at a fixed state."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build(neighborType, searchFactor=3.0):
    N, n, seed = 32, 16, 7
    kappa, rho = 3.7, 0.01
    m = pcp.model(size=N*n, seed=seed)
    m.setModelEnum("rounded")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.setNeighborType(neighborType)
    m.setSearchFactor(searchFactor)
    # Pin maxEdgeLength explicitly so both cells (boxSize=2 -> all-to-all) and
    # balls (ball_radius = searchFactor*0.5 -> all-to-all in a unit box) see
    # the same candidate set. With the auto-compute default this could be
    # small enough that the square cell stencil and circular ball cover
    # slightly different corners, breaking the parity assertion.
    m.setMaxEdgeLength(0.5)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    if neighborType == "balls":
        m.initializeNeighborBall()
        m.updateNeighborBall()
    m.updatePolygonGeometry()
    m.updateForceEnergy()
    return m


m_c = build("cells")
m_b = build("balls")

num_c = np.asarray(m_c.getNumBallNeighbors(), dtype=np.int64)
num_b = np.asarray(m_b.getNumBallNeighbors(), dtype=np.int64)
print(f"candidate counts (cells path): min={num_c.min()} max={num_c.max()} mean={num_c.mean():.1f}")
print(f"candidate counts (balls path): min={num_b.min()} max={num_b.max()} mean={num_b.mean():.1f}")
print(f"per-vertex count diff (cells - balls): "
      f"min={int((num_c - num_b).min())} max={int((num_c - num_b).max())}")

# Compare sets per vertex (positions are identical between m_c and m_b — same seed/setup)
ball_c = np.asarray(m_c.getBallNeighbors(), dtype=np.int64).reshape(-1, m_c.getBallMaxNeighbors())
ball_b = np.asarray(m_b.getBallNeighbors(), dtype=np.int64).reshape(-1, m_b.getBallMaxNeighbors())
N = m_c.getNumVertices()
n_diff = 0
for i in range(N):
    set_c = set(ball_c[i, :num_c[i]].tolist())
    set_b = set(ball_b[i, :num_b[i]].tolist())
    if set_c != set_b:
        n_diff += 1
        if n_diff <= 3:
            only_c = sorted(set_c - set_b)
            only_b = sorted(set_b - set_c)
            print(f"  vertex {i}: |c|={len(set_c)} |b|={len(set_b)}  "
                  f"only_c={only_c[:5]}{'...' if len(only_c)>5 else ''}  "
                  f"only_b={only_b[:5]}{'...' if len(only_b)>5 else ''}")
print(f"vertices where candidate sets differ: {n_diff} / {N}")

E_c = m_c.getEnergy()
E_b = m_b.getEnergy()
F_c = np.asarray(m_c.getForces(), dtype=np.float64)
F_b = np.asarray(m_b.getForces(), dtype=np.float64)

print(f"E (rounded, cells path) = {E_c:.15e}")
print(f"E (rounded, balls path) = {E_b:.15e}")
print(f"|dE|                    = {abs(E_c - E_b):.3e}")
print(f"||F_c - F_b||_inf       = {np.max(np.abs(F_c - F_b)):.3e}")
print(f"||F_c - F_b||_2         = {np.linalg.norm(F_c - F_b):.3e}")

# atomicAdd ordering noise is ~1e-14 for the rounded kernel at this size
assert abs(E_c - E_b) <= 1e-12 * max(abs(E_c), 1.0), f"energy mismatch: {E_c} vs {E_b}"
assert np.max(np.abs(F_c - F_b)) <= 1e-12 * max(np.max(np.abs(F_c)), 1.0)

print(f"\nballs path rebuild count: {m_b.getBallRebuildCount()}")
assert m_b.getBallRebuildCount() >= 1, "ball list never rebuilt"

print("\nROUNDED BALL PARITY OK (single-shot, fixed state).")
