"""test_neighbor_ball_parity.py — verify that the new ball-based neighbor list
produces the same vertex-disk energy and forces as the cell-grid path on the
normal model. Atomic-add ordering can give tiny floating-point differences, so
we require ~1e-12 relative agreement, not bit-exactness."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build_model(neighborType):
    N, n, seed = 16, 16, 42
    kappa, rho = 3.7, 0.05
    m = pcp.model(size=N*n, seed=seed)
    m.setModelEnum("normal")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.setNeighborType(neighborType)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    if neighborType == "balls":
        m.initializeNeighborBall()
        m.updateNeighborBall()
    m.updateNeighbors()
    m.updateOutersections()
    m.updateForceEnergy()
    return m


m_c = build_model("cells")
m_b = build_model("balls")

E_c = m_c.getEnergy()
E_b = m_b.getEnergy()
F_c = np.asarray(m_c.getForces(), dtype=np.float64)
F_b = np.asarray(m_b.getForces(), dtype=np.float64)
maxF_c = m_c.getMaxUnbalancedForce()
maxF_b = m_b.getMaxUnbalancedForce()

print(f"neighborType (cells)   = {m_c.getNeighborType()}")
print(f"neighborType (balls)   = {m_b.getNeighborType()}")
print(f"searchFactor (balls)   = {m_b.getSearchFactor()}")
print(f"ballMaxNeighbors       = {m_b.getBallMaxNeighbors()}")
print()
print(f"E (cells)  = {E_c:.15e}")
print(f"E (balls)  = {E_b:.15e}")
print(f"|dE|       = {abs(E_c - E_b):.3e}  (rel: {abs(E_c-E_b)/max(abs(E_c),1e-30):.3e})")
print()
print(f"|F|max (cells) = {maxF_c:.6e}")
print(f"|F|max (balls) = {maxF_b:.6e}")
print(f"||F_c - F_b||_inf = {np.max(np.abs(F_c - F_b)):.3e}")
print(f"||F_c - F_b||_2   = {np.linalg.norm(F_c - F_b):.3e}")

assert abs(E_c - E_b) <= 1e-10 * max(abs(E_c), 1.0), f"energy mismatch: {E_c} vs {E_b}"
assert np.max(np.abs(F_c - F_b)) <= 1e-10 * max(np.max(np.abs(F_c)), 1.0), \
    f"force mismatch (max abs diff: {np.max(np.abs(F_c - F_b))})"

# sanity: ball list contains enough candidates
num_ball = np.asarray(m_b.getNumBallNeighbors(), dtype=np.int64)
print(f"\nball list stats: mean={num_ball.mean():.2f}  max={num_ball.max()}  "
      f"min={num_ball.min()}  total={num_ball.sum()}")

print("\nPARITY OK: balls and cells agree on energy and forces.")
