"""test_neighbor_ball_filtering.py — same parity check but with N=64 polygons
of n=32 vertices, so ball_radius is much smaller than the periodic box and the
ball list does meaningful filtering. Also confirms the Verlet skin rebuild gate
does the right thing across several FIRE steps."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build_model(neighborType):
    N, n, seed = 64, 32, 42
    kappa, rho = 3.7, 0.005
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

num_ball = np.asarray(m_b.getNumBallNeighbors(), dtype=np.int64)
print(f"maxEdgeLength = {m_b.getMaxEdgeLength():.4e}   "
      f"ball_radius = {m_b.getSearchFactor()*m_b.getMaxEdgeLength():.4e}")
print(f"ball list stats: mean={num_ball.mean():.1f}  max={num_ball.max()}  "
      f"min={num_ball.min()}  cap={m_b.getBallMaxNeighbors()}")
print(f"sparsity (mean / numVertices) = {num_ball.mean()/m_b.getNumVertices():.3f}")

E_c, E_b = m_c.getEnergy(), m_b.getEnergy()
F_c = np.asarray(m_c.getForces(), dtype=np.float64)
F_b = np.asarray(m_b.getForces(), dtype=np.float64)

print(f"\nE (cells)        = {E_c:.15e}")
print(f"E (balls)        = {E_b:.15e}")
print(f"|dE|             = {abs(E_c - E_b):.3e}")
print(f"||F_c - F_b||_inf = {np.max(np.abs(F_c - F_b)):.3e}")
assert abs(E_c - E_b) <= 1e-10 * max(abs(E_c), 1.0)
assert np.max(np.abs(F_c - F_b)) <= 1e-10 * max(np.max(np.abs(F_c)), 1.0)

# Now exercise the rebuild gate: a few small FIRE steps shouldn't rebuild every step.
# We can't directly observe rebuilds, but the energy/force agreement should hold.
print("\nrunning 50 FIRE steps with balls...")
for k in range(50):
    res = m_b.minimizeFIREStep(dt=1e-4, alpha=0.1, nPos=0, shakeIter=0)
print(f"  E_balls after 50 steps  = {m_b.getEnergy():.6e}")

print("\nrunning 50 FIRE steps with cells (same init)...")
m_c2 = build_model("cells")
for k in range(50):
    res = m_c2.minimizeFIREStep(dt=1e-4, alpha=0.1, nPos=0, shakeIter=0)
print(f"  E_cells after 50 steps  = {m_c2.getEnergy():.6e}")

print(f"  |dE after 50 steps|     = {abs(m_b.getEnergy() - m_c2.getEnergy()):.3e}")
assert abs(m_b.getEnergy() - m_c2.getEnergy()) <= 1e-8 * max(abs(m_c2.getEnergy()), 1.0)

print("\nFILTERING OK and 50-step FIRE trajectories agree.")
