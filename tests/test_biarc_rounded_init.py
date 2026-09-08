"""test_biarc_rounded_init.py — reproduce the user's flow up to the rounded
initial state and verify ball reach catches all overlap pairs (E close to the
all-to-all reference)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build_to_rounded_init():
    N, n, seed = 64, 32, 42
    kappa, rho = 3.7, 0.001
    m = pcp.model(size=N*n, seed=seed)
    m.setModelEnum("softBody")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1)
    m.setCompressibility(1)
    m.initializeNeighborBall()
    m.updateForceEnergy()
    m.minimizeFIRE(maxForceThreshold=1e-14, maxSteps=20000, progressBar=False)
    m.setPhi(1.1)
    m.updatePolygonGeometry()
    m.setModelEnum("rounded")
    m.initializeNeighborBall()
    m.updatePolygonGeometry()
    m.updateNeighborBall()
    m.updateForceEnergy()
    return m


m = build_to_rounded_init()
print(f"polygonSize    = {m.getNumVertices()//m.getNumPolygons()}")
print(f"maxEdgeLength  = {m.getMaxEdgeLength():.4e}")
print(f"searchFactor   = {m.getSearchFactor()}")
print(f"mean ball ct   = {np.mean(m.getNumBallNeighbors()):.1f}")
print(f"E (balls)      = {m.getEnergy():.4e}")
print(f"|F|max (balls) = {m.getMaxUnbalancedForce():.4e}")

# Now build with effectively all-to-all by setting a HUGE searchFactor so we
# get a reference answer that processes every cross-polygon pair.
m_ref = build_to_rounded_init()
m_ref.setSearchFactor(1000.0)   # ensure all-to-all
m_ref.updateNeighborBall()
m_ref.updateForceEnergy()
print(f"\nreference (all-to-all):")
print(f"mean ball ct   = {np.mean(m_ref.getNumBallNeighbors()):.1f}")
print(f"E (ref)        = {m_ref.getEnergy():.4e}")
print(f"|F|max (ref)   = {m_ref.getMaxUnbalancedForce():.4e}")
print(f"\n|dE|           = {abs(m.getEnergy() - m_ref.getEnergy()):.3e}")
print(f"rel-dE         = {abs(m.getEnergy() - m_ref.getEnergy())/max(abs(m_ref.getEnergy()),1.0):.3e}")
