"""test_edge_crossing_sufficiency.py — test the user's hypothesis: if every
crossing edge-pair has vertex pairs within 2*L, ball_radius = 2.5*L should
be sufficient for the rounded kernel to find all interactions.

Compare initial rounded energy at three search factors:
  - 2.5 (edge-crossing-sufficient by argument)
  - 2.5 + 2*polygonSize/pi (my current auto-default; covers polygon diameter)
  - 1000 (effectively all-to-all reference)
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build_relaxed(searchFactor):
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
    m.minimizeFIRE(maxForceThreshold=1e-14, maxSteps=3000, progressBar=False)
    m.setPhi(1.1)
    m.updatePolygonGeometry()
    m.setMaxEdgeLength()          # recompute, now reflects relaxed edges
    m.setSearchFactor(searchFactor)
    m.setModelEnum("rounded")
    m.initializeNeighborBall()
    m.updateNeighborBall()
    m.updateForceEnergy()
    return m

for sf in [2.5, 2.5 + 2*32/np.pi, 1000.0]:
    m = build_relaxed(sf)
    print(f"searchFactor={sf:7.2f}  ballR={sf*m.getMaxEdgeLength():.4f}  "
          f"meanBall={np.mean(m.getNumBallNeighbors()):6.1f}  "
          f"E={m.getEnergy():.4e}  |F|max={m.getMaxUnbalancedForce():.4e}")
