"""
check_straddle.py — does each polygon straddle the periodic box boundary?

If A's raw vertices span the x or y wrap, the rounded kernel's per-vertex-thread
MIC unwrapping puts different threads in box-shifted frames, breaking the
telescoping sum of origin-anchored Green's terms in I(A->B).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
import pyCudaPolygon as pcp

N_POLY, N_SIDES = 64, 32
N_LARGE = N_SMALL = N_POLY // 2
KAPPA, RATIO, PHI = 3.7, 1.4, 0.8
A_SMALL = PHI / (N_LARGE * RATIO**2 + N_SMALL)
A_LARGE = RATIO**2 * A_SMALL
EL_LARGE = KAPPA * np.sqrt(A_LARGE) / N_SIDES
EL_SMALL = KAPPA * np.sqrt(A_SMALL) / N_SIDES
DELTA = 0.15 * EL_SMALL

m = pcp.model(N_POLY * N_SIDES, seed=42)
m.generateRandomPolygons(N_POLY, N_SIDES)
m.setModelEnum("rounded")
m.setDelta(DELTA)
m.setTargetAreas(np.array([A_LARGE] * N_LARGE + [A_SMALL] * N_SMALL))
m.setTargetEdgeLengths(np.array([EL_LARGE] * N_LARGE + [EL_SMALL] * N_SMALL))
m.setMaxEdgeLength(2 * np.sqrt(A_LARGE / np.pi) + 2 * DELTA)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateNeighbors()
m.updatePolygonGeometry()
m.setStiffness(1)
m.setCompressibility(1)
m.updateForceEnergy()
m.setModelEnum("softBody")
m.minimizeFIRE(dt=0.1, maxForceThreshold=1e-14)
m.updatePolygonGeometry()
m.setModelEnum("rounded")
m.updateForceEnergy()

positions = np.array(m.getPositions())
nArray    = np.array(m.getnArray())
startIdx  = np.concatenate([[0], np.cumsum(nArray)])

def straddle(p):
    s, n = startIdx[p], nArray[p]
    vx = positions[2*s:2*(s+n):2]
    vy = positions[2*s+1:2*(s+n):2]
    # a non-straddling polygon (diameter ~0.15) has raw span < 0.5;
    # a straddling one has raw vertices near both 0 and 1
    return (vx.max()-vx.min() > 0.5), (vy.max()-vy.min() > 0.5)

bad_polys  = sorted({7, 20, 30, 10, 22})
good_polys = sorted({8, 27, 12, 19, 0, 13})

print("BAD-pair polygons:")
for p in bad_polys:
    sx, sy = straddle(p)
    print(f"  poly {p:>2}: straddle_x={sx}  straddle_y={sy}")
print("good-pair polygons:")
for p in good_polys:
    sx, sy = straddle(p)
    print(f"  poly {p:>2}: straddle_x={sx}  straddle_y={sy}")

n_straddle = sum(any(straddle(p)) for p in range(N_POLY))
print(f"\ntotal straddling polygons: {n_straddle}/{N_POLY}")
