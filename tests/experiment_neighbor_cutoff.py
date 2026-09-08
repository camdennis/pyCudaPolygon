"""
experiment_neighbor_cutoff.py — test whether the bad pairArea values are caused
by the neighbour-cell cutoff (boxSize from maxEdgeLength) being too small to
register deeply-overlapping polygon pairs.

Builds the model twice: once with the production maxEdgeLength (boxSize=6, 3x3
scan reaches +/-1 cell) and once with a coarse grid (boxSize=3, the 3x3 scan
covers the whole box -> every pair fully processed). If the bad pairs come out
correct only with boxSize=3, the cutoff is the bug.
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

BAD  = [(20, 30), (7, 20), (7, 30), (10, 22)]
GOOD = [(8, 27), (12, 19), (0, 13)]
# grid-truth intersection areas from verify_pairarea.py
GRID = {(20,30):1.34842e-2, (7,20):1.09212e-2, (7,30):8.87742e-3,
        (10,22):6.10243e-3, (8,27):8.81897e-3, (12,19):1.08385e-2,
        (0,13):1.36087e-2}

def build(max_edge_length):
    m = pcp.model(N_POLY * N_SIDES, seed=42)
    m.generateRandomPolygons(N_POLY, N_SIDES)
    m.setModelEnum("rounded")
    m.setDelta(DELTA)
    m.setTargetAreas(np.array([A_LARGE] * N_LARGE + [A_SMALL] * N_SMALL))
    m.setTargetEdgeLengths(np.array([EL_LARGE] * N_LARGE + [EL_SMALL] * N_SMALL))
    m.setMaxEdgeLength(max_edge_length)
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
    pa = np.array(m.getPairArea()).reshape(N_POLY, N_POLY)
    return pa, m.getEnergy()

# production grid (maxEdgeLength ~ 0.148 -> boxSize 6)
MEL_PROD = 2 * np.sqrt(A_LARGE / np.pi) + 2 * DELTA
pa6, E6 = build(MEL_PROD)
print(f"production : maxEdgeLength={MEL_PROD:.4f}  boxSize={int(1.0/MEL_PROD)}  "
      f"E0={E6:.5e}")

# coarse grid (maxEdgeLength 0.30 -> boxSize 3 -> 3x3 scan covers whole box)
pa3, E3 = build(0.30)
print(f"coarse     : maxEdgeLength=0.3000  boxSize={int(1.0/0.30)}  E0={E3:.5e}")
print()

print(f"{'pair':>10} {'boxSize6 sum':>14} {'boxSize3 sum':>14} {'grid area':>13} {'verdict':>22}")
for tag, pairs in (("BAD", BAD), ("good", GOOD)):
    for A, B in pairs:
        s6 = pa6[A, B] + pa6[B, A]
        s3 = pa3[A, B] + pa3[B, A]
        g  = GRID[(A, B)]
        ok3 = abs(s3 - g) / g < 0.02
        verdict = "boxSize3 -> CORRECT" if ok3 else "still wrong"
        print(f"({A:>3},{B:>3}) {s6:>14.5e} {s3:>14.5e} {g:>13.5e} {verdict:>22}")
