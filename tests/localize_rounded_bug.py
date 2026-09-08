"""
localize_rounded_bug.py — FD-test every vertex of poly 27 (the polygon with the
known rounded-model force bug) and a control polygon, to see whether the
wrong-force vertices form a localized cluster (indicating a specific arc with
interior crossings affecting its 3-vertex stencil).
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

nArray   = np.array(m.getnArray())
startIdx = np.concatenate([[0], np.cumsum(nArray)])
pos0     = np.array(m.getPositions())

def energy_at(pos):
    m.setPositions(pos)
    m.updateNeighborCells()
    m.updateForceEnergy()
    return m.getEnergy()

m.setPositions(pos0)
m.updateNeighborCells(); m.updateForceEnergy()
F = np.array(m.getForces())
h = 1e-7

def scan(p, label):
    s, n = startIdx[p], nArray[p]
    print(f"\n--- {label}  poly {p}  (vertices {s}..{s+n-1}, local 0..{n-1}) ---")
    print(f"{'loc':>4} {'vertex':>8} {'|F|':>11} "
          f"{'rel err x':>11} {'rel err y':>11}")
    for lv in range(n):
        v = s + lv
        fmag = np.hypot(F[2*v], F[2*v+1])
        errs = []
        for c in (0, 1):
            idx = 2*v + c
            pp = pos0.copy(); pp[idx] += h
            Ep = energy_at(pp)
            pm = pos0.copy(); pm[idx] -= h
            Em = energy_at(pm)
            fd = -(Ep - Em) / (2*h)
            rel = abs(F[idx] - fd) / (abs(fd) + 1e-30)
            errs.append(rel)
        flag = "  <-- WRONG" if max(errs) > 1e-2 else ""
        print(f"{lv:>4} {v:>8} {fmag:>11.4e} "
              f"{errs[0]:>11.2e} {errs[1]:>11.2e}{flag}")

scan(27, "TARGET (known bad)")
scan( 8, "control (known good)")
