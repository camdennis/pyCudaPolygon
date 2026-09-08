"""
test_force_consistency.py — finite-difference check that the rounded model's
force equals -dE/dx. If F != -grad E, FIRE minimises the wrong thing and the
true energy oscillates.
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

nArray   = np.array(m.getnArray())
startIdx = np.concatenate([[0], np.cumsum(nArray)])
pos0     = np.array(m.getPositions())

def straddles(p):
    s, n = startIdx[p], nArray[p]
    vx = pos0[2*s:2*(s+n):2]; vy = pos0[2*s+1:2*(s+n):2]
    return (vx.max()-vx.min() > 0.5) or (vy.max()-vy.min() > 0.5)

def run_fd(model_enum):
    print(f"\n================  model = {model_enum}  ================")

    def energy_at(pos):
        m.setPositions(pos)
        m.updateNeighborCells(); m.updateNeighbors()
        m.updateForceEnergy()
        return m.getEnergy()

    m.setModelEnum(model_enum)
    m.setPositions(pos0)
    m.updateNeighborCells(); m.updateNeighbors(); m.updateForceEnergy()
    E0 = m.getEnergy()
    F  = np.array(m.getForces())          # F = -dE/dx
    print(f"E0 = {E0:.8e}")

    h = 1e-7
    worst = 0.0
    # fixed vertex set (rounded's high-force vertices) so both models are
    # probed at the SAME geometry
    test_vertices = [662, 663, 252, 253, 976, 977, 278, 279, 883, 885]
    poly_of = lambda v: int(np.searchsorted(startIdx, v, side='right') - 1)
    print(f"{'vertex':>8} {'poly':>5} {'strad':>6} {'comp':>5} "
          f"{'F (kernel)':>14} {'-dE/dx (FD)':>14} {'abs err':>11} {'rel err':>10}")
    for v in test_vertices:
        p = poly_of(v)
        for c in (0, 1):
            idx = 2*v + c
            pp = pos0.copy(); pp[idx] += h
            Ep = energy_at(pp)
            pm = pos0.copy(); pm[idx] -= h
            Em = energy_at(pm)
            fd = -(Ep - Em) / (2*h)
            err = abs(F[idx] - fd)
            rel = err / (abs(fd) + 1e-30)
            worst = max(worst, err)
            print(f"{v:>8} {p:>5} {str(straddles(p)):>6} "
                  f"{('x' if c==0 else 'y'):>5} "
                  f"{F[idx]:>14.6e} {fd:>14.6e} {err:>11.2e} {rel:>10.2e}")
    print(f"worst abs err = {worst:.3e}")

run_fd("rounded")
run_fd("areaSquared")
