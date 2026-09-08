"""
test_normal_force.py — finite-difference check that the 'normal' model's force
equals -dE/dx. Mirrors test_force_consistency.py but for the normal model,
which uses updateForceEnergyExteriorKernel + InteriorKernel + the global
sign-flip in model.cpp.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

N, n, seed, kappa, rho = 64, 32, 42, 3.7, 0.001

# build the same way biArcTesting.ipynb does, briefly softBody-relax, then
# switch to "normal" model
m = pcp.model(size=N*n, seed=seed)
m.setModelEnum("softBody")
m.generateRandomPolygons(N, n)
m.setBiPerimeters(kappa)
m.setDelta(rho)
m.setStiffness(1)
m.setCompressibility(1)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateOutersections()
m.updateForceEnergy()
print("softBody relax ...", flush=True)
m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=2000, progressBar=False)
m.setPhi(1.0)

m.setModelEnum("normal")
m.setMaxEdgeLength(0.1)
m.initializeNeighborCells()

def refresh():
    m.updateNeighborCells()
    m.updateNeighbors()
    m.updateOutersections()
    m.updateForceEnergy()

refresh()
nArray   = np.array(m.getnArray())
startIdx = np.concatenate([[0], np.cumsum(nArray)])
pos0     = np.array(m.getPositions())

def energy_at(pos):
    m.setPositions(pos)
    refresh()
    return m.getEnergy()

m.setPositions(pos0); refresh()
E0 = m.getEnergy()
F  = np.array(m.getForces())          # F = -dE/dx
print(f"normal model  E0 = {E0:.6e}   |F|max = {np.max(np.abs(F)):.6e}")

fmag = np.hypot(F[0::2], F[1::2])
top = np.argsort(fmag)[-12:][::-1]    # 12 highest-force vertices
h = 1e-7
print(f"\n{'vertex':>8} {'|F|':>12} {'comp':>5} {'F (kernel)':>14} "
      f"{'-dE/dx (FD)':>14} {'abs err':>11} {'rel err':>10}")
worst_rel = 0.0
worst_abs = 0.0
for v in top:
    for c in (0, 1):
        idx = 2*v + c
        pp = pos0.copy(); pp[idx] += h
        Ep = energy_at(pp)
        pm = pos0.copy(); pm[idx] -= h
        Em = energy_at(pm)
        fd = -(Ep - Em) / (2*h)
        err = abs(F[idx] - fd)
        rel = err / (abs(fd) + 1e-30)
        worst_rel = max(worst_rel, rel)
        worst_abs = max(worst_abs, err)
        print(f"{v:>8} {fmag[v]:>12.5e} {('x' if c==0 else 'y'):>5} "
              f"{F[idx]:>14.6e} {fd:>14.6e} {err:>11.2e} {rel:>10.2e}")
print(f"\nworst abs err = {worst_abs:.3e}   worst rel err = {worst_rel:.3e}")
print("=> normal force is INCONSISTENT with energy" if worst_rel > 1e-3
      else "=> normal force is consistent (F == -dE/dx)")
