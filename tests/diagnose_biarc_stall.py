"""
diagnose_biarc_stall.py — replicate biArcTesting.ipynb up to the stuck state,
then finite-difference the rounded-model force against the energy.

If F == -dE/dx at the stuck state, the maxUnbalancedForce plateau is a genuine
(constrained) minimum. If F != -dE/dx, the force-consistency bug is why FIRE
cannot drive the force down.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

N, n, seed, kappa, rho = 64, 32, 42, 3.7, 0.001

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
m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=3000, progressBar=False)
m.setPhi(1.0)

m.setModelEnum("rounded")
m.setMaxEdgeLength(0.1)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateOutersections()
m.updateForceEnergy()
print("rounded minimise ...", flush=True)
m.minimizeFIRE(maxSteps=1, shakeIter=100, progressBar=False)
m.minimizeFIRE(dt=1e-3, maxSteps=1, shakeIter=100, progressBar=False)
m.minimizeFIRE(dt=1e-8, maxSteps=1000, shakeIter=100, progressBar=False)

print(f"\nstuck state:  E = {m.getEnergy():.6e}   "
      f"maxUnbalancedForce = {m.getMaxUnbalancedForce():.6e}")

# ── finite-difference force vs energy at the stuck state ─────────────────────
nArray   = np.array(m.getnArray())
startIdx = np.concatenate([[0], np.cumsum(nArray)])
pos0     = np.array(m.getPositions())

def energy_at(pos):
    m.setPositions(pos)
    m.updateNeighborCells()
    m.updateForceEnergy()
    return m.getEnergy()

m.setPositions(pos0)
m.updateNeighborCells()
m.updateForceEnergy()
E0 = m.getEnergy()
F  = np.array(m.getForces())          # F = -dE/dx

fmag = np.hypot(F[0::2], F[1::2])
top = np.argsort(fmag)[-12:][::-1]    # 12 highest-force vertices

h = 1e-7
print(f"\n{'vertex':>8} {'|F|':>12} {'comp':>5} {'F (kernel)':>14} "
      f"{'-dE/dx (FD)':>14} {'abs err':>11} {'rel err':>10}")
worst = 0.0
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
        worst = max(worst, rel)
        print(f"{v:>8} {fmag[v]:>12.5e} {('x' if c==0 else 'y'):>5} "
              f"{F[idx]:>14.6e} {fd:>14.6e} {err:>11.2e} {rel:>10.2e}")
print(f"\nworst relative error = {worst:.3e}")
print("=> force is INCONSISTENT with energy" if worst > 1e-3
      else "=> force is consistent; plateau is a genuine/constrained minimum")
