"""
test_fire_floor.py — drive softBody FIRE for 10000 steps step-by-step,
tracking dt, alpha, nPos, energy, |F|. softBody is *not* rigid → no SHAKE
or projection in FIRE step. Goal: find out why |F| floors at ~1e-9.
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

# FIRE state
dt    = 1e-3
alpha = 0.1
nPos  = 0
E     = m.getEnergy()
F     = m.getMaxUnbalancedForce()

print(f"{'step':>6}  {'E':>14}  {'|F|max':>13}  {'dt':>11}  {'alpha':>8}  "
      f"{'nPos':>5}  {'dE':>13}  {'note':<20}", flush=True)
print(f"{0:>6d}  {E:>14.6e}  {F:>13.6e}  {dt:>11.3e}  {alpha:>8.3e}  "
      f"{nPos:>5d}", flush=True)

E_prev = E
dt_prev = dt
nPos_resets = 0
last_log = 0

for step in range(1, 10001):
    E, dt, alpha, nPos = m.minimizeFIREStep(dt, alpha, nPos, shakeIter=0)
    note = ""
    if dt != dt_prev:
        if dt < dt_prev:
            note = "dt halved (P<0)"
            nPos_resets += 1
        else:
            note = "dt ramp"
    dE = E - E_prev

    # log: regular log-spaced + every dt change + every plateau
    do_log = step in (1, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
    do_log = do_log or note.startswith("dt halved") or note.startswith("dt ramp")
    if do_log and step - last_log >= 1:
        # query |F| only on log steps to avoid extra GPU sync every iter
        F = m.getMaxUnbalancedForce()
        print(f"{step:>6d}  {E:>14.6e}  {F:>13.6e}  {dt:>11.3e}  {alpha:>8.3e}  "
              f"{nPos:>5d}  {dE:>13.3e}  {note:<20}", flush=True)
        last_log = step
    dt_prev = dt
    E_prev  = E

print(f"\nTotal P<0 (velocity-resetting) events: {nPos_resets}", flush=True)
F_final = m.getMaxUnbalancedForce()
print(f"final |F|max = {F_final:.6e}   final dt = {dt:.6e}", flush=True)
