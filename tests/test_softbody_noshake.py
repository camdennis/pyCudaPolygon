"""softBody minimisation without SHAKE — does it actually converge?"""

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

print(f"{'after':>10}  {'E':>12}  {'|F|max':>12}", flush=True)
m.updateForceEnergy()
print(f"{'init':>10}  {m.getEnergy():>12.4e}  {m.getMaxUnbalancedForce():>12.4e}", flush=True)

for ns in (500, 1000, 2000, 5000, 10000, 20000):
    m.minimizeFIRE(maxForceThreshold=1e-30, maxSteps=ns,
                   shakeIter=0, progressBar=False)
    m.updateForceEnergy()
    print(f"{ns:>10d}  {m.getEnergy():>12.4e}  {m.getMaxUnbalancedForce():>12.4e}", flush=True)
