"""diagnose_energytoo.py — load energyTooSmall and check whether the suspiciously
low energy is from missed pairs (neighbor cutoff too small)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

path = os.path.join(os.path.dirname(__file__), "energyTooSmall")

m = pcp.model(size=2048)
m.loadModel(path)
m.setModelEnum("rounded")
m.updatePolygonGeometry()

print(f"loaded:  modelEnum={m.getModelEnum()}  delta={m.getDelta():.4e}", flush=True)

for mel in (0.0174, 0.05, 0.1, 0.2, 0.5):
    m.setMaxEdgeLength(mel)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    m.updateNeighbors()
    m.updateForceEnergy()
    E   = m.getEnergy()
    Fmx = m.getMaxUnbalancedForce()
    # also: which boxSize?  numNonzero pairs (via pairArea if available)
    box = int(1.0 / mel)
    try:
        pa = np.array(m.getPairArea()).reshape(64, 64)
        n_nz = int((np.abs(pa + pa.T) > 1e-14).sum() // 2)
    except Exception:
        n_nz = -1
    print(f"  maxEdgeLength = {mel:.4f}  boxSize ≈ {box:>3d}  "
          f"E = {E:.4e}  |F|max = {Fmx:.4e}  nonzero pairs = {n_nz}",
          flush=True)
