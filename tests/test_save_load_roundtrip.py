"""test_save_load_roundtrip.py — verify the refactored .npz save/load.
Build a model, set known scalars (including delta), save, load into a fresh
model, and assert every piece of state matches."""

import sys, os, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

N, n, seed = 8, 16, 7
kappa, rho = 3.7, 0.13

m = pcp.model(size=N*n, seed=seed)
m.setModelEnum("rounded")
m.generateRandomPolygons(N, n)
m.setBiPerimeters(kappa)
m.setDelta(rho)
m.setMaxEdgeLength(0.025)
m.setStiffness(1.0)
m.setCompressibility(1.0)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updatePolygonGeometry()
m.updateForceEnergy()

pos0   = np.asarray(m.getPositions(),         dtype=np.float64).copy()
nArr0  = np.asarray(m.getnArray(),            dtype=np.int64).copy()
tA0    = np.asarray(m.getTargetAreas(),       dtype=np.float64).copy()
tE0    = np.asarray(m.getTargetEdgeLengths(), dtype=np.float64).copy()
me0    = str(m.getModelEnum())
d0     = float(m.getDelta())
mel0   = float(m.getMaxEdgeLength())
st0    = float(m.getStiffness())
cmp0   = float(m.getCompressibility())
E0     = float(m.getEnergy())

print(f"saved: modelEnum={me0}  delta={d0:.4e}  E={E0:.6e}", flush=True)

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "roundtrip")
    m.saveModel(path)
    saved = path + ".npz"
    assert os.path.isfile(saved), f"saver did not produce {saved}"

    # overwrite-guard
    try:
        m.saveModel(path)
    except FileExistsError:
        print("overwrite guard OK", flush=True)
    else:
        raise AssertionError("expected FileExistsError on second save without overwrite=True")
    m.saveModel(path, overwrite=True)

    m2 = pcp.model(size=4, seed=seed+1)
    m2.loadModel(path)
    m2.updatePolygonGeometry()
    m2.updateNeighborCells()
    m2.updateForceEnergy()

    pos1  = np.asarray(m2.getPositions(),         dtype=np.float64)
    nArr1 = np.asarray(m2.getnArray(),            dtype=np.int64)
    tA1   = np.asarray(m2.getTargetAreas(),       dtype=np.float64)
    tE1   = np.asarray(m2.getTargetEdgeLengths(), dtype=np.float64)
    me1   = str(m2.getModelEnum())
    d1    = float(m2.getDelta())
    mel1  = float(m2.getMaxEdgeLength())
    st1   = float(m2.getStiffness())
    cmp1  = float(m2.getCompressibility())
    E1    = float(m2.getEnergy())

    def chk(name, a, b):
        ok = np.array_equal(a, b)
        print(f"  {name:>18s}  match={ok}  (shape {np.shape(a)})", flush=True)
        if not ok:
            raise AssertionError(f"{name} mismatch")

    chk("positions",         pos0, pos1)
    chk("nArray",            nArr0, nArr1)
    chk("targetAreas",       tA0,  tA1)
    chk("targetEdgeLengths", tE0,  tE1)
    assert me0  == me1,  f"modelEnum:       {me0!r} vs {me1!r}"
    assert d0   == d1,   f"delta:           {d0}    vs {d1}"
    assert mel0 == mel1, f"maxEdgeLength:   {mel0}  vs {mel1}"
    assert st0  == st1,  f"stiffness:       {st0}   vs {st1}"
    assert cmp0 == cmp1, f"compressibility: {cmp0}  vs {cmp1}"
    print(f"  scalars match  (delta={d1:.4e}, mel={mel1:.4e})", flush=True)
    print(f"  E_loaded = {E1:.6e}   |dE| = {abs(E1-E0):.3e}", flush=True)

    # path-without-suffix loading should also work
    m3 = pcp.model(size=4, seed=seed+2)
    m3.loadModel(path)
    assert float(m3.getDelta()) == d0
    print("  load without .npz suffix OK", flush=True)

print("\nALL CHECKS PASSED", flush=True)
