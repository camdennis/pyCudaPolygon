import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pyPolygon')))
import numpy as np
import pyCudaPolygon as pcp

def regular_polygon(n, R, cx=0.0, cy=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([cx + R*np.cos(angles), cy + R*np.sin(angles)])

def _build_model(polys_verts, delta, box=1.0):
    all_verts = np.vstack(polys_verts) / box
    nv = len(all_verts)
    sizes = [len(v) for v in polys_verts]
    start_indices = [0]
    for s in sizes: start_indices.append(start_indices[-1] + s)
    m = pcp.model(nv)
    m.setStartIndices(start_indices)
    m.setPositions(all_verts.flatten().tolist())
    m.setModelEnum("rounded")
    m.setDelta(delta / box)
    m.setMaxEdgeLength(0.5)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    m.updateNeighbors()
    m.updatePolygonGeometry()
    return m

def get_energy(polys_verts, delta, box=1.0):
    m = _build_model(polys_verts, delta, box)
    m.updateForceEnergy()
    return m.getEnergy()

n = 5; R = 0.12; delta = 0.015; box = 1.0
v1 = regular_polygon(n, R, 0.5, 0.5)
v2 = regular_polygon(n, R, 0.5+0.175, 0.5)
all_verts = np.vstack([v1, v2])

m = _build_model([v1, v2], delta, box)
m.updateForceEnergy()
F0 = np.array(m.getForces())

eps = 1e-5
nv = len(all_verts)

print(f"{'vtx':4s}  {'coord':5s}  {'ANA_force':12s}  {'FD_force':12s}  {'rel_err':10s}  {'flag'}")
max_err = 0.0
for k in range(nv):
    for d in range(2):
        vp = all_verts.copy(); vp[k,d] += eps
        vm = all_verts.copy(); vm[k,d] -= eps
        Ep = get_energy([vp[:n], vp[n:]], delta, box)
        Em = get_energy([vm[:n], vm[n:]], delta, box)
        fd_force = -(Ep-Em)/(2*eps)
        ana = F0[2*k+d]
        err = abs(ana-fd_force)/(abs(fd_force)+1e-10)
        max_err = max(max_err, err)
        flag = "BAD" if err>0.05 else "ok"
        if abs(fd_force)>1e-8 or abs(ana)>1e-8:
            print(f"v{k:2d}  {'xy'[d]}   {ana:+.4e}  {fd_force:+.4e}  {err:10.3f}  {flag}")

print(f"\nmax_rel_err = {max_err:.4f}")
