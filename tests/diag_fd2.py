import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
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

def energy_at(verts_flat, n, delta, box):
    v1 = verts_flat[:2*n].reshape(n, 2)
    v2 = verts_flat[2*n:].reshape(n, 2)
    m = _build_model([v1, v2], delta, box)
    m.updateForceEnergy()
    return m.getEnergy()

n = 5; R = 0.12; delta = 0.015; box = 1.0
eps = 1e-5; atol = 1e-10

v1 = regular_polygon(n, R, 0.5, 0.5)
v2 = regular_polygon(n, R, 0.5+0.175, 0.5)
all_verts = np.concatenate([v1.flatten(), v2.flatten()])

m = _build_model([v1, v2], delta, box)
m.updateForceEnergy()
F0 = np.array(m.getForces())

print(f"{'vtx':4s}  {'coord':5s}  {'CUDA_force':12s}  {'FD_grad':12s}  {'rel_err':10s}")
max_err = 0.0
max_info = None
check_verts = list(range(min(4, n))) + list(range(n, n + min(4, n)))
for k in check_verts:
    for d in range(2):
        idx = 2*k + d
        vp = all_verts.copy(); vp[idx] += eps
        vm = all_verts.copy(); vm[idx] -= eps
        Ep = energy_at(vp, n, delta, box)
        Em = energy_at(vm, n, delta, box)
        fd_grad = (Ep - Em)/(2*eps)
        cuda_force = F0[idx]
        err = abs(cuda_force + fd_grad)
        scale = abs(fd_grad) + atol
        rel = err / scale
        max_err = max(max_err, rel)
        if rel == max_err:
            max_info = (k, d, cuda_force, fd_grad, rel)
        flag = "BAD" if rel>0.05 else "ok"
        print(f"v{k:2d}  {'xy'[d]}   {cuda_force:+.6e}  {fd_grad:+.6e}  {rel:10.4f}  {flag}")

print(f"\nmax_rel_err = {max_err:.4f}")
if max_info:
    k, d, cf, fg, rel = max_info
    print(f"Worst: v{k}[{'xy'[d]}]  cuda={cf:.4e}  fd_grad={fg:.4e}  both_tiny={abs(fg)<1e-8}")
