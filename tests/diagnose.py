import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyCudaPolygon'))
import pyCudaPolygon as pcp
from pyCudaPolygonLink import libpyCudaPolygon as lpcp
import numpy as np

SAVE = os.path.join(os.path.dirname(__file__), 'testSave')

def load_fresh():
    m = pcp.model(size=1)
    m.loadModel(SAVE)
    m.initForceEnergy()
    m.updatePolygonGeometry()
    m.updateForceEnergy()
    m.projectForce()
    return m

# ── 1. Basic state ──────────────────────────────────────────────────────────
m = load_fresh()
print("=== Initial state ===")
print(f"  numPolygons={m.getNumPolygons()}, n={m.getnArray()[0]}")
print(f"  phi={m.getPhi():.6f}")
print(f"  energy={m.getEnergy():.6e}")
print(f"  maxForce={m.getMaxUnbalancedForce():.6e}")
print(f"  numIntersections={m.getNumIntersections()}")
print(f"  constraintViolation={m.getConstraintViolation()}", flush=True)

# ── 2. Check projected force magnitude ─────────────────────────────────────
forces = m.getForces()
rawForce = np.max(np.abs(forces))
print(f"\n=== Force analysis ===")
print(f"  max |f_raw|   = {rawForce:.6e} (after projectForce)")
print(f"  norm of forces = {np.linalg.norm(forces):.6e}", flush=True)

# ── 3. Step-by-step FIRE tracing ────────────────────────────────────────────
print("\n=== Single FIRE step trace (no rollback) ===")
m2 = load_fresh()
lpcp.Model.resetVelocities(m2)
m2.updatePolygonGeometry()
m2.updateForceEnergy()
m2.projectForce()

dt = 1e-3
alpha = 0.1
nPos = 0
pos_before = m2.getPositions().copy()
e_before = m2.getEnergy()
ni_before = m2.getNumIntersections()
print(f"  Before step: E={e_before:.8e}, ni={ni_before}", flush=True)

energy, dt_new, alpha_new, nPos_new = m2.minimizeFIREStep(dt, alpha, nPos, shakeIter=5)
pos_after = m2.getPositions()
e_after = m2.getEnergy()
ni_after = m2.getNumIntersections()
shake_iters = m2.getLastShakeIters()

max_disp = np.max(np.abs(pos_after - pos_before))
# wrap displacements
d = pos_after - pos_before + 1.5
d %= 1.0
d -= 0.5
max_disp_wrapped = np.max(np.abs(d))

print(f"  After step:  E={e_after:.8e}, ni={ni_after}")
print(f"  dE = {e_after - e_before:.4e}, dt_new={dt_new:.4e}")
print(f"  max_disp (raw)={max_disp:.4e}, wrapped={max_disp_wrapped:.4e}")
print(f"  SHAKE iters used: {shake_iters}")
print(f"  constraintViol after: {m2.getConstraintViolation()}", flush=True)

# ── 4. Near-parallel edge detection ────────────────────────────────────────
print("\n=== Near-parallel edge analysis ===")
m3 = load_fresh()
m3.updatePolygonGeometry()
m3.updateNeighborCells()
m3.updateNeighbors()

positions = m3.getPositions().reshape(-1, 2)
nArray = m3.getnArray()
startIndices = m3.getStartIndices()
nVert = m3.getNumVertices()

def get_edge_vec(positions, i, n_next):
    d = positions[n_next] - positions[i]
    d = d - np.round(d)
    return d

near_parallel_count = 0
near_parallel_examples = []
eps_parallel = 0.01  # sin(angle) < 0.01 → angle < 0.57 degrees

for p in range(m3.getNumPolygons()):
    s = startIndices[p]
    n = nArray[p]
    edge_vecs = []
    for k in range(n):
        ki = s + k
        nk = s + (k + 1) % n
        d = positions[nk] - positions[ki]
        d = d - np.round(d)
        length = np.linalg.norm(d)
        if length > 1e-14:
            edge_vecs.append(d / length)
        else:
            edge_vecs.append(np.array([0.0, 0.0]))
    # Check all pairs of edges within the polygon
    for i in range(n):
        for j in range(i+2, n):
            if i == 0 and j == n-1:
                continue  # adjacent
            cross = abs(edge_vecs[i][0]*edge_vecs[j][1] - edge_vecs[i][1]*edge_vecs[j][0])
            if cross < eps_parallel and np.linalg.norm(edge_vecs[i]) > 0.5:
                near_parallel_count += 1
                if len(near_parallel_examples) < 5:
                    near_parallel_examples.append((p, i, j, cross))

print(f"  Near-parallel intra-polygon edge pairs (cross < {eps_parallel}): {near_parallel_count}")
for p, i, j, cross in near_parallel_examples:
    print(f"    polygon {p}, edges ({i},{j}), |cross|={cross:.2e}")

# Check inter-polygon near-parallel overlapping pairs
neighbors_dict = m3.getNeighbors()
inter_parallel_count = 0
inter_parallel_examples = []
for i, neighbor_list in neighbors_dict.items():
    pi = i
    si = m3.getShapeId()[pi]
    npi = startIndices[si] + (pi - startIndices[si] + 1) % nArray[si]
    ei = positions[npi] - positions[pi]
    ei = ei - np.round(ei)
    li = np.linalg.norm(ei)
    if li < 1e-14:
        continue
    ei /= li
    for j in neighbor_list:
        pj = j
        sj = m3.getShapeId()[pj]
        npj = startIndices[sj] + (pj - startIndices[sj] + 1) % nArray[sj]
        ej = positions[npj] - positions[pj]
        ej = ej - np.round(ej)
        lj = np.linalg.norm(ej)
        if lj < 1e-14:
            continue
        ej /= lj
        cross = abs(ei[0]*ej[1] - ei[1]*ej[0])
        if cross < eps_parallel:
            inter_parallel_count += 1
            if len(inter_parallel_examples) < 5:
                inter_parallel_examples.append((si, sj, i, j, cross))

print(f"  Near-parallel inter-polygon neighbor pairs (cross < {eps_parallel}): {inter_parallel_count}")
for si, sj, i, j, cross in inter_parallel_examples:
    print(f"    shapes ({si},{sj}) edges ({i},{j}), |cross|={cross:.2e}", flush=True)

# ── 5. Energy without projectForce (to see if projection hurts) ────────────
print("\n=== Force before vs after projectForce ===")
m4 = load_fresh()
m4.updatePolygonGeometry()
m4.updateNeighborCells()
m4.updateNeighbors()
m4.updateOutersections()
m4.updateForceEnergy()
f_raw = m4.getForces().copy()
m4.projectForce()
f_proj = m4.getForces().copy()
print(f"  max |f_raw|   = {np.max(np.abs(f_raw)):.6e}")
print(f"  max |f_proj|  = {np.max(np.abs(f_proj)):.6e}")
print(f"  fraction retained = {np.linalg.norm(f_proj)/np.linalg.norm(f_raw):.4f}")

# dot product should be >= 0 if projection preserves direction
dot = np.dot(f_raw, f_proj)
print(f"  dot(f_raw, f_proj) = {dot:.4e}  (positive = same direction)", flush=True)

# ── 6. Energy after a single GD step without SHAKE ─────────────────────────
print("\n=== Can simple GD step decrease energy? ===")
m5 = load_fresh()
m5.updatePolygonGeometry()
m5.updateNeighborCells()
m5.updateNeighbors()
m5.updateOutersections()
m5.updateForceEnergy()
m5.projectForce()
e0 = m5.getEnergy()
ni0 = m5.getNumIntersections()

for dt_test in [1e-3, 1e-4, 1e-5, 1e-6]:
    mt = load_fresh()
    mt.updatePolygonGeometry()
    mt.updateNeighborCells()
    mt.updateNeighbors()
    mt.updateOutersections()
    mt.updateForceEnergy()
    mt.projectForce()
    mt.updatePositions(dt_test)
    mt.updatePolygonGeometry()
    mt.updateNeighborCells()
    mt.updateNeighbors()
    mt.updateOutersections()
    mt.updateForceEnergy()
    mt.projectForce()
    e1 = mt.getEnergy()
    ni1 = mt.getNumIntersections()
    print(f"  dt={dt_test:.1e}: E0={e0:.6e} → E1={e1:.6e}, dE={e1-e0:.3e}, ni: {ni0}→{ni1}")

print("\nDone.", flush=True)
