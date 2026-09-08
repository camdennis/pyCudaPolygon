"""
Run FIRE until it stagnates, save the near-instability state,
then diagnose what's happening there.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyCudaPolygon'))
import pyCudaPolygon as pcp
from pyCudaPolygonLink import libpyCudaPolygon as lpcp
import numpy as np

SAVE = os.path.join(os.path.dirname(__file__), 'testSave')
STUCK = os.path.join(os.path.dirname(__file__), 'nearInstability')

m = pcp.model(size=1)
m.loadModel(SAVE)

print("=== Running FIRE until stuck ===", flush=True)
prev_dt = 1e-3
stuck_step = None
for energy, maxForce, dt, step in m.minimizeFIRELoop(maxForceThreshold=1e-14, dt=1e-3, maxSteps=50000):
    if step % 200 == 0:
        print(f"  step={step:5d}, E={energy:.5e}, F={maxForce:.5e}, dt={dt:.3e}", flush=True)
    # Detect stuck: dt below resolution threshold
    if dt < 1e-7 and stuck_step is None:
        stuck_step = step
        print(f"\n** dt < 1e-7 at step {step}: saving near-instability state **", flush=True)
        m.saveModel(STUCK, overwrite=True)
        break
    if maxForce <= 1e-14:
        print(f"\n** Converged at step {step}! **", flush=True)
        m.saveModel(STUCK, overwrite=True)
        break

print(f"\nFinal: step={step}, E={energy:.6e}, F={maxForce:.6e}, dt={dt:.3e}", flush=True)

if stuck_step is None:
    print("WARNING: Did not detect stuck state - saving final state")
    m.saveModel(STUCK, overwrite=True)

# ── Now diagnose the saved stuck state ──────────────────────────────────────
print("\n=== Diagnosing near-instability state ===", flush=True)
m2 = pcp.model(size=1)
m2.loadModel(STUCK)
m2.updatePolygonGeometry()
m2.updateNeighborCells()
m2.updateNeighbors()
m2.updateOutersections()
m2.updateForceEnergy()
m2.projectForce()

e0 = m2.getEnergy()
ni0 = m2.getNumIntersections()
muf0 = m2.getMaxUnbalancedForce()
cv0 = m2.getConstraintViolation()
print(f"  E={e0:.6e}, ni={ni0}, maxForce={muf0:.6e}")
print(f"  constraintViolation={cv0}", flush=True)

# Test GD steps at various dt to see which reduce energy
print("\n  GD step energy test (no SHAKE):", flush=True)
for dt_test in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    mt = pcp.model(size=1)
    mt.loadModel(STUCK)
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
    print(f"    dt={dt_test:.0e}: dE={e1-e0:.4e}, ni: {ni0}→{ni1}, F={mt.getMaxUnbalancedForce():.4e}")

# ── Check near-parallel inter-polygon pairs at stuck state ──────────────────
print("\n  Near-parallel inter-polygon neighbor pairs:", flush=True)
m3 = pcp.model(size=1)
m3.loadModel(STUCK)
m3.updatePolygonGeometry()
m3.updateNeighborCells()
m3.updateNeighbors()

positions = m3.getPositions().reshape(-1, 2)
nArray = m3.getnArray()
startIndices = m3.getStartIndices()

shapeIds = m3.getShapeId()
neighbors_dict = m3.getNeighbors()

inter_parallel_count = 0
inter_parallel_examples = []
for i, neighbor_list in neighbors_dict.items():
    si = shapeIds[i]
    npi = startIndices[si] + (i - startIndices[si] + 1) % nArray[si]
    ei = positions[npi] - positions[i]
    ei -= np.round(ei)
    li = np.linalg.norm(ei)
    if li < 1e-14: continue
    ei /= li
    for j in neighbor_list:
        sj = shapeIds[j]
        npj = startIndices[sj] + (j - startIndices[sj] + 1) % nArray[sj]
        ej = positions[npj] - positions[j]
        ej -= np.round(ej)
        lj = np.linalg.norm(ej)
        if lj < 1e-14: continue
        ej /= lj
        cross = abs(ei[0]*ej[1] - ei[1]*ej[0])
        if cross < 0.05:
            inter_parallel_count += 1
            if len(inter_parallel_examples) < 10:
                inter_parallel_examples.append((si, sj, i, j, cross))

print(f"  Inter-polygon near-parallel pairs (cross < 0.05): {inter_parallel_count}")
for si, sj, i, j, cross in inter_parallel_examples:
    print(f"    shapes ({si},{sj}) edges ({i%nArray[si]:2d},{j%nArray[sj]:2d}), |sin(θ)|={cross:.4f}")

# ── Force fraction retained after projection ───────────────────────────────
print("\n  Force projection analysis:", flush=True)
m4 = pcp.model(size=1)
m4.loadModel(STUCK)
m4.updatePolygonGeometry()
m4.updateNeighborCells()
m4.updateNeighbors()
m4.updateOutersections()
m4.updateForceEnergy()
f_raw = m4.getForces().copy()
m4.projectForce()
f_proj = m4.getForces().copy()
print(f"    max|f_raw|={np.max(np.abs(f_raw)):.4e}, max|f_proj|={np.max(np.abs(f_proj)):.4e}")
print(f"    norm: {np.linalg.norm(f_raw):.4e} → {np.linalg.norm(f_proj):.4e}  "
      f"({100*np.linalg.norm(f_proj)/np.linalg.norm(f_raw):.1f}% retained)")
print(f"    dot(f_raw, f_proj) = {np.dot(f_raw, f_proj):.4e}")

print("\nDone.", flush=True)
