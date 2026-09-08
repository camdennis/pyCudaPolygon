"""
test_packing64_fire.py

64 bidisperse rounded polygons (32 large + 32 small) with:
  - 32 vertices each, shape index κ=3.7
  - Perimeter ratio large/small = 1.4  (area ratio = 1.96)
  - Total backbone packing fraction φ = 0.8

Polygons are placed on a jittered 8×8 grid with random rotations so the
initial state has significant but not catastrophic overlap.
FIRE is run until E=0 (no overlap) or MAX_STEPS is reached.
drawRounded is saved at step 0 and at the final step.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyCudaPolygon as pcp

# ── parameters ────────────────────────────────────────────────────────────────
N_POLYS   = 64
N_LARGE   = N_POLYS // 2
N_SMALL   = N_POLYS // 2
N_SIDES   = 32
P0        = 3.7
RATIO     = 1.4          # large/small perimeter ratio
PHI       = 0.8          # total backbone packing fraction

# Areas so that N_LARGE*A_LARGE + N_SMALL*A_SMALL = PHI
A_SMALL   = PHI / (N_LARGE * RATIO**2 + N_SMALL)
A_LARGE   = RATIO**2 * A_SMALL

EL_LARGE  = P0 * np.sqrt(A_LARGE) / N_SIDES
EL_SMALL  = P0 * np.sqrt(A_SMALL) / N_SIDES
DELTA     = 0.15 * EL_SMALL

SEED      = 42
MAX_STEPS = 500_000
F_TOL     = 1e-10
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

print(f"N_POLYS={N_POLYS}  N_LARGE={N_LARGE}  N_SMALL={N_SMALL}")
print(f"A_large={A_LARGE:.5f}  A_small={A_SMALL:.5f}")
print(f"EL_large={EL_LARGE:.5f}  EL_small={EL_SMALL:.5f}  delta={DELTA:.5f}")
print(f"phi_check = {N_LARGE*A_LARGE + N_SMALL*A_SMALL:.4f}  (target {PHI})")

# ── shape generation via FIRE + softBody ──────────────────────────────────────
# Place all polygons on a sparse grid, run softBody FIRE (edge+area springs) to
# converge each polygon independently to a random equilateral shape with the
# correct area. No inter-polygon forces, so each relaxes in isolation.
rng = np.random.default_rng(SEED)

def _gen_shapes_fire(n_polys, n_sides, el_large, el_small, a_large, a_small, n_large, rng):
    nv = n_polys * n_sides
    grid_n = int(np.ceil(np.sqrt(n_polys)))
    sp = 1.0 / grid_n

    all_verts = np.zeros((nv, 2))
    for i in range(n_polys):
        el = el_large if i < n_large else el_small
        # Circumradius of regular N-gon with edge length el
        r_circ = el / (2.0 * np.sin(np.pi / n_sides))
        row, col = divmod(i, grid_n)
        cx = (col + 0.5) * sp
        cy = (row + 0.5) * sp
        # Jitter angles and radii to break symmetry
        base_angles = 2.0 * np.pi * np.arange(n_sides) / n_sides
        base_angles += rng.uniform(-0.4 * np.pi / n_sides, 0.4 * np.pi / n_sides, n_sides)
        radii = r_circ * (1.0 + rng.uniform(-0.15, 0.15, n_sides))
        all_verts[i*n_sides:(i+1)*n_sides, 0] = (cx + radii * np.cos(base_angles)) % 1.0
        all_verts[i*n_sides:(i+1)*n_sides, 1] = (cy + radii * np.sin(base_angles)) % 1.0

    start_indices = list(range(0, nv + 1, n_sides))
    sm = pcp.model(nv)
    sm.setStartIndices(start_indices)
    sm.setPositions(all_verts.flatten())
    sm.setModelEnum("softBody")
    sm.setStiffness(1.0)
    sm.setCompressibility(1.0)
    sm.setMaxEdgeLength(0.5)
    sm.initializeNeighborCells()
    sm.updatePolygonGeometry()

    tgt_areas = np.array([a_large if i < n_large else a_small for i in range(n_polys)])
    tgt_el    = np.array([el_large if i < n_large else el_small for i in range(n_polys)])
    sm.setTargetAreas(tgt_areas)
    sm.setTargetEdgeLengths(tgt_el)
    sm.updateForceEnergy()

    sm.minimizeFIRE(maxForceThreshold=1e-8, dt=1e-3, maxSteps=200000, dtMax=0.05)

    raw = sm.getPositions().reshape(-1, 2)
    shapes = []
    for i in range(n_polys):
        v = raw[i*n_sides:(i+1)*n_sides].copy()
        # Chain-unwrap across PBC so polygon is contiguous
        for k in range(1, n_sides):
            diff = v[k] - v[k-1]
            v[k] = v[k-1] + (diff - np.round(diff))
        v -= v.mean(axis=0)
        # Random rotation
        theta = rng.uniform(0, 2.0 * np.pi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        v = v @ np.array([[c_t, -s_t], [s_t, c_t]]).T
        shapes.append(v)
    return shapes

print(f"\nGenerating {N_POLYS} shapes via softBody FIRE …", flush=True)
shapes = _gen_shapes_fire(N_POLYS, N_SIDES, EL_LARGE, EL_SMALL, A_LARGE, A_SMALL, N_LARGE, rng)
print(f"  done", flush=True)

# ── placement: fully random positions ────────────────────────────────────────
all_verts = np.zeros((N_POLYS * N_SIDES, 2))
for i, shape in enumerate(shapes):
    cx = rng.uniform(0.0, 1.0)
    cy = rng.uniform(0.0, 1.0)
    all_verts[i*N_SIDES:(i+1)*N_SIDES] = (shape + [cx, cy]) % 1.0

# ── build model ───────────────────────────────────────────────────────────────
nv = N_POLYS * N_SIDES
start_indices = list(range(0, nv + 1, N_SIDES))

m = pcp.model(nv)
m.setStartIndices(start_indices)
m.setPositions(all_verts.flatten())
m.setModelEnum("rounded")
m.setDelta(DELTA)
m.setMaxEdgeLength(0.5)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateNeighbors()
m.updatePolygonGeometry()
m.updateForceEnergy()

target_areas        = np.array([A_LARGE if i < N_LARGE else A_SMALL for i in range(N_POLYS)])
target_edge_lengths = P0 * np.sqrt(target_areas) / N_SIDES
m.setTargetAreas(target_areas)
m.setTargetEdgeLengths(target_edge_lengths)

E0 = m.getEnergy()
print(f"\nStep 0: E={E0:.4e}  |F|max={m.getMaxUnbalancedForce():.4e}")

# ── drawRounded at step 0 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
m.drawRounded(ax=ax, delta=DELTA)
ax.set_title(f"Step 0   E = {E0:.4e}   φ = {PHI}")
fig.savefig(os.path.join(OUT_DIR, "packing64_fire_step0.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved packing64_fire_step0.png")

# ── FIRE minimization (restartable loop, monitor dt) ─────────────────────────
# Scale dt to the initial force so that max displacement per step ≈ 0.1% EL_SMALL.
# With large forces (dense packing), this prevents polygons from leaping past
# their neighbor cells, which corrupts the intersection energy.
print("\nRunning FIRE …", flush=True)
_F0 = m.getMaxUnbalancedForce()
DT_MAX      = min(5e-3, float(np.sqrt(1e-3 * EL_SMALL / max(_F0, 1.0))))
DT_INIT     = DT_MAX / 10.0
DT_MIN      = DT_INIT * 1e-5
MAX_RESTARTS = 50
print(f"  dt_init={DT_INIT:.2e}  dt_max={DT_MAX:.2e}  (F0={_F0:.2e})", flush=True)

total_steps = 0
restarts    = 0
energy      = E0
maxForce    = _F0

while total_steps < MAX_STEPS and restarts <= MAX_RESTARTS:
    for energy, maxForce, dt, step in m.minimizeFIRELoop(
            maxForceThreshold=F_TOL,
            dt=DT_INIT,
            maxSteps=MAX_STEPS - total_steps,
            dtMax=DT_MAX,
            alphaStart=0.1,
            fAlpha=0.99,
            fInc=1.1,
            fDec=0.5,
            nMin=5,
            shakeIter=10000,
            progressBar=False):
        total_steps += 1
        if total_steps % 100 == 0:
            print(f"  step={total_steps:7d}  E={energy:.4e}  |F|={maxForce:.4e}  dt={dt:.2e}  r={restarts}", flush=True)
            m.updatePolygonGeometry()
            m.updateForceEnergy()
            fig, ax = plt.subplots(figsize=(6, 6))
            m.drawRounded(ax=ax, delta=DELTA)
            ax.set_title(f"Step {total_steps}   E = {energy:.4e}   φ = {PHI}")
            fig.savefig(os.path.join(OUT_DIR, f"packing64_fire_step{total_steps:07d}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
        if maxForce <= F_TOL:
            break
        if dt < DT_MIN:
            break
    else:
        break
    if maxForce <= F_TOL:
        break
    if total_steps >= MAX_STEPS:
        break
    restarts += 1
    print(f"  Restart {restarts} at step {total_steps}: E={energy:.4e}  |F|={maxForce:.4e}  dt={dt:.2e}", flush=True)

steps = total_steps
print(f"\nFinal: step={steps}  E={energy:.4e}  |F|max={m.getMaxUnbalancedForce():.4e}")

# ── drawRounded at final step ─────────────────────────────────────────────────
m.updatePolygonGeometry()
m.updateForceEnergy()
fig, ax = plt.subplots(figsize=(6, 6))
m.drawRounded(ax=ax, delta=DELTA)
ax.set_title(f"Step {steps}   E = {energy:.4e}   φ = {PHI}")
fig.savefig(os.path.join(OUT_DIR, "packing64_fire_final.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved packing64_fire_final.png")
