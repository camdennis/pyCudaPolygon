"""
Generate and minimize a bidisperse rounded-polygon packing.
  64 polygons  ·  32 sides  ·  shape index p0=3.7  ·  phi_backbone=0.8
  Bidisperse area ratio 1.4^2 (large/small).

minimizeFIRE's built-in checkpointing is used so long runs survive interruption.
Neighbor cells are now refreshed inside minimizeFIREStep for the rounded model.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
from scipy.optimize import minimize as scipy_minimize
import pyCudaPolygon as pcp

# ─── parameters ───────────────────────────────────────────────────────────────
N_POLYS     = 2
N_SIDES     = 32
P0          = 3.7       # target shape index  P / sqrt(A)
RATIO       = 1.4       # large/small linear size ratio
PHI         = 0.8       # backbone packing fraction
DELTA       = 0.008     # arc-rounding radius
SEED        = 42
MAX_STEPS   = 500_000
F_TOL       = 1e-10
CHECKPOINT_DIR  = os.path.join(os.path.dirname(__file__), 'ckpt_packing_phi08')
CHECKPOINT_FREQ = 100
SAVE_PATH       = os.path.join(os.path.dirname(__file__), 'packing_rounded_phi08')

# ─── target backbone areas ────────────────────────────────────────────────────
# Use the same per-polygon size as the 64-polygon bidisperse case, regardless of N_POLYS.
N_REF   = 64
n_large = N_POLYS // 2
n_small = N_POLYS // 2
A_small_ref = PHI / (N_REF // 2 + N_REF // 2 * RATIO**2)
A_large_ref = RATIO**2 * A_small_ref
A_small, A_large = A_small_ref, A_large_ref
print(f"Target areas: A_small={A_small:.5f}, A_large={A_large:.5f}")

# ─── shape generation ─────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def _angle_to_verts(angles):
    n = len(angles) + 1
    theta = np.zeros(n)
    for k in range(1, n):
        theta[k] = theta[k-1] + angles[k-1]
    ex, ey = np.cos(theta), np.sin(theta)
    v = np.zeros((n+1, 2))
    for k in range(n):
        v[k+1] = v[k] + np.array([ex[k], ey[k]])
    return v

def _poly_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

def _gen_shape(n, p0, rng):
    target_area = (n / p0) ** 2
    def energy(phi): return 0.5 * np.sum((phi - phi0_ref)**2)
    def grad_e(phi): return phi - phi0_ref
    def cons(phi):
        v = _angle_to_verts(phi)
        cl = v[-1] - v[0]
        area = _poly_area(v[:-1])
        return np.array([cl[0], cl[1], area / target_area - 1.0])
    for _ in range(20):
        phi0_ref = rng.dirichlet(np.ones(n - 1)) * 2 * np.pi
        res = scipy_minimize(energy, phi0_ref, jac=grad_e, method='SLSQP',
            bounds=[(1e-4, np.pi - 1e-4)] * (n - 1),
            constraints=[
                {'type': 'eq',   'fun': cons},
                {'type': 'ineq', 'fun': lambda p: np.sum(p) - np.pi},
                {'type': 'ineq', 'fun': lambda p: 2*np.pi - np.sum(p)},
            ],
            options={'maxiter': 2000, 'ftol': 1e-12})
        if res.success:
            v = _angle_to_verts(res.x)
            backbone = v[:-1]
            backbone -= backbone.mean(axis=0)
            return backbone
    raise RuntimeError(f"Shape generation failed (n={n}, p0={p0})")

print(f"Generating {N_POLYS} shapes (n={N_SIDES}, p0={P0}) …", flush=True)
polys = []
for i in range(N_POLYS):
    shape = _gen_shape(N_SIDES, P0, rng)
    A_tgt = A_large if i < n_large else A_small
    shape *= np.sqrt(A_tgt / abs(_poly_area(shape)))
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    shape = shape @ np.array([[c, -s], [s, c]]).T
    polys.append(shape)
print("  Done.", flush=True)

# ─── placement: stack at centre so they start overlapping ────────────────────
all_verts = np.zeros((N_POLYS * N_SIDES, 2))
for i, shape in enumerate(polys):
    cx, cy = 0.5, 0.5
    verts = (shape + np.array([cx, cy])) % 1.0
    all_verts[i*N_SIDES:(i+1)*N_SIDES] = verts

nv = N_POLYS * N_SIDES
start_indices = list(range(0, nv + 1, N_SIDES))

# ─── build rounded model ──────────────────────────────────────────────────────
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

# Set target areas and average edge lengths so SHAKE can maintain polygon shapes.
target_areas_arr = np.array([A_large if i < n_large else A_small for i in range(N_POLYS)])
target_edge_lengths_arr = P0 * np.sqrt(target_areas_arr) / N_SIDES
m.setTargetAreas(target_areas_arr)
m.setTargetEdgeLengths(target_edge_lengths_arr)

print(f"Initial E={m.getEnergy():.4e}  |F|max={m.getMaxUnbalancedForce():.4e}\n", flush=True)

# ─── FIRE with built-in checkpointing ────────────────────────────────────────
# Neighbor cells are now refreshed inside minimizeFIREStep for the rounded model
# (see model.cpp: needsNeighborUpdate includes rounded).
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
energy, dt, steps, _ = m.minimizeFIRE(
    maxForceThreshold=F_TOL,
    dt=1e-4,
    maxSteps=MAX_STEPS,
    dtMax=5e-3,
    checkpointDir=CHECKPOINT_DIR,
    checkpointFreq=CHECKPOINT_FREQ,
    overwriteCheckpoint=True,
)
print(f"\nDone: E={energy:.6e}  dt={dt:.2e}  steps={steps}")

# ─── save final state ─────────────────────────────────────────────────────────
m.saveModel(SAVE_PATH, overwrite=True)
print(f"Saved to {SAVE_PATH}/")

# ─── visualise ────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    from matplotlib.collections import PatchCollection

    pos = m.getPositions().reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(7, 7))
    patches, colors = [], []
    for i in range(N_POLYS):
        v = pos[i*N_SIDES:(i+1)*N_SIDES]
        patches.append(MPoly(v, closed=True))
        colors.append('steelblue' if i < n_large else 'coral')
    ax.add_collection(PatchCollection(patches, facecolors=colors,
                                      edgecolors='k', linewidths=0.4, alpha=0.75))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(f'Minimized packing  φ={PHI}  p₀={P0}  E={energy:.2e}  steps={steps}')
    fig_path = os.path.join(os.path.dirname(__file__), 'packing_rounded_phi08.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
except Exception as e:
    print(f"(Visualisation skipped: {e})")
