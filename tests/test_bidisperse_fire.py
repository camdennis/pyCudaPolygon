"""
test_bidisperse_fire.py

Two 32-vertex bidisperse rounded polygons with shape index κ=3.7:
  polygon 0 (large): backbone area = 0.8/64
  polygon 1 (small): perimeter 1.4× smaller → area = 0.8/64 / 1.4²

Offset so they overlap on creation.
FIRE on the 'rounded' model until convergence.
drawRounded saved at step 0 and final step into tests/.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
from scipy.optimize import minimize as scipy_minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyCudaPolygon as pcp

# ── parameters ────────────────────────────────────────────────────────────────
N_SIDES   = 32
P0        = 3.7           # shape index κ = perimeter / √area
RATIO     = 1.4           # large/small perimeter ratio
A_LARGE   = 0.8 / 64     # backbone area of large polygon
A_SMALL   = A_LARGE / RATIO**2  # same κ, perimeter 1.4× smaller

# delta: 20% of large-polygon edge length — small enough to be valid for both sizes
EL_LARGE  = P0 * np.sqrt(A_LARGE) / N_SIDES   # ≈ 0.01293
EL_SMALL  = P0 * np.sqrt(A_SMALL) / N_SIDES   # ≈ 0.00923
DELTA     = 0.15 * EL_SMALL                    # ≈ 0.00138 (valid for both sizes)

SEED      = 42
MAX_STEPS = 500_000
F_TOL     = 1e-10
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

print(f"A_large={A_LARGE:.5f}  A_small={A_SMALL:.5f}")
print(f"EL_large={EL_LARGE:.5f}  EL_small={EL_SMALL:.5f}  delta={DELTA:.5f}")

# ── shape generation ──────────────────────────────────────────────────────────
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
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

def _gen_shape(n, p0, target_area, rng):
    """Random equilateral closed polygon with shape index p0 scaled to target_area."""
    unit_area = (n / p0) ** 2
    for _ in range(20):
        phi0_ref = rng.dirichlet(np.ones(n - 1)) * 2 * np.pi
        def energy(phi):  return 0.5 * np.sum((phi - phi0_ref)**2)
        def grad_e(phi):  return phi - phi0_ref
        def cons(phi):
            v = _angle_to_verts(phi)
            cl = v[-1] - v[0]
            return np.array([cl[0], cl[1], _poly_area(v[:-1]) / unit_area - 1.0])
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
            backbone = v[:-1] - v[:-1].mean(axis=0)
            backbone *= np.sqrt(target_area / _poly_area(backbone))
            return backbone
    raise RuntimeError(f"Shape generation failed (n={n}, p0={p0})")

print("Generating shapes …", flush=True)
shape_large = _gen_shape(N_SIDES, P0, A_LARGE, rng)
shape_small = _gen_shape(N_SIDES, P0, A_SMALL, rng)

for shape in (shape_large, shape_small):
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    shape[:] = shape @ np.array([[c, -s], [s, c]]).T

print(f"  large: area={_poly_area(shape_large):.5f}  (target {A_LARGE:.5f})")
print(f"  small: area={_poly_area(shape_small):.5f}  (target {A_SMALL:.5f})")

# ── placement: offset so they clearly overlap ─────────────────────────────────
# Circumradii approx sqrt(A/π): large≈0.063, small≈0.045, sum≈0.108.
# Separation 0.08 < sum → clear overlap; small enough not to wrap.
R_large = np.sqrt(A_LARGE / np.pi)
sep = 0.7 * (R_large + np.sqrt(A_SMALL / np.pi))   # 70% of just-touching
all_verts = np.zeros((2 * N_SIDES, 2))
all_verts[:N_SIDES] = (shape_large + [0.5 - sep/2, 0.5]) % 1.0
all_verts[N_SIDES:] = (shape_small + [0.5 + sep/2, 0.5]) % 1.0
print(f"Centre separation = {sep:.4f}  (R_large+R_small ≈ {R_large + np.sqrt(A_SMALL/np.pi):.4f})")

# ── build model ───────────────────────────────────────────────────────────────
nv = 2 * N_SIDES
m = pcp.model(nv)
m.setStartIndices([0, N_SIDES, nv])
m.setPositions(all_verts.flatten())
m.setModelEnum("rounded")
m.setDelta(DELTA)
m.setMaxEdgeLength(0.5)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateNeighbors()
m.updatePolygonGeometry()
m.updateForceEnergy()

target_areas        = np.array([A_LARGE, A_SMALL])
target_edge_lengths = P0 * np.sqrt(target_areas) / N_SIDES
m.setTargetAreas(target_areas)
m.setTargetEdgeLengths(target_edge_lengths)

E0 = m.getEnergy()
print(f"\nStep 0: E={E0:.4e}  |F|max={m.getMaxUnbalancedForce():.4e}")

# ── drawRounded at step 0 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
m.drawRounded(ax=ax, delta=DELTA)
ax.set_title(f"Step 0   E = {E0:.4e}")
fig.savefig(os.path.join(OUT_DIR, "bidisperse_fire_step0.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved bidisperse_fire_step0.png")

# ── FIRE minimization (restartable loop, monitor dt via generator) ─────────────
# Using minimizeFIRELoop so each call properly re-initializes FIRE state.
# We break early when dt collapses, then let the next call reset velocities/dt.
print("\nRunning FIRE …", flush=True)
DT_INIT     = 1e-3     # same as hexagon demo (larger dt → more progress per step)
DT_MAX      = 5e-2     # same as hexagon demo
DT_MIN      = 1e-6     # dt below this → stalled, restart
MAX_RESTARTS = 30

total_steps = 0
restarts    = 0
energy      = m.getEnergy()
maxForce    = m.getMaxUnbalancedForce()
steps       = 0

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
        if total_steps % 10000 == 0:
            print(f"  step={total_steps:7d}  E={energy:.4e}  |F|={maxForce:.4e}  dt={dt:.2e}  r={restarts}", flush=True)
        if maxForce <= F_TOL or energy < 1e-12:
            break
        if dt < DT_MIN:
            break
    else:
        break   # exhausted maxSteps
    if maxForce <= F_TOL or energy < 1e-12:
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
fig, ax = plt.subplots(figsize=(5, 5))
m.drawRounded(ax=ax, delta=DELTA)
ax.set_title(f"Step {steps}   E = {energy:.4e}")
fig.savefig(os.path.join(OUT_DIR, "bidisperse_fire_final.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved bidisperse_fire_final.png")
