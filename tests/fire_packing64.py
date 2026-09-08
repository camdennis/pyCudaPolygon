"""
fire_packing64.py — 64 bidisperse 32-gons (κ=3.7, ratio=1.4) at φ=0.8, FIRE + SHAKE.

32 large (A_large = 1.96·a) + 32 small (A_small = a),
total backbone area = 64/2·(1.96+1)·a = 0.8  →  a = 0.8/(32·2.96).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyCudaPolygon as pcp

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── parameters ────────────────────────────────────────────────────────────────
N_POLY  = 64
N_SIDES = 32
N_LARGE = N_POLY // 2   # indices 0 … N_LARGE-1
N_SMALL = N_POLY // 2   # indices N_LARGE … N_POLY-1
KAPPA   = 3.7
RATIO   = 1.4           # A_large / A_small = RATIO²
PHI     = 0.8

# N_LARGE·A_large + N_SMALL·A_small = PHI,  A_large = RATIO²·A_small
A_SMALL = PHI / (N_LARGE * RATIO**2 + N_SMALL)
A_LARGE = RATIO**2 * A_SMALL

EL_LARGE = KAPPA * np.sqrt(A_LARGE) / N_SIDES
EL_SMALL = KAPPA * np.sqrt(A_SMALL) / N_SIDES
DELTA    = 0.15 * EL_SMALL

print(f"φ={PHI}  N={N_POLY} ({N_LARGE} large + {N_SMALL} small)  κ={KAPPA}  ratio={RATIO}")
print(f"A_large={A_LARGE:.5f}  A_small={A_SMALL:.5f}")
print(f"EL_large={EL_LARGE:.5f}  EL_small={EL_SMALL:.5f}  δ={DELTA:.5f}")
print(f"Check: {N_LARGE}·A_L + {N_SMALL}·A_S = {N_LARGE*A_LARGE + N_SMALL*A_SMALL:.4f}  (target {PHI})")

# ── build model ───────────────────────────────────────────────────────────────
m = pcp.model(N_POLY * N_SIDES, seed=42)
m.generateRandomPolygons(N_POLY, N_SIDES)
m.setModelEnum("rounded")
m.setDelta(DELTA)

# Set bidisperse targets without calling resetAreas / setPhi
# first half = large, second half = small
target_areas = np.array([A_LARGE] * N_LARGE + [A_SMALL] * N_SMALL)
target_el    = np.array([EL_LARGE] * N_LARGE + [EL_SMALL] * N_SMALL)
m.setTargetAreas(target_areas)
m.setTargetEdgeLengths(target_el)

m.setMaxEdgeLength(2 * np.sqrt(A_LARGE / np.pi) + 2 * DELTA)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateNeighbors()
m.updatePolygonGeometry()

# Relax shapes to equilateral using softBody FIRE before packing
print("Shape relaxation (softBody FIRE) ...", flush=True)
m.setStiffness(1)
m.setCompressibility(1)
m.updateForceEnergy()
m.setModelEnum("softBody")
m.minimizeFIRE(dt=0.1, maxForceThreshold=1e-14)
m.updatePolygonGeometry()
m.setModelEnum("areaSquared")
m.updateForceEnergy()

E0 = m.getEnergy()
print(f"Step 0: E={E0:.4e}  |F|max={m.getMaxUnbalancedForce():.4e}")

# ── draw helper ───────────────────────────────────────────────────────────────
COLOR_LARGE = '#4878cf'   # blue
COLOR_SMALL = '#d65f5f'   # red

import matplotlib.colors as mcolors

# ── draw before ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
m.drawBiarc(ax=ax, psi_frac=0.5)
ax.set_title(f"Step 0  N={N_POLY}  phi={PHI}\nE = {E0:.4e}")
fig.savefig(os.path.join(OUT_DIR, "fire_packing64_before.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fire_packing64_before.png")

# ── FIRE ──────────────────────────────────────────────────────────────────────
DT_INIT      = .05
DT_MAX       = 10
DT_MIN       = 1e-8
F_TOL        = 1e-10
MAX_STEPS    = 2_000_000
MAX_RESTARTS = 200

# Every step for the first 100, then log-spaced up to MAX_STEPS (~20 more)
_ckpt_steps = set(range(1, 101)) | set(int(x) for x in np.geomspace(100, MAX_STEPS, 20).round())

import time as _time

def save_checkpoint(m, step, energy):
    t0 = _time.perf_counter()
    m.updatePolygonGeometry()
    t1 = _time.perf_counter()
    m.updateForceEnergy()
    maxForce = m.getMaxUnbalancedForce()
    t2 = _time.perf_counter()
    fig, ax = plt.subplots(figsize=(6, 6))
    m.drawBiarc(ax=ax, psi_frac=0.5)
    t3 = _time.perf_counter()
    ax.set_title(f"Step {step}  N={N_POLY}  phi={PHI}\nE={energy:.4e}  |F|max={maxForce:.4e}")
    img_path = os.path.join(OUT_DIR, f"fire_packing64_ckpt_{step:08d}.png")
    fig.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    ckpt_dir = os.path.join(OUT_DIR, f"fire_packing64_ckpt_{step:08d}")
    m.saveModel(ckpt_dir, overwrite=True)
    t4 = _time.perf_counter()
    print(f"  [ckpt] step={step}  E={energy:.4e}  |F|={maxForce:.4e}  geom={t1-t0:.2f}s  force={t2-t1:.2f}s  draw={t3-t2:.2f}s  save={t4-t3:.2f}s  total={t4-t0:.2f}s", flush=True)

total_steps = 0
restarts    = 0
energy      = E0
maxForce    = m.getMaxUnbalancedForce()
_step_t0    = _time.perf_counter()

print("\nRunning FIRE ...", flush=True)
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
        _step_t1 = _time.perf_counter()
        total_steps += 1
        if total_steps <= 5 or total_steps % 50000 == 0:
            print(f"  step={total_steps:8d}  E={energy:.4e}  |F|={maxForce:.4e}  dt={dt:.2e}  fire_step={_step_t1-_step_t0:.2f}s  shakeIters={m.getLastShakeIters()}", flush=True)
        if total_steps in _ckpt_steps:
            save_checkpoint(m, total_steps, energy)
        _step_t0 = _time.perf_counter()
        if maxForce <= F_TOL:
            break
        if 0 <= energy < 1e-12:
            break
        if dt < DT_MIN:
            break
    else:
        break
    if maxForce <= F_TOL or (0 <= energy < 1e-12):
        break
    if total_steps >= MAX_STEPS:
        break
    restarts += 1
    print(f"  Restart {restarts} at step {total_steps}: E={energy:.4e}  |F|={maxForce:.4e}  dt={dt:.2e}", flush=True)

print(f"\nFinal: step={total_steps}  E={energy:.4e}  |F|max={m.getMaxUnbalancedForce():.4e}")

# ── draw after ────────────────────────────────────────────────────────────────
m.updatePolygonGeometry()
m.updateForceEnergy()
fig, ax = plt.subplots(figsize=(6, 6))
m.drawBiarc(ax=ax, psi_frac=0.5)
ax.set_title(f"Step {total_steps}  N={N_POLY}  phi={PHI}\nE = {energy:.4e}")
fig.savefig(os.path.join(OUT_DIR, "fire_packing64_after.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fire_packing64_after.png")
