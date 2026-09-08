"""
fire_two_polygon.py — two overlapping rounded hexagons minimized with FIRE.

Uses SmoothedPolygon (smooth_polygon.py) with rigid-body DOFs:
    state  = (tx, ty, theta) per polygon
    energy = total pairwise intersection area
    forces = -packing_overlap_gradient(...)
"""

import sys, os
sys.path.insert(0, '/home/rdennis/Documents/Code/pyPolygon')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smooth_polygon import SmoothedPolygon, Packing, packing_overlap_gradient

# ── geometry ──────────────────────────────────────────────────────────────────
n     = 6
R     = 0.07
EL    = 2 * R * np.sin(np.pi / n)   # edge length
delta = 0.25 * EL                   # rounding radius = 25% of edge

A = SmoothedPolygon.regular(n, R, delta, center=(0.10, 0.50))
B = SmoothedPolygon.regular(n, R, delta, center=(0.17, 0.50))  # overlap ≈ 0.07 < 2R=0.14

packing_init = Packing([A.copy(), B.copy()])
E0 = packing_init.total_overlap_area()
print(f"n={n}  R={R:.4f}  EL={EL:.4f}  delta={delta:.5f}")
print(f"Initial overlap area = {E0:.6e}")

# ── FIRE ──────────────────────────────────────────────────────────────────────
polys      = [A.copy(), B.copy()]
N          = len(polys)
v          = np.zeros((N, 3))   # velocities: [vx, vy, vtheta] per polygon
dt         = 5e-3
dt_max     = 0.05
f_inc      = 1.1
f_dec      = 0.5
alpha      = 0.1
alpha_start = 0.1
f_alpha    = 0.99
n_min      = 5
n_pos      = 0

history    = [E0]
max_steps  = 5000
step       = 0

for step in range(1, max_steps + 1):
    pack = Packing(polys)
    E    = pack.total_overlap_area()
    history.append(E)

    if E < 1e-12:
        print(f"Converged at step {step}  E={E:.4e}")
        break

    G = packing_overlap_gradient(pack)   # (N, 3): dE/d(tx,ty,theta)
    F = -G                               # force = -gradient

    # FIRE: check power P = F · v
    P = float(np.sum(F * v))

    if P > 0.0:
        n_pos += 1
        F_norm = np.linalg.norm(F)
        v_norm = np.linalg.norm(v)
        if F_norm > 0:
            v = (1.0 - alpha) * v + alpha * (F / F_norm) * v_norm
        if n_pos >= n_min:
            dt    = min(dt * f_inc, dt_max)
            alpha = alpha * f_alpha
    else:
        n_pos = 0
        v     = np.zeros_like(v)
        dt    = dt * f_dec
        alpha = alpha_start

    # Velocity half-step and position update
    v     = v + F * dt
    delta_dof = v * dt   # (N, 3)

    new_polys = []
    for i, poly in enumerate(polys):
        dx, dy, dtheta = delta_dof[i]
        c = tuple(poly.vertices.mean(axis=0))
        new_polys.append(poly.translate(dx, dy).rotate(dtheta, center=c))
    polys = new_polys

    if step % 200 == 0:
        print(f"  step={step:5d}  E={E:.4e}  |F|={np.linalg.norm(F):.4e}  dt={dt:.4e}")
else:
    print(f"Reached max steps ({max_steps}), E={E:.4e}")

E_final = Packing(polys).total_overlap_area()
print(f"Final overlap area = {E_final:.6e}  (steps={step})")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

packing_init.plot(axes[0], fill=True, fill_alpha=0.35, show_backbone=True,
                  title=f"Before FIRE\noverlap = {E0:.3e}")

Packing(polys).plot(axes[1], fill=True, fill_alpha=0.35, show_backbone=True,
                    title=f"After FIRE  (step {step})\noverlap = {E_final:.3e}")

axes[2].plot(history, linewidth=1.5)
axes[2].set_xlabel("step")
axes[2].set_ylabel("overlap area")
axes[2].set_title("FIRE convergence")
if max(history) / (min(history) + 1e-15) > 10:
    axes[2].set_yscale("log")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fire_two_polygon.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
