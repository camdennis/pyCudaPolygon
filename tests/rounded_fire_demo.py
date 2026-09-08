"""
rounded_fire_demo.py

1. Build two rounded hexagons (backbone area = 0.8/64 each) overlapping in a 1×1 PBC box.
2. Draw them with drawRounded + force arrows → rounded_before.png
3. Run FIRE minimization (with projectForce + SHAKE) until energy < 1e-14 or polygons separate.
4. Draw final configuration → rounded_after.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pyCudaPolygon as pcp

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def regular_polygon(n, R, cx=0.0, cy=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([cx + R*np.cos(angles), cy + R*np.sin(angles)])


def build_model(polys_verts, delta):
    all_verts = np.vstack(polys_verts)
    nv = len(all_verts)
    sizes = [len(v) for v in polys_verts]
    start_indices = [0]
    for s in sizes:
        start_indices.append(start_indices[-1] + s)

    m = pcp.model(nv)
    m.setStartIndices(start_indices)
    m.setPositions(all_verts.flatten().tolist())
    m.setModelEnum("rounded")
    m.setDelta(delta)
    m.setMaxEdgeLength(0.5)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    m.updateNeighbors()
    m.updatePolygonGeometry()
    return m


def draw_with_forces(m, title, filename):
    """Draw rounded outlines + backbone vertices with force arrows."""
    m.updatePolygonGeometry()
    m.updateForceEnergy()
    m.projectForce()

    forces = np.array(m.getForces())
    pos = np.array(m.getPositions())
    nArray = m.getnArray()

    fig, ax = plt.subplots(figsize=(5, 5))
    m.drawRounded(ax=ax)

    # Overlay force arrows on backbone vertices
    cmap = plt.get_cmap('tab20')
    start = 0
    fstart = 0
    for poly_idx, n in enumerate(nArray):
        color = cmap(poly_idx % cmap.N)
        px = pos[start:start + 2*n][::2]
        py = pos[start:start + 2*n][1::2]
        fx = forces[fstart:fstart + 2*n][::2]
        fy = forces[fstart:fstart + 2*n][1::2]

        # Scale arrows so the longest is ~0.05 box units
        fmag = np.hypot(fx, fy)
        fmax = fmag.max()
        scale = fmax / 0.05 if fmax > 1e-30 else 1.0

        ax.quiver(px, py, fx / scale, fy / scale,
                  angles='xy', scale_units='xy', scale=1,
                  color='k', width=0.004, headwidth=4, headlength=5, zorder=20)

        start += 2*n
        fstart += 2*n

    E = m.getEnergy()
    ax.set_title(f"{title}\nE = {E:.4e}")
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}  (E={E:.4e})")
    return E


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    n = 6
    target_area = 0.8 / 64          # ≈ 0.0125

    # Regular hexagon: area = (3√3/2) R²
    hex_area_coeff = 3 * np.sqrt(3) / 2   # ≈ 2.5981
    R = np.sqrt(target_area / hex_area_coeff)
    edge_len = R                     # hexagon edge length equals circumradius
    delta = 0.25 * edge_len          # rounding radius = 25% of edge

    print(f"n={n}  R={R:.5f}  edge_len={edge_len:.5f}  delta={delta:.5f}")
    print(f"target backbone area = {target_area:.6f}")

    # Place centres so polygons overlap by ~0.5 R
    cx1, cy1 = 0.45, 0.50
    overlap = 0.5 * R               # centre-to-centre < 2R → overlap
    cx2, cy2 = cx1 + 2*R - overlap, cy1

    v1 = regular_polygon(n, R, cx1, cy1)
    v2 = regular_polygon(n, R, cx2, cy2)

    # Wrap into [0,1)
    v1 %= 1.0
    v2 %= 1.0

    m = build_model([v1, v2], delta)

    # ---- Before: draw overlapping configuration with forces ----
    print("\n--- Before minimization ---")
    E0 = draw_with_forces(m, "Before FIRE", os.path.join(out_dir, "rounded_before.png"))

    # ---- FIRE minimization ----
    print("\n--- Running FIRE minimization ---")
    max_steps = 100_000
    log_every = 5_000
    final_step = 0
    final_E = E0

    for energy, maxForce, dt, step in m.minimizeFIRELoop(
            maxForceThreshold=1e-10,
            maxSteps=max_steps,
            dt=1e-3,
            dtMax=0.05,
            alphaStart=0.1,
            fAlpha=0.99,
            fInc=1.1,
            fDec=0.5,
            nMin=5,
            shakeIter=10):

        final_step = step
        final_E = energy

        if step % log_every == 0 or step == 1:
            print(f"  step={step:6d}  E={energy:.4e}  maxF={maxForce:.4e}  dt={dt:.4e}")

        if energy < 1e-14:
            print(f"  Energy converged to zero at step {step}.")
            break
    else:
        print(f"  Reached max steps ({max_steps}).")

    print(f"\n  Final: step={final_step}  E={final_E:.4e}")

    # ---- After: draw separated configuration ----
    print("\n--- After minimization ---")
    draw_with_forces(m, "After FIRE", os.path.join(out_dir, "rounded_after.png"))

    print("\nDone. Plots written to tests/rounded_before.png and tests/rounded_after.png")
