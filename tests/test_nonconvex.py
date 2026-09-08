"""
test_nonconvex.py

Rigorous two-polygon test for non-convex conditions.

Configurations tested (all placed well inside [0,1]², no PBC splitting):
  0. Convex A ∩ Convex B               — sanity check
  1. Dented A ∩ Convex B, overlap away from dent
  2. Dented A ∩ Convex B, overlap ON the dent (hardest for reflex arc)
  3. Dented A ∩ Dented B               — both non-convex

For each: compare model energy with a backbone reference area computed on a
dense grid (ray-casting PIP, no shapely dependency).  The rounded model's
energy approximates the rounded-polygon intersection; for small delta this is
close to the backbone intersection.  A correct implementation must have:
  - E >= 0
  - |E - backbone_ref| / max(backbone_ref, 1e-6) < some tolerance
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

# ─── helpers ─────────────────────────────────────────────────────────────────

def signed_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def ensure_ccw(v):
    """Return v in CCW order."""
    return v if signed_area(v) > 0 else v[::-1]

def regular_poly(n, R, cx=0.0, cy=0.0, theta0=0.0):
    angles = np.linspace(theta0, theta0 + 2*np.pi, n, endpoint=False)
    return ensure_ccw(np.column_stack([cx + R * np.cos(angles),
                                       cy + R * np.sin(angles)]))

def dented_poly(n, R, dent_idx, dent_r, cx=0.0, cy=0.0, theta0=0.0):
    """Regular n-gon with vertex dent_idx pushed inward to radius dent_r.
    Creates one reflex corner (non-convex shape).
    """
    angles = np.linspace(theta0, theta0 + 2*np.pi, n, endpoint=False)
    radii = np.full(n, R, dtype=float)
    radii[dent_idx] = dent_r
    return ensure_ccw(np.column_stack([cx + radii * np.cos(angles),
                                       cy + radii * np.sin(angles)]))

def pip_batch(pts, poly):
    """Ray-casting PIP for N×2 array pts against closed polygon poly (N×2).
    Works correctly for non-convex (simple) polygons."""
    n = len(poly)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dy12 = y2 - y1
        cross = ((y1 <= y) & (y < y2)) | ((y2 <= y) & (y < y1))
        t = np.where(np.abs(dy12) > 1e-15, (y - y1) / dy12, 0.0)
        xc = x1 + t * (x2 - x1)
        inside ^= cross & (xc > x)
    return inside

def backbone_intersection_area(vA, vB, resolution=800):
    """Reference backbone intersection area via dense grid sampling."""
    all_v = np.vstack([vA, vB])
    xlo, xhi = all_v[:, 0].min() - 0.005, all_v[:, 0].max() + 0.005
    ylo, yhi = all_v[:, 1].min() - 0.005, all_v[:, 1].max() + 0.005
    xs = np.linspace(xlo, xhi, resolution, endpoint=False) + (xhi - xlo) / (2 * resolution)
    ys = np.linspace(ylo, yhi, resolution, endpoint=False) + (yhi - ylo) / (2 * resolution)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    cell = (xhi - xlo) * (yhi - ylo) / resolution**2
    return np.sum(pip_batch(pts, vA) & pip_batch(pts, vB)) * cell

def make_model(verts_list, delta):
    all_v = np.vstack(verts_list)
    nv = len(all_v)
    si = [0]
    for v in verts_list:
        si.append(si[-1] + len(v))
    m = pcp.model(nv)
    m.setStartIndices(si)
    m.setPositions(all_v.flatten())
    m.setModelEnum("rounded")
    m.setDelta(delta)
    m.setMaxEdgeLength(0.5)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    m.updateNeighbors()
    m.updatePolygonGeometry()
    m.updateForceEnergy()
    return m

def run_test(label, vA, vB, delta, idx):
    print(f"\n{'='*60}")
    print(f"Test {idx}: {label}")

    aA = abs(signed_area(vA))
    aB = abs(signed_area(vB))
    print(f"  backbone areas: A={aA:.5f}  B={aB:.5f}")
    print(f"  A CCW? {signed_area(vA) > 0}   B CCW? {signed_area(vB) > 0}")

    ref = backbone_intersection_area(vA, vB)
    print(f"  Backbone intersection (grid ref) = {ref:.6e}")

    m = make_model([vA, vB], delta)
    E = m.getEnergy()
    print(f"  Model energy  E                  = {E:.6e}")

    if ref > 1e-8:
        print(f"  E / ref                          = {E/ref:.4f}")
    if E < 0:
        print("  *** ERROR: energy is NEGATIVE ***")
    if ref > 1e-8 and abs(E - ref) / ref > 0.10:
        print(f"  *** WARNING: >10% discrepancy (expected for large delta or rounding) ***")

    # Draw
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    for verts, fc, ec in [(vA, '#aec6cf', '#1f5c7a'), (vB, '#ffecb3', '#7a5200')]:
        patch = mpatches.Polygon(verts, closed=True, alpha=0.5, facecolor=fc, edgecolor=ec, lw=1.5)
        ax.add_patch(patch)
        cx, cy = verts.mean(axis=0)
        ax.scatter(*verts[0], s=40, c='r', zorder=5)  # mark vertex 0
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f"Backbone\nref={ref:.3e}")

    ax = axes[1]
    m.updatePolygonGeometry()
    m.updateForceEnergy()
    m.drawRounded(ax=ax, delta=delta)
    ax.set_title(f"Rounded model\nE={E:.3e}  {'OK' if E >= 0 else 'NEGATIVE!'}")

    fname = os.path.join(OUT_DIR, f"nonconvex_test{idx}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")
    return E, ref

# ─── parameters ──────────────────────────────────────────────────────────────
N     = 8        # vertices per polygon (small for readability)
R     = 0.08     # circumradius
# Edge length of regular N-gon: 2R sin(π/N)
EL    = 2 * R * np.sin(np.pi / N)
DELTA = 0.05 * EL  # small rounding (5% of edge length)

DENT_IDX = 0     # which vertex to dent (angle 0 = rightmost)
DENT_R   = 0.2 * R  # reflex corner radius (deep dent)

# A centered at (0.4, 0.5); dent vertex points RIGHT (toward B)
cxA, cyA = 0.4, 0.5
# B centered at varying positions

print(f"N={N}  R={R:.4f}  EL={EL:.4f}  DELTA={DELTA:.5f}")
print(f"Dent: vertex {DENT_IDX} pushed to r={DENT_R:.4f} (was R={R:.4f})")

results = []

# ─── Test 0: Convex A ∩ Convex B (sanity) ────────────────────────────────────
vA = regular_poly(N, R, cxA, cyA)
vB = regular_poly(N, R, 0.53, 0.5, theta0=0.1)   # overlap ~0.05 gap
E, ref = run_test("Convex A ∩ Convex B (sanity)", vA, vB, DELTA, 0)
results.append(("Convex vs Convex", E, ref))

# ─── Test 1: Dented A ∩ Convex B, overlap AWAY from dent ─────────────────────
# Dent is at angle 0 (right side of A at x≈0.48).
# B is placed above A to overlap on the top — far from dent.
vA = dented_poly(N, R, DENT_IDX, DENT_R, cxA, cyA)
vB = regular_poly(N, R, 0.45, 0.57, theta0=0.0)   # B above A
E, ref = run_test("Dented A ∩ Convex B, overlap AWAY from dent", vA, vB, DELTA, 1)
results.append(("Dented(away)", E, ref))

# ─── Test 2: Dented A ∩ Convex B, overlap ONTO dent region ───────────────────
# B is to the right of A; the dent is on A's right side.
# B's left extent reaches into the dent notch of A.
vA = dented_poly(N, R, DENT_IDX, DENT_R, cxA, cyA)
# B at (0.52, 0.5): left extent ≈ 0.52 - R = 0.44; dent vertex at 0.4+DENT_R=0.416
# B overlaps A's body but also covers the dent region.
vB = regular_poly(N, R, 0.51, 0.5, theta0=0.0)
E, ref = run_test("Dented A ∩ Convex B, overlap ON dent", vA, vB, DELTA, 2)
results.append(("Dented(on-dent)", E, ref))

# ─── Test 3: Dented A ∩ Dented B, dents facing each other ────────────────────
# A's dent faces right (vertex 0), B's dent faces left (vertex N//2).
vA = dented_poly(N, R, 0,      DENT_R, 0.40, 0.5)
vB = dented_poly(N, R, N // 2, DENT_R, 0.52, 0.5)   # vertex N//2 is at angle π (left)
E, ref = run_test("Dented A ∩ Dented B, dents facing each other", vA, vB, DELTA, 3)
results.append(("Both dented", E, ref))

# ─── Test 4: Deep dent — B fully covers the dent cavity ──────────────────────
# B is so large or well-placed that the reflex arc of A is entirely inside B.
DENT_R2 = 0.05 * R  # very deep dent
vA = dented_poly(N, R, 0, DENT_R2, cxA, cyA)
vB = regular_poly(N, 1.2 * R, 0.50, 0.5)  # larger B fully covers A's dent region
E, ref = run_test("Deep dent A fully inside larger B", vA, vB, DELTA, 4)
results.append(("Deep dent, A in B", E, ref))

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print(f"{'Test':<35}  {'E':>12}  {'ref':>12}  {'E/ref':>8}  {'OK?'}")
for name, E, ref in results:
    ratio = f"{E/ref:.4f}" if ref > 1e-8 else "  N/A"
    ok = "OK" if E >= -1e-12 else "NEGATIVE"
    if ref > 1e-8 and abs(E - ref) / ref > 0.10:
        ok += " (large diff)"
    print(f"  {name:<33}  {E:>12.4e}  {ref:>12.4e}  {ratio:>8}  {ok}")
