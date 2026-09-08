"""
verify_pairarea.py — cross-check the CUDA `rounded` model's per-pair signed
intersection area (getPairArea) against an independent grid computation.

For two CCW polygons the pairwise intersection area is mathematically >= 0.
If the kernel reports negative pairArea, either (a) it is a pure sign bug
(|pairArea| has the right magnitude) or (b) the magnitude is also wrong.
This script settles that empirically.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
import pyCudaPolygon as pcp

# ── replicate fire_packing64_rounded.py setup, stop at step 0 ──────────────────
N_POLY, N_SIDES = 64, 32
N_LARGE = N_SMALL = N_POLY // 2
KAPPA, RATIO, PHI = 3.7, 1.4, 0.8

A_SMALL = PHI / (N_LARGE * RATIO**2 + N_SMALL)
A_LARGE = RATIO**2 * A_SMALL
EL_LARGE = KAPPA * np.sqrt(A_LARGE) / N_SIDES
EL_SMALL = KAPPA * np.sqrt(A_SMALL) / N_SIDES
DELTA = 0.15 * EL_SMALL

m = pcp.model(N_POLY * N_SIDES, seed=42)
m.generateRandomPolygons(N_POLY, N_SIDES)
m.setModelEnum("rounded")
m.setDelta(DELTA)
m.setTargetAreas(np.array([A_LARGE] * N_LARGE + [A_SMALL] * N_SMALL))
m.setTargetEdgeLengths(np.array([EL_LARGE] * N_LARGE + [EL_SMALL] * N_SMALL))
m.setMaxEdgeLength(2 * np.sqrt(A_LARGE / np.pi) + 2 * DELTA)
m.initializeNeighborCells()
m.updateNeighborCells()
m.updateNeighbors()
m.updatePolygonGeometry()

m.setStiffness(1)
m.setCompressibility(1)
m.updateForceEnergy()
m.setModelEnum("softBody")
m.minimizeFIRE(dt=0.1, maxForceThreshold=1e-14)
m.updatePolygonGeometry()
m.setModelEnum("rounded")
m.updateForceEnergy()

E0        = m.getEnergy()
positions = np.array(m.getPositions())
delta     = m.getDelta()
nArray    = np.array(m.getnArray())
startIdx  = np.concatenate([[0], np.cumsum(nArray)])
N         = N_POLY
pa        = np.array(m.getPairArea()).reshape(N, N)

print(f"delta={delta:.6e}  E0(getEnergy, two-pass |area|)={E0:.6e}")

# ── geometry helpers ──────────────────────────────────────────────────────────
def unwrap_polygon(p):
    """Vertices of polygon p, periodic-unwrapped into one consistent image."""
    s, n = startIdx[p], nArray[p]
    vx = positions[2*s:2*(s+n):2].copy()
    vy = positions[2*s+1:2*(s+n):2].copy()
    ux = np.empty(n); uy = np.empty(n)
    ux[0], uy[0] = vx[0], vy[0]
    for i in range(1, n):
        dx, dy = vx[i]-vx[i-1], vy[i]-vy[i-1]
        dx -= round(dx); dy -= round(dy)
        ux[i], uy[i] = ux[i-1]+dx, uy[i-1]+dy
    return ux, uy

def smoothed_boundary(ux, uy, n_arc_pts=24):
    """Single-arc rounded boundary as a dense closed polyline (mirrors the
    kernel's computeArcGeom: arcs radius delta at center z, edges ap_i->am_{i+1})."""
    n = len(ux)
    def arc_geom(vx, vy, axx, ayy, wx, wy):
        u_len = np.hypot(axx, ayy); v_len = np.hypot(wx, wy)
        if u_len < 1e-12 or v_len < 1e-12:
            return None
        uhx, uhy = axx/u_len, ayy/u_len
        vhx, vhy = wx/v_len, wy/v_len
        cr = uhx*vhy - uhy*vhx
        dt = uhx*vhx + uhy*vhy
        abs_cr, denom = abs(cr), 1.0 + dt
        if abs_cr < 1e-12 or denom < 1e-12:
            return None
        ell = delta * abs_cr / denom
        amx, amy = vx - ell*uhx, vy - ell*uhy
        apx, apy = vx + ell*vhx, vy + ell*vhy
        inv = 1.0 / abs_cr
        zx = vx + delta*(vhx - uhx)*inv
        zy = vy + delta*(vhy - uhy)*inv
        phi0 = np.arctan2(amy - zy, amx - zx)
        dphi = np.arctan2(cr, dt)
        return amx, amy, apx, apy, zx, zy, phi0, dphi
    geoms = []
    for i in range(n):
        ip, inn = (i-1) % n, (i+1) % n
        geoms.append(arc_geom(ux[i], uy[i],
                              ux[i]-ux[ip], uy[i]-uy[ip],
                              ux[inn]-ux[i], uy[inn]-uy[i]))
    bx, by = [], []
    for i in range(n):
        inn = (i+1) % n
        g_i, g_n = geoms[i], geoms[inn]
        if g_i is not None:
            amx, amy, apx, apy, zx, zy, phi0, dphi = g_i
            npts = max(3, int(abs(dphi)/(2*np.pi)*n_arc_pts) + 2)
            for t in np.linspace(0, dphi, npts):
                bx.append(zx + delta*np.cos(phi0 + t))
                by.append(zy + delta*np.sin(phi0 + t))
        else:
            bx.append(ux[i]); by.append(uy[i])
        if g_n is not None:
            bx.append(g_n[0]); by.append(g_n[1])
        else:
            bx.append(ux[inn]); by.append(uy[inn])
    return np.column_stack([bx, by])

def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)

def points_in_poly(pts, poly):
    """Vectorized ray-casting: pts (K,2), poly (M,2) closed polyline."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    xj, yj = poly[:, 0], poly[:, 1]
    xi, yi = np.roll(xj, 1), np.roll(yj, 1)
    for k in range(len(poly)):
        cond = ((yi[k] > y) != (yj[k] > y))
        xint = (xj[k]-xi[k]) * (y-yi[k]) / (yj[k]-yi[k] + 1e-300) + xi[k]
        inside ^= cond & (x < xint)
    return inside

def grid_intersection_area(polyA, polyB, K=900):
    minx = max(polyA[:,0].min(), polyB[:,0].min())
    maxx = min(polyA[:,0].max(), polyB[:,0].max())
    miny = max(polyA[:,1].min(), polyB[:,1].min())
    maxy = min(polyA[:,1].max(), polyB[:,1].max())
    if minx >= maxx or miny >= maxy:
        return 0.0
    xs = np.linspace(minx, maxx, K)
    ys = np.linspace(miny, maxy, K)
    cell = (xs[1]-xs[0]) * (ys[1]-ys[0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inA = points_in_poly(pts, polyA)
    inB = points_in_poly(pts, polyB)
    return np.sum(inA & inB) * cell

def densify(poly, max_seg=8e-5):
    """Resample a closed polyline so every segment is shorter than max_seg —
    makes the midpoint-inside test in directional_integral accurate."""
    out = []
    n = len(poly)
    for k in range(n):
        p0, p1 = poly[k], poly[(k+1) % n]
        d = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
        m = max(1, int(np.ceil(d / max_seg)))
        for j in range(m):
            out.append(p0 + (p1-p0)*(j/m))
    return np.array(out)

def directional_integral(polyA, polyB):
    """Independent reconstruction of the kernel's I(A->B): sum the origin-anchored
    Green's term 0.5*(x0*y1 - x1*y0) over segments of polyA whose midpoint lies
    inside polyB. polyA should be densified."""
    x0, y0 = polyA[:,0], polyA[:,1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    mids = np.column_stack([0.5*(x0+x1), 0.5*(y0+y1)])
    inside = points_in_poly(mids, polyB)
    terms = 0.5*(x0*y1 - x1*y0)
    return np.sum(terms[inside])

def grid_self_area(poly, K=900):
    """Grid area of a single polygon — must match |shoelace| if the
    point-in-polygon test is correct."""
    minx, maxx = poly[:,0].min(), poly[:,0].max()
    miny, maxy = poly[:,1].min(), poly[:,1].max()
    xs = np.linspace(minx, maxx, K)
    ys = np.linspace(miny, maxy, K)
    cell = (xs[1]-xs[0]) * (ys[1]-ys[0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return points_in_poly(pts, poly).sum() * cell

# ── validate the point-in-polygon / grid machinery ───────────────────────────
print("── point-in-polygon validation ──")
# analytic: circle radius 0.1
th = np.linspace(0, 2*np.pi, 4000, endpoint=False)
circ = np.column_stack([0.3 + 0.1*np.cos(th), 0.4 + 0.1*np.sin(th)])
print(f"  circle r=0.1:  grid={grid_self_area(circ):.6e}  "
      f"analytic={np.pi*0.01:.6e}")
# two offset circles, known lens-area formula
d = 0.12
c2 = np.column_stack([0.3 + d + 0.1*np.cos(th), 0.4 + 0.1*np.sin(th)])
r = 0.1
lens = 2*r*r*np.arccos(d/(2*r)) - 0.5*d*np.sqrt(4*r*r - d*d)
print(f"  two circles d=0.12:  grid_isect="
      f"{grid_intersection_area(circ, c2):.6e}  analytic_lens={lens:.6e}")
print()

# ── build smoothed boundaries; bring B to min-image of A per pair ─────────────
verts = [unwrap_polygon(p) for p in range(N)]
cents = np.array([(np.mean(ux), np.mean(uy)) for ux, uy in verts])

def smoothed_pair(A, B):
    uxA, uyA = verts[A]
    uxB, uyB = verts[B]
    shift = np.round(cents[A] - cents[B])  # min-image B near A
    polyA = smoothed_boundary(uxA, uyA)
    polyB = smoothed_boundary(uxB + shift[0], uyB + shift[1])
    return polyA, polyB

# ── sanity: smoothed polygon orientation + grid-vs-shoelace self test ─────────
neg_shoelace = sum(1 for p in range(N) if shoelace(smoothed_boundary(*verts[p])) < 0)
print(f"smoothed polygons with negative (CW) shoelace: {neg_shoelace}/{N}")
print("  grid_self vs |shoelace| for the polygons used below:")
for p in (7, 20, 30, 8, 27):
    poly = smoothed_boundary(*verts[p])
    print(f"    poly {p:>2}: grid_self={grid_self_area(poly):.6e}  "
          f"|shoelace|={abs(shoelace(poly)):.6e}")
print()

# ── enumerate pairs, classify by signed pairArea sum ──────────────────────────
pairs = []
for A in range(N):
    for B in range(A+1, N):
        s = pa[A, B] + pa[B, A]
        if abs(s) > 1e-12:
            pairs.append((A, B, s))
pairs.sort(key=lambda t: t[2])

signed_total = sum(s for _, _, s in pairs)
abs_total    = sum(abs(s) for _, _, s in pairs)
n_neg = sum(1 for _, _, s in pairs if s < 0)
print(f"nonzero pairs: {len(pairs)}   negative: {n_neg}")
print(f"signed Σ(pairArea) = {signed_total:.6e}   Σ|pairArea| = {abs_total:.6e}")
print()

# ── verify the most-negative pairs and a few positive ones ────────────────────
to_check = pairs[:4] + pairs[-3:]
print(f"per-direction breakdown  (kernel pa  vs  independent I)")
print(f"{'pair':>10} {'dir':>5} {'kernel pa':>13} {'indep I':>13} {'Δ':>12}")
for A, B, s in to_check:
    polyA, polyB = smoothed_pair(A, B)
    dA, dB = densify(polyA), densify(polyB)
    I_AtoB = directional_integral(dA, polyB)
    I_BtoA = directional_integral(dB, polyA)
    true_area = grid_intersection_area(polyA, polyB)
    tag = "NEG" if s < 0 else "pos"
    print(f"({A:>3},{B:>3}) [{tag}]")
    print(f"{'':>10} {'A->B':>5} {pa[A,B]:>13.5e} {I_AtoB:>13.5e} "
          f"{pa[A,B]-I_AtoB:>12.2e}")
    print(f"{'':>10} {'B->A':>5} {pa[B,A]:>13.5e} {I_BtoA:>13.5e} "
          f"{pa[B,A]-I_BtoA:>12.2e}")
    print(f"{'':>10} {'sum':>5} {pa[A,B]+pa[B,A]:>13.5e} "
          f"{I_AtoB+I_BtoA:>13.5e}   (grid intersection area = {true_area:.5e})")
    print()
