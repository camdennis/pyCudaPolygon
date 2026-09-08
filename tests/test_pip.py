"""
test_pip.py — faithful Python port of the kernel's pipSmoothed(), tested against
the verified ray-caster (points_in_poly). If they disagree for points deep
inside a polygon, the inside-test is the deep-overlap bug.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
import pyCudaPolygon as pcp

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

positions = np.array(m.getPositions())
delta     = m.getDelta()
nArray    = np.array(m.getnArray())
startIdx  = np.concatenate([[0], np.cumsum(nArray)])

# ── geometry ──────────────────────────────────────────────────────────────────
def unwrap_polygon(p):
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

def polygon_arc_arrays(p):
    """Return per-vertex arc arrays (bam, bap, bz, bphi0, bdphi) as the kernel
    would see them."""
    ux, uy = unwrap_polygon(p)
    n = len(ux)
    bam = np.zeros((n, 2)); bap = np.zeros((n, 2))
    bz  = np.zeros((n, 2)); bphi0 = np.zeros(n); bdphi = np.zeros(n)
    for i in range(n):
        ip, inn = (i-1) % n, (i+1) % n
        g = arc_geom(ux[i], uy[i], ux[i]-ux[ip], uy[i]-uy[ip],
                     ux[inn]-ux[i], uy[inn]-uy[i])
        if g is None:
            bam[i] = bap[i] = (ux[i], uy[i]); bdphi[i] = 0.0
        else:
            amx, amy, apx, apy, zx, zy, phi0, dphi = g
            bam[i] = (amx, amy); bap[i] = (apx, apy)
            bz[i] = (zx, zy); bphi0[i] = phi0; bdphi[i] = dphi
    return bam, bap, bz, bphi0, bdphi, (ux, uy)

def smoothed_boundary(ux, uy, n_arc_pts=24):
    n = len(ux)
    geoms = [arc_geom(ux[i], uy[i], ux[i]-ux[(i-1)%n], uy[i]-uy[(i-1)%n],
                      ux[(i+1)%n]-ux[i], uy[(i+1)%n]-uy[i]) for i in range(n)]
    bx, by = [], []
    for i in range(n):
        g_i, g_n = geoms[i], geoms[(i+1) % n]
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
            bx.append(ux[(i+1) % n]); by.append(uy[(i+1) % n])
    return np.column_stack([bx, by])

def points_in_poly(pts, poly):
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    xj, yj = poly[:, 0], poly[:, 1]
    xi, yi = np.roll(xj, 1), np.roll(yj, 1)
    for k in range(len(poly)):
        cond = ((yi[k] > y) != (yj[k] > y))
        xint = (xj[k]-xi[k]) * (y-yi[k]) / (yj[k]-yi[k] + 1e-300) + xi[k]
        inside ^= cond & (x < xint)
    return inside

# ── faithful port of kernel pipSmoothed / tauInArc / wrapAngleDiff ───────────
def wrap_angle_diff(x):
    x = np.fmod(x + 3.0*np.pi, 2.0*np.pi) - np.pi
    return np.where(x <= -np.pi, x + 2.0*np.pi, x)

def tau_in_arc(phi, phi0, dphi):
    if abs(dphi) < 1e-12:
        return np.full_like(phi, -1.0)
    diff = wrap_angle_diff(phi - phi0)
    tau = diff / dphi
    return np.where((tau > 1e-9) & (tau < 1.0 - 1e-9), tau, -1.0)

def pip_smoothed(pts, bam, bap, bz, bphi0, bdphi, delta):
    px, py = pts[:, 0], pts[:, 1]
    cnt = np.zeros(len(pts), dtype=np.int64)
    n = len(bam)
    for j in range(n):
        jn = (j + 1) % n
        if abs(bdphi[j]) > 1e-12:
            dy = py - bz[j, 1]
            disc = delta*delta - dy*dy
            valid = disc > 0.0
            sq = np.sqrt(np.where(valid, disc, 0.0))
            for sg in (-1, 1):
                xc = bz[j, 0] + sg*sq
                mask = valid & (xc > px)
                phi = np.arctan2(dy, xc - bz[j, 0])
                tau = tau_in_arc(phi, bphi0[j], bdphi[j])
                cnt += (mask & (tau >= 0.0)).astype(np.int64)
        x1, y1 = bap[j]; x2, y2 = bam[jn]
        cross = ((y1 <= py) & (py < y2)) | ((y2 <= py) & (py < y1))
        t = (py - y1) / (y2 - y1 + 1e-300)
        xint = x1 + t*(x2 - x1)
        cnt += (cross & (xint > px)).astype(np.int64)
    return (cnt & 1) != 0

# ── compare the two inside-tests over each polygon's bounding box ─────────────
print(f"delta={delta:.6e}")
print(f"{'poly':>6} {'grid pts':>10} {'disagree':>10} {'disagree %':>11} "
      f"{'deep disagree':>14}")
for p in (7, 20, 30, 22, 8, 27):
    bam, bap, bz, bphi0, bdphi, (ux, uy) = polygon_arc_arrays(p)
    poly = smoothed_boundary(ux, uy)
    minx, maxx = poly[:,0].min(), poly[:,0].max()
    miny, maxy = poly[:,1].min(), poly[:,1].max()
    pad = 0.1*(maxx-minx)
    K = 400
    xs = np.linspace(minx-pad, maxx+pad, K)
    ys = np.linspace(miny-pad, maxy+pad, K)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    truth = points_in_poly(pts, poly)
    test  = pip_smoothed(pts, bam, bap, bz, bphi0, bdphi, delta)
    disagree = truth != test
    # "deep" = truth-inside points well away from the boundary
    cx, cy = ux.mean(), uy.mean()
    rad = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
    deep = truth & (rad < 0.5*rad[truth].max())
    deep_dis = (disagree & deep).sum()
    print(f"{p:>6} {len(pts):>10} {disagree.sum():>10} "
          f"{100*disagree.mean():>10.3f}% {deep_dis:>14}")
