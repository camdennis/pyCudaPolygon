"""
draw_grid_check.py — visualise the independent grid used by verify_pairarea.py.

Draws, for a chosen bad pair, the two smoothed polygon boundaries and the grid
classification (inside-A, inside-B, inside-both) so the overlap capture can be
checked by eye. The grid here is pure NumPy — no model calls, no maxEdgeLength.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyCudaPolygon as pcp

PAIR = (20, 30)   # worst negative pair from verify_pairarea.py
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rounded",
                    "grid_check.png")

# ── model setup (identical to fire_packing64_rounded.py, stop at step 0) ──────
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
pa        = np.array(m.getPairArea()).reshape(N_POLY, N_POLY)

# ── geometry helpers (same as verify_pairarea.py) ────────────────────────────
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

def smoothed_boundary(ux, uy, n_arc_pts=24):
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

# ── build the pair ───────────────────────────────────────────────────────────
A, B = PAIR
uxA, uyA = unwrap_polygon(A)
uxB, uyB = unwrap_polygon(B)
cA = np.array([uxA.mean(), uyA.mean()])
cB = np.array([uxB.mean(), uyB.mean()])
shift = np.round(cA - cB)
polyA = smoothed_boundary(uxA, uyA)
polyB = smoothed_boundary(uxB + shift[0], uyB + shift[1])

# overlap bounding box = the grid region
minx = max(polyA[:,0].min(), polyB[:,0].min())
maxx = min(polyA[:,0].max(), polyB[:,0].max())
miny = max(polyA[:,1].min(), polyB[:,1].min())
maxy = min(polyA[:,1].max(), polyB[:,1].max())

K = 500
xs = np.linspace(minx, maxx, K)
ys = np.linspace(miny, maxy, K)
cell = (xs[1]-xs[0]) * (ys[1]-ys[0])
gx, gy = np.meshgrid(xs, ys)
pts = np.column_stack([gx.ravel(), gy.ravel()])
inA = points_in_poly(pts, polyA)
inB = points_in_poly(pts, polyB)
both = inA & inB

cls = np.zeros(len(pts))           # 0 outside
cls[inA & ~inB] = 1                # A only
cls[inB & ~inA] = 2                # B only
cls[both]       = 3                # intersection
cls = cls.reshape(K, K)

grid_area = both.sum() * cell
print(f"pair {PAIR}:  kernel pa[A,B]+pa[B,A] = {pa[A,B]+pa[B,A]:.5e}")
print(f"             grid intersection area = {grid_area:.5e}")
print(f"             grid points classified inside-both = {both.sum()}")

# ── draw ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# panel 1: whole pair + grid-region rectangle
ax1.plot(*np.vstack([polyA, polyA[0]]).T, '-', color='#1f77b4', lw=1.5,
         label=f'smoothed poly {A}')
ax1.plot(*np.vstack([polyB, polyB[0]]).T, '-', color='#ff7f0e', lw=1.5,
         label=f'smoothed poly {B}')
ax1.add_patch(plt.Rectangle((minx, miny), maxx-minx, maxy-miny,
              fill=False, edgecolor='k', ls='--', lw=1, label='grid region'))
ax1.set_aspect(1); ax1.legend(fontsize=9)
ax1.set_title(f"pair {PAIR}  —  full smoothed polygons")

# panel 2: grid classification raster + boundaries
cmap = matplotlib.colors.ListedColormap(
    ['white', '#aec7e8', '#ffbb78', '#d62728'])
ax2.imshow(cls, origin='lower', extent=[minx, maxx, miny, maxy],
           cmap=cmap, vmin=0, vmax=3, interpolation='nearest', aspect='equal')
ax2.plot(*np.vstack([polyA, polyA[0]]).T, '-', color='#1f77b4', lw=1.2)
ax2.plot(*np.vstack([polyB, polyB[0]]).T, '-', color='#ff7f0e', lw=1.2)
ax2.set_xlim(minx, maxx); ax2.set_ylim(miny, maxy)
ax2.set_title(f"grid {K}x{K}: red=inside both  (area={grid_area:.4e})")

fig.savefig(OUT, dpi=130, bbox_inches='tight')
print(f"saved {OUT}")
