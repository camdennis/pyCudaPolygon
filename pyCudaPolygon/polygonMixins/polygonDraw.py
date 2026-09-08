import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arc

class Mixin():
    def __init__(self, rng = None):
        # random number generator fallback
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Public entry point: draw(). Dispatches to the right rendering based
    # on the model's current mode and per-polygon-δ flag.
    # ------------------------------------------------------------------
    def draw(self, ax=None, mode=None,
             numbering=False, forces=None, arrowColor='r',
             axisSize=1, ms=3, marker='-o',
             delta=None, n_arc_pts=24, psi_frac=0.5,
             facecolor=None, edgecolor=None, alpha=0.5, linewidth=0.8,
             colors=None):
        """Draw every polygon. Boundary style is auto-detected unless `mode`
        is given:
            mode='backbone' — vertices + straight edges (original behavior)
            mode='rounded'  — single-arc rounded boundary
            mode='biarc'    — area-preserving biarc boundary (uses psi_frac)
            mode=None       — picks 'rounded' if getModelEnum() == 'rounded',
                              else 'backbone'

        Per-polygon δ is auto-applied for rounded / biarc rendering when
        `isPerPolygonDeltaEnabled()` is True (passes the polyDelta array
        through to the per-polygon arc geometry). Override by passing
        `delta=` explicitly (scalar or length-numPolygons array).

        `forces`, `numbering` overlay backbone-vertex markers on top of any
        rounded/biarc fill — pass them and you'll see force quivers and/or
        vertex indices regardless of mode.
        """
        if ax is None:
            _, ax = plt.subplots()

        # Resolve mode
        if mode is None:
            try:
                mode = 'rounded' if self.getModelEnum().lower() == 'rounded' else 'backbone'
            except Exception:
                mode = 'backbone'

        # Resolve δ for rounded/biarc paths
        delta_arg = delta
        if delta_arg is None and mode in ('rounded', 'biarc'):
            delta_arg = self._resolveDelta()

        if mode == 'rounded':
            self._drawRoundedInternal(
                ax, delta_arg, axisSize, n_arc_pts,
                facecolor, edgecolor, alpha, linewidth, colors)
            # Overlay backbone markers/forces if requested.
            if forces is not None or numbering:
                self._drawBackboneInternal(
                    ax, numbering, forces, arrowColor, axisSize, ms, marker,
                    lines=False)
        elif mode == 'biarc':
            self._drawBiarcInternal(
                ax, delta_arg, psi_frac, axisSize, n_arc_pts,
                facecolor, edgecolor, alpha, linewidth, colors)
            if forces is not None or numbering:
                self._drawBackboneInternal(
                    ax, numbering, forces, arrowColor, axisSize, ms, marker,
                    lines=False)
        else:
            # backbone (original behavior)
            self._drawBackboneInternal(
                ax, numbering, forces, arrowColor, axisSize, ms, marker,
                lines=True)

        ax.set_xlim([0, axisSize])
        ax.set_ylim([0, axisSize])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    # ------------------------------------------------------------------
    # Backward-compatible shims: drawRounded / drawBiarc still work, but
    # forward into the unified renderer so behavior matches draw().
    # ------------------------------------------------------------------
    def drawRounded(self, ax=None, delta=None, axisSize=1, n_arc_pts=24,
                    facecolor=None, edgecolor=None, alpha=0.5, linewidth=0.8,
                    colors=None):
        return self.draw(ax=ax, mode='rounded', delta=delta,
                         axisSize=axisSize, n_arc_pts=n_arc_pts,
                         facecolor=facecolor, edgecolor=edgecolor,
                         alpha=alpha, linewidth=linewidth, colors=colors)

    def drawBiarc(self, ax=None, delta=None, psi_frac=0.5, axisSize=1,
                  n_arc_pts=24, facecolor=None, edgecolor=None, alpha=0.5,
                  linewidth=0.8, colors=None):
        return self.draw(ax=ax, mode='biarc', delta=delta, psi_frac=psi_frac,
                         axisSize=axisSize, n_arc_pts=n_arc_pts,
                         facecolor=facecolor, edgecolor=edgecolor,
                         alpha=alpha, linewidth=linewidth, colors=colors)

    # ------------------------------------------------------------------
    # Internal: resolve δ for rendering. Returns either a scalar or a
    # numpy array of length numPolygons.
    # ------------------------------------------------------------------
    def _resolveDelta(self):
        try:
            if self.isPerPolygonDeltaEnabled():
                return np.asarray(self.getPolyDelta())
        except Exception:
            pass
        return self.getDelta()

    @staticmethod
    def _deltaForPoly(delta, poly_idx):
        """Return the δ to use for polygon poly_idx whether δ is scalar or array."""
        if np.ndim(delta) == 0:
            return float(delta)
        return float(delta[poly_idx])

    # ------------------------------------------------------------------
    # Internal: backbone rendering (the original draw() body, split out).
    # `lines=True` plots straight edges between vertices; `lines=False`
    # only plots vertex markers (used as overlay on top of arc fills).
    # ------------------------------------------------------------------
    def _drawBackboneInternal(self, ax, numbering, forces, arrowColor,
                              axisSize, ms, marker, lines):
        def fixPXPY(px, py):
            # Unwrap vertices so consecutive vertices are MIC-close. Vertex 0
            # stays at its original position; subsequent vertices extend
            # outside [0,1] if needed. This preserves the polygon's actual
            # position so it aligns with MC test points / GPU kernels that
            # operate on raw positions.
            n = len(px)
            ox = np.empty(n); oy = np.empty(n)
            ox[0] = px[0]; oy[0] = py[0]
            for k in range(1, n):
                dx = px[k] - ox[k-1]
                dy = py[k] - oy[k-1]
                dx -= np.round(dx)
                dy -= np.round(dy)
                ox[k] = ox[k-1] + dx
                oy[k] = oy[k-1] + dy
            return ox, oy

        pos = self.getPositions()
        nArray = self.getnArray()
        cmap = plt.get_cmap('tab20')
        start = 0
        fStart = 0
        # When overlaying, drop the connecting line so we only see markers.
        eff_marker = marker if lines else marker.replace('-', '')
        for poly_idx, n in enumerate(nArray):
            color = cmap(poly_idx % cmap.N)
            px = pos[start:start + 2*n][::2]
            py = pos[start:start + 2*n][1::2]
            px = np.concatenate((px, [px[0]]))
            py = np.concatenate((py, [py[0]]))
            px, py = fixPXPY(px, py)
            if forces is not None:
                fx = forces[fStart:fStart + 2*n][::2]
                fy = forces[fStart:fStart + 2*n][1::2]
                fx = np.concatenate((fx, [fx[0]]))
                fy = np.concatenate((fy, [fy[0]]))
            for i in range(3):
                for j in range(3):
                    if eff_marker:
                        ax.plot((px + i - 1) * axisSize, (py + j - 1) * axisSize,
                                eff_marker, markersize=ms,
                                color=color, markerfacecolor=color,
                                markeredgecolor=color)
                    if forces is not None:
                        ax.quiver(px + i - 1, py + j - 1, fx, fy,
                                  angles='xy', scale_units='xy', scale=1,
                                  color=arrowColor, width=0.003,
                                  headwidth=3, headlength=4, zorder=10)
            if numbering:
                for k in range(len(px) - 1):
                    textX = (px[k] + 1) % 1
                    textY = (py[k] + 1) % 1
                    ax.text(textX * axisSize, textY * axisSize, str(start//2 + k),
                            fontsize=8, color='k', ha='left', va='bottom')
            start  += 2*n
            fStart += 2*n

    # ------------------------------------------------------------------
    # Internal: single-arc rounded boundary (the original drawRounded body).
    # Now accepts δ as scalar OR length-numPolygons array.
    # ------------------------------------------------------------------
    def _drawRoundedInternal(self, ax, delta, axisSize, n_arc_pts,
                              facecolor, edgecolor, alpha, linewidth, colors):
        def fixPXPY(px, py):
            minX = min(px); minY = min(py)
            px += 1.5 - minX; py += 1.5 - minY
            px %= 1;          py %= 1
            px += minX - 0.5; py += minY - 0.5
            return px, py

        def arc_geom(vx, vy, ux, uy, wx, wy, dd):
            u_len = np.hypot(ux, uy);  v_len = np.hypot(wx, wy)
            if u_len < 1e-12 or v_len < 1e-12:
                return None
            uh_x, uh_y = ux / u_len, uy / u_len
            vh_x, vh_y = wx / v_len, wy / v_len
            cr = uh_x * vh_y - uh_y * vh_x
            dt = uh_x * vh_x + uh_y * vh_y
            abs_cr = abs(cr)
            denom = 1.0 + dt
            if abs_cr < 1e-12 or denom < 1e-12:
                return None
            ell = dd * abs_cr / denom
            am_x = vx - ell * uh_x; am_y = vy - ell * uh_y
            ap_x = vx + ell * vh_x; ap_y = vy + ell * vh_y
            inv_cr = 1.0 / abs_cr
            zx = vx + dd * (vh_x - uh_x) * inv_cr
            zy = vy + dd * (vh_y - uh_y) * inv_cr
            phi0 = np.arctan2(am_y - zy, am_x - zx)
            dphi = np.arctan2(cr, dt)
            return am_x, am_y, ap_x, ap_y, zx, zy, phi0, dphi

        pos = np.array(self.getPositions())
        nArray = self.getnArray()
        cmap = plt.get_cmap('tab20')
        start = 0
        for poly_idx, n in enumerate(nArray):
            dd = self._deltaForPoly(delta, poly_idx)
            if colors is not None:
                base = colors[poly_idx] if poly_idx < len(colors) else cmap(poly_idx % cmap.N)
            elif facecolor is not None:
                base = facecolor
            else:
                base = cmap(poly_idx % cmap.N)
            fc = (*base[:3], alpha)
            ec = base[:3] if edgecolor is None else edgecolor

            raw_px = pos[start:start + 2*n][::2].copy()
            raw_py = pos[start:start + 2*n][1::2].copy()
            px_c = np.concatenate((raw_px, [raw_px[0]]))
            py_c = np.concatenate((raw_py, [raw_py[0]]))
            px_c, py_c = fixPXPY(px_c, py_c)
            px = px_c[:-1]; py = py_c[:-1]

            geoms = []
            for i in range(n):
                ip = (i - 1) % n; inn = (i + 1) % n
                ux = px[i] - px[ip];  uy = py[i] - py[ip]
                wx = px[inn] - px[i]; wy = py[inn] - py[i]
                geoms.append(arc_geom(px[i], py[i], ux, uy, wx, wy, dd))

            for tile_i in range(3):
                for tile_j in range(3):
                    ox = tile_i - 1; oy = tile_j - 1
                    bdy_x, bdy_y = [], []
                    for i in range(n):
                        inn = (i + 1) % n
                        g_i = geoms[i]; g_n = geoms[inn]
                        if g_i is not None:
                            am_x, am_y, ap_x, ap_y, zx, zy, phi0, dphi = g_i
                            npts = max(3, int(abs(dphi) / (2 * np.pi) * n_arc_pts) + 2)
                            for t in np.linspace(0, dphi, npts):
                                bdy_x.append(zx + ox + dd * np.cos(phi0 + t))
                                bdy_y.append(zy + oy + dd * np.sin(phi0 + t))
                        else:
                            bdy_x.append(px[i] + ox); bdy_y.append(py[i] + oy)
                        if g_i is not None and g_n is not None:
                            bdy_x.append(g_n[0] + ox); bdy_y.append(g_n[1] + oy)
                        elif g_i is not None:
                            bdy_x.append(px[inn] + ox); bdy_y.append(py[inn] + oy)
                        elif g_n is not None:
                            bdy_x.append(g_n[0] + ox); bdy_y.append(g_n[1] + oy)
                    if len(bdy_x) < 3:
                        continue
                    bx = np.array(bdy_x) * axisSize
                    by = np.array(bdy_y) * axisSize
                    if np.any((bx >= 0) & (bx <= axisSize) & (by >= 0) & (by <= axisSize)):
                        ax.add_patch(plt.Polygon(np.column_stack([bx, by]),
                                                 closed=True, facecolor=fc,
                                                 edgecolor=ec, linewidth=linewidth))
            start += 2*n

    # ------------------------------------------------------------------
    # Internal: biarc boundary (the original drawBiarc body, parametrized
    # over per-polygon δ).
    # ------------------------------------------------------------------
    def _drawBiarcInternal(self, ax, delta, psi_frac, axisSize, n_arc_pts,
                            facecolor, edgecolor, alpha, linewidth, colors):
        psi_frac = float(np.clip(psi_frac, 1e-3, 1 - 1e-3))

        def fixPXPY(px, py):
            minX = min(px); minY = min(py)
            px += 1.5 - minX; py += 1.5 - minY
            px %= 1;          py %= 1
            px += minX - 0.5; py += minY - 0.5
            return px, py

        def biarc_geom(vx, vy, ux, uy, wx, wy, dd):
            u_len = np.hypot(ux, uy); v_len = np.hypot(wx, wy)
            if u_len < 1e-12 or v_len < 1e-12:
                return None
            uhx, uhy = ux / u_len, uy / u_len
            vhx, vhy = wx / v_len, wy / v_len
            c = uhx * vhy - uhy * vhx
            d = uhx * vhx + uhy * vhy
            if c < 1e-10 or (1 + d) < 1e-10:
                return None
            phi  = np.arctan2(c, d)
            psi1 = psi_frac * phi
            psi2 = phi - psi1
            s1h = np.sin(psi1 / 2); c1h = np.cos(psi1 / 2)
            s2h = np.sin(psi2 / 2); c2h = np.cos(psi2 / 2)
            if s1h < 1e-10 or s2h < 1e-10:
                return None
            sphi2 = np.sin(phi / 2); cphi2 = np.cos(phi / 2)
            nhx, nhy = -uhy, uhx
            P_over_h2 = 0.5 * np.sin(phi) + 0.5 * cphi2**2 * (c1h/s1h + c2h/s2h)
            seg1 = (psi1 - np.sin(psi1)) / s1h**4
            seg2 = (psi2 - np.sin(psi2)) / s2h**4
            F = P_over_h2 - cphi2**2 / 8.0 * (seg1 + seg2)
            if F <= 1e-20:
                return None
            dA_star = dd**2 * (np.tan(phi / 2) - phi / 2)
            if dA_star <= 0:
                return None
            h = np.sqrt(dA_star / F)
            d1 = h * cphi2 / (2.0 * s1h**2)
            el1 = h * sphi2 + h * cphi2 * (c1h / s1h)
            d2 = h * cphi2 / (2.0 * s2h**2)
            el2 = h * sphi2 + h * cphi2 * (c2h / s2h)
            vpx = -c * uhx - d * uhy
            vpy = -c * uhy + d * uhx
            amx = vx - el1 * uhx; amy = vy - el1 * uhy
            apx = vx + el2 * vhx; apy = vy + el2 * vhy
            z1x = amx + d1 * nhx; z1y = amy + d1 * nhy
            z2x = apx + d2 * vpx; z2y = apy + d2 * vpy
            jx = vx + h * (-sphi2 * uhx + cphi2 * nhx)
            jy = vy + h * (-sphi2 * uhy + cphi2 * nhy)
            phi0_1 = np.arctan2(amy - z1y, amx - z1x)
            phi0_2 = np.arctan2(jy - z2y, jx - z2x)
            return (amx, amy, apx, apy, jx, jy, z1x, z1y, z2x, z2y,
                    d1, d2, phi0_1, psi1, phi0_2, psi2)

        pos = np.array(self.getPositions())
        nArray = self.getnArray()
        cmap = plt.get_cmap('tab20')
        start = 0
        for poly_idx, n in enumerate(nArray):
            dd = self._deltaForPoly(delta, poly_idx)
            if colors is not None:
                base = colors[poly_idx] if poly_idx < len(colors) else cmap(poly_idx % cmap.N)
            elif facecolor is not None:
                base = facecolor
            else:
                base = cmap(poly_idx % cmap.N)
            fc = (*base[:3], alpha)
            ec = base[:3] if edgecolor is None else edgecolor

            raw_px = pos[start:start + 2*n][::2].copy()
            raw_py = pos[start:start + 2*n][1::2].copy()
            px_c = np.concatenate((raw_px, [raw_px[0]]))
            py_c = np.concatenate((raw_py, [raw_py[0]]))
            px_c, py_c = fixPXPY(px_c, py_c)
            px = px_c[:-1]; py = py_c[:-1]

            geoms = []
            for i in range(n):
                ip = (i - 1) % n; inn = (i + 1) % n
                ux = px[i] - px[ip];  uy = py[i] - py[ip]
                wx = px[inn] - px[i]; wy = py[inn] - py[i]
                geoms.append(biarc_geom(px[i], py[i], ux, uy, wx, wy, dd))

            n_pts_per_arc = max(3, n_arc_pts // 2)
            for tile_i in range(3):
                for tile_j in range(3):
                    ox = tile_i - 1; oy = tile_j - 1
                    bdy_x, bdy_y = [], []
                    for i in range(n):
                        inn = (i + 1) % n
                        g_i = geoms[i]; g_n = geoms[inn]
                        if g_i is not None:
                            amx, amy, apx, apy, jx, jy, z1x, z1y, z2x, z2y, \
                                d1, d2, phi0_1, psi1, phi0_2, psi2 = g_i
                            for t in np.linspace(0, psi1, n_pts_per_arc):
                                bdy_x.append(z1x + ox + d1 * np.cos(phi0_1 + t))
                                bdy_y.append(z1y + oy + d1 * np.sin(phi0_1 + t))
                            for t in np.linspace(0, psi2, n_pts_per_arc):
                                bdy_x.append(z2x + ox + d2 * np.cos(phi0_2 + t))
                                bdy_y.append(z2y + oy + d2 * np.sin(phi0_2 + t))
                        else:
                            bdy_x.append(px[i] + ox); bdy_y.append(py[i] + oy)
                        if g_n is not None:
                            bdy_x.append(g_n[0] + ox); bdy_y.append(g_n[1] + oy)
                        else:
                            bdy_x.append(px[inn] + ox); bdy_y.append(py[inn] + oy)
                    if len(bdy_x) < 3:
                        continue
                    bx = np.array(bdy_x) * axisSize
                    by = np.array(bdy_y) * axisSize
                    if np.any((bx >= 0) & (bx <= axisSize) & (by >= 0) & (by <= axisSize)):
                        ax.add_patch(plt.Polygon(np.column_stack([bx, by]),
                                                 closed=True, facecolor=fc,
                                                 edgecolor=ec, linewidth=linewidth))
            start += 2*n
