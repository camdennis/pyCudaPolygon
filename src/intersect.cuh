// intersect.cuh — pairwise intersection routines (Phase 0+).
//
// Each routine returns its result by value as a small POD. The current
// scaffolding covers edge-edge (extracted verbatim from
// updateNeighborsKernel in Phase 0). Arc-edge / arc-arc variants land
// in Phase 2/3.
//
// All routines assume their inputs are already minimum-image-wrapped
// in the relative frame the caller is using. The routines themselves
// perform no MIC wrapping.

#ifndef INTERSECT_CUH
#define INTERSECT_CUH

namespace isect {

// Edge–edge intersection.
//
// Edge 1 goes from origin to r = (rx, ry) in the caller's local frame.
// Edge 2 goes from g = (gx, gy) to g + s, where s = (sx, sy). Both
// directions are assumed to be the MIC-wrapped edge vectors; g is the
// MIC-wrapped offset from edge-1's anchor to edge-2's anchor.
//
// `delta` is an endpoint-exclusion margin in the same units as the
// edge vectors: a crossing is reported only if it falls inside the
// interior of both edges by at least delta along each side. For the
// normal model pass delta = 0; for the rounded model pass the polygon
// padding radius.
//
// Output:
//   crossed       : true iff the intersection is inside both edges'
//                   delta-shrunk interior and the edges are not
//                   degenerate / parallel.
//   tt            : parameter along edge 1 at the crossing, in (0, 1).
//   uu            : parameter along edge 2 at the crossing, in (0, 1).
//   positiveDenom : sign of (rx*sy - ry*sx). True when edge 2 sweeps
//                   counter-clockwise from edge 1. Used downstream to
//                   tag which side of edge 1 the crossing came from.

struct EdgeEdgeResult {
    bool   crossed;
    bool   positiveDenom;
    double tt;
    double uu;
};

__device__ inline EdgeEdgeResult edgeEdgeCross(
    double rx, double ry,
    double sx, double sy,
    double gx, double gy,
    double delta
) {
    EdgeEdgeResult out;
    out.crossed = false;
    out.positiveDenom = false;
    out.tt = 0.0;
    out.uu = 0.0;

    constexpr double eps = 1e-12;
    double denom = rx * sy - ry * sx;
    if (fabs(denom) < eps) return out;     // parallel or degenerate

    double tt = (gx * sy - gy * sx) / denom;
    double uu = (gx * ry - gy * rx) / denom;
    double Li = sqrt(rx * rx + ry * ry);
    double Lj = sqrt(sx * sx + sy * sy);
    double tt_lo = (delta > 0.0 && Li > 0.0) ? delta / Li : 0.0;
    double uu_lo = (delta > 0.0 && Lj > 0.0) ? delta / Lj : 0.0;
    if (tt > tt_lo && tt < 1.0 - tt_lo && uu > uu_lo && uu < 1.0 - uu_lo) {
        out.crossed       = true;
        out.positiveDenom = (denom > 0.0);
        out.tt            = tt;
        out.uu            = uu;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Arc helpers (Phase 2).
//
// All caller coordinates must be in a single consistent MIC frame: the arc
// center, edge endpoints, and resulting crossings live in the same image of
// the periodic box. The routines themselves perform no further wrapping --
// they're pure plane geometry on the values they receive.
//
// Arc parameterisation matches notes section 3.2/3.4: centre z, radius
// delta, angles in [phi0, phi0 + dphi] with dphi > 0 (CCW sweep). The
// "tauArc" output parameter normalises this to [0, 1]: tau = 0 at the
// arc's starting tangent point a^-, tau = 1 at the ending tangent point a^+.

struct ArcCrossingHit {
    double tauArc;   // arc parameter in [0, 1]
    double tEdge;    // edge parameter in (0, 1) for edge-arc; unused for arc-arc
    double Xx, Xy;   // crossing point in the caller's frame
};

struct ArcCrossingPair {
    int count;                  // 0, 1, or 2
    ArcCrossingHit hits[2];
};

// Bring x into [0, 2*pi) using a single fmod-like step. Assumes x is
// already roughly in (-2*pi, 2*pi); two correction branches suffice.
__device__ inline double wrap2pi(double x) {
    constexpr double TWO_PI = 6.283185307179586476925286766559;
    if (x < 0.0)      x += TWO_PI;
    if (x >= TWO_PI)  x -= TWO_PI;
    return x;
}

// Edge-arc crossing. Edge runs from (px,py) to (qx,qy); arc is the circle
// of radius delta centred at (zx,zy) restricted to the angular interval
// [phi0, phi0 + dphi]. Up to 2 crossings emitted; each requires the edge
// parameter t in (0, 1) and the arc parameter tau in [0, 1].
//
// Math: substitute r(t) = p + t*e into |r - z|^2 = delta^2 to get a
// quadratic in t (notes section 4.6.1).
__device__ inline ArcCrossingPair edgeArcCross(
    double px, double py, double qx, double qy,
    double zx, double zy, double delta,
    double phi0, double dphi)
{
    ArcCrossingPair out; out.count = 0;
    double ex = qx - px, ey = qy - py;
    double a  = ex*ex + ey*ey;
    if (a < 1e-28) return out;        // degenerate edge

    double fx = px - zx, fy = py - zy;
    double b = 2.0 * (fx*ex + fy*ey);
    double c = fx*fx + fy*fy - delta*delta;
    double disc = b*b - 4.0*a*c;
    if (disc < 0.0) return out;
    double sd = sqrt(disc);
    double t_arr[2] = { (-b - sd) / (2.0*a), (-b + sd) / (2.0*a) };

    for (int k = 0; k < 2; ++k) {
        double t = t_arr[k];
        if (t <= 0.0 || t >= 1.0) continue;
        double Xx = px + t*ex;
        double Xy = py + t*ey;
        double rel = wrap2pi(atan2(Xy - zy, Xx - zx) - phi0);
        if (rel > dphi) continue;
        out.hits[out.count].tauArc = (dphi > 0.0) ? rel / dphi : 0.0;
        out.hits[out.count].tEdge  = t;
        out.hits[out.count].Xx     = Xx;
        out.hits[out.count].Xy     = Xy;
        ++out.count;
    }
    return out;
}

// Arc-arc crossing. Two arcs of radii deltaA, deltaB centred at zA, zB,
// restricted to their respective angular intervals.
//
// Math: two-circle intersection via the radical-axis construction
// (notes section 4.6.2). Up to 2 crossings; each candidate must lie in
// both arcs' angular ranges.
__device__ inline ArcCrossingPair arcArcCross(
    double zAx, double zAy, double deltaA, double phi0A, double dphiA,
    double zBx, double zBy, double deltaB, double phi0B, double dphiB)
{
    ArcCrossingPair out; out.count = 0;
    double dx = zBx - zAx, dy = zBy - zAy;
    double d2 = dx*dx + dy*dy;
    if (d2 < 1e-28) return out;       // concentric or coincident centres
    double d  = sqrt(d2);

    double a2 = (d2 + deltaA*deltaA - deltaB*deltaB) / (2.0*d);
    double h2 = deltaA*deltaA - a2*a2;
    if (h2 < 0.0) return out;         // circles don't meet
    double h  = sqrt(h2);

    double sx  = dx / d,  sy  = dy / d;
    double spx = -sy,     spy = sx;
    double mx  = zAx + a2*sx;
    double my  = zAy + a2*sy;
    double Xc[2][2] = { { mx + h*spx, my + h*spy },
                        { mx - h*spx, my - h*spy } };

    for (int k = 0; k < 2; ++k) {
        double Xx = Xc[k][0], Xy = Xc[k][1];

        double relA = wrap2pi(atan2(Xy - zAy, Xx - zAx) - phi0A);
        if (relA > dphiA) continue;
        double relB = wrap2pi(atan2(Xy - zBy, Xx - zBx) - phi0B);
        if (relB > dphiB) continue;

        out.hits[out.count].tauArc = (dphiA > 0.0) ? relA / dphiA : 0.0;
        out.hits[out.count].tEdge  = (dphiB > 0.0) ? relB / dphiB : 0.0;
        out.hits[out.count].Xx     = Xx;
        out.hits[out.count].Xy     = Xy;
        ++out.count;

        // For arc-arc the second slot also stores B's parameter via tEdge.
        // Callers that need both should read tauArc (A's) and tEdge (B's).
    }
    return out;
}

} // namespace isect

#endif // INTERSECT_CUH
