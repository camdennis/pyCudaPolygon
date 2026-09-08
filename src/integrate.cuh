// integrate.cuh — per-segment Green's-theorem contributions.
//
// Phase C ("Assemble") of the feature-pair refactor: given the sorted
// crossings on an A-feature, walk consecutive crossings and contribute
// the Green's-theorem integral piece between them to the running
// energy and forces.
//
// The chord helpers (chordArea / chordGradients) are the shared
// primitive across all three models. The arc helpers add the
// circular-segment correction term on top of the chord shoelace
// (see notes.pdf section 4.4) and land in Phase 2/3.

#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH

#include <cuda_runtime.h>
#include "geom.cuh"

namespace integrate {

// Chord shoelace contribution to the Green's-theorem area integral
// for a straight segment from pt1 to pt2, with respect to a per-pair
// anchor. Operates on periodically-wrapped relative coordinates so
// the result telescopes correctly when summed around a closed loop
// that may straddle the periodic box.
//
// Mathematical form: returns r1 x r2 where r_i = wrap(pt_i - anchor).
// This is twice the signed area of triangle (anchor, pt1, pt2); when
// these triangles are summed around a closed loop their total equals
// the loop's enclosed area (the anchor contributions telescope to
// zero).
//
// Was: device function h() in kernels.cuh.
__device__ inline double chordArea(double2 pt1, double2 pt2, double2 anchor) {
    double2 r1 = {pt1.x - anchor.x, pt1.y - anchor.y};
    double2 r2 = {pt2.x - anchor.x, pt2.y - anchor.y};
    r1 = wrap2(r1);
    r2 = wrap2(r2);
    double2 d = wrap2({r2.x - r1.x, r2.y - r1.y});
    return 0.5*((r1.x + r2.x) * d.y - (r1.y + r2.y) * d.x);
}

// Gradients of chordArea with respect to its first two arguments.
// d(chordArea)/d(pt1) = (r2.y, -r2.x), d/d(pt2) = (-r1.y, r1.x).
// Was: device function g12() in kernels.cuh.
__device__ inline void chordGradients(double2 pt1, double2 pt2, double2 anchor,
                                       double2& g1, double2& g2) {
    double2 r1 = {pt1.x - anchor.x, pt1.y - anchor.y};
    double2 r2 = {pt2.x - anchor.x, pt2.y - anchor.y};
    r1 = wrap2(r1);
    r2 = wrap2(r2);
    g1 = { r2.y, -r2.x};
    g2 = {-r1.y,  r1.x};
}

// Arc-segment contribution: chordArea plus the circular-segment correction.
//
// (notes section 4.4) For an arc sub-interval with endpoints X0, X1 in the
// caller's MIC frame and arc span dpsi (= span in radians of the
// sub-interval, signed for orientation), the signed area is
//
//   delta A_arc = (1/2)(X0 x X1)_anchored + (1/2) * delta^2 * (dpsi - sin dpsi)
//
// The chord shoelace piece is exactly chordArea (anchored + periodic-wrapped)
// so that contributions telescope correctly across the closed boundary even
// when polygons straddle the periodic box. The segment correction is purely
// local to the arc -- no anchor or wrapping needed.
__device__ inline double arcArea(double2 pt1, double2 pt2, double2 anchor,
                                  double delta, double dpsi) {
    double seg = 0.5 * delta * delta * (dpsi - sin(dpsi));
    return chordArea(pt1, pt2, anchor) + seg;
}

} // namespace integrate

#endif // INTEGRATE_CUH
