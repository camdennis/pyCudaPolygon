// geom.cuh — periodic-boundary helpers shared across the codebase.
//
// All polygon kernels operate in a unit periodic box; relative
// coordinates need to be folded into [-0.5, 0.5) so MIC distances
// behave correctly. These helpers are isolated here so the
// intersection / integration / parity-walk headers can use them
// without pulling in the full kernels.cuh.

#ifndef GEOM_CUH
#define GEOM_CUH

#include <cuda_runtime.h>
#include <math.h>

__device__ inline double wrapPeriodic(double x) {
    // Wrap x into [-0.5, 0.5)
    x += 1.5;
    while (x > 1.0) x -= 1.0;
    return x - 0.5;
}

__device__ inline double wrap(double x) {
    double y = x + 1.5;
    y = y - floor(y);   // fractional part in [0,1)
    return y - 0.5;
}

__device__ inline double2 wrap2(double2 v) {
    v.x = wrap(v.x);
    v.y = wrap(v.y);
    return v;
}

#endif // GEOM_CUH
