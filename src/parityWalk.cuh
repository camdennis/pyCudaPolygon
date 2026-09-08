// parityWalk.cuh — per-feature walk over sorted crossings (Phase 0 stub).
//
// Phase C ("Assemble") of the feature-pair refactor.
//
// After Phase B emits a flat list of crossing records and the host
// sorts them by (fA, paramA), one thread per A-feature walks its
// contiguous slice. Two jobs:
//
//   1. Recover the inside/outside flag at each sub-segment by walking
//      crossings in parameter order, toggling parity per shape.
//   2. Call into integrate::edgeChord / integrate::arcChord for each
//      sub-segment that falls inside any partner polygon, accumulating
//      energy and forces.
//
// This header is a stub. The walk itself is straightforward but the
// data layout (crossing record fields, per-shape parity bookkeeping)
// is decided in Phase 1 once we know exactly what the normal model
// needs to consume. Re-using the existing
// updateForceEnergyExteriorKernel / updateForceEnergyInteriorKernel
// logic as the reference implementation.

#ifndef PARITY_WALK_CUH
#define PARITY_WALK_CUH

namespace parity {

// (Phase 1) Walk an A-feature's sorted crossings.
//
// fA              : the A-feature ID this thread owns
// crossings_start : pointer to its first crossing in the sorted array
// crossings_end   : pointer just past its last crossing
// (plus positions, shapeId, force, energy buffers)
//
// __device__ inline void walkOneFeature(
//     unsigned int fA,
//     const Crossing* crossings_start,
//     const Crossing* crossings_end,
//     const double* positions,
//     const int*    shapeId,
//     double*       force,
//     double*       energy);

} // namespace parity

#endif // PARITY_WALK_CUH
