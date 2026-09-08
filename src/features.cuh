// features.cuh — Feature-pair refactor (Phase 0)
//
// A "feature" is one boundary segment of a polygon. Three types are
// supported across the normal / single-arc-rounded / biarc models:
//
//   EDGE  : the straight segment from one tangent point to the next.
//           For the normal model, this is just the segment from
//           vertex i to vertex (i+1).
//   ARC1  : the first arc at vertex i (rounded/biarc model).
//           For the single-arc rounded model, this IS the (only) arc.
//   ARC2  : the second arc at vertex i (biarc model only).
//
// Each feature is owned by a single vertex (its anchor vertex). The
// global vertex index is g_i = startIndices[s] + i_local. A feature's
// 32-bit ID packs (g_i, type) with the type in the low two bits:
//
//   featureId = (g_i << 2) | type
//
// This is equivalent to featureId = 4*g_i + type when type < 4.

#ifndef FEATURES_CUH
#define FEATURES_CUH

namespace feat {

enum FeatureType : unsigned char {
    EDGE = 0,
    ARC1 = 1,
    ARC2 = 2,
    // 3 reserved (we have 2 bits free in the low bits of a uint32_t ID)
};

constexpr unsigned int TYPE_MASK    = 0x3u;
constexpr unsigned int VERTEX_SHIFT = 2;

__host__ __device__ inline unsigned int encode(int globalVertex, FeatureType t) {
    return (static_cast<unsigned int>(globalVertex) << VERTEX_SHIFT)
         | static_cast<unsigned int>(t);
}

__host__ __device__ inline int vertexOf(unsigned int fid) {
    return static_cast<int>(fid >> VERTEX_SHIFT);
}

__host__ __device__ inline FeatureType typeOf(unsigned int fid) {
    return static_cast<FeatureType>(fid & TYPE_MASK);
}

// -----------------------------------------------------------------------------
// Feature-pair data structures (Phase A output, Phase B input).
// -----------------------------------------------------------------------------

// One pair of features (one from polygon A, one from polygon B) that Phase B
// will probe for a geometric crossing. Stored as two 32-bit encoded feature
// IDs; the type bits live in the low 2 bits of each (see encode/decode above).
struct FeaturePair {
    unsigned int fA;
    unsigned int fB;
};

// One crossing record emitted by Phase B and consumed by Phase C. The fields
// after fA, paramA are auxiliary data Phase C uses to compute internal terms.
// paramA is a float to keep the 64-bit packed sort key (fA<<32 | floatBits(paramA))
// preserving (fA, paramA) lex order under unsigned radix sort.
struct Crossing {
    unsigned int fA;
    unsigned int fB;
    float        paramA;     // parameter along A-feature, in [0, 1]; used as sort key
    float        paramB;     // parameter along B-feature
    double       Xx, Xy;     // crossing point (caller's local frame)
};

} // namespace feat

#endif // FEATURES_CUH
