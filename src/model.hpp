#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <tuple>
#include <cufft.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <complex>
#include <math.h>
#include <curand_kernel.h>
#include "enumTypes.h"
#include "cuda_check.h"

using namespace std;

class Model {
public:
    Model(int size_);
    ~Model();

    // initializers

    void initializeNeighborCells();
    void initializeNeighborBall();
    void initializeRandomSeed(const unsigned long long seed_);

    // deallocators

    void deallocateAll();

    // helpers

    void sortKeys(int endBit);

    // setters

    void setNumVertices(int size);
    void setPositions(const vector<double>& positions_);
    void setForces(const vector<double>& forces_);
    void setStartIndices(const vector<int>& startIndices_);
    void setModelEnum(simControlStruct::modelEnum modelType_);
    void setMaxEdgeLength(double maxEdgeLength);
    void setSearchFactor(double searchFactor_);
    double getSearchFactor() const;
    void setNeighborType(simControlStruct::neighborTypeEnum neighborType_);
    string getNeighborType() const;
    void setTargetEdgeLengths(const vector<double>& targetEdgeLengths_);
    void setTargetAreas(const vector<double>& targetAreas_);
    void setStiffness(const double stiffness_);
    void setCompressibility(const double compressibility_);
    double getStiffness() const;
    double getCompressibility() const;
    void setDelta(double delta_);
    double getDelta() const;

    // Per-polygon uniform-delta mode (area-preserving rounded). One radius per
    // polygon, recomputed each step from the current corner angles so that
    // the rounded polygon area stays at its stored target. See the kernel
    // comments in kernels.cuh for the math. Enable freezes the current
    // per-polygon area-loss as the target.
    void enablePerPolygonDelta();
    void disablePerPolygonDelta();
    bool isPerPolygonDeltaEnabled() const;
    void updatePolyDelta();
    vector<double> getPolyDelta() const;
    vector<double> getPolyDeltaTargets() const;
    // Directly set polyDelta_d (size numPolygons). Useful for tests that want
    // bounded varying-across-polygons δ without going through the closed-form
    // (which can blow up near degenerate corner geometry). Sets the
    // usePerPolygonDelta flag.
    void setPolyDelta(const vector<double>& deltas);

    // getters

    int getNumVertices() const;
    int getNumPolygons() const;
    vector<int> getShapeId() const;
    vector<double> getPositions() const;
    vector<int> getStartIndices() const;
    vector<int> getNeighborCells() const;
    vector<int> getBallNeighbors() const;
    vector<int> getNumBallNeighbors() const;
    int getBallMaxNeighbors() const;
    long long getBallRebuildCount() const;

    // Phase 2 (test-only for now): runs Phase A on the current ball list
    // and returns the number of feature-pair candidates emitted.
    // featureSetSize: 1 (edge-only), 2 (single-arc rounded), 3 (biarc).
    int runFeaturePairPhaseA(int featureSetSize);
    int getFeaturePairCapacity() const;

    // Phase B (test-only): per-pair crossing detection. Must call
    // runFeaturePairPhaseA first. Returns the number of crossings emitted.
    // delta is the rounding radius (pass 0 for normal edge-edge semantics).
    int runFeaturePairPhaseB(double delta);
    int getCrossingCapacity() const;

    // Phase C scaffolding (installment 1): sort the Phase B output in place
    // by (fA, paramA) and run-length-encode the sorted fA stream. Must be
    // called after runFeaturePairPhaseB. Returns the number of distinct
    // A-features that have at least one crossing. The per-feature walk kernel
    // (Phase C, installment 2) consumes the resulting slices.
    int runFeaturePairPhaseC_sort();
    int getLastNumUniqueFA() const;
    // Host-side getters for inspection / testing.
    vector<unsigned int> getSortedCrossingFA() const;
    vector<float>        getSortedCrossingParamA() const;
    vector<unsigned int> getSortedCrossingFB() const;
    vector<double>       getSortedCrossingX() const;       // 2*lastCrossingCount doubles
    vector<unsigned int> getFeatureUniqueFA() const;
    vector<unsigned int> getFeatureLengths() const;
    // Phase C installment 2: per-A-feature inside chord-area sum. Must call
    // runFeaturePairPhaseC_sort first. Returns numUniqueFA doubles (same
    // ordering as getFeatureUniqueFA()).
    vector<double>       runFeaturePairPhaseC_walkArea();
    // Cross-feature-aware walk: sorts crossings per A-polygon in boundary
    // order and walks each polygon's full closed boundary tracking parity
    // per partner shape globally. Fixes the cross-feature bug in walkArea.
    // Returns per-polygon chord-area sums; total overlap = sum / 2.
    vector<double>       runFeaturePairPhaseC_walkAreaPerPolygon();
    // Phase C installment 3: walk-area + edge-edge force accumulation. Force
    // is written into an internal device buffer (zeroed at launch). Returns
    // the area vector (same as walkArea); call getPhaseCForce() for forces.
    vector<double>       runFeaturePairPhaseC_walkAreaAndForce();
    // Phase C installment 4: dual-traced edge-edge force (delta>0 supported).
    // Result and force buffer share the same layout as runFeaturePairPhaseC_walkAreaAndForce.
    vector<double>       runFeaturePairPhaseC_walkAreaAndForceDual();
    vector<double>       getPhaseCForce() const;        // 2*size doubles
    vector<int> getBoxCounts() const;
    vector<int> getNeighborIndices() const;
    vector<int> getNumNeighbors() const;
    vector<int> getNeighbors() const;
    double getMaxEdgeLength() const;
    string getModelEnum() const;
    vector<double> getForces() const;
    void resetMaxActualNeighbors();
    vector<bool> getInsideFlag() const;
    vector<double> getPerimeters() const;
    vector<int> getIntersectionsCounter() const;
    vector<double> getTU() const;
    vector<double> getUT() const;
    vector<int> getShapeCounts() const;
    vector<uint64_t> getIntersections() const;
    vector<uint32_t> getKeys() const;
    int getNumIntersections() const;
    vector<uint64_t> getOutersections() const;
    unsigned long long getRandomSeed();
    vector<double> getAreas() const;
    double getEnergy() const;
    vector<double> getPairArea() const;
    vector<double> getTargetEdgeLengths() const;
    vector<double> getTargetAreas() const;
    vector<double> getCOM() const;
    double getMaxUnbalancedForce() const;
    vector<double> getEdgeLengths() const;
    double getOverlapArea() const;
    vector<double> getConstraints() const;

    // updaters

    void updatePolygonGeometry();
    void projectForce();
    int  shakeProject(int nIter, double tol = 1e-15);
    int  getLastShakeIters() const;
    void saveTentativePositions();
    double getMaxEffectiveForce(double dt, minimizerEnum minimizerType) const;
    void updateNeighborCells();
    void updateNeighborBall();
    void updateNeighbors();
    void updateOverlapArea(int pointDensity_);
    void updateIntersectionsCounter();
    void updateValidAndCounts();
    void updateOutersections();
    void updateCompactedIntersections();
    void updateForceEnergy();
    void updatePositions(double dt);
    void resetVelocities();
    std::tuple<double, double, double, int> minimizeFIREStep(double dt, double alpha, int nPos, double dtMax = 0.1, double alphaStart = 0.1, double fAlpha = 0.99, double fInc = 1.1, double fDec = 0.5, int nMin = 5, int shakeIter = 5, double rollbackRelTol = 1e-10, double rollbackAbsTol = 1e-14);
    std::tuple<double, double, int> minimizeFIRE(double maxForceThreshold, double dtInit, int maxSteps, double dtMax = 0.1, double alphaStart = 0.1, double fAlpha = 0.99, double fInc = 1.1, double fDec = 0.5, int nMin = 5, int shakeIter = 5, double rollbackRelTol = 1e-10, double rollbackAbsTol = 1e-14);

    // misc:
    void resetAreas();

private:
    simControlStruct simControl;
    int size, numPolygons;
    unsigned long long seed;
    curandState* globalState;
    double* energy;
    double* positions;
    int* startIndices;
    double* areas;
    double* targetEdgeLengths;
    int* countPerBox;
    int* boxId;
    int* neighborIndices;
    int* cellLocation;
    int* shapeId;
    int* neighbors;
    int boxesUsed;
    int maxNeighbors = 100;
    int boxSize;
    int* numNeighbors;
    bool updateMaxNeighbors = false;
    double* maxEdgeLength;
    // Verlet-style ball-neighbor list (neighborType == balls)
    int*    ballNeighbors        = nullptr;   // size * ballMaxNeighbors, per-vertex flat list
    int*    numBallNeighbors     = nullptr;   // size, true count per vertex
    int*    maxActualBallNeighbors = nullptr; // single int, device-side reduce target
    double* ballRefPositions     = nullptr;   // 2*size, positions at last rebuild
    double* ballDispScratch      = nullptr;   // size, per-vertex displacement scratch
    int     ballMaxNeighbors     = 100;
    double  searchFactor         = 2.5;       // ball_radius = searchFactor * maxEdgeLength
    double  cachedBallRadius     = 0.0;       // radius used at last rebuild
    bool    ballListValid        = false;     // false until first build
    long long ballRebuildCount   = 0;         // incremented each time the list is rebuilt

    // Feature-pair refactor scaffolding (Phase A output, Phase B/C input).
    // Not wired into updateForceEnergy yet; reachable through the test-only
    // method runFeaturePairPhaseA(). Treated as opaque bytes by host code:
    // the device-side type is feat::FeaturePair (two uint32). We allocate
    // 2 * featurePairCapacity ints so we never need to include features.cuh
    // from model.cpp.
    unsigned int* featurePairBuf      = nullptr;   // 2 * featurePairCapacity ints
    int*          featurePairCount_d  = nullptr;   // single int, device-side atomic counter
    int           featurePairCapacity = 0;
    // Phase B output: opaque byte buffer of feat::Crossing structs. Size of
    // one Crossing is fetched via crossingByteSize() from the CUDA TU.
    void*         crossingBuf         = nullptr;
    int*          crossingCount_d     = nullptr;
    int           crossingCapacity    = 0;
    int           lastFeaturePairCount = 0;        // remember Phase A's count for Phase B
    int           lastCrossingCount    = 0;        // remember Phase B's count for Phase C
    // Phase C scaffolding (installment 1): sort + run-length-encode.
    // - crossingTMP            : same byte-size as crossingBuf (scratch for gather)
    // - crossingKeys           : numCrossings uint64 (composite (fA, paramA) key)
    // - crossingIdx            : numCrossings uint32 (perm / fA scratch)
    // - featureUniqueFA_d      : numCrossings uint32 (RLE unique fA stream)
    // - featureLengths_d       : numCrossings uint32 (RLE run lengths)
    // - featureNumUnique_d     : single int (RLE output count)
    void*         crossingTMP            = nullptr;
    uint64_t*     crossingKeys           = nullptr;
    uint32_t*     crossingIdx            = nullptr;
    uint32_t*     featureUniqueFA_d      = nullptr;
    uint32_t*     featureLengths_d       = nullptr;
    int*          featureNumUnique_d     = nullptr;
    int           phaseC_scratchCapacity = 0;
    int           lastNumUniqueFA        = 0;
    // Phase C installment 2: per-feature walk scratch and output.
    uint32_t*     featureSliceStart_d    = nullptr;   // numUniqueFA uint32
    double*       featureAreaOut_d       = nullptr;   // numUniqueFA double
    // Phase C installment 3: per-vertex force accumulator (2 * size doubles).
    // Lives separate from `force` so the test path doesn't clobber whatever
    // updateForceEnergy stages there.
    double*       phaseCForce_d          = nullptr;
    double* force;
    int* maxActualNeighbors;
    bool* inside;
    double* perimeters;
    int pointDensity = -1;
    double overlapArea = 0.0;
    int* intersectionsCounter;
    int* valid;
    uint64_t* outputIdx;
    int* shapeCounts;
    uint64_t* intersections;
    double2* tu, *tuTMP, *ut, *utTMP;
    int numIntersections = 0;
    uint64_t* outersections, *outersectionsTMP;
    uint32_t* keys;
    int* next, *prev;
    double stiffness = 0.0;
    double compressibility = 0.0;
    double delta = 0.0;
    // Per-polygon uniform-delta mode (area-preserving rounded). One double per
    // polygon. Targets are frozen at enable-time from the current corner
    // geometry and the scalar `delta`; polyDelta_d is recomputed each step.
    double* polyDelta_d         = nullptr;   // numPolygons doubles
    double* polyDeltaTargets_d  = nullptr;   // numPolygons doubles
    bool    usePerPolygonDelta  = false;
    int* shapeStart;
    int* shapeEnd;
    double* edgeLengths;
    double* comX;
    double* comY;
    double* comParts;
    double* areaParts;
    double* targetAreas;
    double* constraints;
    double* constraintNormSq;
    double* mgsIp;
    double* forceProjIp;
    cusolverDnHandle_t cusolverHandle = nullptr;
    int polygonSize = 0;
    double* roundedScratch = nullptr; // per-vertex scratch for rounded kernel
    int     roundedStride  = 0;       // doubles per vertex in roundedScratch
    double* pairArea       = nullptr; // numPolygons² scratch for areaSquared mode
    double* edgeGradTMP = nullptr;
    double* uMat = nullptr;
    double* singularValuesTMP = nullptr;
    double* vMatTMP = nullptr;
    int* solverInfoTMP = nullptr;
    double* qAreaVec = nullptr;
    double* cusolverWorkspace = nullptr;
    int cusolverWorkspaceSize = 0;
    double* hRnrmF = nullptr;
    double* xpbdArea;
    double* xpbdGradNormSq;
    double* positionsTMP;
    double* positionsTMP2;
    double* effForceMagTMP;
    double* velocities;
    double* fireScratchTMP;
    double* fireResultTMP;
    int*    shakeItersTMP;
    int     lastShakeIters;
};

#endif
