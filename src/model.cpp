#include <iostream>
#include <vector>
#include "model.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <complex>
#include <math.h>
#include <algorithm>
#include <curand_kernel.h>
#include "enumTypes.h"

using namespace std;

// Declaration of CUDA helper functions (defined in model.cu)
// initializers:
extern "C" void initializeRandomStates(curandState *globalState, unsigned long long seed, int gridSize);
// helpers
extern "C" void sortKeysCUDA(uint64_t* d_keys, int numItems,int beginBit, int endBit, uint32_t* d_perm_out);
extern "C" void applyPermutationCUDA_int64(const uint64_t* d_input, const uint32_t* d_perm, uint64_t* d_output, int numItems);
extern "C" void applyPermutationCUDA_double2(const double2* d_input, const uint32_t* d_perm, double2* d_output, int numItems);
extern "C" void computeNextPrevCUDA(int* next, int* prev, int* startIndices, int* shapeId, int size);
// updaters:
extern "C" void updatePolygonGeometryCUDA(int numVertices, int numPolygons, double* positions, int* startIndices, int* shapeId, int* next, int* prev, double* edgeLengths, double* areaParts, double* comParts, double* area, double* comX, double* comY, double* maxEdgeLength, double* constraints, double* constraintNormSq);
extern "C" void projectForceCUDA(int numVertices, int numPolygons, int n,
    const int* shapeId, const int* startIndices, const int* next,
    const double* positions, const double* constraints,
    double* edgeGradTMP, double* uMat, double* singularValuesTMP, double* vMatTMP,
    int* solverInfoTMP, double* qAreaVec, cusolverDnHandle_t handle,
    double* workspace, int workspaceSize, double* hRnrmF, double* force);
extern "C" int xpbdProjectCUDA(int numVertices, int numPolygons, int nIter, int* startIndices, int* shapeId, int* next, int* prev, double* positions, const double* targetEdgeLengths, const double* targetAreas, double* d_area, double* d_gradNormSq, double tol, double* convTMP);
extern "C" int  shakeProjectCUDA(int numPolygons, int n, const int* startIndices, const int* next, const int* prev, double* positions, const double* targetEdgeLengths, const double* targetAreas, int maxIter, double tol, int* maxIterOut);
extern "C" void saveTentativePositionsCUDA(int numVertices, const double* positions, double* positionsTMP);
extern "C" double getMaxEffectiveForceCUDA(int numVertices, const double* positions, const double* positionsTMP, const double* force, double scale, double* effForceMagTMP);
extern "C" void updateNeighborCellsCUDA(double* positions, int* startIndices, int* shapeId, int numPolygons, int size, int boxSize, int* cellLocation, int* countPerBox, int* boxId, int& boxesUsed, int* neighborIndices);
extern "C" void updateShapeIdCUDA(int* shapeId, int* startIndices, int size, int numPolygons);
extern "C" int updateNeighborsCUDA(int* shapeId, int* startIndices, double* positions, int* cellLocation, int* neighborIndices, int size, int* neighbors, int* numNeighbors, int maxNeighbors, int boxSize, int* countPerBox, int* maxActualNeighbors, double2* tu, bool*);
extern "C" void updateOverlapAreaCUDA(int* shapeId, int* startIndices, int pointDensity, int* intersectionsCounter, int* neighborIndices, int size, int boxSize, int* countPerBox, double* positions, double& overlapArea);
extern "C" int updateValidAndCountsCUDA(int numVertices, int* neighbors, int* numNeighbors, int maxNeighbors, bool* insideFlag, int* shapeIds, int numShapes, int* valid, int* shapeCounts, uint64_t* outputIdx);
extern "C" void updateCompactedIntersectionsCUDA(int numVertices, int maxNeighbors, int* neighbors, bool* insideFlag, int* shapeIds, int* startIndices, int* valid, uint64_t* outputIdx, uint64_t* intersections, int numIntersections, double2* tu, double2* tuTMP);
extern "C" void updateOutersectionsCUDA(const uint64_t* intersections, const double2* tu, const double2* ut, const int* startIndices, int numIntersections, uint64_t* outersections);
extern "C" void updateForceEnergyExteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const double2* tu, const double2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy);
extern "C" void updateForceEnergyInteriorCUDA(int numVertices, int numIntersections, const uint64_t* intersections, const uint64_t* outersections, const double2* tu, const double2* ut, const double* positions, const int* next, const int* prev, const int* shapeId, const int* startIndices, double* force, double* energy, int numPolygons, int* shapeStart, int* shapeEnd);
extern "C" void updatePositionsCUDA(int numVertices, double* positions, const double* force, double dt);
extern "C" void updateForceEnergyEdgeCUDA(int numVertices, const double* positions, const double* targetEdgeLengths, const double* edgeLengths, const int* next, const int* prev, const int* shapeId, double* force, double* energy, double stiffness);
extern "C" void updateForceEnergyAreaCUDA(int numVertices, const int* shapeId, const int* next, const int* prev, const double* positions, const double* areas, const double* targetAreas, const int* startIndices, double* force, double* energy, double compressibility);
extern "C" void updateShapeRangesCUDA(int numPolygons, int numVertices, int numIntersections, const uint64_t* intersections, int* shapeStart, int* shapeEnd);
// getters
extern "C" double getMaxUnbalancedForceCUDA(int numVertices, double* force);
// FIRE
extern "C" void   updatePositionAndVelocityFIRECUDA(int numVertices, double* positions, double* velocities, const double* force, double dt);
extern "C" void   updateVelocityFIRECUDA(int numVertices, double* velocities, const double* force, double dt);
extern "C" double bendVelocityTowardsForceFIRECUDA(int numVertices, double* velocities, const double* force, double alpha, double* scratch, double* result);
extern "C" void   rederiveVelocityFromDisplacementFIRECUDA(int numVertices, double* vel, const double* posNew, const double* posOld, double dt);
// misc
extern "C" void resetAreasCUDA(const int numVertices, const int* shapeId, double* positions, const double* areas, const double* targetAreas, const double* comX, const double* comY);

// Constructor

Model::Model(int size_)
    : size(size_),
      positions(nullptr), force(nullptr), maxActualNeighbors(nullptr), globalState(nullptr),
      countPerBox(nullptr), boxId(nullptr), neighborIndices(nullptr), cellLocation(nullptr),
      shapeId(nullptr), neighbors(nullptr), numNeighbors(nullptr),
      inside(nullptr), perimeters(nullptr), intersectionsCounter(nullptr),
      valid(nullptr), outputIdx(nullptr), shapeCounts(nullptr), intersections(nullptr),
      tu(nullptr), ut(nullptr), tuTMP(nullptr), utTMP(nullptr), outersections(nullptr),
      outersectionsTMP(nullptr), keys(nullptr), next(nullptr), prev(nullptr), startIndices(nullptr),
      areas(nullptr), targetAreas(nullptr), targetEdgeLengths(nullptr),
      shapeStart(nullptr), shapeEnd(nullptr), edgeLengths(nullptr), maxEdgeLength(nullptr), comX(nullptr), comY(nullptr), areaParts(nullptr), comParts(nullptr),
      constraintNormSq(nullptr), mgsIp(nullptr), forceProjIp(nullptr),
      xpbdArea(nullptr), xpbdGradNormSq(nullptr),
      positionsTMP(nullptr), positionsTMP2(nullptr), effForceMagTMP(nullptr),
      velocities(nullptr), fireScratchTMP(nullptr), fireResultTMP(nullptr),
      shakeItersTMP(nullptr), lastShakeIters(0)
{
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaMalloc((void**)&positions, 2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&comParts, 2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&areaParts, 2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&constraints, 6 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&edgeLengths, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&maxEdgeLength, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&force, size * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&positionsTMP,  2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&positionsTMP2, 2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&effForceMagTMP, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&velocities, 2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&fireScratchTMP, 2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&fireResultTMP, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&shakeItersTMP, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&maxActualNeighbors, sizeof(int)));
    int init = INT_MIN;
    CUDA_CHECK(cudaMemcpy(maxActualNeighbors, &init, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&globalState, sizeof(curandState) * size));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
    setModelEnum(simControl.modelType);
}

Model::~Model() {
    if (positions) CUDA_CHECK_NOABORT(cudaFree(positions));
    if (force) CUDA_CHECK_NOABORT(cudaFree(force));
    if (energy) CUDA_CHECK_NOABORT(cudaFree(energy));
    if (maxActualNeighbors) CUDA_CHECK_NOABORT(cudaFree(maxActualNeighbors));
    if (globalState) CUDA_CHECK_NOABORT(cudaFree(globalState));
    // From initializeNeighborCells
    if (countPerBox) CUDA_CHECK_NOABORT(cudaFree(countPerBox));
    if (boxId) CUDA_CHECK_NOABORT(cudaFree(boxId));
    if (cellLocation) CUDA_CHECK_NOABORT(cudaFree(cellLocation));
    if (neighborIndices) CUDA_CHECK_NOABORT(cudaFree(neighborIndices));
    if (neighbors) CUDA_CHECK_NOABORT(cudaFree(neighbors));
    if (inside) CUDA_CHECK_NOABORT(cudaFree(inside));
    if (shapeId) CUDA_CHECK_NOABORT(cudaFree(shapeId));
    if (numNeighbors) CUDA_CHECK_NOABORT(cudaFree(numNeighbors));
    if (valid) CUDA_CHECK_NOABORT(cudaFree(valid));
    if (outputIdx) CUDA_CHECK_NOABORT(cudaFree(outputIdx));
    if (shapeCounts) CUDA_CHECK_NOABORT(cudaFree(shapeCounts));
    if (intersections) CUDA_CHECK_NOABORT(cudaFree(intersections));
    if (tu) CUDA_CHECK_NOABORT(cudaFree(tu));
    if (ut) CUDA_CHECK_NOABORT(cudaFree(ut));
    if (tuTMP) CUDA_CHECK_NOABORT(cudaFree(tuTMP));
    if (utTMP) CUDA_CHECK_NOABORT(cudaFree(utTMP));
    if (keys) CUDA_CHECK_NOABORT(cudaFree(keys));
    if (outersections) CUDA_CHECK_NOABORT(cudaFree(outersections));
    if (outersectionsTMP) CUDA_CHECK_NOABORT(cudaFree(outersectionsTMP));
    if (next) CUDA_CHECK_NOABORT(cudaFree(next));
    if (prev) CUDA_CHECK_NOABORT(cudaFree(prev));
    if (edgeLengths) CUDA_CHECK_NOABORT(cudaFree(edgeLengths));
    if (maxEdgeLength) CUDA_CHECK_NOABORT(cudaFree(maxEdgeLength));
    // From setStartIndices
    if (areas) CUDA_CHECK_NOABORT(cudaFree(areas));
    if (perimeters) CUDA_CHECK_NOABORT(cudaFree(perimeters));
    if (startIndices) CUDA_CHECK_NOABORT(cudaFree(startIndices));
    if (constraintNormSq) CUDA_CHECK_NOABORT(cudaFree(constraintNormSq));
    if (mgsIp) CUDA_CHECK_NOABORT(cudaFree(mgsIp));
    if (forceProjIp) CUDA_CHECK_NOABORT(cudaFree(forceProjIp));
    if (xpbdArea) CUDA_CHECK_NOABORT(cudaFree(xpbdArea));
    if (xpbdGradNormSq) CUDA_CHECK_NOABORT(cudaFree(xpbdGradNormSq));
    if (positionsTMP)  CUDA_CHECK_NOABORT(cudaFree(positionsTMP));
    if (positionsTMP2) CUDA_CHECK_NOABORT(cudaFree(positionsTMP2));
    if (effForceMagTMP) CUDA_CHECK_NOABORT(cudaFree(effForceMagTMP));
    if (velocities) CUDA_CHECK_NOABORT(cudaFree(velocities));
    if (fireScratchTMP) CUDA_CHECK_NOABORT(cudaFree(fireScratchTMP));
    if (fireResultTMP)            CUDA_CHECK_NOABORT(cudaFree(fireResultTMP));
    if (shakeItersTMP)            CUDA_CHECK_NOABORT(cudaFree(shakeItersTMP));
    if (edgeGradTMP) CUDA_CHECK_NOABORT(cudaFree(edgeGradTMP));
    if (uMat) CUDA_CHECK_NOABORT(cudaFree(uMat));
    if (singularValuesTMP) CUDA_CHECK_NOABORT(cudaFree(singularValuesTMP));
    if (vMatTMP) CUDA_CHECK_NOABORT(cudaFree(vMatTMP));
    if (solverInfoTMP) CUDA_CHECK_NOABORT(cudaFree(solverInfoTMP));
    if (qAreaVec) CUDA_CHECK_NOABORT(cudaFree(qAreaVec));
    if (cusolverWorkspace) CUDA_CHECK_NOABORT(cudaFree(cusolverWorkspace));
    if (hRnrmF) { delete[] hRnrmF; hRnrmF = nullptr; }
    if (cusolverHandle) { cusolverDnDestroy(cusolverHandle); cusolverHandle = nullptr; }
    // From updateOverlapArea
    if (intersectionsCounter) CUDA_CHECK_NOABORT(cudaFree(intersectionsCounter));
}

// resetters

void Model::resetMaxActualNeighbors() {
    int init = INT_MIN;
    CUDA_CHECK(cudaMemcpy(maxActualNeighbors, &init, sizeof(int), cudaMemcpyHostToDevice));
}

void Model::deallocateAll() {
    CUDA_CHECK_NOABORT(cudaFree(positions));
//    delete [] C;
}

// initializers

void Model::initializeNeighborCells() {
    if (countPerBox)     { CUDA_CHECK_NOABORT(cudaFree(countPerBox));     countPerBox     = nullptr; }
    if (boxId)           { CUDA_CHECK_NOABORT(cudaFree(boxId));           boxId           = nullptr; }
    if (cellLocation)    { CUDA_CHECK_NOABORT(cudaFree(cellLocation));    cellLocation    = nullptr; }
    if (neighborIndices) { CUDA_CHECK_NOABORT(cudaFree(neighborIndices)); neighborIndices = nullptr; }
    if (neighbors)       { CUDA_CHECK_NOABORT(cudaFree(neighbors));       neighbors       = nullptr; }
    if (inside)          { CUDA_CHECK_NOABORT(cudaFree(inside));          inside          = nullptr; }
    if (shapeId)         { CUDA_CHECK_NOABORT(cudaFree(shapeId));         shapeId         = nullptr; }
    if (next)            { CUDA_CHECK_NOABORT(cudaFree(next));            next            = nullptr; }
    if (prev)            { CUDA_CHECK_NOABORT(cudaFree(prev));            prev            = nullptr; }
    if (numNeighbors)    { CUDA_CHECK_NOABORT(cudaFree(numNeighbors));    numNeighbors    = nullptr; }
    if (valid)           { CUDA_CHECK_NOABORT(cudaFree(valid));           valid           = nullptr; }
    if (outputIdx)       { CUDA_CHECK_NOABORT(cudaFree(outputIdx));       outputIdx       = nullptr; }
    if (shapeCounts)     { CUDA_CHECK_NOABORT(cudaFree(shapeCounts));     shapeCounts     = nullptr; }
    if (intersections)   { CUDA_CHECK_NOABORT(cudaFree(intersections));   intersections   = nullptr; }
    if (tu)              { CUDA_CHECK_NOABORT(cudaFree(tu));              tu              = nullptr; }
    if (ut)              { CUDA_CHECK_NOABORT(cudaFree(ut));              ut              = nullptr; }
    if (tuTMP)           { CUDA_CHECK_NOABORT(cudaFree(tuTMP));           tuTMP           = nullptr; }
    if (utTMP)           { CUDA_CHECK_NOABORT(cudaFree(utTMP));           utTMP           = nullptr; }
    if (outersections)   { CUDA_CHECK_NOABORT(cudaFree(outersections));   outersections   = nullptr; }
    if (outersectionsTMP){ CUDA_CHECK_NOABORT(cudaFree(outersectionsTMP));outersectionsTMP= nullptr; }
    if (keys)            { CUDA_CHECK_NOABORT(cudaFree(keys));            keys            = nullptr; }

    int numBoxes = boxSize * boxSize;
    CUDA_CHECK(cudaMalloc((void**)&countPerBox, numBoxes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&boxId, numBoxes * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&cellLocation, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&neighborIndices, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&neighbors, maxNeighbors * size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&inside, maxNeighbors * size * sizeof(bool)));

    CUDA_CHECK(cudaMalloc((void**)&shapeId, size * sizeof(int)));
    updateShapeIdCUDA(shapeId, startIndices, size, numPolygons);
    CUDA_CHECK(cudaMalloc(&next, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&prev, size * sizeof(int)));
    computeNextPrevCUDA(next, prev, startIndices, shapeId, size);

    CUDA_CHECK(cudaMalloc((void**)&numNeighbors, size * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&valid, maxNeighbors * size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&outputIdx, maxNeighbors * size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&shapeCounts, numPolygons * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&intersections, maxNeighbors * size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&tu, maxNeighbors * size * sizeof(double2)));
    CUDA_CHECK(cudaMalloc((void**)&ut, maxNeighbors * size * sizeof(double2)));
    CUDA_CHECK(cudaMalloc((void**)&tuTMP, maxNeighbors * size * sizeof(double2)));
    CUDA_CHECK(cudaMalloc((void**)&utTMP, maxNeighbors * size * sizeof(double2)));

    CUDA_CHECK(cudaMalloc((void**)&outersections, maxNeighbors * size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&outersectionsTMP, maxNeighbors * size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&keys, maxNeighbors * size * sizeof(uint32_t)));
}

void Model::initializeRandomSeed(const unsigned long long seed_) {
    seed = seed_;
    initializeRandomStates(globalState, seed, size);
}

// helpers

void Model::sortKeys(int endBit) {
    sortKeysCUDA(intersections, numIntersections, 0, endBit, keys);
}

vector<uint32_t> Model::getKeys() const {
    vector<uint32_t> keys_(numIntersections);
    if (numIntersections > 0) {
        CUDA_CHECK(cudaMemcpy(keys_.data(), keys, numIntersections * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    return keys_;
}

// updaters

void Model::updateNeighborCells() {
    updateNeighborCellsCUDA(positions, startIndices, shapeId, numPolygons, size, boxSize, cellLocation, countPerBox, boxId, boxesUsed, neighborIndices);
}

void Model::updateNeighbors() {
    // The neighbors array has a size maxNeighbors * numVertices
    // numNeighbors is an array of size numVertices that says how many neighbors
    // are in each
    // first attempt
    int newActualNeighbors = updateNeighborsCUDA(shapeId, startIndices, positions, cellLocation,
                              neighborIndices, size, neighbors, numNeighbors,
                              maxNeighbors, boxSize, countPerBox, maxActualNeighbors, tu, inside);
    if (newActualNeighbors > maxNeighbors) {
        // read required max from device
        // warn the user
        std::cerr << "Warning: neighbor buffer overflow: maxNeighbors=" << maxNeighbors
                  << " required=" << newActualNeighbors << ". Resizing and retrying.\n";

        // resize neighbors buffer to accommodate required value (at least hostMaxActual)
        int newMax = max(maxNeighbors * 2 + 1, newActualNeighbors);
        CUDA_CHECK_NOABORT(cudaFree(neighbors));
        CUDA_CHECK_NOABORT(cudaFree(inside));
        CUDA_CHECK_NOABORT(cudaFree(valid));
        CUDA_CHECK_NOABORT(cudaFree(outputIdx));
        CUDA_CHECK_NOABORT(cudaFree(intersections));
        CUDA_CHECK_NOABORT(cudaFree(tu));
        CUDA_CHECK_NOABORT(cudaFree(outersections));
        CUDA_CHECK_NOABORT(cudaFree(ut));
        CUDA_CHECK_NOABORT(cudaFree(tuTMP));
        CUDA_CHECK_NOABORT(cudaFree(outersectionsTMP));
        CUDA_CHECK_NOABORT(cudaFree(utTMP));
        CUDA_CHECK_NOABORT(cudaFree(keys));
        CUDA_CHECK(cudaMalloc((void**)&neighbors, newMax * size * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&inside, newMax * size * sizeof(bool)));
        CUDA_CHECK(cudaMalloc((void**)&valid, newMax * size * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&outputIdx, newMax * size * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&intersections, newMax * size * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&tu, newMax * size * sizeof(double2)));
        CUDA_CHECK(cudaMalloc((void**)&outersections, newMax * size * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&ut, newMax * size * sizeof(double2)));
        CUDA_CHECK(cudaMalloc((void**)&tuTMP, newMax * size * sizeof(double2)));
        CUDA_CHECK(cudaMalloc((void**)&outersectionsTMP, newMax * size * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&utTMP, newMax * size * sizeof(double2)));
        CUDA_CHECK(cudaMalloc((void**)&keys, newMax * size * sizeof(uint32_t)));
        maxNeighbors = newMax;

        // retry once
        int ok = updateNeighborsCUDA(shapeId, startIndices, positions, cellLocation,
                                      neighborIndices, size, neighbors, numNeighbors,
                                      maxNeighbors, boxSize, countPerBox, maxActualNeighbors, tu, inside);
        if (ok > maxNeighbors) {
            std::cerr << "Warning: updateNeighbors still failed after resizing to " << maxNeighbors << "\n";
        }
    }
    resetMaxActualNeighbors();
}

void Model::updateValidAndCounts() {
    numIntersections = updateValidAndCountsCUDA(size, neighbors, numNeighbors, maxNeighbors, inside, shapeId, numPolygons, valid, shapeCounts, outputIdx);
}

void Model::updatePolygonGeometry() {
    updatePolygonGeometryCUDA(size, numPolygons, positions, startIndices, shapeId, next, prev, edgeLengths, areaParts, comParts, areas, comX, comY, maxEdgeLength, constraints, constraintNormSq);
}

int Model::shakeProject(int nIter, double tol) {
    lastShakeIters = shakeProjectCUDA(numPolygons, polygonSize, startIndices, next, prev,
                                      positions, targetEdgeLengths, targetAreas, nIter, tol,
                                      shakeItersTMP);
    return lastShakeIters;
}

int Model::getLastShakeIters() const {
    return lastShakeIters;
}

void Model::saveTentativePositions() {
    saveTentativePositionsCUDA(size, positions, positionsTMP);
}

double Model::getMaxEffectiveForce(double dt, minimizerEnum minimizerType) const {
    double scale = (minimizerType == minimizerEnum::GD) ? (1.0 / dt) : (2.0 / (dt * dt));
    return getMaxEffectiveForceCUDA(size, positions, positionsTMP, force, scale, effForceMagTMP);
}

void Model::projectForce() {
    projectForceCUDA(size, numPolygons, polygonSize,
        shapeId, startIndices, next,
        positions, constraints,
        edgeGradTMP, uMat, singularValuesTMP, vMatTMP,
        solverInfoTMP, qAreaVec, cusolverHandle,
        cusolverWorkspace, cusolverWorkspaceSize, hRnrmF, force);
}


void Model::updateCompactedIntersections() {
    updateCompactedIntersectionsCUDA(size, maxNeighbors, neighbors, inside, shapeId, startIndices, valid, outputIdx, intersections, numIntersections, tu, tuTMP);
}

void Model::updateOutersections() {
    numIntersections = updateValidAndCountsCUDA(size, neighbors, numNeighbors, maxNeighbors, inside, shapeId, numPolygons, valid, shapeCounts, outputIdx);
    updateCompactedIntersectionsCUDA(size, maxNeighbors, neighbors, inside, shapeId, startIndices, valid, outputIdx, intersections, numIntersections, tu, tuTMP);
    sortKeysCUDA(intersections, numIntersections, 0, 64, keys);
    applyPermutationCUDA_double2(tu, keys, tuTMP, numIntersections);
    updateOutersectionsCUDA(intersections, tuTMP, utTMP, startIndices, numIntersections, outersectionsTMP);
    sortKeysCUDA(intersections, numIntersections, 0, 48, keys);
    applyPermutationCUDA_double2(tuTMP, keys, tu, numIntersections);
    applyPermutationCUDA_double2(utTMP, keys, ut, numIntersections);
    applyPermutationCUDA_int64(outersectionsTMP, keys, outersections, numIntersections);
}

void Model::updateForceEnergy() {
    CUDA_CHECK(cudaMemset(force, 0, size * 2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(energy, 0, sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());
    switch (simControl.modelType) {
        case simControlStruct::modelEnum::normal:
            updateForceEnergyExteriorCUDA(size, numIntersections, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy);
            updateShapeRangesCUDA(numPolygons, size, numIntersections, intersections, shapeStart, shapeEnd);
            updateForceEnergyInteriorCUDA(size, numIntersections, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy, numPolygons, shapeStart, shapeEnd);
            return;
        case simControlStruct::modelEnum::softBody:
            updateForceEnergyEdgeCUDA(size, positions, targetEdgeLengths, edgeLengths, next, prev, shapeId, force, energy, stiffness);
            updateForceEnergyAreaCUDA(size, shapeId, next, prev, positions, areas, targetAreas, startIndices, force, energy, compressibility);
            return;
        case simControlStruct::modelEnum::edgeOnly:
            updateForceEnergyEdgeCUDA(size, positions, targetEdgeLengths, edgeLengths, next, prev, shapeId, force, energy, stiffness);
            return;
        case simControlStruct::modelEnum::areaOnly:
            updateForceEnergyAreaCUDA(size, shapeId, next, prev, positions, areas, targetAreas, startIndices, force, energy, compressibility);
            return;
        case simControlStruct::modelEnum::hybrid:
            updateForceEnergyExteriorCUDA(size, numIntersections, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy);
            updateShapeRangesCUDA(numPolygons, size, numIntersections, intersections, shapeStart, shapeEnd);
            updateForceEnergyInteriorCUDA(size, numIntersections, intersections, outersections, tu, ut, positions, next, prev, shapeId, startIndices, force, energy, numPolygons, shapeStart, shapeEnd);
            updateForceEnergyEdgeCUDA(size, positions, targetEdgeLengths, edgeLengths, next, prev, shapeId, force, energy, stiffness);
            updateForceEnergyAreaCUDA(size, shapeId, next, prev, positions, areas, targetAreas, startIndices, force, energy, compressibility);
            return;
        case simControlStruct::modelEnum::abnormal:
            return;
        default:
            return;
    }
}

void Model::updatePositions(double dt) {
    updatePositionsCUDA(size, positions, force, dt);
}

void Model::resetVelocities() {
    CUDA_CHECK(cudaMemset(velocities, 0, 2 * size * sizeof(double)));
}

std::tuple<double, double, double, int> Model::minimizeFIREStep(double dt, double alpha, int nPos, double dtMax, double alphaStart, double fAlpha, double fInc, double fDec, int nMin, int shakeIter) {
    bool needsIntersections = (simControl.modelType == simControlStruct::modelEnum::normal
                            || simControl.modelType == simControlStruct::modelEnum::hybrid);
    bool isRigid = (simControl.modelType == simControlStruct::modelEnum::normal);
    bool shakeFailed = false;
    // Save pre-step positions and energy for rollback on SHAKE failure or energy increase.
    CUDA_CHECK(cudaMemcpy(positionsTMP2, positions, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice));
    double ePre = getEnergy();
    updatePositionAndVelocityFIRECUDA(size, positions, velocities, force, dt);
    updatePolygonGeometry();
    if (needsIntersections) {
        updateNeighborCells();
        updateNeighbors();
        if (isRigid && shakeIter > 0) {
            // positionsTMP = post-Verlet positions, used for velocity rederivation on success.
            CUDA_CHECK(cudaMemcpy(positionsTMP, positions, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice));
            shakeProject(shakeIter, 1e-15);
            updatePolygonGeometry();
            if (lastShakeIters >= shakeIter) {
                // First attempt hit the limit — retry with 10x more iterations.
                shakeProject(shakeIter * 10, 1e-15);
                updatePolygonGeometry();
                if (lastShakeIters >= shakeIter * 10) {
                    shakeFailed = true;
                }
            }
            if (shakeFailed) {
                // Roll back to pre-step positions (near-zero violations) and reset FIRE.
                CUDA_CHECK(cudaMemcpy(positions, positionsTMP2, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice));
                updatePolygonGeometry();
                CUDA_CHECK(cudaMemset(velocities, 0, 2 * size * sizeof(double)));
            } else {
                rederiveVelocityFromDisplacementFIRECUDA(size, velocities, positions, positionsTMP, dt);
            }
        }
        updateOutersections();
    }
    updateForceEnergy();
    if (isRigid) projectForce();
    if (shakeFailed) {
        dt = dt * fDec;
        alpha = alphaStart;
        nPos = 0;
        return {getEnergy(), dt, alpha, nPos};
    }
    // Roll back and reset if energy increased by more than floating-point noise.
    // Mixed tolerance prevents dt → 0 collapse from GPU-reduction non-determinism.
    double ePost = getEnergy();
    if (ePost > ePre + fmax(fabs(ePre) * 1e-10, 1e-14)) {
        CUDA_CHECK(cudaMemcpy(positions, positionsTMP2, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice));
        updatePolygonGeometry();
        if (needsIntersections) {
            updateNeighborCells();
            updateNeighbors();
            updateOutersections();
            updateForceEnergy();
            if (isRigid) projectForce();
        } else {
            updateForceEnergy();
        }
        CUDA_CHECK(cudaMemset(velocities, 0, 2 * size * sizeof(double)));
        dt = dt * fDec;
        alpha = alphaStart;
        nPos = 0;
        return {ePre, dt, alpha, nPos};
    }
    updateVelocityFIRECUDA(size, velocities, force, dt);
    double P = bendVelocityTowardsForceFIRECUDA(size, velocities, force, alpha, fireScratchTMP, fireResultTMP);
    if (P > 0.0) {
        if (++nPos >= nMin) {
            double newDt = fmin(dt * fInc, dtMax);
            if (newDt > dt) alpha = alpha * fAlpha;
            dt = newDt;
        }
    } else {
        CUDA_CHECK(cudaMemset(velocities, 0, 2 * size * sizeof(double)));
        dt = dt * fDec;
        alpha = alphaStart;
        nPos = 0;
    }
    return {getEnergy(), dt, alpha, nPos};
}

std::tuple<double, double, int> Model::minimizeFIRE(double maxForceThreshold, double dtInit, int maxSteps, double dtMax, double alphaStart, double fAlpha, double fInc, double fDec, int nMin, int shakeIter) {
    resetVelocities();
    double dt = dtInit;
    double alpha = alphaStart;
    int nPos = 0;
    bool isRigid = (simControl.modelType == simControlStruct::modelEnum::normal);
    updatePolygonGeometry();
    updateForceEnergy();
    if (isRigid) projectForce();
    for (int step = 0; step < maxSteps; ++step) {
        double energy;
        std::tie(energy, dt, alpha, nPos) = minimizeFIREStep(dt, alpha, nPos, dtMax, alphaStart, fAlpha, fInc, fDec, nMin, shakeIter);
        if (getMaxUnbalancedForceCUDA(size, force) <= maxForceThreshold)
            return {energy, dt, step + 1};
    }
    return {getEnergy(), dt, maxSteps};
}

// misc
void Model::resetAreas() {
    resetAreasCUDA(size, shapeId, positions, areas, targetAreas, comX, comY);
}

// setters

void Model::setMaxEdgeLength(double maxEdgeLength_) {
    boxSize = floor(1.0 / maxEdgeLength_);
    CUDA_CHECK(cudaMemcpy(maxEdgeLength, &maxEdgeLength_, sizeof(double), cudaMemcpyHostToDevice));
}

void Model::setModelEnum(simControlStruct::modelEnum modelType_) {
    simControl.modelType = modelType_;
}

void Model::setPositions(const vector<double>& positionsData) {
    if (positionsData.size() != size * 2) {
        cout << "Update numVertices before setting the positions" << endl;
        return;
    }
    CUDA_CHECK(cudaMemcpy(positions, positionsData.data(), 2 * size * sizeof(double), cudaMemcpyHostToDevice));
}

void Model::setForces(const vector<double>& forcesData) {
    // This can be deleted. I am using it for diagnostics
    // and don't have plans to implement a feature which
    // would use this.
    CUDA_CHECK(cudaMemcpy(force, forcesData.data(), 2 * size * sizeof(double), cudaMemcpyHostToDevice));
}

void Model::setStartIndices(const vector<int>& startIndicesData) {
    numPolygons = startIndicesData.size() - 1;
    if (areas)             { CUDA_CHECK(cudaFree(areas));             areas             = nullptr; }
    if (targetAreas)       { CUDA_CHECK(cudaFree(targetAreas));       targetAreas       = nullptr; }
    if (targetEdgeLengths) { CUDA_CHECK(cudaFree(targetEdgeLengths)); targetEdgeLengths = nullptr; }
    if (comX)              { CUDA_CHECK(cudaFree(comX));              comX              = nullptr; }
    if (comY)              { CUDA_CHECK(cudaFree(comY));              comY              = nullptr; }
    if (perimeters)        { CUDA_CHECK(cudaFree(perimeters));        perimeters        = nullptr; }
    if (startIndices)      { CUDA_CHECK(cudaFree(startIndices));      startIndices      = nullptr; }
    if (shapeStart)        { CUDA_CHECK(cudaFree(shapeStart));        shapeStart        = nullptr; }
    if (shapeEnd)          { CUDA_CHECK(cudaFree(shapeEnd));          shapeEnd          = nullptr; }
    if (constraintNormSq)  { CUDA_CHECK(cudaFree(constraintNormSq));  constraintNormSq  = nullptr; }
    if (mgsIp)             { CUDA_CHECK(cudaFree(mgsIp));             mgsIp             = nullptr; }
    if (forceProjIp)       { CUDA_CHECK(cudaFree(forceProjIp));       forceProjIp       = nullptr; }
    CUDA_CHECK(cudaMalloc((void**)&areas,             numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&targetAreas,       numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&targetEdgeLengths, numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&comX,              numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&comY,              numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&perimeters,        numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&startIndices,      (numPolygons + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(startIndices, startIndicesData.data(), (numPolygons + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&shapeStart,                numPolygons * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&shapeEnd,                  numPolygons * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&constraintNormSq,  3 * numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&mgsIp,             numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&forceProjIp,       3 * numPolygons * sizeof(double)));
    if (xpbdArea)              { CUDA_CHECK(cudaFree(xpbdArea));              xpbdArea              = nullptr; }
    if (xpbdGradNormSq)        { CUDA_CHECK(cudaFree(xpbdGradNormSq));        xpbdGradNormSq        = nullptr; }
    CUDA_CHECK(cudaMalloc((void**)&xpbdArea,              numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&xpbdGradNormSq,        numPolygons * sizeof(double)));

    // n+1 individual constraint projection buffers
    polygonSize = startIndicesData[1] - startIndicesData[0];
    int n = polygonSize;
    if (edgeGradTMP)       { CUDA_CHECK(cudaFree(edgeGradTMP));       edgeGradTMP       = nullptr; }
    if (uMat)              { CUDA_CHECK(cudaFree(uMat));              uMat              = nullptr; }
    if (singularValuesTMP) { CUDA_CHECK(cudaFree(singularValuesTMP)); singularValuesTMP = nullptr; }
    if (vMatTMP)           { CUDA_CHECK(cudaFree(vMatTMP));           vMatTMP           = nullptr; }
    if (solverInfoTMP)     { CUDA_CHECK(cudaFree(solverInfoTMP));     solverInfoTMP     = nullptr; }
    if (qAreaVec)          { CUDA_CHECK(cudaFree(qAreaVec));          qAreaVec          = nullptr; }
    if (cusolverWorkspace) { CUDA_CHECK(cudaFree(cusolverWorkspace)); cusolverWorkspace = nullptr; }
    if (hRnrmF)            { delete[] hRnrmF;                         hRnrmF            = nullptr; }
    long long matStride = (long long)2*n * n;
    CUDA_CHECK(cudaMalloc((void**)&edgeGradTMP,       matStride * numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&uMat,              matStride * numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&singularValuesTMP, (long long)n * numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&vMatTMP,           (long long)n * n * numPolygons * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&solverInfoTMP,     numPolygons * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&qAreaVec,          2 * size * sizeof(double)));
    hRnrmF = new double[numPolygons];
    // Query cuSolver workspace size (lwork is count of doubles)
    CUSOLVER_CHECK(cusolverDnDgesvdaStridedBatched_bufferSize(
        cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, n,
        2*n, n,
        edgeGradTMP, 2*n, matStride,
        singularValuesTMP, (long long)n,
        uMat, 2*n, matStride,
        vMatTMP, n, (long long)n*n,
        &cusolverWorkspaceSize, numPolygons));
    CUDA_CHECK(cudaMalloc((void**)&cusolverWorkspace, (size_t)cusolverWorkspaceSize * sizeof(double)));
}

void Model::setNumVertices(int numVertices_) {
    size = numVertices_;
    if (positions)     { CUDA_CHECK(cudaFree(positions));     positions     = nullptr; }
    if (edgeLengths)   { CUDA_CHECK(cudaFree(edgeLengths));   edgeLengths   = nullptr; }
    if (force)         { CUDA_CHECK(cudaFree(force));         force         = nullptr; }
    if (velocities)    { CUDA_CHECK(cudaFree(velocities));    velocities    = nullptr; }
    if (fireScratchTMP){ CUDA_CHECK(cudaFree(fireScratchTMP));fireScratchTMP= nullptr; }
    if (fireResultTMP) { CUDA_CHECK(cudaFree(fireResultTMP)); fireResultTMP = nullptr; }
    if (comParts)      { CUDA_CHECK(cudaFree(comParts));      comParts      = nullptr; }
    if (areaParts)     { CUDA_CHECK(cudaFree(areaParts));     areaParts     = nullptr; }
    if (constraints)   { CUDA_CHECK(cudaFree(constraints));   constraints   = nullptr; }
    if (positionsTMP)  { CUDA_CHECK(cudaFree(positionsTMP));  positionsTMP  = nullptr; }
    if (positionsTMP2) { CUDA_CHECK(cudaFree(positionsTMP2)); positionsTMP2 = nullptr; }
    if (effForceMagTMP){ CUDA_CHECK(cudaFree(effForceMagTMP));effForceMagTMP= nullptr; }
    if (globalState)   { CUDA_CHECK(cudaFree(globalState));   globalState   = nullptr; }
    CUDA_CHECK(cudaMalloc((void**)&positions,    2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&edgeLengths,  size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&force,        size * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&velocities,   2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&fireScratchTMP, 2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&fireResultTMP,  sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&comParts,       2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&areaParts,      2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&constraints,    6 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&positionsTMP,   2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&positionsTMP2,  2 * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&effForceMagTMP, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&globalState,    sizeof(curandState) * size));
}

void Model::setTargetEdgeLengths(const vector<double>& targetEdgeLengthsData) {
    CUDA_CHECK(cudaMemcpy(targetEdgeLengths, targetEdgeLengthsData.data(), numPolygons * sizeof(double), cudaMemcpyHostToDevice));
}

void Model::setTargetAreas(const vector<double>& targetAreasData) {
    CUDA_CHECK(cudaMemcpy(targetAreas, targetAreasData.data(), numPolygons * sizeof(double), cudaMemcpyHostToDevice));
}

void Model::setStiffness(const double stiffness_) {
    stiffness = stiffness_;
}

void Model::setCompressibility(const double compressibility_) {
    compressibility = compressibility_;
}

double Model::getStiffness() const {
    return stiffness;
}

double Model::getCompressibility() const {
    return compressibility;
}

// getters

string Model::getModelEnum() const {
    switch (simControl.modelType) {
        case simControlStruct::modelEnum::normal:   return "normal";
        case simControlStruct::modelEnum::edgeOnly: return "edgeOnly";
        case simControlStruct::modelEnum::areaOnly: return "areaOnly";
        case simControlStruct::modelEnum::softBody: return "softBody";
        case simControlStruct::modelEnum::hybrid:   return "hybrid";
        case simControlStruct::modelEnum::abnormal: return "abnormal";
        default: return std::to_string(static_cast<int>(simControl.modelType));
    }
}

int Model::getNumVertices() const {
    return size;
}

int Model::getNumPolygons() const {
    return numPolygons;
}

unsigned long long Model::getRandomSeed() {
    return seed;
}

double Model::getEnergy() const {
    double energy_ = 0.0;
    CUDA_CHECK(cudaMemcpy(&energy_, energy, sizeof(double), cudaMemcpyDeviceToHost));
    return energy_;
}

vector<int> Model::getShapeId() const {
    vector<int> shapeId_(size);
    CUDA_CHECK(cudaMemcpy(shapeId_.data(), shapeId, size * sizeof(int), cudaMemcpyDeviceToHost));
    return shapeId_;
}

vector<double> Model::getPositions() const {
    vector<double> positions_(2 * size);
    CUDA_CHECK(cudaMemcpy(positions_.data(), positions, 2 * size * sizeof(double), cudaMemcpyDeviceToHost));
    return positions_;
}

vector<double> Model::getForces() const {
    vector<double> force_(2 * size);
    CUDA_CHECK(cudaMemcpy(force_.data(), force, 2 * size * sizeof(double), cudaMemcpyDeviceToHost));
    return force_;
}

vector<int> Model::getIntersectionsCounter() const {
    vector<int> intersectionsCounter_(pointDensity * pointDensity);
    CUDA_CHECK(cudaMemcpy(intersectionsCounter_.data(), intersectionsCounter, pointDensity * pointDensity * sizeof(int), cudaMemcpyDeviceToHost));
    return intersectionsCounter_;
}

double Model::getMaxEdgeLength() const {
    double val = 0.0;
    CUDA_CHECK(cudaMemcpy(&val, maxEdgeLength, sizeof(double), cudaMemcpyDeviceToHost));
    return val;
}

vector<int> Model::getNeighbors() const {
    vector<int> neighbors_(maxNeighbors * size);
    CUDA_CHECK(cudaMemcpy(neighbors_.data(), neighbors, maxNeighbors * size * sizeof(int), cudaMemcpyDeviceToHost));
    return neighbors_;
}

vector<bool> Model::getInsideFlag() const {
    size_t n = static_cast<size_t>(maxNeighbors) * static_cast<size_t>(size);
    if (n == 0) return vector<bool>();

    // temporary contiguous buffer that matches device layout (bytes)
    vector<unsigned char> tmp(n);
    // copy from device (device buffer 'inside' was allocated with sizeof(bool))
    CUDA_CHECK(cudaMemcpy(tmp.data(), inside, n * sizeof(bool), cudaMemcpyDeviceToHost));

    // convert to vector<bool>
    vector<bool> inside_(n);
    for (size_t i = 0; i < n; ++i) inside_[i] = tmp[i] ? true : false;
    return inside_;
}

vector<double> Model::getTU() const {
    if (numIntersections == 0) return vector<double>();
    // temporary contiguous buffer that matches device layout (bytes)
    vector<double2> tu_(numIntersections);
    CUDA_CHECK(cudaMemcpy(tu_.data(), tu, numIntersections * sizeof(double2), cudaMemcpyDeviceToHost));
    vector<double> sol(2 * numIntersections);
    for (int i = 0; i < numIntersections; i++) {
        sol[2 * i] = static_cast<double>(tu_[i].x);
        sol[2 * i + 1] = static_cast<double>(tu_[i].y);
    }
    return sol;
}

vector<double> Model::getUT() const {
    if (numIntersections == 0) return vector<double>();
    // temporary contiguous buffer that matches device layout (bytes)
    vector<double2> ut_(numIntersections);
    CUDA_CHECK(cudaMemcpy(ut_.data(), ut, numIntersections * sizeof(double2), cudaMemcpyDeviceToHost));
    vector<double> sol(2 * numIntersections);
    for (int i = 0; i < numIntersections; i++) {
        sol[2 * i] = static_cast<double>(ut_[i].x);
        sol[2 * i + 1] = static_cast<double>(ut_[i].y);
    }
    return sol;
}

vector<int> Model::getNumNeighbors() const {
    vector<int> numNeighbors_(size);
    CUDA_CHECK(cudaMemcpy(numNeighbors_.data(), numNeighbors, size * sizeof(int), cudaMemcpyDeviceToHost));
    return numNeighbors_;
}

vector<int> Model::getStartIndices() const {
    vector<int> startIndices_(numPolygons + 1);
    CUDA_CHECK(cudaMemcpy(startIndices_.data(), startIndices, (numPolygons + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    return startIndices_;
}

vector<double> Model::getAreas() const {
    vector<double> areas_(numPolygons);
    CUDA_CHECK(cudaMemcpy(areas_.data(), areas, numPolygons * sizeof(double), cudaMemcpyDeviceToHost));
    return areas_;
}

vector<double> Model::getTargetAreas() const {
    vector<double> targetAreas_(numPolygons);
    CUDA_CHECK(cudaMemcpy(targetAreas_.data(), targetAreas, numPolygons * sizeof(double), cudaMemcpyDeviceToHost));
    return targetAreas_;
}

vector<double> Model::getConstraints() const {
    vector<double> constraints_(size * 6);
    CUDA_CHECK(cudaMemcpy(constraints_.data(), constraints, size * 6 * sizeof(double), cudaMemcpyDeviceToHost));
    return constraints_;
}

vector<double> Model::getCOM() const {
    vector<double> com_(numPolygons * 2);
    CUDA_CHECK(cudaMemcpy(com_.data(), comX, numPolygons * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(com_.data() + numPolygons, comY, numPolygons * sizeof(double), cudaMemcpyDeviceToHost));
    return com_;
}

vector<double> Model::getPerimeters() const {
    vector<double> perimeters_(numPolygons);
    CUDA_CHECK(cudaMemcpy(perimeters_.data(), perimeters, numPolygons * sizeof(double), cudaMemcpyDeviceToHost));
    return perimeters_;
}

vector<int> Model::getNeighborCells() const {
    vector<int> neighborCells_(size);
    CUDA_CHECK(cudaMemcpy(neighborCells_.data(), cellLocation, size * sizeof(int), cudaMemcpyDeviceToHost));
    return neighborCells_;
}

vector<int> Model::getNeighborIndices() const {
    vector<int> neighborIndices_(size);
    CUDA_CHECK(cudaMemcpy(neighborIndices_.data(), neighborIndices, size * sizeof(int), cudaMemcpyDeviceToHost));
    return neighborIndices_;
}

vector<int> Model::getBoxCounts() const {
    vector<int> countPerBox_(boxSize * boxSize);
    CUDA_CHECK(cudaMemcpy(countPerBox_.data(), countPerBox, boxSize * boxSize * sizeof(int), cudaMemcpyDeviceToHost));
    return countPerBox_;
}

void Model::updateOverlapArea(int pointDensity_) {
    // allocate or reallocate the device-side counter buffer if density changed
    if (pointDensity != pointDensity_) {
        if (intersectionsCounter != nullptr) {
            CUDA_CHECK_NOABORT(cudaFree(intersectionsCounter));
            intersectionsCounter = nullptr;
        }
        size_t total = static_cast<size_t>(pointDensity_) * static_cast<size_t>(pointDensity_);
        CUDA_CHECK(cudaMalloc((void**)&intersectionsCounter, total * sizeof(int)));
        // remember new density
        pointDensity = pointDensity_;
    }

    // ensure the buffer is zeroed (CUDA kernel may overwrite but zeroing is cheap)
    size_t total = static_cast<size_t>(pointDensity) * static_cast<size_t>(pointDensity);
    CUDA_CHECK(cudaMemset(intersectionsCounter, 0, total * sizeof(int)));

    // call CUDA routine that computes and returns the raw sum of intersectionsCounter entries
    updateOverlapAreaCUDA(shapeId, startIndices, pointDensity, intersectionsCounter, neighborIndices, size, boxSize, countPerBox, positions, overlapArea
    );

    // normalize to fraction of sampled points -> overlap area estimate in [0,1]
    overlapArea /= static_cast<double>(pointDensity * pointDensity);
}

vector<int> Model::getShapeCounts() const {
    vector<int> shapeCounts_(numPolygons);
    CUDA_CHECK(cudaMemcpy(shapeCounts_.data(), shapeCounts, numPolygons * sizeof(int), cudaMemcpyDeviceToHost));
    return shapeCounts_;
}

vector<uint64_t> Model::getIntersections() const {
    vector<uint64_t> intersections_(numIntersections);
    if (numIntersections > 0) {
        CUDA_CHECK(cudaMemcpy(intersections_.data(), intersections, numIntersections * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }
    return intersections_;
}

int Model::getNumIntersections() const {
    return numIntersections;
}

vector<uint64_t> Model::getOutersections() const {
    vector<uint64_t> outersections_(numIntersections);
    if (numIntersections > 0) {
        CUDA_CHECK(cudaMemcpy(outersections_.data(), outersections, numIntersections * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }
    return outersections_;
}

vector<double> Model::getTargetEdgeLengths() const {
    vector<double> targetEdgeLengths_(numPolygons);
    CUDA_CHECK(cudaMemcpy(targetEdgeLengths_.data(), targetEdgeLengths, numPolygons * sizeof(double), cudaMemcpyDeviceToHost));
    return targetEdgeLengths_;
}

double Model::getMaxUnbalancedForce() const {
    return getMaxUnbalancedForceCUDA(size, force);
}

vector<double> Model::getEdgeLengths() const {
    vector<double> edgeLengths_(size);
    CUDA_CHECK(cudaMemcpy(edgeLengths_.data(), edgeLengths, size * sizeof(double), cudaMemcpyDeviceToHost));
    return edgeLengths_;
}

double Model::getOverlapArea() const {
    return overlapArea;
}
