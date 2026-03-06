#include <iostream>
#include <vector>
#include "model.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
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
extern "C" void applyPermutationCUDA_float2(const float2* d_input, const uint32_t* d_perm, float2* d_output, int numItems);
extern "C" void computeNextPrevCUDA(int* next, int* prev, int* startIndices, int* shapeId, int size);
// updaters:
extern "C" void updateAreasCUDA(double* areas, double* positions, int* startIndices, int numPolygons);
extern "C" void updateNeighborCellsCUDA(double* positions, int* startIndices, int* shapeId, int numPolygons, int size, int boxSize, int* cellLocation, int* countPerBox, int* boxId, int& boxesUsed, int* neighborIndices);
extern "C" void updateShapeIdCUDA(int* shapeId, int* startIndices, int size, int numPolygons);
extern "C" int updateNeighborsCUDA(int* shapeId, int* startIndices, double* positions, int* cellLocation, int* neighborIndices, int size, int* neighbors, int* numNeighbors, int maxNeighbors, int boxSize, int* countPerBox, double a, int* maxActualNeighbors);
extern "C" void updateContactsCUDA(const int* shapeId, const int* startIndices, const double* positions, const int size, const int* neighbors, const int* numNeighbors, const int maxNeighbors, bool* inside, float2* tu, int* contacts, int* numContacts);
extern "C" void updatePerimetersCUDA(double* perimeters, double* positions, int* startIndices, int numPolygons);
extern "C" void updateOverlapAreaCUDA(int* shapeId, int* startIndices, int pointDensity, int* intersectionsCounter, int* neighborIndices, int size, int boxSize, int* countPerBox, double* positions, double& overlapArea);
extern "C" int updateValidAndCountsCUDA(int numVertices, int* contacts, int* numContacts, int maxNeighbors, bool* insideFlag, int* shapeIds, int numShapes, int* valid, int* shapeCounts, uint64_t* outputIdx);
extern "C" void updateCompactedIntersectionsCUDA(int numVertices, int maxNeighbors, int* contacts, bool* insideFlag, int* shapeIds, int* startIndices, int* valid, uint64_t* outputIdx, uint64_t* intersections, int numIntersections, float2* tu);
extern "C" void updateOutersectionsCUDA(const uint64_t* intersections, const float2* tu, const float2* ut, const int* startIndices, int numIntersections, uint64_t* outersections);

// Constructor
Model::Model(int size_)
    : size(size_),
      positions(nullptr), forces(nullptr), maxActualNeighbors(nullptr), globalState(nullptr),
      countPerBox(nullptr), boxId(nullptr), neighborIndices(nullptr), cellLocation(nullptr),
      shapeId(nullptr), neighbors(nullptr), contacts(nullptr), numNeighbors(nullptr),
      numContacts(nullptr), inside(nullptr), perimeters(nullptr), intersectionsCounter(nullptr),
      valid(nullptr), outputIdx(nullptr), shapeCounts(nullptr), intersections(nullptr),
      tu(nullptr), ut(nullptr), tuTMP(nullptr), utTMP(nullptr), outersections(nullptr),
      outersectionsTMP(nullptr), keys(nullptr),
      startIndices(nullptr), areas(nullptr)
{
    cudaFree(0);
    cudaMalloc((void**)&positions, 2 * size * sizeof(double));
    cudaMalloc((void**)&forces, size * 2 * sizeof(double));

    cudaMalloc(&maxActualNeighbors, sizeof(int));
    int init = INT_MIN;
    cudaMemcpy(maxActualNeighbors, &init, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&globalState, sizeof(curandState) * size);
    setModelEnum(simControl.modelType);
}

Model::~Model() {
    if (positions) cudaFree(positions);
    if (forces) cudaFree(forces);
    if (maxActualNeighbors) cudaFree(maxActualNeighbors);
    if (globalState) cudaFree(globalState);    
    // From initializeNeighborCells
    if (countPerBox) cudaFree(countPerBox);
    if (boxId) cudaFree(boxId);
    if (cellLocation) cudaFree(cellLocation);
    if (neighborIndices) cudaFree(neighborIndices);
    if (neighbors) cudaFree(neighbors);
    if (contacts) cudaFree(contacts);
    if (inside) cudaFree(inside);
    if (shapeId) cudaFree(shapeId);
    if (numNeighbors) cudaFree(numNeighbors);
    if (numContacts) cudaFree(numContacts);
    if (valid) cudaFree(valid);
    if (outputIdx) cudaFree(outputIdx);
    if (shapeCounts) cudaFree(shapeCounts);
    if (intersections) cudaFree(intersections);
    if (tu) cudaFree(tu);
    if (ut) cudaFree(ut);
    if (tuTMP) cudaFree(tuTMP);
    if (utTMP) cudaFree(utTMP);
    if (keys) cudaFree(keys);
    if (outersections) cudaFree(outersections);
    if (outersectionsTMP) cudaFree(outersectionsTMP);
    if (next) cudaFree(next);
    if (prev) cudaFree(prev);
    // From setStartIndices
    if (areas) cudaFree(areas);
    if (perimeters) cudaFree(perimeters);
    if (startIndices) cudaFree(startIndices);
    
    // From updateOverlapArea
    if (intersectionsCounter) cudaFree(intersectionsCounter);
}

// resetters

void Model::resetMaxActualNeighbors() {
    int init = INT_MIN;
    cudaMemcpy(maxActualNeighbors, &init, sizeof(int), cudaMemcpyHostToDevice);
}

void Model::deallocateAll() {
    cudaFree(positions);
//    delete [] C;
}

// initializers

void Model::initializeNeighborCells() {
    int numBoxes = boxSize * boxSize;
    cudaMalloc((void**)&countPerBox, numBoxes * sizeof(int));
    cudaMalloc((void**)&boxId, numBoxes * sizeof(int));

    cudaMalloc((void**)&cellLocation, size * sizeof(int));
    cudaMalloc((void**)&neighborIndices, size * sizeof(int));
    cudaMalloc((void**)&neighbors, maxNeighbors * size * sizeof(int));
    cudaMalloc((void**)&contacts, maxNeighbors * size * sizeof(int));
    cudaMalloc((void**)&inside, maxNeighbors * size * sizeof(bool));

    cudaMalloc((void**)&shapeId, size * sizeof(int));
    updateShapeIdCUDA(shapeId, startIndices, size, numPolygons);
    cudaMalloc(&next, size * sizeof(int));
    cudaMalloc(&prev, size * sizeof(int));
    computeNextPrevCUDA(next, prev, startIndices, shapeId, size);    

    cudaMalloc((void**)&numNeighbors, size * sizeof(int));
    cudaMalloc((void**)&numContacts, size * sizeof(int));
    
    cudaMalloc((void**)&valid, maxNeighbors * size * sizeof(int));
    cudaMalloc((void**)&outputIdx, maxNeighbors * size * sizeof(uint64_t));
    cudaMalloc((void**)&shapeCounts, numPolygons * sizeof(int));

    cudaMalloc((void**)&intersections, maxNeighbors * size * sizeof(uint64_t));
    cudaMalloc((void**)&tu, maxNeighbors * size * sizeof(float2));
    cudaMalloc((void**)&ut, maxNeighbors * size * sizeof(float2));
    cudaMalloc((void**)&tuTMP, maxNeighbors * size * sizeof(float2));
    cudaMalloc((void**)&utTMP, maxNeighbors * size * sizeof(float2));
        
    cudaMalloc((void**)&outersections, maxNeighbors * size * sizeof(uint64_t));
    cudaMalloc((void**)&outersectionsTMP, maxNeighbors * size * sizeof(uint64_t));
    cudaMalloc((void**)&keys, maxNeighbors * size * sizeof(uint32_t));
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
        cudaMemcpy(keys_.data(), keys, numIntersections * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    return keys_;
}

// updaters

void Model::updateNeighborCells() {
    updateNeighborCellsCUDA(positions, startIndices, shapeId, numPolygons, size, boxSize, cellLocation, countPerBox, boxId, boxesUsed, neighborIndices);
}

void Model::updateNeighbors(double a) {
    // The neighbors array has a size maxNeighbors * numVertices
    // numNeighbors is an array of size numVertices that says how many neighbors
    // are in each
    // first attempt
    int newActualNeighbors = updateNeighborsCUDA(shapeId, startIndices, positions, cellLocation,
                              neighborIndices, size, neighbors, numNeighbors,
                              maxNeighbors, boxSize, countPerBox, a, maxActualNeighbors);
    if (newActualNeighbors > maxNeighbors) {
        // read required max from device
        // warn the user
        std::cerr << "Warning: neighbor buffer overflow: maxNeighbors=" << maxNeighbors
                  << " required=" << newActualNeighbors << ". Resizing and retrying.\n";

        // resize neighbors buffer to accommodate required value (at least hostMaxActual)
        int newMax = max(maxNeighbors * 2 + 1, newActualNeighbors);
        cudaFree(neighbors);
        cudaFree(inside);
        cudaFree(valid);
        cudaFree(outputIdx);
        cudaFree(intersections);
        cudaFree(tu);
        cudaFree(outersections);
        cudaFree(ut);
        cudaFree(tuTMP);
        cudaFree(outersectionsTMP);
        cudaFree(utTMP);
        cudaFree(contacts);
        cudaFree(keys);
        cudaMalloc((void**)&neighbors, newMax * size * sizeof(int));
        cudaMalloc((void**)&inside, newMax * size * sizeof(bool));
        cudaMalloc((void**)&valid, newMax * size * sizeof(int));
        cudaMalloc((void**)&outputIdx, newMax * size * sizeof(uint64_t));
        cudaMalloc((void**)&intersections, newMax * size * sizeof(uint64_t));
        cudaMalloc((void**)&tu, newMax * size * sizeof(float2));
        cudaMalloc((void**)&outersections, newMax * size * sizeof(uint64_t));
        cudaMalloc((void**)&ut, newMax * size * sizeof(float2));
        cudaMalloc((void**)&tuTMP, newMax * size * sizeof(float2));
        cudaMalloc((void**)&outersectionsTMP, newMax * size * sizeof(uint64_t));
        cudaMalloc((void**)&utTMP, newMax * size * sizeof(float2));
        cudaMalloc((void**)&contacts, newMax * size * sizeof(int));
        cudaMalloc((void**)&keys, newMax * size * sizeof(uint32_t));
        maxNeighbors = newMax;

        // retry once
        int ok = updateNeighborsCUDA(shapeId, startIndices, positions, cellLocation,
                                      neighborIndices, size, neighbors, numNeighbors,
                                      maxNeighbors, boxSize, countPerBox, a, maxActualNeighbors);
        if (ok > maxNeighbors) {
            std::cerr << "Warning: updateNeighbors still failed after resizing to " << maxNeighbors << "\n";
        }
    }
    resetMaxActualNeighbors();
}

void Model::updateContacts() {
    // The neighbors array has a size maxNeighbors * numVertices
    // numNeighbors is an array of size numVertices that says how many neighbors
    // are in each
    // first attempt    
    updateContactsCUDA(shapeId, startIndices, positions, 
        size, neighbors, numNeighbors, maxNeighbors, inside, tu, contacts, numContacts);
}

void Model::updateValidAndCounts() {
    numIntersections = updateValidAndCountsCUDA(size, contacts, numContacts, maxNeighbors, inside, shapeId, numPolygons, valid, shapeCounts, outputIdx);
}

void Model::updateAreas() {
    cudaMemset(areas, 0, numPolygons * sizeof(double));
    updateAreasCUDA(areas, positions, startIndices, numPolygons);
}

void Model::updatePerimeters() {
    cudaMemset(perimeters, 0, numPolygons * sizeof(double));
    updatePerimetersCUDA(perimeters, positions, startIndices, numPolygons);
}

void Model::updateCompactedIntersections() {
    updateCompactedIntersectionsCUDA(size, maxNeighbors, contacts, inside, shapeId, startIndices, valid, outputIdx, intersections, numIntersections, tu);
}

void Model::updateOutersections() {
    numIntersections = updateValidAndCountsCUDA(size, contacts, numContacts, maxNeighbors, inside, shapeId, numPolygons, valid, shapeCounts, outputIdx);
    updateCompactedIntersectionsCUDA(size, maxNeighbors, contacts, inside, shapeId, startIndices, valid, outputIdx, intersections, numIntersections, tu);
    sortKeysCUDA(intersections, numIntersections, 0, 64, keys);
    applyPermutationCUDA_float2(tu, keys, tuTMP, numIntersections);
    updateOutersectionsCUDA(intersections, tuTMP, utTMP, startIndices, numIntersections, outersectionsTMP);
    sortKeysCUDA(intersections, numIntersections, 0, 48, keys);
    applyPermutationCUDA_float2(tuTMP, keys, tu, numIntersections);
    applyPermutationCUDA_float2(utTMP, keys, ut, numIntersections);
    applyPermutationCUDA_int64(outersectionsTMP, keys, outersections, numIntersections);
}

// setters

void Model::setMaxEdgeLength(double maxEdgeLength_) {
    maxEdgeLength = maxEdgeLength_;
    boxSize = floor(1.0 / maxEdgeLength);
}

void Model::setModelEnum(simControlStruct::modelEnum modelType_) {
    simControl.modelType = modelType_;
}

void Model::setPositions(const vector<double>& positionsData) {
    cudaMemcpy(positions, positionsData.data(), 2 * size * sizeof(double), cudaMemcpyHostToDevice);
}

void Model::setStartIndices(const vector<int>& startIndicesData) {
    numPolygons = startIndicesData.size() - 1;
    cudaMalloc((void**)&areas, numPolygons * sizeof(double));
    cudaMalloc((void**)&perimeters, numPolygons * sizeof(double));
    cudaMalloc((void**)&startIndices, (numPolygons + 1) * sizeof(int));
    cudaMemcpy(startIndices, startIndicesData.data(), (numPolygons + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

void Model::setNumVertices(int numVertices_) {
    size = numVertices_;
}

// getters

string Model::getModelEnum() const {
    switch (simControl.modelType) {
        case simControlStruct::modelEnum::normal:   return "normal";
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

vector<int> Model::getShapeId() const {
    vector<int> shapeId_(size);
    cudaMemcpy(shapeId_.data(), shapeId, size * sizeof(int), cudaMemcpyDeviceToHost);
    return shapeId_;
}

vector<double> Model::getPositions() const {
    vector<double> positions_(2 * size);
    cudaMemcpy(positions_.data(), positions, 2 * size * sizeof(double), cudaMemcpyDeviceToHost);
    return positions_;
}

vector<int> Model::getIntersectionsCounter() const {
    vector<int> intersectionsCounter_(pointDensity * pointDensity);
    cudaMemcpy(intersectionsCounter_.data(), intersectionsCounter, pointDensity * pointDensity * sizeof(int), cudaMemcpyDeviceToHost);
    return intersectionsCounter_;
}

double Model::getMaxEdgeLength() const {
    return maxEdgeLength;
}

vector<int> Model::getNeighbors() const {
    vector<int> neighbors_(maxNeighbors * size);
    cudaMemcpy(neighbors_.data(), neighbors, maxNeighbors * size * sizeof(int), cudaMemcpyDeviceToHost);
    return neighbors_;
}

vector<int> Model::getContacts() const {
    vector<int> contacts_(maxNeighbors * size);
    cudaMemcpy(contacts_.data(), contacts, maxNeighbors * size * sizeof(int), cudaMemcpyDeviceToHost);
    return contacts_;
}

vector<bool> Model::getInsideFlag() const {
    size_t n = static_cast<size_t>(maxNeighbors) * static_cast<size_t>(size);
    if (n == 0) return vector<bool>();

    // temporary contiguous buffer that matches device layout (bytes)
    vector<unsigned char> tmp(n);
    // copy from device (device buffer 'inside' was allocated with sizeof(bool))
    cudaMemcpy(tmp.data(), inside, n * sizeof(bool), cudaMemcpyDeviceToHost);

    // convert to vector<bool>
    vector<bool> inside_(n);
    for (size_t i = 0; i < n; ++i) inside_[i] = tmp[i] ? true : false;
    return inside_;
}

vector<double> Model::getTU() const {
    if (numIntersections == 0) return vector<double>();
    // temporary contiguous buffer that matches device layout (bytes)
    vector<float2> tu_(numIntersections);
    cudaMemcpy(tu_.data(), tu, numIntersections * sizeof(float2), cudaMemcpyDeviceToHost);
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
    vector<float2> ut_(numIntersections);
    cudaMemcpy(ut_.data(), ut, numIntersections * sizeof(float2), cudaMemcpyDeviceToHost);
    vector<double> sol(2 * numIntersections);
    for (int i = 0; i < numIntersections; i++) {
        sol[2 * i] = static_cast<double>(ut_[i].x);
        sol[2 * i + 1] = static_cast<double>(ut_[i].y);
    }
    return sol;
}

vector<int> Model::getNumNeighbors() const {
    vector<int> numNeighbors_(size);
    cudaMemcpy(numNeighbors_.data(), numNeighbors, size * sizeof(int), cudaMemcpyDeviceToHost);
    return numNeighbors_;
}

vector<int> Model::getNumContacts() const {
    vector<int> numContacts_(size);
    cudaMemcpy(numContacts_.data(), numContacts, size * sizeof(int), cudaMemcpyDeviceToHost);
    return numContacts_;
}

vector<int> Model::getStartIndices() const {
    vector<int> startIndices_(numPolygons + 1);
    cudaMemcpy(startIndices_.data(), startIndices, (numPolygons + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    return startIndices_;
}

vector<double> Model::getAreas() const {
    vector<double> areas_(numPolygons);
    cudaMemcpy(areas_.data(), areas, numPolygons * sizeof(double), cudaMemcpyDeviceToHost);
    return areas_;
}

vector<double> Model::getPerimeters() const {
    vector<double> perimeters_(numPolygons);
    cudaMemcpy(perimeters_.data(), perimeters, numPolygons * sizeof(double), cudaMemcpyDeviceToHost);
    return perimeters_;
}

vector<int> Model::getNeighborCells() const {
    vector<int> neighborCells_(size);
    cudaMemcpy(neighborCells_.data(), cellLocation, size * sizeof(int), cudaMemcpyDeviceToHost);
    return neighborCells_;
}

vector<int> Model::getNeighborIndices() const {
    vector<int> neighborIndices_(size);
    cudaMemcpy(neighborIndices_.data(), neighborIndices, size * sizeof(int), cudaMemcpyDeviceToHost);
    return neighborIndices_;
}

vector<int> Model::getBoxCounts() const {
    vector<int> countPerBox_(boxSize * boxSize);
    cudaMemcpy(countPerBox_.data(), countPerBox, boxSize * boxSize * sizeof(int), cudaMemcpyDeviceToHost);
    return countPerBox_;
}

vector<double> Model::getForces() const {
    vector<double> forces_(size * 2);
    cudaMemcpy(forces_.data(), forces, size * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    return forces_;
}

void Model::updateOverlapArea(int pointDensity_) {
    // allocate or reallocate the device-side counter buffer if density changed
    if (pointDensity != pointDensity_) {
        if (intersectionsCounter != nullptr) {
            cudaFree(intersectionsCounter);
            intersectionsCounter = nullptr;
        }
        size_t total = static_cast<size_t>(pointDensity_) * static_cast<size_t>(pointDensity_);
        cudaMalloc((void**)&intersectionsCounter, total * sizeof(int));
        // remember new density
        pointDensity = pointDensity_;
    }

    // ensure the buffer is zeroed (CUDA kernel may overwrite but zeroing is cheap)
    size_t total = static_cast<size_t>(pointDensity) * static_cast<size_t>(pointDensity);
    cudaMemset(intersectionsCounter, 0, total * sizeof(int));

    // call CUDA routine that computes and returns the raw sum of intersectionsCounter entries
    updateOverlapAreaCUDA(
        shapeId,
        startIndices,
        pointDensity,
        intersectionsCounter,
        neighborIndices,
        size,
        boxSize,
        countPerBox,
        positions,
        overlapArea
    );

    // normalize to fraction of sampled points -> overlap area estimate in [0,1]
    overlapArea /= static_cast<double>(pointDensity * pointDensity);
}

double Model::getOverlapArea() const {
    return overlapArea;
}

vector<int> Model::getShapeCounts() const {
    vector<int> shapeCounts_(numPolygons);
    cudaMemcpy(shapeCounts_.data(), shapeCounts, numPolygons * sizeof(int), cudaMemcpyDeviceToHost);
    return shapeCounts_;
}

vector<uint64_t> Model::getIntersections() const {
    vector<uint64_t> intersections_(numIntersections);
    if (numIntersections > 0) {
        cudaMemcpy(intersections_.data(), intersections, numIntersections * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    return intersections_;
}

int Model::getNumIntersections() const {
    return numIntersections;
}

vector<uint64_t> Model::getOutersections() const {
    vector<uint64_t> outersections_(numIntersections);
    if (numIntersections > 0) {
        cudaMemcpy(outersections_.data(), outersections, numIntersections * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    return outersections_;
}