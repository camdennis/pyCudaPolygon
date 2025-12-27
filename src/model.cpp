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
extern "C" void initializeRandomStates(curandState *globalState, unsigned long long seed, int gridSize);
extern "C" void updateAreasCUDA(double* areas, double* positions, int* startIndices, int numPolygons);
extern "C" void updateNeighborCellsCUDA(double* positions, int* startIndices, int* shapeId, int numPolygons, int size, int boxSize, int* cellLocation, int* countPerBox, int* boxId, int& boxesUsed, int* neighborIndices);
extern "C" void updateShapeIdCUDA(int* shapeId, int* startIndices, int size, int numPolygons);
extern "C" int updateNeighborsCUDA(int* shapeId, int* startIndices, double* positions, int* cellLocation, int* neighborIndices, int size, int* neighbors, int* numNeighbors, int maxNeighbors, int boxSize, int* countPerBox, double a, int* maxActualNeighbors, bool* inside, double* t, double* u);
extern "C" void updatePerimetersCUDA(double* perimeters, double* positions, int* startIndices, int numPolygons);
extern "C" void updateOverlapAreaCUDA(
    int* shapeId,
    int* startIndices,
    int pointDensity,
    int* intersectionCounter,
    int* neighborIndices,
    int size,
    int boxSize,
    int* countPerBox,
    double* positions,
    double& overlapArea
);

// Constructor
Model::Model(int size_) : size(size_) {
    cudaFree(0);
    cudaMalloc((void**)&positions, 2 * size * sizeof(double));
    cudaMalloc((void**)&forces, size * 2 * sizeof(double));

    cudaMalloc(&maxActualNeighbors, sizeof(int));
    int init = INT_MIN;
    cudaMemcpy(maxActualNeighbors, &init, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&globalState, sizeof(curandState) * size);
    setModelEnum(simControl.modelType);
}

void Model::resetMaxActualNeighbors() {
    int init = INT_MIN;
    cudaMemcpy(maxActualNeighbors, &init, sizeof(int), cudaMemcpyHostToDevice);
}

void Model::deallocateAll() {
    cudaFree(positions);
//    delete [] C;
}

void Model::setMaxEdgeLength(double maxEdgeLength_) {
    maxEdgeLength = maxEdgeLength_;
    boxSize = floor(1.0 / maxEdgeLength);
}

void Model::initializeNeighborCells() {
    int numBoxes = boxSize * boxSize;
    cudaMalloc((void**)&countPerBox, numBoxes * sizeof(int));
    cudaMalloc((void**)&boxId, numBoxes * sizeof(int));

    cudaMalloc((void**)&cellLocation, size * sizeof(int));
    cudaMalloc((void**)&neighborIndices, size * sizeof(int));
    cudaMalloc((void**)&neighbors, maxNeighbors * size * sizeof(int));
    cudaMalloc((void**)&inside, maxNeighbors * size * sizeof(bool));

    cudaMalloc((void**)&t, maxNeighbors * size * sizeof(double));
    cudaMalloc((void**)&u, maxNeighbors * size * sizeof(double));

    cudaMalloc((void**)&shapeId, size * sizeof(int));
    updateShapeIdCUDA(shapeId, startIndices, size, numPolygons);

    cudaMalloc((void**)&numNeighbors, size * sizeof(int));
}

void Model::updateNeighborCells() {
    updateNeighborCellsCUDA(positions, startIndices, shapeId, numPolygons, size, boxSize, cellLocation, countPerBox, boxId, boxesUsed, neighborIndices);
}

void Model::updateNeighbors(double a) {
    // first attempt
    int newActualNeighbors = updateNeighborsCUDA(shapeId, startIndices, positions, cellLocation,
                              neighborIndices, size, neighbors, numNeighbors,
                              maxNeighbors, boxSize, countPerBox, a, maxActualNeighbors, inside, t, u);
    if (newActualNeighbors > maxNeighbors) {
        // read required max from device
        // warn the user
        std::cerr << "Warning: neighbor buffer overflow: maxNeighbors=" << maxNeighbors
                  << " required=" << newActualNeighbors << ". Resizing and retrying.\n";

        // resize neighbors buffer to accommodate required value (at least hostMaxActual)
        int newMax = max(maxNeighbors * 2 + 1, newActualNeighbors);
        cudaFree(neighbors);
        cudaFree(inside);
        cudaFree(t);
        cudaFree(u);
        cudaMalloc((void**)&neighbors, newMax * size * sizeof(int));
        cudaMalloc((void**)&inside, newMax * size * sizeof(bool));
        cudaMalloc((void**)&t, newMax * size * sizeof(bool));
        cudaMalloc((void**)&u, newMax * size * sizeof(bool));
        maxNeighbors = newMax;

        // retry once
        int ok = updateNeighborsCUDA(shapeId, startIndices, positions, cellLocation,
                                      neighborIndices, size, neighbors, numNeighbors,
                                      maxNeighbors, boxSize, countPerBox, a, maxActualNeighbors, inside, t, u);
        if (ok > maxNeighbors) {
            std::cerr << "Warning: updateNeighbors still failed after resizing to " << maxNeighbors << "\n";
        }
    }

    // reset device-side max tracker for next run
//    resetMaxActualNeighbors();
}

void Model::setModelEnum(simControlStruct::modelEnum modelType_) {
    simControl.modelType = modelType_;
}

string Model::getModelEnum() const {
    // Return a human-friendly name if you have known enum values,
    // otherwise fall back to the numeric value.
    // If you want specific names, replace the numeric fallback with a switch
    // mapping simControl.modelType to "normal"/"abnormal", etc.
    return std::to_string(static_cast<int>(simControl.modelType));
}

void Model::initializeRandomSeed(const unsigned long long seed_) {
    seed = seed_;
    initializeRandomStates(globalState, seed, size);
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

// Function to return the result matrix
int Model::getNumVertices() const {
    return size;
}

// Function to return the result matrix
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
    size_t n = static_cast<size_t>(maxNeighbors) * static_cast<size_t>(size);
    if (n == 0) return vector<double>();
    // temporary contiguous buffer that matches device layout (bytes)
    vector<double> tu_(2 * n);
    cudaMemcpy(tu_.data(), t, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tu_.data() + n, u, n * sizeof(double), cudaMemcpyDeviceToHost);
    return tu_;
}

vector<int> Model::getNumNeighbors() const {
    vector<int> numNeighbors_(size);
    cudaMemcpy(numNeighbors_.data(), numNeighbors, size * sizeof(int), cudaMemcpyDeviceToHost);
    return numNeighbors_;
}

vector<int> Model::getStartIndices() const {
    vector<int> startIndices_(numPolygons + 1);
    cudaMemcpy(startIndices_.data(), startIndices, (numPolygons + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    return startIndices_;
}

void Model::updateAreas() {
    cudaMemset(areas, 0, numPolygons * sizeof(double));
    updateAreasCUDA(areas, positions, startIndices, numPolygons);
}

void Model::updatePerimeters() {
    cudaMemset(perimeters, 0, numPolygons * sizeof(double));
    updatePerimetersCUDA(perimeters, positions, startIndices, numPolygons);
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

void Model::setNumVertices(int numVertices_) {
    size = numVertices_;
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

// The neighbor cells are ordered so we can loop over the 
// polygons in a cell. For each point, you count the
// number of polygons it's in.
// NOTE: This function modifies device-side intersectionsCounter and returns
//       a normalized overlap area (fraction of points in overlapping polygons).
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

    // call CUDA routine that computes and returns the raw sum of intersectionCounter entries
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