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

// Declaration of pointwiseMultiply function (defined in model.cu)
extern "C" void initializeRandomStates(curandState *globalState, unsigned long long seed, int gridSize);
extern "C" void updateAreasCUDA(double* areas, double* positions, int* startIndices, int numShapes);
extern "C" void updateNeighborCellsCUDA(double* positions, int* startIndices, int* shapeId, int numShapes, int size, int boxSize, int* cellLocation, int* countPerBox, int* boxId, int& boxesUsed, int* neighborIndices);
extern "C" void updateShapeIdCUDA(int* shapeId, int* startIndices, int size, int numShapes);
extern "C" void updateNeighborsCUDA(int* shapeId, int* startIndices, double* positions, int* cellLocation, int* neighborIndices, int size, int* neighbors, int* numNeighbors, int maxNeighbors, int boxSize, int* countPerBox, double a);

// Constructor
Model::Model(int size_) : size(size_) {
    cudaFree(0);
    cudaMalloc((void**)&positions, 2 * size * sizeof(double));

    cudaMalloc((void**)&globalState, sizeof(curandState) * size);
    setModelEnum(simControl.modelType);
}

void Model::deallocateAll() {
    cudaFree(positions);
//    delete [] C;
}

void Model::setMaxEdgeLength(double maxEdgeLength_) {
    maxEdgeLength = maxEdgeLength_;
    boxSize = ceil(1.0 / maxEdgeLength);
}

void Model::initializeNeighborCells() {
    int numBoxes = boxSize * boxSize;
    cudaMalloc((void**)&countPerBox, numBoxes * sizeof(int));
    cudaMalloc((void**)&boxId, numBoxes * sizeof(int));

    cudaMalloc((void**)&cellLocation, size * sizeof(int));
    cudaMalloc((void**)&neighborIndices, size * sizeof(int));
    cudaMalloc((void**)&neighbors, maxNeighbors * size * sizeof(int));

    cudaMalloc((void**)&shapeId, size * sizeof(int));
    updateShapeIdCUDA(shapeId, startIndices, size, numShapes);

    cudaMalloc((void**)&numNeighbors, size * sizeof(int));
}

void Model::updateNeighborCells() {
    updateNeighborCellsCUDA(positions, startIndices, shapeId, numShapes, size, boxSize, cellLocation, countPerBox, boxId, boxesUsed, neighborIndices);
}

void Model::updateNeighbors(double a) {
    updateNeighborsCUDA(shapeId, startIndices, positions, cellLocation, neighborIndices, size, neighbors, numNeighbors, maxNeighbors, boxSize, countPerBox, a);
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
    numShapes = startIndicesData.size() - 1;
    cudaMalloc((void**)&areas, numShapes * sizeof(double));
    cudaMalloc((void**)&startIndices, (numShapes + 1) * sizeof(double));
    cudaMemcpy(startIndices, startIndicesData.data(), (numShapes + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

// Function to return the result matrix
int Model::getNumVertices() const {
    return size;
}

unsigned long long Model::getRandomSeed() {
    return seed;
}

vector<double> Model::getPositions() const {
    vector<double> positions_(2 * size);
    cudaMemcpy(positions_.data(), positions, 2 * size * sizeof(double), cudaMemcpyDeviceToHost);
    return positions_;
}

double Model::getMaxEdgeLength() const {
    return maxEdgeLength;
}

vector<int> Model::getNeighbors() const {
    vector<int> neighbors_(maxNeighbors * size);
    cudaMemcpy(neighbors_.data(), neighbors, maxNeighbors * size * sizeof(int), cudaMemcpyDeviceToHost);
    return neighbors_;
}

vector<int> Model::getNumNeighbors() const {
    vector<int> numNeighbors_(size);
    cudaMemcpy(numNeighbors_.data(), numNeighbors, size * sizeof(int), cudaMemcpyDeviceToHost);
    return numNeighbors_;
}

vector<int> Model::getStartIndices() const {
    vector<int> startIndices_(numShapes + 1);
    cudaMemcpy(startIndices_.data(), startIndices, (numShapes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    return startIndices_;
}

void Model::updateAreas() {
    updateAreasCUDA(areas, positions, startIndices, numShapes);
}

vector<double> Model::getAreas() const {
    vector<double> areas_(numShapes);
    cudaMemcpy(areas_.data(), areas, numShapes * sizeof(double), cudaMemcpyDeviceToHost);
    return areas_;
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

