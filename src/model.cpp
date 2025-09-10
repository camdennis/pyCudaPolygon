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

void Model::setModelEnum(simControlStruct::modelEnum modelType_) {
    simControl.modelType = modelType_;
}

void Model::initializeRandomSeed(const unsigned long long seed_) {
    seed = seed_;
    initializeRandomStates(globalState, seed, size);
}

void Model::setPositions(const vector<double>& positionsData) {
    cudaMemcpy(positions, positionsData.data(), 2 * size * sizeof(double), cudaMemcpyHostToDevice);
}

// Function to return the result matrix
const int Model::getNumVertices() const {
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

void Model::setNumVertices(int numVertices_) {
    size = numVertices_;
}