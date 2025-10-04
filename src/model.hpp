#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>
#include <complex>
#include <math.h>
#include <curand_kernel.h>
#include "enumTypes.h"

using namespace std;

class Model {
public:
    Model(int size_);             // Constructor declaration
    void setNumVertices(int size);
    void setPositions(const vector<double>& positions_);
    int getNumVertices() const;
    vector<double> getPositions() const;
    void setStartIndices(const vector<int>& startIndices_);
    vector<int> getStartIndices() const;
    void setModelEnum(simControlStruct::modelEnum modelType_);
    void deallocateAll();
    void initializeRandomSeed(const unsigned long long seed_);
    unsigned long long getRandomSeed();
    void updateAreas();
    void setMaxEdgeLength(double maxEdgeLength);
    vector<double> getAreas() const;
    void updateNeighborCells();
    void updateNeighbors();
    void initializeNeighborCells();
    vector<int> getNeighborCells() const;
    vector<int> getBoxCounts() const;
    vector<int> getNeighborIndices() const;
    void updateNeighbors(double a);

private:
    simControlStruct simControl;  // Instance of simControlStruct
    int size, numShapes;
    unsigned long long seed;
    curandState* globalState;
    double* positions;
    int* startIndices;
    double* areas;
    int* countPerBox; //vector of count per box
    int* boxId;
    int* neighborIndices;
    int* cellLocation;
    int* shapeId;
    int* neighbors;
    int boxesUsed;
    int maxNeighbors = 10;
    int boxSize;
    int* numNeighbors;
    bool updateMaxNeighbors = false;
};

#endif
