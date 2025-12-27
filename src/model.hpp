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
    int getNumPolygons() const;
    vector<int> getShapeId() const;
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
    void initializeNeighborCells();
    vector<int> getNeighborCells() const;
    vector<int> getBoxCounts() const;
    vector<int> getNeighborIndices() const;
    void updateNeighbors(double a);
    vector<int> getNumNeighbors() const;
    vector<int> getNeighbors() const;
    double getMaxEdgeLength() const;
    string getModelEnum() const;
    vector<double> getForces() const;
    void resetMaxActualNeighbors();
    vector<bool> getInsideFlag() const;
    void updatePerimeters();
    vector<double> getPerimeters() const;

    void updateOverlapArea(int pointDensity_);
    double getOverlapArea() const;
    void updateIntersectionsCounter();
    vector<int> getIntersectionsCounter() const;
    vector<double> getTU() const;

private:
    simControlStruct simControl;  // Instance of simControlStruct
    int size, numPolygons;
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
    double maxEdgeLength;
    double* forces;
    int* maxActualNeighbors;
    bool* inside;
    double* perimeters;
    int pointDensity = -1;
    double overlapArea = 0.0;
    // device buffer for per-sample intersection counts (added)
    int* intersectionsCounter;
    double* t, *u;
};

#endif
