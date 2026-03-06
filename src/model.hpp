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
    Model(int size_);
    ~Model();

    // initializers

    void initializeNeighborCells();
    void initializeRandomSeed(const unsigned long long seed_);

    // deallocators

    void deallocateAll();

    // helpers

    void sortKeys(int endBit);

    // setters

    void setNumVertices(int size);
    void setPositions(const vector<double>& positions_);
    void setStartIndices(const vector<int>& startIndices_);
    void setModelEnum(simControlStruct::modelEnum modelType_);
    void setMaxEdgeLength(double maxEdgeLength);

    // getters

    int getNumVertices() const;
    int getNumPolygons() const;
    vector<int> getShapeId() const;
    vector<double> getPositions() const;
    vector<int> getStartIndices() const;
    vector<int> getNeighborCells() const;
    vector<int> getBoxCounts() const;
    vector<int> getNeighborIndices() const;
    vector<int> getNumNeighbors() const;
    vector<int> getNumContacts() const;
    vector<int> getNeighbors() const;
    vector<int> getContacts() const;
    double getMaxEdgeLength() const;
    string getModelEnum() const;
    vector<double> getForces() const;
    void resetMaxActualNeighbors();
    vector<bool> getInsideFlag() const;
    vector<double> getPerimeters() const;
    double getOverlapArea() const;
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

    // updaters

    void updateAreas();
    void updateNeighborCells();
    void updateNeighbors(double a);
    void updateContacts();
    void updatePerimeters();
    void updateOverlapArea(int pointDensity_);
    void updateIntersectionsCounter();
    void updateValidAndCounts();
    void updateOutersections();
    void updateCompactedIntersections();
    void updateForceEnergy();

private:
    simControlStruct simControl;
    int size, numPolygons;
    unsigned long long seed;
    curandState* globalState;
    double* energy;
    double* positions;
    int* startIndices;
    double* areas;
    int* countPerBox;
    int* boxId;
    int* neighborIndices;
    int* cellLocation;
    int* shapeId;
    int* neighbors;
    int* contacts;
    int boxesUsed;
    int maxNeighbors = 100;
    int boxSize;
    int* numNeighbors;
    int* numContacts;
    bool updateMaxNeighbors = false;
    double maxEdgeLength;
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
    float2* tu, *tuTMP, *ut, *utTMP;
    int numIntersections = 0;
    uint64_t* outersections, *outersectionsTMP;
    uint32_t* keys;
    int* next, *prev;
};

#endif
