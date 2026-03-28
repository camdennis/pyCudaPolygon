#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>
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
    void setEdgeLengths(const vector<double>& edgeLengths_);
    void setStiffness(const double stiffness_);

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
    vector<double> getNorm2() const;
    double getEnergy() const;
    vector<double> getEdgeLengths() const;
    vector<double> getConstraintForces() const;
    double getMaxUnbalancedForce() const;
    vector<double> getConstraints() const;
    vector<double> getProjection() const;
    double getOverlapArea() const;

    // updaters

    void updateAreas();
    void updateNeighborCells();
    void updateNeighbors();
    void updatePerimeters();
    void updateOverlapArea(int pointDensity_);
    void updateIntersectionsCounter();
    void updateValidAndCounts();
    void updateOutersections();
    void updateCompactedIntersections();
    void updateForceEnergy();
    void updatePositions(double dt);
    void updateConstraintForces();

private:
    simControlStruct simControl;
    int size, numPolygons;
    unsigned long long seed;
    curandState* globalState;
    double* energy;
    double* positions;
    int* startIndices;
    int* startDOF, *endDOF;
    double* areas;
    double* edgeLengths;
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
    double stiffness;
    int* shapeStart;
    int* shapeEnd;
    double* constraints, *constraintsTMP;
    double* norm2, *norm2TMP;
    size_t norm2TMPStorageBytes = 0;
    double* proj;
    double* constraintForce;
    double maxUnbalancedForce;
};

#endif
