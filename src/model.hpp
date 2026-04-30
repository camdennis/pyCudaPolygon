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
    void setTargetEdgeLengths(const vector<double>& targetEdgeLengths_);
    void setTargetAreas(const vector<double>& targetAreas_);
    void setStiffness(const double stiffness_);
    void setCompressibility(const double compressibility_);
    double getStiffness() const;
    double getCompressibility() const;

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
    double getEnergy() const;
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
    void updateNeighbors();
    void updateOverlapArea(int pointDensity_);
    void updateIntersectionsCounter();
    void updateValidAndCounts();
    void updateOutersections();
    void updateCompactedIntersections();
    void updateForceEnergy();
    void updatePositions(double dt);
    void resetVelocities();
    std::tuple<double, double, double, int> minimizeFIREStep(double dt, double alpha, int nPos, double dtMax = 0.1, double alphaStart = 0.1, double fAlpha = 0.99, double fInc = 1.1, double fDec = 0.5, int nMin = 5, int shakeIter = 5);
    std::tuple<double, double, int> minimizeFIRE(double maxForceThreshold, double dtInit, int maxSteps, double dtMax = 0.1, double alphaStart = 0.1, double fAlpha = 0.99, double fInc = 1.1, double fDec = 0.5, int nMin = 5, int shakeIter = 5);

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
