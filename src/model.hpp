#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>
#include <complex>
#include <math.h>
#include <curand_kernel.h>
#include "enumTypes.h"

class Model {
public:
    Model(int size);             // Constructor declaration
    void setNumVertices(int size);
    void setPositions(const std::vector<double>& positions_);
    const int getNumVertices() const;
    std::vector<double> getPositions() const;
    void setModelEnum(simControlStruct::modelEnum modelType_);
    void deallocateAll();
    void initializeRandomSeed(const unsigned long long seed_);
    unsigned long long getRandomSeed();

private:
    simControlStruct simControl;  // Instance of simControlStruct
    int size;
    unsigned long long seed;
    curandState* globalState;
    double* positions;
};

#endif
