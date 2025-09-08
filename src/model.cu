#include <cuda_runtime.h>
#include <iostream>
#include <cufft.h>
#include <complex>
#include <curand_kernel.h>
#include <float.h>

static const dim3 myBlockDim(16, 16);
static const int blockSize = 256;


__global__ void initStatesKernel(curandState *globalState, unsigned long long seed, int gridSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Flatten 2D index to 1D index
    int idx = row * gridSize + col;
    if (row < gridSize && col < gridSize) {
       curand_init(seed, idx, 0, &globalState[idx]);
    }
}

extern "C" void initializeRandomStates(curandState *globalState, unsigned long long int seed, int gridSize) {
    dim3 gridDim((gridSize + myBlockDim.x - 1) / myBlockDim.x, (gridSize + myBlockDim.y - 1) / myBlockDim.y);
    initStatesKernel<<<gridDim, myBlockDim>>>(globalState, seed, gridSize);
    cudaDeviceSynchronize();
}