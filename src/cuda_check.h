#pragma once
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstdlib>

// Checks a CUDA API call and aborts on error.
// Use for all CUDA calls outside of destructors.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Checks a CUDA API call and prints on error without aborting.
// Use inside destructors where exit() would skip other cleanup.
#define CUDA_CHECK_NOABORT(call)                                                  \
    do {                                                                          \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error (non-fatal) at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(_err));                 \
        }                                                                         \
    } while (0)

#define CUSOLVER_CHECK(call)                                                      \
    do {                                                                          \
        cusolverStatus_t _err = (call);                                           \
        if (_err != CUSOLVER_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuSolver error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, (int)_err);                               \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Checks the last kernel launch error and aborts on error.
// Call immediately after a <<<>>> kernel launch.
#define CUDA_CHECK_KERNEL()                                                       \
    do {                                                                          \
        cudaError_t _err = cudaGetLastError();                                    \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(_err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)
