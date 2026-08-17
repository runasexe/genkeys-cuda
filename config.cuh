#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "[!] CUDA ERROR: " << cudaGetErrorString(err) \
                      << " (" << #call << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "[!] CUDA KERNEL LAUNCH ERROR: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

#define MAX_RESULTS_PER_BATCH 64

#endif // CONFIG_CUH
