#ifndef ADD_SCALAR_CUH
#define ADD_SCALAR_CUH

#include <stdint.h>
#include <cuda_runtime.h>

// Adds a 64-bit integer to a 256-bit little-endian integer in-place.
// The base is modified.
__host__ __device__ inline void add_scalar256(uint8_t scalar[32], uint64_t addend) {
    uint64_t carry = addend;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (carry == 0) return;
        uint64_t sum = (uint64_t)scalar[i] + (carry & 0xFF);
        scalar[i] = (uint8_t)(sum & 0xFF);
        carry = (carry >> 8) + (sum >> 8);
    }
}

// Same as above but returns the result in a separate array out.
__host__ __device__ inline void add_scalar256_out(uint8_t out[32], const uint8_t base[32], uint64_t addend) {
    uint64_t carry = addend;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        uint64_t sum = (uint64_t)base[i] + (carry & 0xFF);
        out[i] = (uint8_t)(sum & 0xFF);
        carry = (carry >> 8) + (sum >> 8);
    }
}

#endif // ADD_SCALAR_CUH
