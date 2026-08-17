#ifndef GPU_SEED_GEN_CUH
#define GPU_SEED_GEN_CUH

#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_ROTL32(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define CUDA_QUARTER_ROUND(a, b, c, d) \
    a += b; d ^= a; d = CUDA_ROTL32(d, 16); \
    c += d; b ^= c; b = CUDA_ROTL32(b, 12); \
    a += b; d ^= a; d = CUDA_ROTL32(d, 8);  \
    c += d; b ^= c; b = CUDA_ROTL32(b, 7);

__device__ inline void chacha8_block_gpu(const uint32_t input_state[16], uint8_t output_bytes32[32]) {
    uint32_t x[16];
    
    // Load input state into thread registers
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        x[i] = input_state[i];
    }

    // 4 double rounds = 8 rounds total (ChaCha8)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Column round
        CUDA_QUARTER_ROUND(x[0], x[4], x[8],  x[12]);
        CUDA_QUARTER_ROUND(x[1], x[5], x[9],  x[13]);
        CUDA_QUARTER_ROUND(x[2], x[6], x[10], x[14]);
        CUDA_QUARTER_ROUND(x[3], x[7], x[11], x[15]);

        // Diagonal round
        CUDA_QUARTER_ROUND(x[0], x[5], x[10], x[15]);
        CUDA_QUARTER_ROUND(x[1], x[6], x[11], x[12]);
        CUDA_QUARTER_ROUND(x[2], x[7], x[8],  x[13]);
        CUDA_QUARTER_ROUND(x[3], x[4], x[9],  x[14]);
    }

    // Add input state back to working state
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        x[i] += input_state[i];
    }

    // Write first 32 bytes (8 words) to output
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        reinterpret_cast<uint32_t*>(output_bytes32)[i] = x[i];
    }
}

#endif // GPU_SEED_GEN_CUH
