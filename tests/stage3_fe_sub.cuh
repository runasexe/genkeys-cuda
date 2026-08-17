#ifndef STAGE3_FE_SUB_CUH
#define STAGE3_FE_SUB_CUH

#include <iostream>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

__global__ void test_stage3_sub_kernel(int* d_result, uint8_t* d_out) {
    const uint8_t B_X_bytes[32] = {
        0x1a, 0xd5, 0x25, 0x8f, 0x60, 0x2d, 0x56, 0xc9,
        0xb2, 0xa7, 0x25, 0x95, 0x60, 0xc7, 0x2c, 0x69,
        0x5c, 0xdc, 0xd6, 0xfd, 0x31, 0xe2, 0xa4, 0xc0,
        0xfe, 0x53, 0x6e, 0xcd, 0xd3, 0x36, 0x69, 0x21
    };

    fe a, zero, diff_self, diff_zero;
    fe_frombytes(a, B_X_bytes);
    fe_0(zero);

    // Test 1: A - A == 0
    fe_sub(diff_self, a, a);
    uint8_t out_zero[32];
    fe_tobytes(out_zero, diff_self);

    // Test 2: A - 0 == A
    fe_sub(diff_zero, a, zero);
    fe_tobytes(d_out, diff_zero);

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (out_zero[i] != 0 || d_out[i] != B_X_bytes[i]) {
            ok = false;
        }
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage3_fe_sub(bool verbose) {
    if (verbose) std::cout << "[TEST  3] Field Subtraction (fe_sub): ";
    int *d_res, h_res = 0;
    uint8_t *d_out;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, 32));

    test_stage3_sub_kernel<<<1, 1>>>(d_res, d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_out));

    if (h_res) {
        if (verbose) std::cout << "PASS" << std::endl;
        return true;
    } else {
        std::cout << "FAIL" << std::endl;
        return false;
    }
}

#endif // STAGE3_FE_SUB_CUH
