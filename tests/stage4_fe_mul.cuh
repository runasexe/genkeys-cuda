#ifndef STAGE4_FE_MUL_CUH
#define STAGE4_FE_MUL_CUH

#include <iostream>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

__global__ void test_stage4_mul_kernel(int* d_result, uint8_t* d_out) {
    const uint8_t B_Y_bytes[32] = {
        0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66
    };
    const uint8_t five_bytes[32] = {
        0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };

    fe by, five, one, res_five, res_one;
    fe_frombytes(by, B_Y_bytes);
    fe_frombytes(five, five_bytes);
    fe_1(one);

    // Test 1: 5 * (4/5) == 4
    fe_mul(res_five, by, five);
    fe_tobytes(d_out, res_five);

    // Test 2: (4/5) * 1 == 4/5
    fe_mul(res_one, by, one);
    uint8_t out_one[32];
    fe_tobytes(out_one, res_one);

    bool ok = (d_out[0] == 4);
    for (int i = 1; i < 32; ++i) {
        if (d_out[i] != 0) ok = false;
    }
    for (int i = 0; i < 32; ++i) {
        if (out_one[i] != B_Y_bytes[i]) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage4_fe_mul(bool verbose) {
    if (verbose) std::cout << "[TEST  4] Field Multiplication (fe_mul): ";
    int *d_res, h_res = 0;
    uint8_t *d_out;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, 32));

    test_stage4_mul_kernel<<<1, 1>>>(d_res, d_out);
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

#endif // STAGE4_FE_MUL_CUH
