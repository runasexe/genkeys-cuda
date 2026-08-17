#ifndef STAGE1_FIELD_ARITHMETIC_CUH
#define STAGE1_FIELD_ARITHMETIC_CUH

#include <iostream>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

__global__ void test_stage1_kernel(int* d_result) {
    fe one, res, res_inv;
    fe_1(one);
    fe_mul(res, one, one);
    fe_invert(res_inv, one);

    uint8_t bytes_one[32], bytes_inv[32];
    fe_tobytes(bytes_one, res);
    fe_tobytes(bytes_inv, res_inv);

    bool ok = (bytes_one[0] == 1) && (bytes_inv[0] == 1);
    for (int i = 1; i < 32; ++i) {
        if (bytes_one[i] != 0 || bytes_inv[i] != 0) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage1_field_arithmetic(bool verbose) {
    if (verbose) std::cout << "[TEST  1] Field Arithmetic (1*1=1 & Inverse): ";
    int *d_res, h_res = 0;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    test_stage1_kernel<<<1, 1>>>(d_res);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));

    if (h_res) {
        if (verbose) std::cout << "PASS" << std::endl;
        return true;
    } else {
        std::cout << "FAIL" << std::endl;
        return false;
    }
}

#endif // STAGE1_FIELD_ARITHMETIC_CUH
