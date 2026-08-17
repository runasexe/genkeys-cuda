#ifndef STAGE5_FIELD_ENCODING_CUH
#define STAGE5_FIELD_ENCODING_CUH

#include <iostream>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

__global__ void test_stage5_kernel(int* d_result, uint8_t* d_out_x, uint8_t* d_out_y) {
    const uint8_t B_X_bytes[32] = {
        0x1a, 0xd5, 0x25, 0x8f, 0x60, 0x2d, 0x56, 0xc9,
        0xb2, 0xa7, 0x25, 0x95, 0x60, 0xc7, 0x2c, 0x69,
        0x5c, 0xdc, 0xd6, 0xfd, 0x31, 0xe2, 0xa4, 0xc0,
        0xfe, 0x53, 0x6e, 0xcd, 0xd3, 0x36, 0x69, 0x21
    };
    const uint8_t B_Y_bytes[32] = {
        0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66
    };

    fe bx, by;
    fe_frombytes(bx, B_X_bytes);
    fe_frombytes(by, B_Y_bytes);
    fe_tobytes(d_out_x, bx);
    fe_tobytes(d_out_y, by);

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (d_out_x[i] != B_X_bytes[i] || d_out_y[i] != B_Y_bytes[i]) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage5_field_encoding(bool verbose) {
    if (verbose) std::cout << "[TEST  5] Field Encoding Roundtrip (fe_frombytes <-> fe_tobytes): ";
    int *d_res, h_res = 0;
    uint8_t *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, 32));
    CUDA_CHECK(cudaMalloc(&d_y, 32));

    test_stage5_kernel<<<1, 1>>>(d_res, d_x, d_y);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    if (h_res) {
        if (verbose) std::cout << "PASS" << std::endl;
        return true;
    } else {
        std::cout << "FAIL" << std::endl;
        return false;
    }
}

#endif // STAGE5_FIELD_ENCODING_CUH
