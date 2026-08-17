#ifndef STAGE7_POINT_ADDITION_CUH
#define STAGE7_POINT_ADDITION_CUH

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

static void print_hex_bytes_point_add(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

__global__ void test_stage7_kernel(int* d_result, uint8_t* d_add_x, uint8_t* d_add_y) {
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

    ge_p3 B;
    fe_frombytes(B.X, B_X_bytes);
    fe_frombytes(B.Y, B_Y_bytes);
    fe_1(B.Z);
    fe_mul(B.T, B.X, B.Y);

    ge_p3 Neutral;
    fe_0(Neutral.X);
    fe_1(Neutral.Y);
    fe_1(Neutral.Z);
    fe_0(Neutral.T);

    ge_p3 R_add;
    ge_p3_add(&R_add, &Neutral, &B);

    fe Z_inv, final_x, final_y;
    fe_invert(Z_inv, R_add.Z);
    fe_mul(final_x, R_add.X, Z_inv);
    fe_mul(final_y, R_add.Y, Z_inv);
    fe_tobytes(d_add_x, final_x);
    fe_tobytes(d_add_y, final_y);

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (d_add_x[i] != B_X_bytes[i] || d_add_y[i] != B_Y_bytes[i]) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage7_point_addition(bool verbose) {
    if (verbose) std::cout << "[TEST  7] Extended Point Addition (ge_p3_add): ";
    int *d_res, h_res = 0;
    uint8_t *d_x, *d_y, h_x[32], h_y[32];
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, 32));
    CUDA_CHECK(cudaMalloc(&d_y, 32));

    test_stage7_kernel<<<1, 1>>>(d_res, d_x, d_y);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_x, d_x, 32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, 32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    if (h_res) {
        if (verbose) std::cout << "PASS" << std::endl;
        return true;
    } else {
        std::cout << "FAIL" << std::endl;
        std::cout << "  Got X: "; print_hex_bytes_point_add(h_x, 32);
        std::cout << "  Got Y: "; print_hex_bytes_point_add(h_y, 32);
        return false;
    }
}

#endif // STAGE7_POINT_ADDITION_CUH
