#ifndef STAGE8_POINT_DOUBLING_CUH
#define STAGE8_POINT_DOUBLING_CUH

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

static void print_hex_bytes_point_dbl(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

__global__ void test_stage8_kernel(int* d_result, uint8_t* d_dbl_x, uint8_t* d_dbl_y) {
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

    const uint8_t exp_2B_X[32] = {
        0x0e, 0xce, 0x43, 0x28, 0x4e, 0xa1, 0xc5, 0x83,
        0x5f, 0xa4, 0xd7, 0x15, 0x45, 0x8e, 0x0d, 0x08,
        0xac, 0xe7, 0x33, 0x18, 0x7d, 0x3b, 0x04, 0x3d,
        0x6c, 0x04, 0x5a, 0x9f, 0x4c, 0x38, 0xab, 0x36
    };
    const uint8_t exp_2B_Y[32] = {
        0xc9, 0xa3, 0xf8, 0x6a, 0xae, 0x46, 0x5f, 0x0e,
        0x56, 0x51, 0x38, 0x64, 0x51, 0x0f, 0x39, 0x97,
        0x56, 0x1f, 0xa2, 0xc9, 0xe8, 0x5e, 0xa2, 0x1d,
        0xc2, 0x29, 0x23, 0x09, 0xf3, 0xcd, 0x60, 0x22
    };

    ge_p3 B;
    fe_frombytes(B.X, B_X_bytes);
    fe_frombytes(B.Y, B_Y_bytes);
    fe_1(B.Z);
    fe_mul(B.T, B.X, B.Y);

    ge_p3 B_2;
    ge_p3_dbl(&B_2, &B);

    fe Z_inv, final_x, final_y;
    fe_invert(Z_inv, B_2.Z);
    fe_mul(final_x, B_2.X, Z_inv);
    fe_mul(final_y, B_2.Y, Z_inv);
    fe_tobytes(d_dbl_x, final_x);
    fe_tobytes(d_dbl_y, final_y);

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (d_dbl_x[i] != exp_2B_X[i] || d_dbl_y[i] != exp_2B_Y[i]) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage8_point_doubling(bool verbose) {
    if (verbose) std::cout << "[TEST  8] Point Doubling Primitive (ge_p3_dbl: 2*Base = 2B): ";
    int *d_res, h_res = 0;
    uint8_t *d_x, *d_y, h_x[32], h_y[32];
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, 32));
    CUDA_CHECK(cudaMalloc(&d_y, 32));

    test_stage8_kernel<<<1, 1>>>(d_res, d_x, d_y);
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
        std::cout << "  Got 2B X: "; print_hex_bytes_point_dbl(h_x, 32);
        std::cout << "  Got 2B Y: "; print_hex_bytes_point_dbl(h_y, 32);
        return false;
    }
}

#endif // STAGE8_POINT_DOUBLING_CUH
