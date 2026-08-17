#ifndef STAGE9_MIXED_ADDITION_CUH
#define STAGE9_MIXED_ADDITION_CUH

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

static void print_hex_bytes_mixed_add(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

__global__ void test_stage9_kernel(int* d_result, uint8_t* d_madd_x, uint8_t* d_madd_y) {
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

    ge_p3 h;
    fe_0(h.X);
    fe_1(h.Y);
    fe_1(h.Z);
    fe_0(h.T);

    ge_precomp t0;
    select_precomp(&t0, 0, 1);
    ge_madd(&h, &h, &t0);

    fe Z_inv, final_x, final_y;
    fe_invert(Z_inv, h.Z);
    fe_mul(final_x, h.X, Z_inv);
    fe_mul(final_y, h.Y, Z_inv);
    fe_tobytes(d_madd_x, final_x);
    fe_tobytes(d_madd_y, final_y);

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (d_madd_x[i] != B_X_bytes[i] || d_madd_y[i] != B_Y_bytes[i]) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage9_mixed_addition(bool verbose) {
    if (verbose) std::cout << "[TEST  9] Mixed Addition Primitive (ge_madd: Neutral + base[0][0] = Base): ";
    int *d_res, h_res = 0;
    uint8_t *d_x, *d_y, h_x[32], h_y[32];
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, 32));
    CUDA_CHECK(cudaMalloc(&d_y, 32));

    test_stage9_kernel<<<1, 1>>>(d_res, d_x, d_y);
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
        std::cout << "  Got X: "; print_hex_bytes_mixed_add(h_x, 32);
        std::cout << "  Got Y: "; print_hex_bytes_mixed_add(h_y, 32);
        return false;
    }
}

#endif // STAGE9_MIXED_ADDITION_CUH
