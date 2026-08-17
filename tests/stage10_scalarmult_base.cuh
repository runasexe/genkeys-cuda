#ifndef STAGE10_SCALARMULT_BASE_CUH
#define STAGE10_SCALARMULT_BASE_CUH

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"

static void print_hex_bytes_scalarmult(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

__global__ void test_stage10_kernel(int* d_result, uint8_t* d_pub_out) {
    const uint8_t golden_clamped_scalar[32] = {
        0x58, 0x51, 0x41, 0x64, 0xda, 0x42, 0x62, 0x19,
        0xe5, 0xba, 0x42, 0xab, 0xeb, 0xb9, 0x1c, 0x5d,
        0xe5, 0xc0, 0xcb, 0x6d, 0xe5, 0x3a, 0xac, 0xc1,
        0x6e, 0xa4, 0x37, 0x0a, 0x38, 0x31, 0x0b, 0x46
    };

    ge_scalarmult_base(d_pub_out, golden_clamped_scalar);

    const uint8_t expected_pub[32] = {
        0xc3, 0x82, 0x59, 0xd8, 0x96, 0x1a, 0x21, 0xf7,
        0x39, 0xed, 0xa2, 0x7d, 0x4c, 0xf1, 0x6b, 0xce,
        0x8f, 0x57, 0x3f, 0xae, 0xff, 0x70, 0xf4, 0xca,
        0xbc, 0x9b, 0xfb, 0xa7, 0x6f, 0xa6, 0x49, 0x17
    };

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (d_pub_out[i] != expected_pub[i]) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage10_scalarmult_base(bool verbose) {
    if (verbose) std::cout << "[TEST 10] Base Scalar Multiplication (ge_scalarmult_base): ";
    int *d_res, h_res = 0;
    uint8_t *d_pub, h_pub[32];
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pub, 32));

    test_stage10_kernel<<<1, 1>>>(d_res, d_pub);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pub, d_pub, 32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_pub));

    if (h_res) {
        if (verbose) std::cout << "PASS" << std::endl;
        return true;
    } else {
        std::cout << "FAIL" << std::endl;
        std::cout << "  Computed Public Key: "; print_hex_bytes_scalarmult(h_pub, 32);
        return false;
    }
}

#endif // STAGE10_SCALARMULT_BASE_CUH
