#ifndef STAGE6_SHA512_CUH
#define STAGE6_SHA512_CUH

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../sha512_cuda.cuh"

static void print_hex_bytes_sha(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

__global__ void test_stage6_kernel(int* d_result, uint8_t* d_sha_out) {
    const uint8_t seed_test[32] = {
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60,
        0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c, 0xc4,
        0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x91,
        0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae, 0x7f, 0x60
    };
    sha512_32bytes(seed_test, d_sha_out);

    const uint8_t expected_sha_prefix[32] = {
        0x5d, 0x51, 0x41, 0x64, 0xda, 0x42, 0x62, 0x19,
        0xe5, 0xba, 0x42, 0xab, 0xeb, 0xb9, 0x1c, 0x5d,
        0xe5, 0xc0, 0xcb, 0x6d, 0xe5, 0x3a, 0xac, 0xc1,
        0x6e, 0xa4, 0x37, 0x0a, 0x38, 0x31, 0x0b, 0x86
    };

    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (d_sha_out[i] != expected_sha_prefix[i]) ok = false;
    }
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage6_sha512(bool verbose) {
    if (verbose) std::cout << "[TEST  6] SHA-512 Core Hash Verification (sha512_32bytes): ";
    int *d_res, h_res = 0;
    uint8_t *d_sha, h_sha[64];
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sha, 64));

    test_stage6_kernel<<<1, 1>>>(d_res, d_sha);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sha, d_sha, 64, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_sha));

    if (h_res) {
        if (verbose) std::cout << "PASS" << std::endl;
        return true;
    } else {
        std::cout << "FAIL" << std::endl;
        std::cout << "  Got SHA-512 prefix: "; print_hex_bytes_sha(h_sha, 32);
        return false;
    }
}

#endif // STAGE6_SHA512_CUH
