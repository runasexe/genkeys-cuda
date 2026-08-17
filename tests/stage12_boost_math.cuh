#ifndef STAGE12_BOOST_MATH_CUH
#define STAGE12_BOOST_MATH_CUH

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../ed25519_cuda.cuh"
#include "../add_scalar.cuh"

__global__ void test_stage12_kernel(int* d_result) {
    // 1. Initial clamped scalar
    uint8_t start_az[32] = {
        0x58, 0x51, 0x41, 0x64, 0xda, 0x42, 0x62, 0x19,
        0xe5, 0xba, 0x42, 0xab, 0xeb, 0xb9, 0x1c, 0x5d,
        0xe5, 0xc0, 0xcb, 0x6d, 0xe5, 0x3a, 0xac, 0xc1,
        0x6e, 0xa4, 0x37, 0x0a, 0x38, 0x31, 0x0b, 0x46 // bit 254 is set, bit 0,1,2 are 0
    };

    // 2. Start point
    ge_p3 P;
    ge_scalarmult_base(&P, start_az);

    // 3. Fast addition loop (1000 iterations of +8G)
    const int ITERS = 1000;
    const ge_precomp* p8G = &base[0][7];
    for (int i = 0; i < ITERS; ++i) {
        ge_p1p1 r;
        ge_madd(&r, &P, p8G);
        ge_p1p1_to_p3(&P, &r);
    }
    
    uint8_t fast_pubkey[32];
    ge_p3_tobytes(fast_pubkey, &P);

    // 4. Exact full scalar multiplication
    uint8_t exact_az[32];
    add_scalar256_out(exact_az, start_az, 8ULL * ITERS);
    
    ge_p3 exact_P;
    ge_scalarmult_base(&exact_P, exact_az);
    uint8_t exact_pubkey[32];
    ge_p3_tobytes(exact_pubkey, &exact_P);

    // 5. Compare
    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (fast_pubkey[i] != exact_pubkey[i]) {
            ok = false;
        }
    }
    
    d_result[0] = ok ? 1 : 0;
}

inline bool test_stage12_boost_math(bool verbose) {
    if (verbose) std::cout << "[TEST 12] Extreme Boost Math (ge_madd loop vs scalarmult): ";
    int *d_res, h_res = 0;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));

    test_stage12_kernel<<<1, 1>>>(d_res);
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

#endif // STAGE12_BOOST_MATH_CUH
