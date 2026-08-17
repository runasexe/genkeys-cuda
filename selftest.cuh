#ifndef SELFTEST_CUH
#define SELFTEST_CUH

#include <iostream>

#include "tests/stage1_field_arithmetic.cuh"
#include "tests/stage2_fe_add.cuh"
#include "tests/stage3_fe_sub.cuh"
#include "tests/stage4_fe_mul.cuh"
#include "tests/stage5_field_encoding.cuh"
#include "tests/stage6_sha512.cuh"
#include "tests/stage7_point_addition.cuh"
#include "tests/stage8_point_doubling.cuh"
#include "tests/stage9_mixed_addition.cuh"
#include "tests/stage10_scalarmult_base.cuh"
#include "tests/stage11_create_public_key.cuh"
#include "tests/stage12_boost_math.cuh"

inline bool run_full_selftest(bool verbose = false) {
    if (verbose) {
        std::cout << "[*] Running diagnostic self-tests..." << std::endl;
    }

    bool pass1  = test_stage1_field_arithmetic(verbose);
    bool pass2  = test_stage2_fe_add(verbose);
    bool pass3  = test_stage3_fe_sub(verbose);
    bool pass4  = test_stage4_fe_mul(verbose);
    bool pass5  = test_stage5_field_encoding(verbose);
    bool pass6  = test_stage6_sha512(verbose);
    bool pass7  = test_stage7_point_addition(verbose);
    bool pass8  = test_stage8_point_doubling(verbose);
    bool pass9  = test_stage9_mixed_addition(verbose);
    bool pass10 = test_stage10_scalarmult_base(verbose);
    bool pass11 = test_stage11_create_public_key(verbose);
    bool pass12 = test_stage12_boost_math(verbose);

    bool all_passed = pass1 && pass2 && pass3 && pass4 && pass5 && pass6 && pass7 && pass8 && pass9 && pass10 && pass11 && pass12;

    if (all_passed) {
        std::cout << "[+] All diagnostic selftests passed!" << std::endl;
    } else {
        std::cerr << "[!] ERROR: One or more selftest stages FAILED!" << std::endl;
    }

    return all_passed;
}

#endif // SELFTEST_CUH
