#ifndef ED25519_CUDA_CUH
#define ED25519_CUDA_CUH

#include <stdint.h>

#include "sha512_cuda.cuh"
#include "ed25519_cuda_fe.h"
#include "ed25519_cuda_fe.cu"
#include "ed25519_cuda_ge.h"
#include "ed25519_base_table.cuh"
#include "ed25519_cuda_ge.cu"

__device__ inline void ge_p3_add(ge_p3 *r, const ge_p3 *p1, const ge_p3 *p2) {
    ge_cached q2;
    ge_p1p1 p1p1;
    ge_p3_to_cached(&q2, p2);
    ge_add(&p1p1, p1, &q2);
    ge_p1p1_to_p3(r, &p1p1);
}

__device__ inline void ge_p3_dbl(ge_p3 *r, const ge_p3 *p) {
    ge_p1p1 t;
    ge_p3_dbl(&t, p);
    ge_p1p1_to_p3(r, &t);
}

__device__ inline void ge_madd(ge_p3 *r, const ge_p3 *p, const ge_precomp *q) {
    ge_p1p1 t;
    ge_madd(&t, p, q);
    ge_p1p1_to_p3(r, &t);
}

__device__ inline void ge_scalarmult_base(uint8_t *pk, const uint8_t *a) {
    ge_p3 h;
    ge_scalarmult_base(&h, a);
    ge_p3_tobytes(pk, &h);
}

__device__ inline void ed25519_create_public_key(uint8_t *public_key, const uint8_t *private_key) {
    uint8_t az[64];
    sha512_32bytes(private_key, az);

    az[0] &= 248;
    az[31] &= 127;
    az[31] |= 64;

    ge_scalarmult_base(public_key, az);
}

#endif // ED25519_CUDA_CUH
