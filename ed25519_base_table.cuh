#ifndef ED25519_BASE_TABLE_CUH
#define ED25519_BASE_TABLE_CUH

#include "ed25519_cuda_ge.h"

// Place 60KB precomputed base table in GPU global read-only memory (.rodata)
__device__ static const ge_precomp base[32][8] = {
#include "base_table.inc"
};

__device__ inline void select_precomp(ge_precomp *r, int pos, int val) {
    if (val > 0) {
        const ge_precomp *p = &base[pos][val - 1];
        for (int i = 0; i < 10; ++i) {
            r->yplusx[i] = p->yplusx[i];
            r->yminusx[i] = p->yminusx[i];
            r->xy2d[i] = p->xy2d[i];
        }
    } else if (val < 0) {
        const ge_precomp *p = &base[pos][-val - 1];
        for (int i = 0; i < 10; ++i) {
            r->yplusx[i] = p->yminusx[i];
            r->yminusx[i] = p->yplusx[i];
            r->xy2d[i] = -p->xy2d[i];
        }
    } else {
        for (int i = 0; i < 10; ++i) {
            r->yplusx[i] = (i == 0) ? 1 : 0;
            r->yminusx[i] = (i == 0) ? 1 : 0;
            r->xy2d[i] = 0;
        }
    }
}

#endif // ED25519_BASE_TABLE_CUH
