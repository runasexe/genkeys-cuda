#ifndef GENKEYS_CUDA_CUH
#define GENKEYS_CUDA_CUH

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#include "config.cuh"
#include "global_state.cuh"

#include "cpu_seed_gen.cuh"
#include "gpu_seed_gen.cuh"

#include "sha512_cuda.cuh"
#include "ed25519_cuda.cuh"

#include "yggdrasil_cuda.cuh"
#include "add_scalar.cuh"

__constant__ ge_precomp constant_8G;

// Device function to count leading zero bits in a 32-byte public key
__device__ inline uint32_t count_leading_zero_bits(const uint8_t pubkey[32]) {
    uint32_t total_zeros = 0;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(pubkey);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = __byte_perm(p[i], 0, 0x0123);
        if (w == 0) {
            total_zeros += 32;
        } else {
            total_zeros += __clz(w);
            break;
        }
    }
    return total_zeros;
}

// CUDA Kernel for Ed25519 vanity key search
__global__ void vanity_search_kernel(
    const uint8_t* __restrict__ base_seed,
    uint32_t min_zeros,
    FoundKey* __restrict__ results,
    uint32_t* __restrict__ result_count,
    uint32_t max_results,
    uint64_t total_threads
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    uint32_t state[16];

    // 1. ChaCha constants ("expand 32-byte k", Words 0-3)
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;

    // 2. Load 32-byte base seed for this batch (Words 4-11)
    const uint32_t* seed_ptr = reinterpret_cast<const uint32_t*>(base_seed);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[4 + i] = seed_ptr[i];
    }

    // 3. Worker ID (uint64_t) split into two 32-bit words (Words 12-13)
    state[12] = static_cast<uint32_t>(tid);
    state[13] = static_cast<uint32_t>(tid >> 32);

    // 4. Zero padding for remaining state words (Words 14-15)
    state[14] = 0;
    state[15] = 0;

    // Construct 32-byte candidate seed for this thread
    uint8_t thread_seed[32];
    chacha8_block_gpu(state, thread_seed);

    // Derive Ed25519 public key
    uint8_t pubkey[32];
    ed25519_create_public_key(pubkey, thread_seed);

    // Count leading zero bits in generated public key
    uint32_t zeros = count_leading_zero_bits(pubkey);

    // Filter results matching minimum threshold using atomic operations
    if (zeros >= min_zeros) {
        uint32_t idx = atomicAdd(result_count, 1);
        if (idx < max_results) {
            results[idx].zeros = zeros;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                results[idx].seed[k] = thread_seed[k];
                results[idx].pubkey[k] = pubkey[k];
            }
        }
    }
}

// Returns true if candidate cannot meet the leading zero threshold (enabling early exit)
__device__ inline bool should_early_exit(const fe h, uint32_t min_zeros) {
    if (min_zeros < 8) return false;

    // 1. Compute carry chain q (11 branchless instructions)
    int32_t q = (19 * h[9] + (((int32_t) 1) << 24)) >> 25;
    q = (h[0] + q) >> 26;
    q = (h[1] + q) >> 25;
    q = (h[2] + q) >> 26;
    q = (h[3] + q) >> 25;
    q = (h[4] + q) >> 26;
    q = (h[5] + q) >> 25;
    q = (h[6] + q) >> 26;
    q = (h[7] + q) >> 25;
    q = (h[8] + q) >> 26;
    q = (h[9] + q) >> 25;

    // 2. Compute exact h0 before high-bit reduction
    int32_t h0 = h[0] + 19 * q;

    // --- Fast low-byte verification (without full fe_tobytes serialization) ---

    // Check bytes 0-1 (16 bits)
    if (min_zeros >= 16) {
        // Exit early if lowest 16 bits are non-zero
        if ((h0 & 0xFFFF) != 0) return true; 

        // Check byte 2 (24 bits total)
        if (min_zeros >= 24) {
            if ((h0 & 0xFFFFFF) != 0) return true;

            // Check bytes 3-5 (up to 48 bits)
            if (min_zeros >= 32) {
                // Compute carry0 and h1 to inspect byte 3
                int32_t carry0 = h0 >> 26;
                int32_t h1 = h[1] + carry0;
                
                // Byte 3: s[3] = (h0 >> 24) | (h1 << 2)
                uint32_t s3 = ((h0 >> 24) | (h1 << 2)) & 0xFF;
                if (s3 != 0) return true;

                if (min_zeros >= 40) {
                    // Byte 4: s[4] = h1 >> 6
                    uint32_t s4 = (h1 >> 6) & 0xFF;
                    if (s4 != 0) return true;

                    if (min_zeros >= 48) {
                        // Byte 5: s[5] = h1 >> 14
                        uint32_t s5 = (h1 >> 14) & 0xFF;
                        if (s5 != 0) return true;
                    }
                }
            }
        }
    } else {
        // Check byte 0 only (8 bits)
        if ((h0 & 0xFF) != 0) return true;
    }

    // Leading low bytes matched zero; proceed with full candidate evaluation
    return false;
}

#define BATCH_SIZE 1024

__global__ void vanity_search_boost_kernel(
    const uint8_t* __restrict__ base_scalar,
    uint32_t min_zeros,
    FoundKey* __restrict__ results,
    uint32_t* __restrict__ result_count,
    uint32_t max_results,
    uint64_t total_threads,
    uint32_t iters_per_thread
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    uint8_t k_thread[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) k_thread[i] = base_scalar[i];

    uint64_t addend = 8ULL * tid * iters_per_thread;
    add_scalar256(k_thread, addend);

    ge_p3 P;
    ge_scalarmult_base(&P, k_thread);

    for (uint32_t i = 0; i < iters_per_thread; i += BATCH_SIZE) {
        fe Z_arr[BATCH_SIZE];
        fe Y_arr[BATCH_SIZE];
        
        for (int j = 0; j < BATCH_SIZE; ++j) {
            fe_copy(Y_arr[j], P.Y);
            fe_copy(Z_arr[j], P.Z);
            
            ge_p1p1 r;
            ge_madd(&r, &P, &constant_8G);
            ge_p1p1_to_p3(&P, &r);
        }

        fe acc[BATCH_SIZE];
        fe_copy(acc[0], Z_arr[0]);

        for (int j = 1; j < BATCH_SIZE; ++j) {
            fe_mul(acc[j], acc[j - 1], Z_arr[j]);
        }

        fe inv;
        fe_invert(inv, acc[BATCH_SIZE - 1]);

        for (int j = BATCH_SIZE - 1; j >= 1; --j) {
            fe z_inv_j;
            fe_mul(z_inv_j, inv, acc[j - 1]);
            fe_mul(inv, inv, Z_arr[j]);
            fe_copy(Z_arr[j], z_inv_j);
        }
        fe_copy(Z_arr[0], inv);

        for (int j = 0; j < BATCH_SIZE; ++j) {
            fe y;
            // Multiply affine-equivalent Y by inverted Z (computed in-place via Montgomery batch inversion)
            fe_mul(y, Y_arr[j], Z_arr[j]);

            if (should_early_exit(y, min_zeros)) {
                continue;
            }
            
            alignas(4) uint8_t pubkey[32];
            fe_tobytes(pubkey, y);
            
            // Ignore sign bit for fast zero counting
            pubkey[31] &= 0x7F;

            if (i + j < iters_per_thread) {
                uint32_t zeros = count_leading_zero_bits(pubkey);
                if (zeros >= min_zeros) {
                    uint32_t idx = atomicAdd(result_count, 1);
                    if (idx < max_results) {
                        results[idx].zeros = zeros;
                        uint8_t exact_scalar[32];
                        add_scalar256_out(exact_scalar, k_thread, 8ULL * (i + j));
                        
                        #pragma unroll
                        for (int k = 0; k < 32; ++k) {
                            results[idx].seed[k] = exact_scalar[k];
                        }
                        
                        ge_p3 exact_P;
                        ge_scalarmult_base(&exact_P, exact_scalar);
                        ge_p3_tobytes(results[idx].pubkey, &exact_P);
                    }
                }
            }
        }
    }
}

static void print_hex(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    std::cout << std::dec;
}

// Host loop executing vanity key generation and search
inline void run_vanity_search(
    int device_id,
    GlobalState& global_state,
    uint64_t max_iterations,
    uint32_t blocks,
    uint32_t threads_per_block,
    uint32_t max_results,
    uint64_t summary,
    bool summary_newline,
    bool address,
    bool vanity_mode,
    uint32_t iters_per_thread
) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    // Dynamic grid sizing based on SM count and max threads per SM
    const uint32_t num_sm = prop.multiProcessorCount;
    if (threads_per_block == 0) {
        threads_per_block = 256;
    }
    if (blocks == 0) {
        const uint32_t max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        const uint32_t blocks_per_sm = max_threads_per_sm / threads_per_block;
        blocks = num_sm * (blocks_per_sm > 0 ? blocks_per_sm : 2);
    }
    const uint64_t total_threads = (uint64_t)blocks * threads_per_block;
    if (max_results == 0) {
        max_results = 256;
    }

    std::cout << "[*] Starting CUDA vanity key search" << std::endl;
    std::cout << "[*] Device:          " << prop.name << " (" << num_sm << " SMs)" << std::endl;
    std::cout << "[*] Grid Config:     " << blocks << " blocks x " << threads_per_block << " threads" << std::endl;
    std::cout << "[*] Batch Size:      " << total_threads << " threads (" << (total_threads / 1000000.0) << " Mkeys/batch)" << std::endl;
    std::cout << "[*] Max Iterations:  " << (max_iterations == 0 ? "Infinite (0)" : std::to_string(max_iterations)) << std::endl;
    std::cout << "[*] Target Zeros:    " << global_state.get_best_zeros() << " bits minimum" << std::endl;
    
    if (vanity_mode) {
        std::cout << "[*] Mode:            Extreme Boost Vanity (Chunking, ge_madd)" << std::endl;
        std::cout << "[*] Iters/Thread:    " << iters_per_thread << std::endl;
        
        ge_precomp h_8G;
        CUDA_CHECK(cudaMemcpyFromSymbol(&h_8G, base, sizeof(ge_precomp), 7 * sizeof(ge_precomp), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyToSymbol(constant_8G, &h_8G, sizeof(ge_precomp)));
        
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        std::cout << "[*] Mode:            Standard Seed Search" << std::endl;
    }

    // Allocate GPU buffers
    uint8_t* d_base_seed;
    FoundKey* d_results;
    uint32_t* d_result_count;

    CUDA_CHECK(cudaMalloc(&d_base_seed, 32));
    CUDA_CHECK(cudaMalloc(&d_results, max_results * sizeof(FoundKey)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));

    // Create CUDA Stream for Async Overlapped Operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CPUSeedGenerator seed_gen;

    uint64_t batch_count = 0;
    std::vector<FoundKey> h_results(max_results);

    auto start_time = std::chrono::high_resolution_clock::now();

    while (true) {
        if (max_iterations > 0 && batch_count >= max_iterations) {
            std::cout << "\n[*] Reached maximum iterations limit (" << max_iterations << "). Search finished." << std::endl;
            break;
        }

        uint32_t current_min_zeros = global_state.get_best_zeros();

        if (vanity_mode) {
            uint8_t h_base_scalar[32];
            seed_gen.generate_seed(h_base_scalar);
            h_base_scalar[0] &= 248;
            h_base_scalar[31] &= 127;
            h_base_scalar[31] |= 64;
            
            CUDA_CHECK(cudaMemcpyAsync(d_base_seed, h_base_scalar, 32, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(d_result_count, 0, sizeof(uint32_t), stream));
            
            vanity_search_boost_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_base_seed,
                current_min_zeros,
                d_results,
                d_result_count,
                max_results,
                total_threads,
                iters_per_thread
            );
        } else {
            uint8_t h_base_seed[32] = {0};
            seed_gen.generate_seed(h_base_seed);
            
            CUDA_CHECK(cudaMemcpyAsync(d_base_seed, h_base_seed, 32, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(d_result_count, 0, sizeof(uint32_t), stream));
            
            vanity_search_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_base_seed,
                current_min_zeros,
                d_results,
                d_result_count,
                max_results,
                total_threads
            );
        }
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Retrieve result count
        uint32_t h_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (vanity_mode) {
            global_state.add_checked_keys(total_threads * iters_per_thread);
        } else {
            global_state.add_checked_keys(total_threads);
        }
        batch_count++;

        // Process found keys
        if (h_count > 0) {
            uint32_t fetch_count = (h_count > max_results) ? max_results : h_count;
            CUDA_CHECK(cudaMemcpyAsync(h_results.data(), d_results, fetch_count * sizeof(FoundKey), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (uint32_t i = 0; i < fetch_count; ++i) {
                if (global_state.check_and_update(h_results[i].zeros, h_results[i].seed[0] != 0 ? h_results[i] : h_results[i])) {
                    std::cout << "[+] Found " << h_results[i].zeros << " (0x"
                              << std::hex << std::setw(2) << std::setfill('0') << (int)h_results[i].zeros << std::dec
                              << "): ";
                    
                    if (vanity_mode) {
                        std::cout << "AZ ";
                    }
                    print_hex(h_results[i].seed, 32);
                    std::cout << " ";
                    print_hex(h_results[i].pubkey, 32);
                    if (address) {
                        std::cout << " " << pubkey_to_yggdrasil_ipv6(h_results[i].pubkey, h_results[i].zeros);
                    }
                    std::cout << std::endl;
                }
            }
        }

        // Print progress summary periodically based on user-configured interval
        if ((summary > 0) && (summary == 1 || batch_count % summary == 0 || (max_iterations > 0 && batch_count == max_iterations))) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
            uint64_t searched = global_state.total_keys_checked.load();
            double mkeys_per_sec = (searched / 1000000.0) / elapsed_sec;

            std::cerr << "[*] Summary: " << (searched / 1000000.0) << " Mkeys | "
                      << "Speed: " << std::fixed << std::setprecision(2) << mkeys_per_sec << " Mkeys/s | "
                      << "Best: " << global_state.get_best_zeros() << " bits\r";
            if (summary_newline) {
                std::cerr << std::endl;
            }
            std::cerr << std::flush;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_base_seed));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_result_count));
}

#endif // GENKEYS_CUDA_CUH
