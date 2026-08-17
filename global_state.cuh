#ifndef GLOBAL_STATE_CUH
#define GLOBAL_STATE_CUH

#include <stdint.h>
#include <vector>
#include <atomic>
#include <iostream>
#include <iomanip>
#include "config.cuh"

// Struct for storing key match results
struct FoundKey {
    uint32_t zeros;
    uint8_t seed[32];
    uint8_t pubkey[32];
};

// Global state tracking search progress and results
struct GlobalState {
    std::atomic<uint64_t> total_keys_checked{0};
    uint32_t current_best_zeros;
    uint32_t max_best_zeros;
    std::vector<FoundKey> best_keys;

    GlobalState(uint32_t initial_min_zeros = 32, uint32_t initial_max_zeros = 0) : current_best_zeros(initial_min_zeros), max_best_zeros(initial_max_zeros) {}

    uint32_t get_best_zeros() const {
        return current_best_zeros;
    }

    bool check_and_update(uint32_t zeros, const FoundKey& key) {
        if (zeros > current_best_zeros) {
            if (max_best_zeros > 0 && zeros > max_best_zeros) {
                zeros = max_best_zeros;
            }
            current_best_zeros = zeros;
            best_keys.push_back(key);
            return true;
        } else if (zeros == current_best_zeros) {
            best_keys.push_back(key);
            return true;
        }
        return false;
    }

    void add_checked_keys(uint64_t count) {
        total_keys_checked.fetch_add(count, std::memory_order_relaxed);
    }
};

#endif // GLOBAL_STATE_CUH
