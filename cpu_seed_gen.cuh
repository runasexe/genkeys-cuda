#ifndef CPU_SEED_GEN_CUH
#define CPU_SEED_GEN_CUH

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <sys/random.h>

// 32-bit left rotation
inline uint32_t rotl32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

// ChaCha quarter-round function
#define CHACHA_QUARTER_ROUND(a, b, c, d) \
    a += b; d ^= a; d = rotl32(d, 16); \
    c += d; b ^= c; b = rotl32(b, 12); \
    a += b; d ^= a; d = rotl32(d, 8);  \
    c += d; b ^= c; b = rotl32(b, 7);

// Full ChaCha20 block function (20 rounds) for CPU seed generation
inline void chacha20_block(const uint32_t key[8], uint64_t counter, const uint8_t salt[8], uint8_t output[64]) {
    uint32_t x[16];

    // ChaCha constants ("expand 32-byte k")
    x[0] = 0x61707865;
    x[1] = 0x3320646e;
    x[2] = 0x79622d32;
    x[3] = 0x6b206574;

    // Key (32 bytes / 8 words)
    for (int i = 0; i < 8; ++i) {
        x[4 + i] = key[i];
    }

    // 64-bit counter (split into two 32-bit words) and Salt/Nonce (8 bytes)
    x[12] = static_cast<uint32_t>(counter);
    x[13] = static_cast<uint32_t>(counter >> 32);
    
    std::memcpy(&x[14], salt, 8);

    uint32_t input_state[16];
    std::memcpy(input_state, x, sizeof(x));

    // 10 double rounds = 20 rounds
    for (int i = 0; i < 10; ++i) {
        // Column round
        CHACHA_QUARTER_ROUND(x[0], x[4], x[8],  x[12]);
        CHACHA_QUARTER_ROUND(x[1], x[5], x[9],  x[13]);
        CHACHA_QUARTER_ROUND(x[2], x[6], x[10], x[14]);
        CHACHA_QUARTER_ROUND(x[3], x[7], x[11], x[15]);

        // Diagonal round
        CHACHA_QUARTER_ROUND(x[0], x[5], x[10], x[15]);
        CHACHA_QUARTER_ROUND(x[1], x[6], x[11], x[12]);
        CHACHA_QUARTER_ROUND(x[2], x[7], x[8],  x[13]);
        CHACHA_QUARTER_ROUND(x[3], x[4], x[9],  x[14]);
    }

    // Add input state back to working state
    for (int i = 0; i < 16; ++i) {
        x[i] += input_state[i];
    }
    
    std::memcpy(output, x, 64);
}

class CPUSeedGenerator {
private:
    uint32_t chacha_key[8]; 
    uint64_t iteration_counter;
    uint8_t nonce[8] = {0};
    
    void get_os_entropy(uint32_t* buffer_32) {
        uint8_t temp_buffer[32];
        ssize_t res = getrandom(temp_buffer, 32, 0);
        if (res != 32) {
            throw std::runtime_error("Failed to read entropy");
        }
        
        std::memcpy(buffer_32, temp_buffer, 32);
        std::memset(temp_buffer, 0, sizeof(temp_buffer));
    }

    void reseed() {
        uint32_t fresh_entropy[8];
        get_os_entropy(fresh_entropy);
        
        // XOR fresh entropy with current key
        for (int i = 0; i < 8; i++) {
            chacha_key[i] ^= fresh_entropy[i];
        }
        
        iteration_counter = 0; 
        
        // Securely wipe temporary entropy buffer
        std::memset(fresh_entropy, 0, sizeof(fresh_entropy));
    }

public:
    CPUSeedGenerator() {
        get_os_entropy(chacha_key);
        iteration_counter = 0;
    }

    void generate_seed(uint8_t out_seed32[32]) {
        // Reseed with OS entropy every 2^16 (65536) iterations
        if (iteration_counter >= 65536) {
            reseed();
        }

        uint8_t chacha_output[64];
        
        chacha20_block(chacha_key, iteration_counter++, nonce, chacha_output);

        // Copy first 32 bytes to output seed
        std::memcpy(out_seed32, chacha_output, 32);
        
        // Securely erase local output buffer
        std::memset(chacha_output, 0, sizeof(chacha_output));
    }
};

#endif // CPU_SEED_GEN_CUH