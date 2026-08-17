#ifndef YGGDRASIL_CUDA_CUH
#define YGGDRASIL_CUDA_CUH

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <array>
#include <string>
#include <sstream>

// Convert Ed25519 public key to Yggdrasil IPv6 address
inline std::string pubkey_to_yggdrasil_ipv6(const uint8_t pubkey[32], uint8_t zeros) {
    std::array<uint8_t, 16> addr = {0};
    
    // 1. Set Yggdrasil address prefix (0x02)
    addr[0] = 0x02; 
    
    // 2. Set leading zero bit count
    addr[1] = zeros; 
    
    // 3. Pack remaining inverted public key bits into IPv6 address bytes
    int skip_bits = zeros + 1; 
    int dest_byte_idx = 2;
    uint8_t current_byte = 0;
    int bits_in_current_byte = 0;

    for (int bit_idx = skip_bits; bit_idx < 256 && dest_byte_idx < 16; ++bit_idx) {
        int byte_pos = bit_idx / 8;
        int bit_pos = 7 - (bit_idx % 8);
        uint8_t orig_bit = (pubkey[byte_pos] >> bit_pos) & 1;
        
        // Invert bit per Yggdrasil address specification
        uint8_t inv_bit = orig_bit ^ 1;

        current_byte = (current_byte << 1) | inv_bit;
        bits_in_current_byte++;

        if (bits_in_current_byte == 8) {
            addr[dest_byte_idx++] = current_byte;
            current_byte = 0;
            bits_in_current_byte = 0;
        }
    }
    
    // 4. Canonical IPv6 string formatting (RFC 5952)
    uint16_t words[8];
    for (int i = 0; i < 8; ++i) {
        words[i] = (addr[i * 2] << 8) | addr[i * 2 + 1];
    }

    // Find longest sequence of consecutive zero 16-bit words
    int best_start = -1, best_len = 0;
    int cur_start = -1, cur_len = 0;
    for (int i = 0; i < 8; ++i) {
        if (words[i] == 0) {
            if (cur_start == -1) cur_start = i;
            cur_len++;
        } else {
            if (cur_len > best_len) {
                best_len = cur_len;
                best_start = cur_start;
            }
            cur_start = -1;
            cur_len = 0;
        }
    }
    if (cur_len > best_len) {
        best_len = cur_len;
        best_start = cur_start;
    }
    // RFC 5952: do not compress a single 16-bit zero word
    if (best_len < 2) {
        best_start = -1;
    }

    // Build formatted IPv6 string
    std::ostringstream oss;
    oss << std::hex;
    
    int i = 0;
    while (i < 8) {
        if (i == best_start) {
            oss << "::";
            i += best_len; // Skip compressed zero words
        } else {
            // Append colon separator if not at start and preceding block was not "::"
            if (i > 0 && i != best_start + best_len) {
                oss << ":";
            }
            oss << words[i];
            i++;
        }
    }
    
    return oss.str();
}

#endif // YGGDRASIL_CUDA_CUH
