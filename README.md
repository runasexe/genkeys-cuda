# genkeys-cuda

High-performance CUDA implementation for generating **Ed25519 vanity keys** and **Yggdrasil Network IPv6 addresses**, optimized for modern NVIDIA GPUs with dedicated architectural tuning for the **NVIDIA GeForce RTX 2080 Ti** (Turing architecture, `sm_75`).

---

## Overview

`genkeys-cuda` leverages the massive parallel computing capabilities of NVIDIA CUDA GPUs to search for Ed25519 private/public key pairs whose public keys (or derived Yggdrasil IPv6 addresses) contain specific leading zero bit patterns.

### Key Highlights
- **Optimized for RTX 2080 Ti (`sm_75`)**: Custom tuning for 68 Streaming Multiprocessors (SMs), 4,352 CUDA cores, tuned register budgets, L1 cache preference, and 16 KB thread stack configuration.
- **Dual Generation Engines**:
  - **Standard Seed Search (`vanity_search_kernel`)**: Generates cryptographically secure seeds via host ChaCha20 entropy + on-device ChaCha8 expansion and derives full Ed25519 public keys per thread.
  - **Extreme Boost Vanity Mode (`vanity_search_boost_kernel`)**: Uses consecutive point addition (`+8G` jump in twisted Edwards projective space via `ge_madd`), **Montgomery batch coordinate inversion** (amortizing 1,024 field inversions into 1 inversion + multiplications), and **branchless early-exit filters** to achieve extreme search throughput.
- **Yggdrasil Network Support**: Built-in canonical RFC 5952 IPv6 address generation from Ed25519 public keys with leading zero bit counting and `::` zero-run compression.
- **Integrated 12-Stage Self-Test Suite**: Built-in verification testing all mathematical layers (field arithmetic, inversions, SHA-512, RFC 8032 test vectors, point additions, doublings, and boost mode consistency).
- **Asynchronous Execution Pipeline**: Overlapped GPU compute and host-device memory transfers using CUDA streams and pinned/device memory structures.

---

## Architectural & Mathematical Details

### 1. Curve25519 / Ed25519 Field Arithmetic
- Field elements ($\mathbb{F}_{2^{255}-19}$) are represented using 10 limbs in radix $2^{25.5}$ (`int32_t fe[10]`).
- Complete field operations: addition (`fe_add`), subtraction (`fe_sub`), multiplication (`fe_mul`), squaring (`fe_sq`, `fe_sq2`), constant-time selection (`fe_cmov`), and modular inversion via Fermat's Little Theorem ($a^{2^{255}-21} \equiv a^{-1} \pmod{2^{255}-19}$).

### 2. Group Operations on Twisted Edwards Curve
- Extended twisted Edwards coordinates: $-x^2 + y^2 = 1 + d x^2 y^2$ with $d = -121665/121666$.
- Representations: Projective (`ge_p2`), Extended (`ge_p3`), Completed (`ge_p1p1`), Precomputed Duif (`ge_precomp`), and Cached (`ge_cached`).
- Precomputed base point table `base[32][8]` (60 KB) mapped to GPU read-only global memory (`.rodata`) for fast radix-16 window scalar multiplication.

### 3. Extreme Boost Vanity Engine
The boost mode generates consecutive candidates by stepping the clamped private scalar $k$ by 8, which corresponds to adding the constant curve point $8G$:
$$P_{i+1} = P_i + 8G$$
- **Montgomery Batch Inversion**: Instead of performing costly modular inversions ($Z^{-1}$) for each point individually, 1,024 projective coordinates $Z_0, Z_1, \dots, Z_{1023}$ are inverted together in a single batch using only **1 field inversion** and $3(N-1)$ field multiplications.
- **Early-Exit Low-Byte Filter (`should_early_exit`)**: Directly inspects the least significant limb carry chain ($q$ and $h_0$) to discard non-matching candidates in ~11 instructions, bypassing expensive full `fe_tobytes` encoding for over 99.999% of non-matching keys.
- **Exact Scalar Recovery**: When a match meets the threshold, the exact 256-bit scalar is reconstructed and verified against a full `ge_scalarmult_base` derivation before emitting the result.

### 4. Custom Low-Register SHA-512
- Device kernel `sha512_32bytes` processes fixed 32-byte seed buffers with an unrolled 16-word circular register buffer (`W[16]`), preventing register spilling and preserving warp occupancy on Turing SMs.

---

## System Requirements

- **GPU**: NVIDIA CUDA-compatible GPU (RTX 2080 Ti / Turing `sm_75` recommended, Pascal/Ampere/Ada/Hopper supported).
- **CUDA Toolkit**: Version 11.0 or higher (`nvcc`).
- **C++ Compiler**: C++17 compatible host compiler:
  - **Windows**: Microsoft Visual C++ (MSVC 2019 or 2022 via Visual Studio Build Tools).
  - **Linux**: GCC 9+ or Clang 10+.

---

## Compilation

### Linux
```bash
# Build for RTX 2080 Ti (sm_75)
nvcc -O3 -std=c++17 -arch=sm_75 main.cu -o genkeys-cuda

# Build for other architectures (e.g. Ampere RTX 3080/3090 = sm_86, Ada RTX 4090 = sm_89)
nvcc -O3 -std=c++17 -arch=sm_86 main.cu -o genkeys-cuda
```

### Windows (PowerShell / Visual Studio Developer Command Prompt)
```powershell
# Build for RTX 2080 Ti (sm_75)
nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler "/O2 /fp:fast" main.cu -o genkeys-cuda.exe
```

### Supported GPU Architecture Targets
| GPU Generation | Architecture Code | Representative GPUs |
|---|---|---|
| Pascal | `sm_61` | GTX 1080 Ti, GTX 1070 |
| Volta | `sm_70` | Titan V, Tesla V100 |
| **Turing** (Target) | **`sm_75`** | **RTX 2080 Ti, RTX 2080, RTX 2070, GTX 1660** |
| Ampere | `sm_80` / `sm_86` | A100 (`sm_80`), RTX 3090, RTX 3080 (`sm_86`) |
| Ada Lovelace | `sm_89` | RTX 4090, RTX 4080 |
| Hopper | `sm_90` | H100 |

---

## Usage & Command-Line Options

```
Usage: genkeys-cuda [options]

Options:
  -d, --device <ID>     Select CUDA GPU device ID (e.g. -d 0 or -d=0)
  -b, --blocks <N>      Custom grid block count (default: auto-calculated per SM)
  -t, --threads <N>     Custom threads per block (default: 256)
  -r, --results <N>     Results batch buffer size (default: 256)
      --min <N>         Initial minimum leading zero bits threshold (default: 32)
      --max <N>         Maximum zero bits threshold range (0 = unbounded, default: 0)
  -i, --iter <N>        Maximum batch iterations to run (0 = infinite, default: 0)
  -v, --verbose         Enable verbose diagnostic self-test output
  -s, --summary <N>     Enable summary stats every N iterations
      --summary-newline Print summary on a new line instead of carriage return
  -a, --address         Print derived Yggdrasil IPv6 address for matched keys
      --vanity <N>      Enable extreme boost vanity mode with N iterations per thread
  -h, --help            Show this help message
```

---

## Examples

### 1. List Available CUDA Devices
Running without arguments or specifying `-d -1` lists all detected CUDA GPUs:
```bash
./genkeys-cuda -d -1
```

### 2. Standard Search for Yggdrasil Keys (Min 32 Zero Bits)
```bash
./genkeys-cuda -d 0 --min 32 -a -s 10
```

### 3. Extreme Boost Vanity Search (Recommended for High Throughput)
Runs boost mode with 4,096 iterations per thread, searching for keys with at least 40 leading zero bits:
```bash
./genkeys-cuda -d 0 --vanity 4096 --min 40 -a -s 5 --summary-newline
```

### 4. Diagnostic Self-Test (Verbose)
Runs the complete 12-stage validation suite with detailed output before starting the search:
```bash
./genkeys-cuda -d 0 -v -i 1
```

---

## Self-Test Suite Stages

When launched, `genkeys-cuda` runs an internal diagnostic suite before key search to guarantee cryptographic integrity:

| Stage | Test Name | Target Function / Property |
|---|---|---|
| **1** | Field Arithmetic | $1 \times 1 = 1$ and modular inversion $1^{-1} \equiv 1$ in $\mathbb{F}_{2^{255}-19}$ |
| **2** | Field Addition | Identity $A + 0 = A$ and round-trip $(A + B) - B = A$ |
| **3** | Field Subtraction | Self-subtraction $A - A = 0$ and identity $A - 0 = A$ |
| **4** | Field Multiplication | Curve parameter check $5 \times (4/5) \equiv 4$ and identity |
| **5** | Field Encoding | Round-trip consistency `fe_frombytes` $\leftrightarrow$ `fe_tobytes` |
| **6** | SHA-512 Core | Kernel `sha512_32bytes` output against RFC standard test vector |
| **7** | Point Addition | Extended point addition `ge_p3_add` ($\mathcal{O} + B = B$) |
| **8** | Point Doubling | Projective point doubling `ge_p3_dbl` ($2 \times B = 2B$) |
| **9** | Mixed Addition | Precomputed Duif point addition `ge_madd` ($\mathcal{O} + base[0][0] = B$) |
| **10** | Base Scalarmult | Full base point scalar multiplication `ge_scalarmult_base` |
| **11** | Public Key Derivation | Complete Ed25519 keypair generation against 6 RFC 8032 test vectors |
| **12** | Boost Math Consistency | Verifies 1,000 iterative `ge_madd(+8G)` steps match exact scalar multiplication |

---

## Output Format

When a candidate matching the leading zero criteria is found, it is output to `stdout`:
```
[+] Found <zeros> (<zeros_hex>): [AZ] <seed_hex_32_bytes> <pubkey_hex_32_bytes> [<yggdrasil_ipv6>]
```

- `zeros`: Count of leading zero bits in the generated Ed25519 public key.
- `AZ`: Indicates the key was derived in boost mode (private clamped scalar output).
- `seed_hex`: 32-byte private seed (or clamped scalar in boost mode) in hexadecimal.
- `pubkey_hex`: 32-byte Ed25519 public key in hexadecimal.
- `yggdrasil_ipv6`: Canonical IPv6 address formatted per RFC 5952 (when `-a` / `--address` is enabled).

---

## Repository Structure

```
genkeys-cuda/
├── add_scalar.cuh              # 256-bit scalar addition helpers (add_scalar256, add_scalar256_out)
├── base_table.inc              # 60 KB precomputed Ed25519 base point table data
├── config.cuh                  # CUDA check macros and global constants
├── cpu_seed_gen.cuh            # CPU-side ChaCha20 CSPRNG seed generator with OS entropy
├── ed25519_base_table.cuh      # Constant base point table definition in GPU memory
├── ed25519_cuda.cuh            # Ed25519 top-level interface (ge_p3_add, ed25519_create_public_key)
├── ed25519_cuda_fe.cu          # 10-limb field arithmetic implementation (\Z / 2^255-19)
├── ed25519_cuda_fe.h           # Field element type definitions and function declarations
├── ed25519_cuda_ge.cu          # Twisted Edwards group operations and scalar multiplication
├── ed25519_cuda_ge.h           # Group element structs (ge_p2, ge_p3, ge_p1p1, ge_precomp, ge_cached)
├── genkeys_cuda.cuh            # Search kernels (vanity_search_kernel, vanity_search_boost_kernel)
├── global_state.cuh            # Search state coordinator, atomic counters, and key result store
├── gpu_seed_gen.cuh            # GPU-accelerated ChaCha8 per-thread seed generation
├── main.cu                     # CLI entry point, GPU initialization, argument parser
├── selftest.cuh                # Diagnostic test runner executing all 12 stages
├── sha512_cuda.cuh             # GPU SHA-512 implementation (standard + 16-word circular buffer)
├── yggdrasil_cuda.cuh          # Ed25519 public key to canonical Yggdrasil IPv6 address converter
├── tests/                      # Diagnostic test stages 1 to 12
│   ├── stage1_field_arithmetic.cuh
│   ├── stage2_fe_add.cuh
│   ├── stage3_fe_sub.cuh
│   ├── stage4_fe_mul.cuh
│   ├── stage5_field_encoding.cuh
│   ├── stage6_sha512.cuh
│   ├── stage7_point_addition.cuh
│   ├── stage8_point_doubling.cuh
│   ├── stage9_mixed_addition.cuh
│   ├── stage10_scalarmult_base.cuh
│   ├── stage11_create_public_key.cuh
│   └── stage12_boost_math.cuh
└── README.md                   # Project documentation
```

## License, Provenance & Legal Notes

### Upstream Provenance & Public Domain Primitives
- **Ed25519 / Curve25519 Mathematical Core**: The underlying finite field ($\mathbb{F}_{2^{255}-19}$) arithmetic (`fe_*`), group operations on twisted Edwards curves (`ge_*`), and precomputed base table (`base_table.inc`) are derived from the reference implementation (**ref10**) of **SUPERCOP** (*System for Unified Performance Evaluation of Related Cryptographic Operations*), authored by **Daniel J. Bernstein**, **Nils Duif**, **Tanja Lange**, **Peter Schwabe**, and **Bo-Yin Yang**.
- **Public Domain Status of SUPERCOP**: In accordance with the original authors' intent, the reference SUPERCOP/ref10 implementation was explicitly dedicated to the **Public Domain**.
- **Andrew Moon's SUPERCOP Mirror**: The repository integrates primitives from [floodyberry/supercop](https://github.com/floodyberry/supercop) (by Andrew Moon), which likewise carries no restrictive proprietary copyright and serves as a public-domain / open-source port of Bernstein's cryptographic library.

### Project Licensing
- This project is distributed under the terms of the **[MIT License](LICENSE)**. See the [LICENSE](LICENSE) file for complete terms and permissions.
- The CUDA kernels (`vanity_search_kernel`, `vanity_search_boost_kernel`), Montgomery batch inversion pipeline, early-exit filter heuristics, diagnostic test suite, and Yggdrasil address encoder are licensed under MIT, while incorporating public-domain mathematical primitives from SUPERCOP.
- You are free to copy, modify, distribute, benchmark, sublicense, and integrate this code into personal, academic, or commercial projects.

### Disclaimer of Warranty
> [!CAUTION]
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
>
> Users are solely responsible for ensuring secure key management, entropy validation, and operational security when deploying generated cryptographic keys.

---

## References

- **Ed25519 Specification (RFC 8032)**: [Edwards-Curve Digital Signature Algorithm (EdDSA)](https://datatracker.ietf.org/doc/html/rfc8032)
- **IPv6 Text Representation (RFC 5952)**: [A Recommendation for IPv6 Text Representation](https://datatracker.ietf.org/doc/html/rfc5952)
- **SUPERCOP Benchmark Suite**: [System for Unified Performance Evaluation of Related Cryptographic Operations](https://bench.cr.yp.to/supercop.html)
- **floodyberry/supercop**: [GitHub Repository](https://github.com/floodyberry/supercop)
- **Yggdrasil Network**: [Decentralized Mesh Networking Protocol](https://yggdrasil-network.github.io/)

