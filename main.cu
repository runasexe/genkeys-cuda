#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

#include "config.cuh"
#include "global_state.cuh"

#include "sha512_cuda.cuh"
#include "ed25519_cuda.cuh"

#include "selftest.cuh"
#include "genkeys_cuda.cuh"

static void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]" << std::endl << std::endl
              << "Options:" << std::endl
              << "  -d, --device <ID>     Select CUDA GPU device ID (e.g. -d=0)" << std::endl
              << "  -b, --blocks <N>      Custom grid block count" << std::endl
              << "  -t, --threads <N>     Custom threads per block (default: 256)" << std::endl
              << "  -r, --results <N>     Results batch size" << std::endl
              << "      --min <N>         Initial minimum zero bits threshold (default: 32)" << std::endl
              << "      --max <N>         Maximum zero bits threshold range (default: 0)" << std::endl
              << "  -i, --iter <N>        Maximum batch iterations to run (0 = infinite, default: 0)" << std::endl
              << "  -v, --verbose         Enable verbose diagnostic self-test output" << std::endl
              << "  -s, --summary <N>     Enable summary stats each N iterations" << std::endl
              << "      --summary-newline Summary at new line" << std::endl
              << "  -a, --address         Print yggdrasil address" << std::endl
              << "      --vanity <N>      Enable extreme boost vanity mode with N iterations per thread" << std::endl
              << "  -h, --help            Show this help message" << std::endl;
}

static void list_gpus() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "[!] No CUDA devices detected!" << std::endl;
        return;
    }

    std::cout << "Available CUDA GPU devices:" << std::endl;
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  ID " << i << ": " << prop.name
                  << " (Compute " << prop.major << "." << prop.minor
                  << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB VRAM)" << std::endl;
    }
    std::cout << std::endl << "Please specify a valid GPU ID: genkeys-cuda -d <ID>" << std::endl;
}

int main(int argc, char** argv) {
    int device_id = -1;
    bool verbose = false;
    uint64_t summary = 0;
    bool summary_newline = false;
    bool address = false;
    uint64_t max_iterations = 0;
    uint32_t min_zeros = 32;
    uint32_t max_zeros = 0;
    uint32_t blocks = 0;
    uint32_t threads = 0;
    uint32_t max_results = 0;
    uint32_t iters_per_thread = 4096;
    bool vanity_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--summary-newline") {
            summary_newline = true;
        } else if (arg == "-a" || arg == "--address") {
            address = true;
        } else if (arg == "--vanity") {
            if (i + 1 < argc) {
                iters_per_thread = (uint32_t)std::atoi(argv[++i]);
                if (iters_per_thread > 0) vanity_mode = true;
            }
        } else if (arg.rfind("--vanity=", 0) == 0) {
            iters_per_thread = (uint32_t)std::atoi(arg.substr(9).c_str());
            if (iters_per_thread > 0) vanity_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-d" || arg == "--device") {
            if (i + 1 < argc) {
                device_id = std::atoi(argv[++i]);
            }
        } else if (arg.rfind("-d=", 0) == 0) {
            device_id = std::atoi(arg.substr(3).c_str());
        } else if (arg.rfind("--device=", 0) == 0) {
            device_id = std::atoi(arg.substr(9).c_str());
        } else if (arg == "-s" || arg == "--summary") {
            if (i + 1 < argc) {
                summary = std::strtoull(argv[++i], nullptr, 10);
            }
        } else if (arg.rfind("-s=", 0) == 0) {
            summary = std::strtoull(arg.substr(3).c_str(), nullptr, 10);
        } else if (arg.rfind("--summary=", 0) == 0) {
            summary = std::strtoull(arg.substr(10).c_str(), nullptr, 10);
        } else if (arg == "-b" || arg == "--blocks") {
            if (i + 1 < argc) {
                blocks = (uint32_t)std::atoi(argv[++i]);
            }
        } else if (arg.rfind("-b=", 0) == 0) {
            blocks = (uint32_t)std::atoi(arg.substr(3).c_str());
        } else if (arg.rfind("--blocks=", 0) == 0) {
            blocks = (uint32_t)std::atoi(arg.substr(9).c_str());
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                threads = (uint32_t)std::atoi(argv[++i]);
            }
        } else if (arg.rfind("-t=", 0) == 0) {
            threads = (uint32_t)std::atoi(arg.substr(3).c_str());
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads = (uint32_t)std::atoi(arg.substr(10).c_str());
        } else if (arg == "-r" || arg == "--results") {
            if (i + 1 < argc) {
                max_results = (uint32_t)std::atoi(argv[++i]);
            }
        } else if (arg.rfind("-r=", 0) == 0) {
            max_results = (uint32_t)std::atoi(arg.substr(3).c_str());
        } else if (arg.rfind("--results=", 0) == 0) {
            max_results = (uint32_t)std::atoi(arg.substr(10).c_str());
        } else if (arg == "--min") {
            if (i + 1 < argc) {
                min_zeros = (uint32_t)std::atoi(argv[++i]);
            }
        } else if (arg.rfind("--min=", 0) == 0) {
            min_zeros = (uint32_t)std::atoi(arg.substr(6).c_str());
        } else if (arg == "--max") {
            if (i + 1 < argc) {
                max_zeros = (uint32_t)std::atoi(argv[++i]);
            }
        } else if (arg.rfind("--max=", 0) == 0) {
            max_zeros = (uint32_t)std::atoi(arg.substr(6).c_str());
        } else if (arg == "-i" || arg == "--iter") {
            if (i + 1 < argc) {
                max_iterations = std::strtoull(argv[++i], nullptr, 10);
            }
        } else if (arg.rfind("-i=", 0) == 0) {
            max_iterations = std::strtoull(arg.substr(3).c_str(), nullptr, 10);
        } else if (arg.rfind("--iter=", 0) == 0) {
            max_iterations = std::strtoull(arg.substr(7).c_str(), nullptr, 10);
        }
    }

    if (device_id < 0) {
        list_gpus();
        return 0;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_id >= device_count) {
        std::cerr << "[!] Invalid GPU ID " << device_id << ". Max device ID is " << (device_count - 1) << "." << std::endl;
        list_gpus();
        return 1;
    }

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "[*] Selected GPU ID " << device_id << ": " << prop.name << std::endl;

    if (!run_full_selftest(verbose)) {
        std::cerr << "[!] Self-test FAILED! Key generation aborted." << std::endl;
        return 1;
    }

    GlobalState global_state(min_zeros, max_zeros);

    // Launch vanity search loop on selected GPU with GlobalState coordination and max iteration limit
    run_vanity_search(device_id, global_state, max_iterations, blocks, threads, max_results, summary, summary_newline, address, vanity_mode, iters_per_thread);

    return 0;
}
