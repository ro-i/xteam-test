// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

// xteam_bench.cpp — OpenMP cross-team performance & correctness benchmark

#include <unistd.h>

#include "omp.h"

#include "common.h"

Config conf;

// Selected arrays sizes for quick run
#if NOLOOP
const std::vector<uint64_t> array_sizes_quick{XTEAM_TOTAL_NUM_THREADS};
#else
const std::vector<uint64_t> array_sizes_quick{177777777};
#endif // NOLOOP

// Array sizes for full run
#if NOLOOP
const std::vector<uint64_t> array_sizes{1,
                                        100,
                                        1024,
                                        2048,
                                        4096,
                                        8192,
                                        XTEAM_TOTAL_NUM_THREADS / 2,
                                        XTEAM_TOTAL_NUM_THREADS - 1,
                                        XTEAM_TOTAL_NUM_THREADS};
#else
const std::vector<uint64_t> array_sizes{
    1,     100,     1024,    2048,     4096,     8192,      10000,
    81920, 1000000, 4194304, 23445657, 41943040, 100000000, 177777777};
#endif // NOLOOP

// =========================================================================
// Common resources and related functions
// =========================================================================
//
// brought up / freed via init_common() / cleanup_common() around run_bench_op()

// Device-side buffer used by evict_device_cache()
static char *evict_buf;

static void init_common() {
  if (conf.evict_cache)
    evict_buf = target_alloc<char>(CACHE_EVICT_BYTES, omp_get_default_device());
}

static void cleanup_common() {
  if (evict_buf) {
    omp_target_free(evict_buf, omp_get_default_device());
    evict_buf = nullptr;
  }
}

void evict_device_cache() {
  uint64_t n = CACHE_EVICT_BYTES;
#pragma omp target teams distribute parallel for is_device_ptr(evict_buf)      \
    num_teams(XTEAM_NUM_TEAMS) num_threads(XTEAM_NUM_THREADS)
  for (uint64_t i = 0; i < n; i++)
    evict_buf[i] = static_cast<char>(i);
}

static void usage(std::string_view argv0) {
  std::cout
      << "Usage: " << argv0
      << " [-b <int>] [-e] [-q] [-Q <uint64_t>] [-r] [-s] [-w <int>] [-h]\n"
      << "  -b N: Benchmark iterations (default: auto-scaled such that "
         "the runtime per test is ~"
      << AUTO_SCALE_TIME << " second (min " << BENCH_MIN_ITERS
      << " iterations))\n"
      << "  -e: Evict the GPU L2/MALL cache before each iteration "
         "(cold-cache mode). Most relevant for small/medium array sizes "
         "that would otherwise stay resident in cache across iterations.\n"
      << "  -q: Quick run (test only one array size)\n"
      << "  -Q N: Quick run with custom array size N\n"
      << "  -r: Run non-simulation tests\n"
      << "  -s: Run simulation tests\n"
      << "  -w N: Warmup iterations (default: 2)\n"
      << "  -h: Show this help message\n"

      << "\nPseudocode of how the benchmark binaries run the tests:\n"
      << "  for each data type in alphabetical order (e.g. double, int, "
         "long):\n"
      << "    for each array size in numerical order:\n"
      << "      for each test:\n"
      << "        for each warmup iteration:\n"
      << "          run the test and check the result against the gold "
         "result\n"
      << "        for each timed benchmark iteration:\n"
      << "          run the test and check the result against the gold "
         "result\n";
}

// =========================================================================
// Main
// =========================================================================
int main(int argc, char *const *argv) {
  int opt;
  std::optional<uint64_t> custom_quick_array_size;

  while ((opt = getopt(argc, argv, "b:eqQ:rsw:h")) != -1) {
    switch (opt) {
    case 'b':
      conf.bench_iters = std::stoi(optarg);
      conf.auto_scale = false;
      break;
    case 'e':
      conf.evict_cache = true;
      break;
    case 'q':
      conf.quick_run = true;
      break;
    case 'Q':
      conf.quick_run = true;
      custom_quick_array_size = std::stoull(optarg);
      break;
    case 'r':
      conf.run = true;
      break;
    case 's':
      conf.run_sim = true;
      break;
    case 'w':
      conf.warmup_iters = std::stoi(optarg);
      break;
    case 'h':
      usage(argv[0]);
      return EXIT_SUCCESS;
    default:
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (!conf.run && !conf.run_sim) {
    std::cerr << "error: at least one of -r or -s must be specified\n";
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (custom_quick_array_size)
    conf.array_sizes = {*custom_quick_array_size};
  else if (conf.quick_run)
    conf.array_sizes = array_sizes_quick;
  else
    conf.array_sizes = array_sizes;

  if (conf.auto_scale)
    conf.bench_iters = std::numeric_limits<int>::max() - conf.warmup_iters;

  std::cout << std::format(
      "{} benchmark for {} (quick run: {}, auto-scale: {}, evict cache: {}, "
      "codegen autodetection: {}) "
      "- {} warmup, {} timed iterations - {} teams, {} threads\n",
      bench_op_name, COMPILER_LABEL, conf.quick_run ? "true" : "false",
      conf.auto_scale ? "true" : "false", conf.evict_cache ? "true" : "false",
      CODEGEN_AUTODETECTION ? "true" : "false", conf.warmup_iters,
      conf.auto_scale ? "auto-scaled" : std::to_string(conf.bench_iters),
      XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS);

  init_common();
  run_bench_op();
  cleanup_common();

  return EXIT_SUCCESS;
}
