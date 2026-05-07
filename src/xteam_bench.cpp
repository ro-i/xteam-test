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
const std::array<uint64_t, 1> array_sizes_quick{XTEAM_TOTAL_NUM_THREADS};
#else
const std::array<uint64_t, 1> array_sizes_quick{177777777};
#endif // NOLOOP

// Array sizes for full run
#if NOLOOP
const std::array<uint64_t, 9> array_sizes{1,
                                          100,
                                          1024,
                                          2048,
                                          4096,
                                          8192,
                                          XTEAM_TOTAL_NUM_THREADS / 2,
                                          XTEAM_TOTAL_NUM_THREADS - 1,
                                          XTEAM_TOTAL_NUM_THREADS};
#else
const std::array<uint64_t, 14> array_sizes{
    1,     100,     1024,    2048,     4096,     8192,      10000,
    81920, 1000000, 4194304, 23445657, 41943040, 100000000, 177777777};
#endif // NOLOOP

// =========================================================================
// Templated per-type benchmark runner
// =========================================================================

static void usage(std::string_view argv0) {
  std::cout << "Usage: " << argv0
            << " [-b <int>] [-q] [-r] [-s] [-w <int>] [-h]\n"
            << "  -b N: Benchmark iterations (default: auto-scaled such that "
               "the runtime per test is ~"
            << AUTO_SCALE_TIME << " second (min " << BENCH_MIN_ITERS
            << " iterations))\n"
            << "  -q: Quick run (test only one array size)\n"
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

  while ((opt = getopt(argc, argv, "b:qrsw:h")) != -1) {
    switch (opt) {
    case 'b':
      conf.bench_iters = std::stoi(optarg);
      conf.auto_scale = false;
      break;
    case 'q':
      conf.quick_run = true;
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

  if (conf.quick_run)
    conf.array_sizes.assign(array_sizes_quick.begin(), array_sizes_quick.end());
  else
    conf.array_sizes.assign(array_sizes.begin(), array_sizes.end());

  if (conf.auto_scale)
    conf.bench_iters = std::numeric_limits<int>::max() - conf.warmup_iters;

  std::cout << std::format(
      "{} benchmark for {} (quick run: {}, auto-scale: {}, codegen "
      "autodetection: {}) "
      "- {} warmup, {} timed iterations - {} teams, {} threads\n",
      bench_op_name, COMPILER_LABEL, conf.quick_run ? "true" : "false",
      conf.auto_scale ? "true" : "false",
      CODEGEN_AUTODETECTION ? "true" : "false", conf.warmup_iters,
      conf.auto_scale ? "auto-scaled" : std::to_string(conf.bench_iters),
      XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS);

  run_bench_op();

  return EXIT_SUCCESS;
}
