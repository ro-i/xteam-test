// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

// xteam_bench.cpp — OpenMP cross-team performance & correctness benchmark

#include <unistd.h>

#include "omp.h"

#include "bench_common.h"
#include "xteam_simulations_common.h"

#ifdef AOMP
#include "xteam_simulations_aomp.h"
#elif defined(TRUNK)
#include "xteam_simulations_trunk.h"
#elif defined(TRUNK_DEV)
#include "xteam_simulations_trunk_dev.h"
#elif defined(AOMP_DEV)
#include "xteam_simulations_aomp_dev.h"
#endif

static Config conf;

// =========================================================================
// GPU cross-team reduction kernels
// =========================================================================

template <typename T> T reduce_sum(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(+ : s)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(+ : s)
#endif
  for (uint64_t i = 0; i < n; i++)
    s += in[i];
  return s;
}

template <typename T> T reduce_max(const T *__restrict in, uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(max : m)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(max : m)
#endif
  for (uint64_t i = 0; i < n; i++)
    m = std::max(m, in[i]);
  return m;
}

template <typename T> T reduce_min(const T *__restrict in, uint64_t n) {
  T m = red_identity<T, RedOp::Min>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(min : m)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(min : m)
#endif
  for (uint64_t i = 0; i < n; i++)
    m = std::min(m, in[i]);
  return m;
}

template <typename T>
T reduce_dot(const T *__restrict a, const T *__restrict b, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(+ : s)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(+ : s)
#endif
  for (uint64_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

// =========================================================================
// GPU cross-team scan kernels (compilation unsupported by vanilla AOMP)
// =========================================================================

#ifdef AOMP_DEV
template <typename T>
void scan_sum_incl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, + : s)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
#endif
  for (uint64_t i = 0; i < n; i++) {
    s += in[i];
#pragma omp scan inclusive(s)
    out[i] = s;
  }
}

template <typename T>
void scan_sum_excl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, + : s)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
#endif
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
#pragma omp scan exclusive(s)
    s += in[i];
  }
}

template <typename T>
void scan_max_incl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, max : m)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, max : m)
#endif
  for (uint64_t i = 0; i < n; i++) {
    m = std::max(m, in[i]);
#pragma omp scan inclusive(m)
    out[i] = m;
  }
}

template <typename T>
void scan_max_excl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, max : m)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, max : m)
#endif
  for (uint64_t i = 0; i < n; i++) {
    out[i] = m;
#pragma omp scan exclusive(m)
    m = std::max(m, in[i]);
  }
}

template <typename T>
void scan_min_incl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = red_identity<T, RedOp::Min>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, min : m)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, min : m)
#endif
  for (uint64_t i = 0; i < n; i++) {
    m = std::min(m, in[i]);
#pragma omp scan inclusive(m)
    out[i] = m;
  }
}

template <typename T>
void scan_min_excl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = red_identity<T, RedOp::Min>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, min : m)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, min : m)
#endif
  for (uint64_t i = 0; i < n; i++) {
    out[i] = m;
#pragma omp scan exclusive(m)
    m = std::min(m, in[i]);
  }
}

template <typename T>
void scan_dot_incl(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, + : s)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
#endif
  for (uint64_t i = 0; i < n; i++) {
    s += a[i] * b[i];
#pragma omp scan inclusive(s)
    out[i] = s;
  }
}

template <typename T>
void scan_dot_excl(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#if CODEGEN_AUTODETECTION
#pragma omp target teams distribute parallel for reduction(inscan, + : s)
#else
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
#endif
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
#pragma omp scan exclusive(s)
    s += a[i] * b[i];
  }
}
#else  // defined(AOMP_DEV)
template <typename T>
void scan_sum_incl(const T *__restrict in, T *__restrict out, uint64_t n) {}
template <typename T>
void scan_sum_excl(const T *__restrict in, T *__restrict out, uint64_t n) {}
template <typename T>
void scan_max_incl(const T *__restrict in, T *__restrict out, uint64_t n) {}
template <typename T>
void scan_max_excl(const T *__restrict in, T *__restrict out, uint64_t n) {}
template <typename T>
void scan_min_incl(const T *__restrict in, T *__restrict out, uint64_t n) {}
template <typename T>
void scan_min_excl(const T *__restrict in, T *__restrict out, uint64_t n) {}
template <typename T>
void scan_dot_incl(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {}
template <typename T>
void scan_dot_excl(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {}
#endif // defined(AOMP_DEV)

// =========================================================================
// Benchmark harness (OMP-specific run functions)
// =========================================================================

template <typename T, bool is_fp, SimulationLike Sim, typename Kernel,
          typename... Inputs>
std::optional<TimingResult> run_bench_scan(Kernel kernel, T *out, const T *gold,
                                           uint64_t n, const std::string &label,
                                           Sim *sim, Inputs... inputs) {
  std::vector<double> times(conf.bench_iters_scan);
  for (int t = 0; t < conf.warmup_iters + conf.bench_iters_scan; t++) {
    if (sim)
      sim->reset_device();
    auto t1 = Clock::now();
    kernel(inputs..., out, n);
    auto t2 = Clock::now();
    if (t >= conf.warmup_iters) {
      times[t - conf.warmup_iters] =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
              .count();
    }
#pragma omp target update from(out[0 : n])
    if (!check<T, is_fp>(out, gold, n, label))
      return std::nullopt;
  }

  return create_timing_result(times, n, sizeof(T) * n * sizeof...(Inputs));
}

template <typename T, bool is_fp, SimulationLike Sim, typename Kernel,
          typename... Inputs>
std::optional<TimingResult> run_bench_reduce(Kernel kernel, T gold, uint64_t n,
                                             const std::string &label, Sim *sim,
                                             Inputs... inputs) {
  std::vector<double> times(conf.bench_iters_reduction);
  for (int t = 0; t < conf.warmup_iters + conf.bench_iters_reduction; t++) {
    if (sim)
      sim->reset_device();
    auto t1 = Clock::now();
    T result = kernel(inputs..., n);
    auto t2 = Clock::now();
    if (t >= conf.warmup_iters) {
      times[t - conf.warmup_iters] =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
              .count();
    }
    if (!check_single<T, is_fp>(result, gold, label))
      return std::nullopt;
  }

  return create_timing_result(times, n, sizeof(T) * n * sizeof...(Inputs));
}

// =========================================================================
// Templated per-type benchmark runner
// =========================================================================

template <typename T, bool is_fp> void run_type(const char *type_name) {
  for (uint64_t n : conf.array_sizes) {
    T *in1 = alloc<T>(n);
    T *in2 = alloc<T>(n);
    T *out = alloc<T>(n);
    init_data<T, is_fp>(in1, in2, n);
    std::optional<TimingResult> r;

#pragma omp target enter data map(to : in1[0 : n], in2[0 : n], out[0 : n])

#ifdef AOMP
    SimulationAOMP<T> *simulation = new SimulationAOMP<T>();
#elif defined(TRUNK)
    SimulationTrunk<T> *simulation = new SimulationTrunk<T>();
#elif defined(TRUNK_DEV)
    SimulationTrunkDev<T> *simulation = new SimulationTrunkDev<T>();
#elif defined(AOMP_DEV)
    SimulationAOMPDev<T> *simulation = new SimulationAOMPDev<T>();
#else
    SimulationNoop<T> *simulation = new SimulationNoop<T>();
#endif
    simulation->init_device();

    if (conf.reduction || conf.reduction_simulation) {
      // Cross-team reductions (codegen + simulations)
      T gold_sum = gold_reduce_sum(in1, n);
      T gold_max = gold_reduce_max(in1, n);
      T gold_min = gold_reduce_min(in1, n);
      T gold_dot = gold_reduce_dot(in1, in2, n);

      if (conf.reduction) {
        // sum reduction
        r = run_bench_reduce<T, is_fp>(reduce_sum<T>, gold_sum, n, "red_sum",
                                       simulation, in1);
        print_result("red_sum", type_name, n, r);
        // max reduction
        r = run_bench_reduce<T, is_fp>(reduce_max<T>, gold_max, n, "red_max",
                                       simulation, in1);
        print_result("red_max", type_name, n, r);
        // min reduction
        r = run_bench_reduce<T, is_fp>(reduce_min<T>, gold_min, n, "red_min",
                                       simulation, in1);
        print_result("red_min", type_name, n, r);
        // dot reduction
        r = run_bench_reduce<T, is_fp>(reduce_dot<T>, gold_dot, n, "red_dot",
                                       simulation, in1, in2);
        print_result("red_dot", type_name, n, r);
      }

      if (conf.reduction_simulation) {
        // sum reduction
        for (const auto &[name, func] :
             simulation->template get_all_reduce_variants<RedOp::Sum>()) {
          r = run_bench_reduce<T, is_fp>(func, gold_sum, n, name, simulation,
                                         in1);
          print_result(name, type_name, n, r);
        }
        // max reduction
        for (const auto &[name, func] :
             simulation->template get_all_reduce_variants<RedOp::Max>()) {
          r = run_bench_reduce<T, is_fp>(func, gold_max, n, name, simulation,
                                         in1);
          print_result(name, type_name, n, r);
        }
        // min reduction
        for (const auto &[name, func] :
             simulation->template get_all_reduce_variants<RedOp::Min>()) {
          r = run_bench_reduce<T, is_fp>(func, gold_min, n, name, simulation,
                                         in1);
          print_result(name, type_name, n, r);
        }
        // dot reduction
        for (const auto &[name, func] :
             simulation->get_all_reduce_dot_variants()) {
          r = run_bench_reduce<T, is_fp>(func, gold_dot, n, name, simulation,
                                         in1, in2);
          print_result(name, type_name, n, r);
        }
      }
    }

    if (conf.scan || conf.scan_simulation) {
      // Cross-team scans (codegen + simulations)
      T *gold = alloc<T>(n);

      simulation->reset_device();

      // ================================================================
      // inclusive sum scan
      // ================================================================
      gold_inclusive_sum(in1, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_sum_incl<T>, out, gold, n, "incl_sum",
                                     simulation, in1);
        print_result("incl_sum", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->template get_all_scan_incl_variants<RedOp::Sum>()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // exclusive sum scan
      // ================================================================
      gold_exclusive_sum(in1, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_sum_excl<T>, out, gold, n, "excl_sum",
                                     simulation, in1);
        print_result("excl_sum", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->template get_all_scan_excl_variants<RedOp::Sum>()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // inclusive max scan
      // ================================================================
      gold_inclusive_max(in1, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_max_incl<T>, out, gold, n, "incl_max",
                                     simulation, in1);
        print_result("incl_max", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->template get_all_scan_incl_variants<RedOp::Max>()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // exclusive max scan
      // ================================================================
      gold_exclusive_max(in1, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_max_excl<T>, out, gold, n, "excl_max",
                                     simulation, in1);
        print_result("excl_max", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->template get_all_scan_excl_variants<RedOp::Max>()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // inclusive min scan
      // ================================================================
      gold_inclusive_min(in1, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_min_incl<T>, out, gold, n, "incl_min",
                                     simulation, in1);
        print_result("incl_min", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->template get_all_scan_incl_variants<RedOp::Min>()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // exclusive min scan
      // ================================================================
      gold_exclusive_min(in1, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_min_excl<T>, out, gold, n, "excl_min",
                                     simulation, in1);
        print_result("excl_min", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->template get_all_scan_excl_variants<RedOp::Min>()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // inclusive dot scan
      // ================================================================
      gold_inclusive_dot(in1, in2, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_dot_incl<T>, out, gold, n, "incl_dot",
                                     simulation, in1, in2);
        print_result("incl_dot", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->get_all_scan_dot_incl_variants()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1, in2);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // exclusive dot scan
      // ================================================================
      gold_exclusive_dot(in1, in2, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_dot_excl<T>, out, gold, n, "excl_dot",
                                     simulation, in1, in2);
        print_result("excl_dot", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->get_all_scan_dot_excl_variants()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1, in2);
          print_result(name, type_name, n, r);
        }
      }

      free(gold);
    }

#pragma omp target exit data map(delete : in1[0 : n], in2[0 : n], out[0 : n])

    free(in1);
    free(in2);
    free(out);
    simulation->free_device();
    delete simulation;
  }
}

static void usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0
      << " [-b <int>] [-B <int>] [-q] [-r] [-s] [-R] [-S] [-w <int>] [-h]\n";
  std::cout << "  -b N: Benchmark iterations for reduction\n";
  std::cout << "  -B N: Benchmark iterations for scan\n";
  std::cout << "  -q: Quick run (test only one array size)\n";
  std::cout << "  -r: Run reduction tests\n";
  std::cout << "  -s: Run scan tests\n";
  std::cout << "  -R: Run reduction simulations\n";
  std::cout << "  -S: Run scan simulations\n";
  std::cout << "  -w N: Warmup iterations\n";
  std::cout << "  -h: Show this help message\n";
}

// =========================================================================
// Main
// =========================================================================
int main(int argc, char **argv) {
  int opt;

  while ((opt = getopt(argc, argv, "b:B:qrsRSw:h")) != -1) {
    switch (opt) {
    case 'b':
      conf.bench_iters_reduction = std::stoi(optarg);
      break;
    case 'B':
      conf.bench_iters_scan = std::stoi(optarg);
      break;
    case 'q':
      conf.quick_run = true;
      break;
    case 'r':
      conf.reduction = true;
      break;
    case 's':
#ifdef AOMP_DEV
      conf.scan = true;
#endif
      if (!conf.scan)
        std::cerr << "warning: scan codegen unsupported on this compiler\n";
      break;
    case 'R':
      conf.reduction_simulation = true;
      break;
    case 'S':
#ifndef TRUNK
      conf.scan_simulation = true;
#endif
      if (!conf.scan_simulation)
        std::cerr << "warning: scan simulations unsupported on this compiler\n";
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
  if (!conf.reduction && !conf.scan && !conf.reduction_simulation &&
      !conf.scan_simulation) {
    std::cerr << "error: at least one of -r, -s, -R, -S must be specified\n";
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (conf.quick_run)
    conf.array_sizes.assign(array_sizes_quick.begin(), array_sizes_quick.end());
  else
    conf.array_sizes.assign(array_sizes.begin(), array_sizes.end());

  std::cout << std::format(
      "xteam benchmark (quick run: {}) — {} warmup, {} timed "
      "iterations "
      "(reduction), {} timed iterations (scan), "
      "{} teams, {} threads, codegen autodetection: {}\n",
      conf.quick_run ? "true" : "false", conf.warmup_iters,
      conf.bench_iters_reduction, conf.bench_iters_scan, XTEAM_NUM_TEAMS,
      XTEAM_NUM_THREADS, CODEGEN_AUTODETECTION ? "true" : "false");

  std::cout << "Array sizes: ";
  for (uint64_t sz : conf.array_sizes)
    std::cout << std::format(" {}", sz);
  std::cout << "\n\n";

  print_header();

  std::cout << "\n--- int ---\n";
  run_type<int, false>("int");

  std::cout << "\n--- long ---\n";
  run_type<long, false>("long");

  std::cout << "\n--- double ---\n";
  run_type<double, true>("double");

  return EXIT_SUCCESS;
}
