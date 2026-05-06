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
#elif defined(TRUNK_JD)
#include "xteam_simulations_trunk_jd.h"
#elif defined(TRUNK_DEV)
#include "xteam_simulations_trunk_dev.h"
#elif defined(AOMP_DEV)
#include "xteam_simulations_aomp_dev.h"
#endif

static Config conf;

// =========================================================================
// GPU cross-team reduction kernels where the AOMP codegen patterns match.
// (Excluding min since it doesn't offer more insight than max.)
// =========================================================================

template <typename T> T red_sum(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s += in[i];
  return s;
}

template <typename T> T red_max(const T *__restrict in, uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        max : m)
  for (uint64_t i = 0; i < n; i++)
    m = std::max(m, in[i]);
  return m;
}

template <typename T>
T red_dot(const T *__restrict a, const T *__restrict b, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

// Combined reduction (sum and max) in a single loop.
template <typename T> T red_combined(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        + : s) reduction(max : m)
  for (uint64_t i = 0; i < n; i++) {
    s += in[i];
    m = std::max(m, in[i]);
  }
  return (s / 2) + (m / 2);
}

double red_pi(uint64_t n) {
  double pi = 0.0;

  // https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : pi)
  for (uint64_t i = 0; i < n; i++) {
    double term = 1.0 / (2 * i + 1);
    pi += (i & 0x1) ? -term : term;
  }

  return pi * 4.0;
}

// =========================================================================
// GPU cross-team reduction kernels where the AOMP codegen doesn't pattern
// match the optimized codegen.
// =========================================================================

// Multiplication isn't detected by AOMP's pattern matching.
template <typename T> T red_mult(const T *__restrict in, uint64_t n) {
  T m = red_identity<T, RedOp::Mult>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(* : m)
  for (uint64_t i = 0; i < n; i++)
    m *= in[i];
  return m;
}

// Indirect reduction (sum) op isn't detected by AOMP's pattern matching.
template <typename T> T red_indirect(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  auto accumulate = [](T a, T b) { return a + b; };
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s = accumulate(s, in[i]);
  return s;
}

// Combined reduction (sum and max) in separate loops.
template <typename T>
T red_combined_separate(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  T m = red_identity<T, RedOp::Max>();
#pragma omp target map(tofrom : s, m)
#pragma omp teams TEAMS reduction(+ : s) reduction(max : m)
  {
#pragma omp distribute parallel for THREADS reduction(+ : s)
    for (uint64_t i = 0; i < n; i++)
      s += in[i];

#pragma omp distribute parallel for THREADS reduction(max : m)
    for (uint64_t i = 0; i < n; i++)
      m = std::max(m, in[i]);
  }
  return (s / 2) + (m / 2);
}

// Have a reduction in a kernel that is also doing something completely
// unrelated to the reduction (pure register work, no memory ops).
template <typename T> T red_kernel_part(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();

#pragma omp target map(tofrom : s)
#pragma omp teams TEAMS reduction(+ : s)
  {
#pragma omp distribute parallel for THREADS reduction(+ : s)
    for (uint64_t i = 0; i < n; i++)
      s += in[i];

    // Just do something, without actually doing anything
#pragma omp parallel THREADS
    {
      int tid = omp_get_thread_num();
      T x = static_cast<T>(tid);
      for (int j = 0; j < 100; j++)
        x = x * static_cast<T>(0.9) + static_cast<T>(j);
      if (x == static_cast<T>(-1))
        s += x;
    }
  }

  return s;
}

// =========================================================================
// GPU cross-team scan kernels (compilation unsupported by vanilla AOMP)
// =========================================================================

#ifdef AOMP_DEV
template <typename T>
void scan_sum_incl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    s += in[i];
#pragma omp scan inclusive(s)
    out[i] = s;
  }
}

template <typename T>
void scan_sum_excl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
#pragma omp scan exclusive(s)
    s += in[i];
  }
}

template <typename T>
void scan_max_incl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, max : m)
  for (uint64_t i = 0; i < n; i++) {
    m = std::max(m, in[i]);
#pragma omp scan inclusive(m)
    out[i] = m;
  }
}

template <typename T>
void scan_max_excl(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, max : m)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = m;
#pragma omp scan exclusive(m)
    m = std::max(m, in[i]);
  }
}

template <typename T>
void scan_dot_incl(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
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
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
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
void scan_dot_incl(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {}
template <typename T>
void scan_dot_excl(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {}
#endif // defined(AOMP_DEV)

// =========================================================================
// Benchmark harness (OMP-specific run functions)
// =========================================================================

template <typename T, bool is_fp, SimulationLike<T> Sim, typename Kernel,
          typename... Inputs>
std::optional<TimingResult> run_bench_scan(Kernel kernel, T *out, const T *gold,
                                           uint64_t n, std::string_view label,
                                           Sim *sim, Inputs... inputs) {
  std::vector<double> times;
  double total_time = 0.0;

  for (int t = 0; t < conf.warmup_iters + conf.bench_iters_scan; t++) {
    if (sim)
      sim->reset_device();
    auto t1 = Clock::now();
    kernel(inputs..., out, n);
    auto t2 = Clock::now();
#pragma omp target update from(out[0 : n])
    if (!check<T, is_fp>(out, gold, n, label))
      return std::nullopt;

    if (t < conf.warmup_iters)
      continue;
    double d = duration_cast(t2 - t1).count();
    times.push_back(d);
    if (conf.auto_scale) {
      total_time += d;
      if (total_time >= AUTO_SCALE_TIME && times.size() >= BENCH_MIN_ITERS)
        break;
    }
  }

  return create_timing_result(times, sizeof(T) * n * sizeof...(Inputs));
}

template <typename T, bool is_fp, SimulationLike<T> Sim, typename Kernel,
          typename... Inputs>
std::optional<TimingResult> run_bench_red(Kernel kernel, T gold, uint64_t n,
                                          std::string_view label, Sim *sim,
                                          Inputs... inputs) {
  std::vector<double> times;
  double total_time = 0.0;

  for (int t = 0; t < conf.warmup_iters + conf.bench_iters_reduction; t++) {
    if (sim)
      sim->reset_device();
    auto t1 = Clock::now();
    T result = kernel(inputs..., n);
    auto t2 = Clock::now();
    if (!check_single<T, is_fp>(result, gold, label))
      return std::nullopt;

    if (t < conf.warmup_iters)
      continue;
    double d = duration_cast(t2 - t1).count();
    times.push_back(d);
    if (conf.auto_scale) {
      total_time += d;
      if (total_time >= AUTO_SCALE_TIME && times.size() >= BENCH_MIN_ITERS)
        break;
    }
  }

  return create_timing_result(times, sizeof(T) * n * sizeof...(Inputs));
}

// Run a simple reduction (e.g., sum/max/min/mult/or) and all its simulation
// variants.
template <typename T, bool is_fp, RedOp Op, SimulationLike<T> Sim,
          typename Kernel>
void run_red_simple(Kernel kernel, const T *in, uint64_t n,
                    std::string_view type_name, Sim *sim) {
  std::optional<TimingResult> r;

  T gold = gold_red<T, Op>(in, n);
  if (conf.reduction) {
    r = run_bench_red<T, is_fp>(kernel, gold, n, red_op_to_str<Op>("red_{}"),
                                sim, in);
    print_result(red_op_to_str<Op>("red_{}"), type_name, n, r);
  }
  if (conf.reduction_simulation) {
    for (const auto &[name, func] : sim->template get_all_red_variants<Op>()) {
      r = run_bench_red<T, is_fp>(func, gold, n, name, sim, in);
      print_result(name, type_name, n, r);
    }
  }
}

// Run a simple scan (e.g., sum/max/min/mult) and all its simulation variants.
template <typename T, bool is_fp, RedOp Op, ScanMode Mode,
          SimulationLike<T> Sim, typename Kernel>
void run_scan_simple(Kernel kernel, T *gold, const T *in, T *out, uint64_t n,
                     std::string_view type_name, Sim *sim) {
  std::optional<TimingResult> r;

  gold_scan<T, Op, Mode>(in, gold, n);
  if (conf.scan) {
    r = run_bench_scan<T, is_fp>(kernel, out, gold, n,
                                 scan_op_to_str<Op, Mode>("scan_{}"), sim, in);
    print_result(scan_op_to_str<Op, Mode>("scan_{}"), type_name, n, r);
  }
  if (conf.scan_simulation) {
    for (const auto &[name, func] :
         sim->template get_all_scan_variants<Op, Mode>()) {
      r = run_bench_scan<T, is_fp>(func, out, gold, n, name, sim, in);
      print_result(name, type_name, n, r);
    }
  }
}

// =========================================================================
// Templated per-type benchmark runner
// =========================================================================

template <typename T, bool is_fp> void run_type(std::string_view type_name) {
  std::optional<TimingResult> r;

#ifdef AOMP
  SimulationAOMP<T> *simulation = new SimulationAOMP<T>();
#elif defined(TRUNK)
  SimulationTrunk<T> *simulation = new SimulationTrunk<T>();
#elif defined(TRUNK_JD)
  SimulationTrunkJD<T> *simulation = new SimulationTrunkJD<T>();
#elif defined(TRUNK_DEV)
  SimulationTrunkDev<T> *simulation = new SimulationTrunkDev<T>();
#elif defined(AOMP_DEV)
  SimulationAOMPDev<T> *simulation = new SimulationAOMPDev<T>();
#else
  SimulationNoop<T> *simulation = new SimulationNoop<T>();
#endif
  simulation->init_device();

  for (uint64_t n : conf.array_sizes) {
    T *in1 = alloc<T>(n);
    T *in2 = alloc<T>(n);
    T *out = alloc<T>(n);
    init_data<T, is_fp>(in1, in2, n);

#pragma omp target enter data map(to : in1[0 : n], in2[0 : n], out[0 : n])

    simulation->reset_device();

    if (conf.reduction || conf.reduction_simulation) {
      // Cross-team reductions (codegen + simulations)
      T gold;

      // ================================================================
      // dot reduction
      // ================================================================
      gold = gold_red_dot(in1, in2, n);
      if (conf.reduction) {
        r = run_bench_red<T, is_fp>(red_dot<T>, gold, n, "red_dot", simulation,
                                    in1, in2);
        print_result("red_dot", type_name, n, r);
      }
      if (conf.reduction_simulation) {
        for (const auto &[name, func] :
             simulation->get_all_red_dot_variants()) {
          r = run_bench_red<T, is_fp>(func, gold, n, name, simulation, in1,
                                      in2);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // max reduction
      // ================================================================
      run_red_simple<T, is_fp, RedOp::Max>(red_max<T>, in1, n, type_name,
                                           simulation);

      // ================================================================
      // sum reduction
      // ================================================================
      run_red_simple<T, is_fp, RedOp::Sum>(red_sum<T>, in1, n, type_name,
                                           simulation);

      if (!conf.quick_run || std::is_same_v<T, double>) {
        // ================================================================
        // mult reduction
        // ================================================================
        run_red_simple<T, is_fp, RedOp::Mult>(red_mult<T>, in1, n, type_name,
                                              simulation);

        // ================================================================
        // indirect reduction (sum)
        // ================================================================
        gold = gold_red<T, RedOp::Sum>(in1, n);
        if (conf.reduction) {
          r = run_bench_red<T, is_fp>(red_indirect<T>, gold, n, "red_indirect",
                                      simulation, in1);
          print_result("red_indirect", type_name, n, r);
        }

        // ================================================================
        // reduction (sum) in a kernel that is also doing something completely
        // unrelated to the reduction.
        // ================================================================
        gold = gold_red<T, RedOp::Sum>(in1, n);
        if (conf.reduction) {
          r = run_bench_red<T, is_fp>(red_kernel_part<T>, gold, n,
                                      "red_kernel_part", simulation, in1);
          print_result("red_kernel_part", type_name, n, r);
        }

        // ================================================================
        // combined reduction - in the same loop ...
        // ================================================================
        gold = (gold_red<T, RedOp::Sum>(in1, n) / 2) +
               (gold_red<T, RedOp::Max>(in1, n) / 2);
        if (conf.reduction) {
          r = run_bench_red<T, is_fp>(red_combined<T>, gold, n, "red_combined",
                                      simulation, in1);
          print_result("red_combined", type_name, n, r);
        }
        // ================================================================
        // ... and in separate loops
        // ================================================================
        if (conf.reduction) {
          r = run_bench_red<T, is_fp>(red_combined_separate<T>, gold, n,
                                      "red_combined_separate", simulation, in1);
          print_result("red_combined_separate", type_name, n, r);
        }
      }
    }

    if (conf.scan || conf.scan_simulation) {
      // Cross-team scans (codegen + simulations)
      T *gold = alloc<T>(n);

      simulation->reset_device();

      // ================================================================
      // exclusive dot scan
      // ================================================================
      gold_scan_dot<T, ScanMode::Excl>(in1, in2, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_dot_excl<T>, out, gold, n,
                                     "scan_dot_excl", simulation, in1, in2);
        print_result("scan_dot_excl", type_name, n, r);
      }

      if (conf.scan_simulation) {
        for (const auto &[name, func] :
             simulation->get_all_scan_dot_excl_variants()) {
          r = run_bench_scan<T, is_fp>(func, out, gold, n, name, simulation,
                                       in1, in2);
          print_result(name, type_name, n, r);
        }
      }

      // ================================================================
      // inclusive dot scan
      // ================================================================
      gold_scan_dot<T, ScanMode::Incl>(in1, in2, gold, n);

      if (conf.scan) {
        r = run_bench_scan<T, is_fp>(scan_dot_incl<T>, out, gold, n,
                                     "scan_dot_incl", simulation, in1, in2);
        print_result("scan_dot_incl", type_name, n, r);
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
      // exclusive max scan
      // ================================================================
      run_scan_simple<T, is_fp, RedOp::Max, ScanMode::Excl>(
          scan_max_excl<T>, gold, in1, out, n, type_name, simulation);

      // ================================================================
      // inclusive max scan
      // ================================================================
      run_scan_simple<T, is_fp, RedOp::Max, ScanMode::Incl>(
          scan_max_incl<T>, gold, in1, out, n, type_name, simulation);

      // ================================================================
      // exclusive sum scan
      // ================================================================
      run_scan_simple<T, is_fp, RedOp::Sum, ScanMode::Excl>(
          scan_sum_excl<T>, gold, in1, out, n, type_name, simulation);

      // ================================================================
      // inclusive sum scan
      // ================================================================
      run_scan_simple<T, is_fp, RedOp::Sum, ScanMode::Incl>(
          scan_sum_incl<T>, gold, in1, out, n, type_name, simulation);

      free(gold);
    }

#pragma omp target exit data map(delete : in1[0 : n], in2[0 : n], out[0 : n])

    free(in1);
    free(in2);
    free(out);
  }

  simulation->free_device();
  delete simulation;

  if (conf.reduction && std::is_same_v<T, double>) {
    // ================================================================
    // reduction computing Pi
    // ================================================================
    double gold_pi = std::numbers::pi;
    uint64_t n = 5000000000;
    if (conf.reduction) {
      r = run_bench_red<double, true>(
          red_pi, gold_pi, n, "red_pi",
          static_cast<SimulationNoop<double> *>(nullptr));
      print_result("red_pi", type_name, n, r);
    }
  }
}

static void usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0
      << " [-b <int>] [-B <int>] [-q] [-r] [-s] [-R] [-S] [-w <int>] [-h]\n"
      << "  -a: auto-scale benchmark iterations such that the runtime "
         "per test is ~"
      << AUTO_SCALE_TIME << " second (min " << BENCH_MIN_ITERS
      << " iterations)\n"
      << "  -b N: Benchmark iterations for reduction\n"
      << "  -B N: Benchmark iterations for scan\n"
      << "  -q: Quick run (test only one array size)\n"
      << "  -r: Run reduction tests\n"
      << "  -s: Run scan tests\n"
      << "  -R: Run reduction simulations\n"
      << "  -S: Run scan simulations\n"
      << "  -w N: Warmup iterations\n"
      << "  -h: Show this help message\n"

      << "\nNote that at least one of -r, -s, -R, -S must be specified.\n"
      << "\nPseudocode of how the benchmark binaries run the tests:\n"
      << "  for each data type in alphabetical order (e.g. double, int, "
         "long):\n"
      << "    for each array size in numerical order:\n"
      << "      for each test type in alphabetical order (first all "
         "reductions, then all scans):\n"
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
int main(int argc, char **argv) {
  int opt;

  while ((opt = getopt(argc, argv, "ab:B:qrsRSw:h")) != -1) {
    switch (opt) {
    case 'a':
      conf.auto_scale = true;
      break;
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
        std::cerr << "warning: scan codegen unsupported on this compiler - "
                     "ignoring\n";
      break;
    case 'R':
      conf.reduction_simulation = true;
      break;
    case 'S':
#if !defined(TRUNK) && !defined(TRUNK_DEV)
      conf.scan_simulation = true;
#endif
      if (!conf.scan_simulation)
        std::cerr << "warning: scan simulations unsupported on this compiler - "
                     "ignoring\n";
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

  if (conf.auto_scale) {
    conf.bench_iters_reduction =
        std::numeric_limits<int>::max() - conf.warmup_iters;
    conf.bench_iters_scan = std::numeric_limits<int>::max() - conf.warmup_iters;
  }

  std::cout << std::format(
      "xteam benchmark (quick run: {}, auto-scale: {}) — {} warmup, {} timed "
      "iterations "
      "(reduction), {} timed iterations (scan), "
      "{} teams, {} threads, codegen autodetection: {}\n",
      conf.quick_run ? "true" : "false", conf.auto_scale ? "true" : "false",
      conf.warmup_iters,
      conf.auto_scale ? "auto-scaled"
                      : std::to_string(conf.bench_iters_reduction),
      conf.auto_scale ? "auto-scaled" : std::to_string(conf.bench_iters_scan),
      XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS,
      CODEGEN_AUTODETECTION ? "true" : "false");

  std::cout << "Array sizes: ";
  for (uint64_t sz : conf.array_sizes)
    std::cout << " " << fmt_num_sep(std::format("{}", sz));
  std::cout << "\n\n";

  print_header();

  std::cout << "\n--- double ---\n";
  run_type<double, true>("double");

  std::cout << "\n--- uint ---\n";
  run_type<unsigned, false>("uint");

  std::cout << "\n--- ulong ---\n";
  run_type<unsigned long, false>("ulong");

  return EXIT_SUCCESS;
}
