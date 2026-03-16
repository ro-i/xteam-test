// xteam_bench.cpp — OpenMP cross-team performance & correctness benchmark

#include <omp.h>

#include "bench_common.h"

#if SCAN_SIMULATION || REDUCTION_SIMULATION
#ifdef AOMP
#include "xteam_simulations_aomp.h"
#else
#include "xteam_simulations.h"
#endif
#else
template <typename T> void init_device_sim() {}
template <typename T> void reset_device_sim() {}
template <typename T> void free_device_sim() {}
#endif

// =========================================================================
// GPU cross-team reduction kernels
// =========================================================================

template <typename T> T reduce_sum(const T *__restrict in, uint64_t n) {
  T s = T(0);
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
  T m = std::numeric_limits<T>::lowest();
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
  T m = std::numeric_limits<T>::max();
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
  T s = T(0);
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
// GPU cross-team scan kernels
// =========================================================================

#if !defined(AOMP) && SCAN_TEST
template <typename T>
void scan_incl_sum(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = T(0);
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
void scan_excl_sum(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = T(0);
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
void scan_incl_max(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::lowest();
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
void scan_excl_max(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::lowest();
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
void scan_incl_min(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::max();
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
void scan_excl_min(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::max();
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
void scan_incl_dot(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {
  T s = T(0);
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
void scan_excl_dot(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {
  T s = T(0);
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
#endif // !AOMP && SCAN_TEST

// =========================================================================
// Benchmark harness (OMP-specific run functions)
// =========================================================================

template <typename T, bool is_fp, typename Kernel, typename... Inputs>
std::optional<TimingResult> run_bench_scan(Kernel kernel, T *out, const T *gold,
                                           uint64_t n, const char *label,
                                           bool is_sim, Inputs... inputs) {
  std::vector<double> times(BENCH_ITERS_SCAN);
  for (int t = 0; t < WARMUP_ITERS + BENCH_ITERS_SCAN; t++) {
    if (is_sim)
      reset_device_sim<T>();
    auto t1 = Clock::now();
    kernel(inputs..., out, n);
    auto t2 = Clock::now();
    if (t >= WARMUP_ITERS) {
      times[t - WARMUP_ITERS] =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
              .count();
    }
#pragma omp target update from(out[0 : n])
    if (!check<T, is_fp>(out, gold, n, label))
      return std::nullopt;
  }

  return create_timing_result(times, n, sizeof(T) * n * sizeof...(Inputs));
}

template <typename T, bool is_fp, typename Kernel, typename... Inputs>
std::optional<TimingResult> run_bench_reduce(Kernel kernel, T gold, uint64_t n,
                                             const char *label,
                                             Inputs... inputs) {
  std::vector<double> times(BENCH_ITERS_REDUCTION);
  for (int t = 0; t < WARMUP_ITERS + BENCH_ITERS_REDUCTION; t++) {
    auto t1 = Clock::now();
    T result = kernel(inputs..., n);
    auto t2 = Clock::now();
    if (t >= WARMUP_ITERS) {
      times[t - WARMUP_ITERS] =
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
  for (uint64_t n : array_sizes) {
    T *in1 = alloc<T>(n);
    T *in2 = alloc<T>(n);
    T *out = alloc<T>(n);
    init_data<T, is_fp>(in1, in2, n);
    std::optional<TimingResult> r;

#pragma omp target enter data map(to : in1[0 : n], in2[0 : n], out[0 : n])

    // Cross-team reductions (codegen + simulations)
    T gold_val;
    T *gold = static_cast<T *>(malloc(sizeof(T) * n));

#if REDUCTION_SIMULATION
    init_device_sim<T>();
#endif

    gold_val = gold_reduce_sum(in1, n);
#if REDUCTION_TEST
    r = run_bench_reduce<T, is_fp>(reduce_sum<T>, gold_val, n, "red_sum", in1);
    print_result("red_sum", type_name, n, r);
#endif
#if REDUCTION_SIMULATION
    r = run_bench_reduce<T, is_fp>(reduce_sim<T, ScanOp::Sum>, gold_val, n,
                                   "red_sum_sim", in1);
    print_result("red_sum_sim", type_name, n, r);
#endif

    gold_val = gold_reduce_max(in1, n);
#if REDUCTION_TEST
    r = run_bench_reduce<T, is_fp>(reduce_max<T>, gold_val, n, "red_max", in1);
    print_result("red_max", type_name, n, r);
#endif
#if REDUCTION_SIMULATION
    r = run_bench_reduce<T, is_fp>(reduce_sim<T, ScanOp::Max>, gold_val, n,
                                   "red_max_sim", in1);
    print_result("red_max_sim", type_name, n, r);
#endif

    gold_val = gold_reduce_min(in1, n);
#if REDUCTION_TEST
    r = run_bench_reduce<T, is_fp>(reduce_min<T>, gold_val, n, "red_min", in1);
    print_result("red_min", type_name, n, r);
#endif
#if REDUCTION_SIMULATION
    r = run_bench_reduce<T, is_fp>(reduce_sim<T, ScanOp::Min>, gold_val, n,
                                   "red_min_sim", in1);
    print_result("red_min_sim", type_name, n, r);
#endif

    gold_val = gold_reduce_dot(in1, in2, n);
#if REDUCTION_TEST
    r = run_bench_reduce<T, is_fp>(reduce_dot<T>, gold_val, n, "red_dot", in1,
                                   in2);
    print_result("red_dot", type_name, n, r);
#endif
#if REDUCTION_SIMULATION
    r = run_bench_reduce<T, is_fp>(reduce_dot_sim<T>, gold_val, n,
                                   "red_dot_sim", in1, in2);
    print_result("red_dot_sim", type_name, n, r);
#endif

#if REDUCTION_SIMULATION
    free_device_sim<T>();
#endif

    // Cross-team scans (codegen + simulations)

#if SCAN_SIMULATION
    init_device_sim<T>();
#endif

    gold_inclusive_sum(in1, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_incl_sum<T>, out, gold, n, "incl_sum",
                                 false, in1);
    print_result("incl_sum", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_incl_sim<T, ScanOp::Sum>, out, gold, n,
                                 "incl_sum_sim", true, in1);
    print_result("incl_sum_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_incl_sim_v1<T, ScanOp::Sum>, out, gold, n,
                                 "incl_sum_sim_v1", true, in1);
    print_result("incl_sum_sim_v1", type_name, n, r);
#endif
#endif

    gold_exclusive_sum(in1, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_excl_sum<T>, out, gold, n, "excl_sum",
                                 false, in1);
    print_result("excl_sum", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_excl_sim<T, ScanOp::Sum>, out, gold, n,
                                 "excl_sum_sim", true, in1);
    print_result("excl_sum_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_excl_sim_v1<T, ScanOp::Sum>, out, gold, n,
                                 "excl_sum_sim_v1", true, in1);
    print_result("excl_sum_sim_v1", type_name, n, r);
#endif
#endif

    gold_inclusive_max(in1, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_incl_max<T>, out, gold, n, "incl_max",
                                 false, in1);
    print_result("incl_max", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_incl_sim<T, ScanOp::Max>, out, gold, n,
                                 "incl_max_sim", true, in1);
    print_result("incl_max_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_incl_sim_v1<T, ScanOp::Max>, out, gold, n,
                                 "incl_max_sim_v1", true, in1);
    print_result("incl_max_sim_v1", type_name, n, r);
#endif
#endif

    gold_exclusive_max(in1, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_excl_max<T>, out, gold, n, "excl_max",
                                 false, in1);
    print_result("excl_max", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_excl_sim<T, ScanOp::Max>, out, gold, n,
                                 "excl_max_sim", true, in1);
    print_result("excl_max_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_excl_sim_v1<T, ScanOp::Max>, out, gold, n,
                                 "excl_max_sim_v1", true, in1);
    print_result("excl_max_sim_v1", type_name, n, r);
#endif
#endif

    gold_inclusive_min(in1, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_incl_min<T>, out, gold, n, "incl_min",
                                 false, in1);
    print_result("incl_min", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_incl_sim<T, ScanOp::Min>, out, gold, n,
                                 "incl_min_sim", true, in1);
    print_result("incl_min_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_incl_sim_v1<T, ScanOp::Min>, out, gold, n,
                                 "incl_min_sim_v1", true, in1);
    print_result("incl_min_sim_v1", type_name, n, r);
#endif
#endif

    gold_exclusive_min(in1, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_excl_min<T>, out, gold, n, "excl_min",
                                 false, in1);
    print_result("excl_min", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_excl_sim<T, ScanOp::Min>, out, gold, n,
                                 "excl_min_sim", true, in1);
    print_result("excl_min_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_excl_sim_v1<T, ScanOp::Min>, out, gold, n,
                                 "excl_min_sim_v1", true, in1);
    print_result("excl_min_sim_v1", type_name, n, r);
#endif
#endif

    gold_inclusive_dot(in1, in2, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_incl_dot<T>, out, gold, n, "incl_dot",
                                 false, in1, in2);
    print_result("incl_dot", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_incl_dot_sim<T>, out, gold, n,
                                 "incl_dot_sim", true, in1, in2);
    print_result("incl_dot_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_incl_dot_sim_v1<T>, out, gold, n,
                                 "incl_dot_sim_v1", true, in1, in2);
    print_result("incl_dot_sim_v1", type_name, n, r);
#endif
#endif

    gold_exclusive_dot(in1, in2, gold, n);
#if !defined(AOMP) && SCAN_TEST
    r = run_bench_scan<T, is_fp>(scan_excl_dot<T>, out, gold, n, "excl_dot",
                                 false, in1, in2);
    print_result("excl_dot", type_name, n, r);
#endif
#if SCAN_SIMULATION
    r = run_bench_scan<T, is_fp>(scan_excl_dot_sim<T>, out, gold, n,
                                 "excl_dot_sim", true, in1, in2);
    print_result("excl_dot_sim", type_name, n, r);
#ifndef AOMP
    r = run_bench_scan<T, is_fp>(scan_excl_dot_sim_v1<T>, out, gold, n,
                                 "excl_dot_sim_v1", true, in1, in2);
    print_result("excl_dot_sim_v1", type_name, n, r);
#endif
#endif

#if SCAN_SIMULATION
    free_device_sim<T>();
#endif

#pragma omp target exit data map(delete : in1[0 : n], in2[0 : n], out[0 : n])

    free(in1);
    free(in2);
    free(out);
    free(gold);
  }
}

// =========================================================================
// Main
// =========================================================================
int main(int argc, char **argv) {
  std::cout << std::format("xteam benchmark — {} warmup, {} timed iterations "
                           "(reduction), {} timed iterations (scan), "
                           "{} teams, {} threads, codegen autodetection: {}\n",
                           WARMUP_ITERS, BENCH_ITERS_REDUCTION,
                           BENCH_ITERS_SCAN, XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS,
                           CODEGEN_AUTODETECTION ? "true" : "false");
  std::cout << "Array sizes: ";
  for (uint64_t sz : array_sizes)
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
