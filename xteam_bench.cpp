// xteam_bench.cpp — OpenMP cross-team performance & correctness benchmark

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "common.h"
#if (SCAN_SIMULATION || REDUCTION_SIMULATION) && !defined(AOMP)
#include "xteam_simulations.h"
#endif

#if QUICK_RUN
static const std::array<uint64_t, 1> array_sizes{41943040};
#else
static const std::array<uint64_t, 14> array_sizes{
    1,     100,     1024,    2048,     4096,     8192,      10000,
    81920, 1000000, 4194304, 23445657, 41943040, 100000000, 177777777};
#endif

using Clock = std::chrono::high_resolution_clock;

struct TimingResult {
  double min_s, max_s, avg_s;
  double best_mbps, avg_mbps;
};

// =========================================================================
// Utility functions
// =========================================================================

template <typename T, bool is_fp> void init_data(T *arr1, T *arr2, uint64_t n) {
  srand(42); // always generate the same "random" numbers
  for (uint64_t i = 0; i < n; i++) {
    if constexpr (is_fp) {
      arr1[i] = T((rand() % 100) / 100.0);
      arr2[i] = T((rand() % 100) / 100.0);
    } else {
      arr1[i] = T(rand() % 1000);
      arr2[i] = T(rand() % 1000);
    }
  }
}

// =========================================================================
// Gold (CPU) reference implementations
// =========================================================================

// Gold cross-team reductions
template <typename T> T gold_reduce_sum(const T *in, uint64_t n) {
  T a = T(0);
  for (uint64_t i = 0; i < n; i++)
    a += in[i];
  return a;
}
template <typename T> T gold_reduce_max(const T *in, uint64_t n) {
  T a = std::numeric_limits<T>::lowest();
  for (uint64_t i = 0; i < n; i++)
    a = std::max(a, in[i]);
  return a;
}
template <typename T> T gold_reduce_min(const T *in, uint64_t n) {
  T a = std::numeric_limits<T>::max();
  for (uint64_t i = 0; i < n; i++)
    a = std::min(a, in[i]);
  return a;
}
template <typename T> T gold_reduce_dot(const T *a, const T *b, uint64_t n) {
  T s = T(0);
  for (uint64_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

// Gold cross-team scans
template <typename T> void gold_inclusive_sum(const T *in, T *out, uint64_t n) {
  T a = T(0);
  for (uint64_t i = 0; i < n; i++) {
    a += in[i];
    out[i] = a;
  }
}
template <typename T> void gold_exclusive_sum(const T *in, T *out, uint64_t n) {
  T a = T(0);
  for (uint64_t i = 0; i < n; i++) {
    out[i] = a;
    a += in[i];
  }
}
template <typename T> void gold_inclusive_max(const T *in, T *out, uint64_t n) {
  T a = std::numeric_limits<T>::lowest();
  for (uint64_t i = 0; i < n; i++) {
    a = std::max(a, in[i]);
    out[i] = a;
  }
}
template <typename T> void gold_exclusive_max(const T *in, T *out, uint64_t n) {
  T a = std::numeric_limits<T>::lowest();
  for (uint64_t i = 0; i < n; i++) {
    out[i] = a;
    a = std::max(a, in[i]);
  }
}
template <typename T> void gold_inclusive_min(const T *in, T *out, uint64_t n) {
  T a = std::numeric_limits<T>::max();
  for (uint64_t i = 0; i < n; i++) {
    a = std::min(a, in[i]);
    out[i] = a;
  }
}
template <typename T> void gold_exclusive_min(const T *in, T *out, uint64_t n) {
  T a = std::numeric_limits<T>::max();
  for (uint64_t i = 0; i < n; i++) {
    out[i] = a;
    a = std::min(a, in[i]);
  }
}
template <typename T>
void gold_inclusive_dot(const T *a, const T *b, T *out, uint64_t n) {
  T s = T(0);
  for (uint64_t i = 0; i < n; i++) {
    s += a[i] * b[i];
    out[i] = s;
  }
}
template <typename T>
void gold_exclusive_dot(const T *a, const T *b, T *out, uint64_t n) {
  T s = T(0);
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
    s += a[i] * b[i];
  }
}

// =========================================================================
// GPU cross-team reduction kernels
// =========================================================================

template <typename T> T reduce_sum(const T *__restrict in, uint64_t n) {
  T s = T(0);
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s += in[i];
  return s;
}

template <typename T> T reduce_max(const T *__restrict in, uint64_t n) {
  T m = std::numeric_limits<T>::lowest();
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(max : m)
  for (uint64_t i = 0; i < n; i++)
    if (in[i] > m)
      m = in[i];
  return m;
}

template <typename T> T reduce_min(const T *__restrict in, uint64_t n) {
  T m = std::numeric_limits<T>::max();
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(min : m)
  for (uint64_t i = 0; i < n; i++)
    if (in[i] < m)
      m = in[i];
  return m;
}

template <typename T>
T reduce_dot(const T *__restrict a, const T *__restrict b, uint64_t n) {
  T s = T(0);
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

// =========================================================================
// GPU cross-team scan kernels
// =========================================================================

#ifndef AOMP
template <typename T>
void scan_incl_sum(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = T(0);
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    s += in[i];
#pragma omp scan inclusive(s)
    out[i] = s;
  }
}

template <typename T>
void scan_excl_sum(const T *__restrict in, T *__restrict out, uint64_t n) {
  T s = T(0);
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
#pragma omp scan exclusive(s)
    s += in[i];
  }
}

template <typename T>
void scan_incl_max(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::lowest();
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, max : m)
  for (uint64_t i = 0; i < n; i++) {
    if (in[i] > m)
      m = in[i];
#pragma omp scan inclusive(m)
    out[i] = m;
  }
}

template <typename T>
void scan_excl_max(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::lowest();
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, max : m)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = m;
#pragma omp scan exclusive(m)
    if (in[i] > m)
      m = in[i];
  }
}

template <typename T>
void scan_incl_min(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::max();
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, min : m)
  for (uint64_t i = 0; i < n; i++) {
    if (in[i] < m)
      m = in[i];
#pragma omp scan inclusive(m)
    out[i] = m;
  }
}

template <typename T>
void scan_excl_min(const T *__restrict in, T *__restrict out, uint64_t n) {
  T m = std::numeric_limits<T>::max();
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, min : m)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = m;
#pragma omp scan exclusive(m)
    if (in[i] < m)
      m = in[i];
  }
}

template <typename T>
void scan_incl_dot(const T *__restrict a, const T *__restrict b,
                   T *__restrict out, uint64_t n) {
  T s = T(0);
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
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
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) reduction(inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
#pragma omp scan exclusive(s)
    s += a[i] * b[i];
  }
}
#endif // AOMP

// =========================================================================
// Benchmark harness
// =========================================================================

template <typename T, bool is_fp>
bool check_single(T computed, T gold, const char *label,
                  std::optional<uint64_t> index = std::nullopt) {
  if constexpr (!is_fp) {
    if (computed == gold)
      return true;
    if (index)
      std::cerr << std::format("FAIL {} at {}: got {}, expected {}\n", label,
                               *index, computed, gold);
    else
      std::cerr << std::format("FAIL {}: got {}, expected {}\n", label,
                               computed, gold);
    return false;
  }
  double g = (double)gold, c = (double)computed;
  double rel = (g != 0.0) ? std::abs((c - g) / g) : std::abs(c - g);
  if (rel <= 1e-6)
    return true;
  if (index)
    std::cerr << std::format("FAIL {} at {}: got {}, expected {} (rel={})\n",
                             label, *index, c, g, rel);
  else
    std::cerr << std::format("FAIL {}: got {}, expected {} (rel={})\n", label,
                             c, g, rel);
  return false;
}

template <typename T, bool is_fp>
bool check(const T *computed, const T *gold, uint64_t n, const char *label) {
  for (uint64_t i = 0; i < n; i++) {
    if (!check_single<T, is_fp>(computed[i], gold[i], label, i))
      return false;
  }
  return true;
}

TimingResult create_timing_result(const std::vector<double> &times, uint64_t n,
                                  uint64_t data_bytes) {
  auto [mn, mx] = std::minmax_element(times.begin(), times.end());
  double avg =
      std::accumulate(times.begin(), times.end(), 0.0) / (double)(times.size());

  return TimingResult{*mn, *mx, avg, 1e-6 * data_bytes / *mn,
                      1e-6 * data_bytes / avg};
}

template <typename T, bool is_fp, typename Kernel, typename... Inputs>
std::optional<TimingResult> run_bench_scan(Kernel kernel, T *out, const T *gold,
                                           uint64_t n, const char *label,
                                           Inputs... inputs) {
  std::vector<double> times(BENCH_ITERS);
  for (int t = 0; t < WARMUP_ITERS + BENCH_ITERS; t++) {
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
  std::vector<double> times(BENCH_ITERS);
  for (int t = 0; t < WARMUP_ITERS + BENCH_ITERS; t++) {
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

void print_result(const char *test, const char *type, uint64_t n,
                  const std::optional<TimingResult> &r) {
  if (!r) {
    std::cerr << std::format("{:<24} {:<8} {:>10}  FAIL\n", test, type, n);
    return;
  }
  std::cout << std::format("{:<24} {:<8} {:>10}  {:>10.6f}  {:>10.6f}  "
                           "{:>10.6f}  {:>10.0f}  {:>10.0f}\n",
                           test, type, n, r->min_s, r->max_s, r->avg_s,
                           r->best_mbps, r->avg_mbps);
}

// =========================================================================
// Templated per-type benchmark runner
// =========================================================================

template <typename T, bool is_fp> void run_type(const char *type_name) {
  for (uint64_t n : array_sizes) {
    T *in1 = alloc<T>(n);
    T *in2 = alloc<T>(n);
    T *out = alloc<T>(n);
    T *gold = alloc<T>(n);
    init_data<T, is_fp>(in1, in2, n);
    std::optional<TimingResult> r;

#pragma omp target enter data map(to : in1[0 : n], in2[0 : n], out[0 : n],     \
                                      gold[0 : n])

    // Cross-team reductions
    T gold_val;

    gold_val = gold_reduce_sum(in1, n);
    r = run_bench_reduce<T, is_fp>(reduce_sum<T>, gold_val, n, "red_sum", in1);
    print_result("red_sum", type_name, n, r);

    gold_val = gold_reduce_max(in1, n);
    r = run_bench_reduce<T, is_fp>(reduce_max<T>, gold_val, n, "red_max", in1);
    print_result("red_max", type_name, n, r);

    gold_val = gold_reduce_min(in1, n);
    r = run_bench_reduce<T, is_fp>(reduce_min<T>, gold_val, n, "red_min", in1);
    print_result("red_min", type_name, n, r);

    gold_val = gold_reduce_dot(in1, in2, n);
    r = run_bench_reduce<T, is_fp>(reduce_dot<T>, gold_val, n, "red_dot", in1,
                                   in2);
    print_result("red_dot", type_name, n, r);

#ifndef AOMP
#if REDUCTION_SIMULATION
    // Cross-team reductions (simulation)
    init_device_sim<T>();

    gold_val = gold_reduce_sum(in1, n);
    r = run_bench_reduce<T, is_fp>(reduce_sum_sim<T>, gold_val, n, "red_sum_sim", in1);
    print_result("red_sum_sim", type_name, n, r);

    gold_val = gold_reduce_max(in1, n);
    r = run_bench_reduce<T, is_fp>(reduce_max_sim<T>, gold_val, n, "red_max_sim", in1);
    print_result("red_max_sim", type_name, n, r);

    gold_val = gold_reduce_min(in1, n);
    r = run_bench_reduce<T, is_fp>(reduce_min_sim<T>, gold_val, n, "red_min_sim", in1);
    print_result("red_min_sim", type_name, n, r);

    gold_val = gold_reduce_dot(in1, in2, n);
    r = run_bench_reduce<T, is_fp>(reduce_dot_sim<T>, gold_val, n, "red_dot_sim", in1, in2);
    print_result("red_dot_sim", type_name, n, r);

    free_device_sim<T>();
#endif // REDUCTION_SIMULATION

    // Cross-team scans
    gold_inclusive_sum(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_sum<T>, out, gold, n, "incl_sum",
                                 in1);
    print_result("incl_sum", type_name, n, r);

    gold_exclusive_sum(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_sum<T>, out, gold, n, "excl_sum",
                                 in1);
    print_result("excl_sum", type_name, n, r);

    gold_inclusive_max(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_max<T>, out, gold, n, "incl_max",
                                 in1);
    print_result("incl_max", type_name, n, r);

    gold_exclusive_max(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_max<T>, out, gold, n, "excl_max",
                                 in1);
    print_result("excl_max", type_name, n, r);

    gold_inclusive_min(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_min<T>, out, gold, n, "incl_min",
                                 in1);
    print_result("incl_min", type_name, n, r);

    gold_exclusive_min(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_min<T>, out, gold, n, "excl_min",
                                 in1);
    print_result("excl_min", type_name, n, r);

    gold_inclusive_dot(in1, in2, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_dot<T>, out, gold, n, "incl_dot",
                                 in1, in2);
    print_result("incl_dot", type_name, n, r);

    gold_exclusive_dot(in1, in2, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_dot<T>, out, gold, n, "excl_dot",
                                 in1, in2);
    print_result("excl_dot", type_name, n, r);

#if SCAN_SIMULATION
    // Cross-team scans (simulation)
    init_device_sim<T>();

    gold_inclusive_sum(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_sum_sim<T>, out, gold, n,
                                 "incl_sum_sim", in1);
    print_result("incl_sum_sim", type_name, n, r);

    gold_exclusive_sum(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_sum_sim<T>, out, gold, n,
                                 "excl_sum_sim", in1);
    print_result("excl_sum_sim", type_name, n, r);

    gold_inclusive_max(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_max_sim<T>, out, gold, n,
                                 "incl_max_sim", in1);
    print_result("incl_max_sim", type_name, n, r);

    gold_exclusive_max(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_max_sim<T>, out, gold, n,
                                 "excl_max_sim", in1);
    print_result("excl_max_sim", type_name, n, r);

    gold_inclusive_min(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_min_sim<T>, out, gold, n,
                                 "incl_min_sim", in1);
    print_result("incl_min_sim", type_name, n, r);

    gold_exclusive_min(in1, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_min_sim<T>, out, gold, n,
                                 "excl_min_sim", in1);
    print_result("excl_min_sim", type_name, n, r);

    gold_inclusive_dot(in1, in2, gold, n);
    r = run_bench_scan<T, is_fp>(scan_incl_dot_sim<T>, out, gold, n,
                                 "incl_dot_sim", in1, in2);
    print_result("incl_dot_sim", type_name, n, r);

    gold_exclusive_dot(in1, in2, gold, n);
    r = run_bench_scan<T, is_fp>(scan_excl_dot_sim<T>, out, gold, n,
                                 "excl_dot_sim", in1, in2);
    print_result("excl_dot_sim", type_name, n, r);

    free_device_sim<T>();
#endif // SCAN_SIMULATION

#endif // AOMP

#pragma omp target exit data map(delete : in1[0 : n], in2[0 : n], out[0 : n],  \
                                     gold[0 : n])

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
  std::cout << std::format("xteam benchmark — {} warmup, {} timed iterations\n",
                           WARMUP_ITERS, BENCH_ITERS);
  std::cout << "Array sizes: ";
  for (uint64_t sz : array_sizes)
    std::cout << std::format(" {}", sz);
  std::cout << "\n\n";

  std::cout << std::format(
      "{:>24} {:>8} {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}\n", "test",
      "type", "N", "min(s)", "max(s)", "avg(s)", "best MB/s", "avg MB/s");
  std::cout << std::format(
      "{:->24} {:->8} {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}\n",
      "------------------------", "--------", "----------", "----------",
      "----------", "----------", "----------", "----------");

  std::cout << "\n--- int ---\n";
  run_type<int, false>("int");

  std::cout << "\n--- long ---\n";
  run_type<long, false>("long");

  std::cout << "\n--- double ---\n";
  run_type<double, true>("double");

  return EXIT_SUCCESS;
}
