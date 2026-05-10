// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "omp.h"

#ifndef XTEAM_NUM_THREADS
#define XTEAM_NUM_THREADS 512
#endif
#ifndef XTEAM_NUM_TEAMS
#define XTEAM_NUM_TEAMS 208 // gfx90a number of CUs: 104
#endif
// If true, let codegen for reduction/scan determine num_teams and num_threads.
#ifndef CODEGEN_AUTODETECTION
#define CODEGEN_AUTODETECTION 0
#endif
#if CODEGEN_AUTODETECTION
#define TEAMS
#define THREADS
#else // CODEGEN_AUTODETECTION
#define TEAMS num_teams(XTEAM_NUM_TEAMS)
#define THREADS num_threads(XTEAM_NUM_THREADS)
#endif // CODEGEN_AUTODETECTION
#define TEAMS_THREADS TEAMS THREADS
// If true, use no-loop scan codegen.
#ifndef NOLOOP
#define NOLOOP 0
#endif

// Represents the total of threads in the Grid
#define XTEAM_TOTAL_NUM_THREADS (XTEAM_NUM_TEAMS * XTEAM_NUM_THREADS)

// Benchmark minimum number of measured iterations (as a lower bound in case the
// test is so slow that it wouldn't get enough iterations in the auto-scale
// timeframe)
#define BENCH_MIN_ITERS 10
// Auto-scale timeframe in seconds. Benchmarks will be repeated until they reach
// at least this amount of seconds.
#define AUTO_SCALE_TIME 1.0

// Floating point absolute and relative tolerance for comparison.
#define FP_ABS_TOL 1e-12
#define FP_REL_TOL 1e-6

// default alignment for aligned_alloc
#define ALIGNMENT 128

// Size of the device-side buffer used to evict the GPU cache between iterations
// Must be larger than any L2/MALL/Infinity-cache we expect to run on.
#define CACHE_EVICT_BYTES (512ull * 1024 * 1024)

#ifdef AOMP
#define COMPILER_LABEL "aomp"
#elif defined(AOMP_DEV)
#define COMPILER_LABEL "aomp_dev"
#elif defined(TRUNK)
#define COMPILER_LABEL "trunk"
#elif defined(TRUNK_DEV)
#define COMPILER_LABEL "trunk_dev"
#elif defined(TRUNK_JD)
#define COMPILER_LABEL "trunk_jd"
#else
#define COMPILER_LABEL "unknown"
#endif

#define duration_cast(x)                                                       \
  std::chrono::duration_cast<std::chrono::duration<double>>(x)
using Clock = std::chrono::steady_clock;

// Selected arrays sizes for quick run
extern const std::vector<uint64_t> array_sizes_quick;

// Array sizes for full run
#if NOLOOP
extern const std::vector<uint64_t> array_sizes;
#else
extern const std::vector<uint64_t> array_sizes;
#endif // NOLOOP

// Benchmark operation declarations.
// These are expected to be defined in the source file for the operation.
extern std::string bench_op_name;
extern void run_bench_op();

struct Config {
  bool auto_scale = true;
  bool quick_run = false;
  // Whether to run the non-simulation tests.
  bool run = false;
  // Whether to run the simulation tests.
  bool run_sim = false;
  // Whether to evict the GPU L2/MALL cache between iterations (cold-cache
  // mode).
  bool evict_cache = false;
  int warmup_iters = 2;
  int bench_iters = 10;
  std::vector<uint64_t> array_sizes;
};
// Global conf object, instantiated in xteam_bench.cpp
extern Config conf;

struct TimingResult {
  double min_s, max_s, avg_s;
  double best_mbps, avg_mbps;
};

// =========================================================================
// Utility functions
// =========================================================================

template <typename T> inline T *alloc(uint64_t n) {
  if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
    std::cerr << std::format("alloc size overflow n={} sizeof(T)={}\n", n,
                             sizeof(T));
    exit(EXIT_FAILURE);
  }
  size_t bytes = sizeof(T) * n;
  bytes = ((bytes + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

  T *ret = static_cast<T *>(aligned_alloc(ALIGNMENT, bytes));
  if (!ret) {
    std::cerr << std::format("aligned_alloc failed bytes={}\n", bytes);
    exit(EXIT_FAILURE);
  }
  return ret;
}

template <typename T> inline T *target_alloc(uint64_t n, int devid) {
  if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
    std::cerr << std::format("target_alloc size overflow n={} sizeof(T)={}\n",
                             n, sizeof(T));
    exit(EXIT_FAILURE);
  }
  T *ret = static_cast<T *>(omp_target_alloc(sizeof(T) * n, devid));
  if (!ret) {
    std::cerr << std::format("omp_target_alloc failed n={} devid={}\n", n,
                             devid);
    exit(EXIT_FAILURE);
  }
  return ret;
}

// Evict the GPU's L2/MALL cache by writing to a large enough device buffer.
void evict_device_cache();

// Deterministic initialization for reproducibility.
template <typename T> inline void init_data(T *arr1, T *arr2, uint64_t n) {
  srand(42);
  for (uint64_t i = 0; i < n; i++) {
    if constexpr (std::is_floating_point_v<T>) {
      arr1[i] = T((rand() % 100) / 100.0);
      arr2[i] = T((rand() % 100) / 100.0);
    } else {
      arr1[i] = T(rand() % 1000);
      arr2[i] = T(rand() % 1000);
    }
  }
}

// =========================================================================
// RedOp-generic helpers
// =========================================================================

enum class RedOp { Sum, Max, Min, Mult };
enum class ScanMode { Incl, Excl };

template <typename T, RedOp Op> constexpr T red_identity() {
  if constexpr (Op == RedOp::Sum)
    return T(0);
  else if constexpr (Op == RedOp::Max)
    return std::numeric_limits<T>::lowest();
  else if constexpr (Op == RedOp::Min)
    return std::numeric_limits<T>::max();
  else if constexpr (Op == RedOp::Mult)
    return T(1);
  else
    static_assert(!std::is_same_v<T, T>, "Unsupported red op");
}

template <typename T, RedOp Op> constexpr T red_combine(T a, T b) {
  if constexpr (Op == RedOp::Sum)
    return a + b;
  else if constexpr (Op == RedOp::Max)
    return std::max(a, b);
  else if constexpr (Op == RedOp::Min)
    return std::min(a, b);
  else if constexpr (Op == RedOp::Mult)
    return a * b;
  else
    static_assert(!std::is_same_v<T, T>, "Unsupported red op");
}

template <RedOp Op>
inline constexpr std::string red_op_to_str(std::string_view fmt) {
  if constexpr (Op == RedOp::Sum)
    return std::vformat(fmt, std::make_format_args("sum"));
  else if constexpr (Op == RedOp::Max)
    return std::vformat(fmt, std::make_format_args("max"));
  else if constexpr (Op == RedOp::Min)
    return std::vformat(fmt, std::make_format_args("min"));
  else if constexpr (Op == RedOp::Mult)
    return std::vformat(fmt, std::make_format_args("mult"));
  else
    static_assert(!std::is_same_v<RedOp, RedOp>, "Unsupported red op");
}

// =========================================================================
// Benchmark harness utilities
// =========================================================================

template <typename T>
inline bool check_single(T computed, T gold, std::string_view label,
                         std::optional<uint64_t> index = std::nullopt) {
  if constexpr (!std::is_floating_point_v<T>) {
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
  double g = static_cast<double>(gold), c = static_cast<double>(computed);
  double abs_err = std::abs(c - g);
  double scale = std::max({1.0, std::abs(g), std::abs(c)});
  double rel = abs_err / scale;
  if (abs_err <= FP_ABS_TOL || rel <= FP_REL_TOL)
    return true;
  if (index)
    std::cerr << std::format(
        "FAIL {} at {}: got {}, expected {} (abs={}, rel={})\n", label, *index,
        c, g, abs_err, rel);
  else
    std::cerr << std::format("FAIL {}: got {}, expected {} (abs={}, rel={})\n",
                             label, c, g, abs_err, rel);
  return false;
}

template <typename T>
inline bool check(const T *computed, const T *gold, uint64_t n,
                  std::string_view label) {
  for (uint64_t i = 0; i < n; i++) {
    if (!check_single<T>(computed[i], gold[i], label, i))
      return false;
  }
  return true;
}

inline TimingResult create_timing_result(const std::vector<double> &times,
                                         uint64_t data_bytes) {
  if (times.empty()) {
    std::cerr << "internal error: no timing samples collected\n";
    return TimingResult{0.0, 0.0, 0.0, 0.0, 0.0};
  }
  auto [mn, mx] = std::minmax_element(times.begin(), times.end());
  double avg = std::accumulate(times.begin(), times.end(), 0.0) /
               static_cast<double>(times.size());
  double best_mbps = (*mn > 0.0) ? (1e-6 * data_bytes / *mn) : 0.0;
  double avg_mbps = (avg > 0.0) ? (1e-6 * data_bytes / avg) : 0.0;
  return TimingResult{*mn, *mx, avg, best_mbps, avg_mbps};
}

// Add locale-independent thousand separators to make visual number parsing
// easier
inline std::string fmt_num_sep(std::string s) {
  for (int pos = s.length() - 3; pos > 0; pos -= 3)
    s.insert(pos, ",");
  return s;
}

inline void print_array_sizes() {
  std::cout << "Array sizes: ";
  for (uint64_t sz : conf.array_sizes)
    std::cout << " " << fmt_num_sep(std::format("{}", sz));
  std::cout << "\n\n";
}

inline void print_header() {
  std::cout << std::format(
      "{:>24} {:>8} {:>15}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}\n", "test",
      "type", "N", "min(s)", "max(s)", "avg(s)", "best MB/s", "avg MB/s");
  std::cout << std::format(
      "{:->24} {:->8} {:->15}  {:->10}  {:->10}  {:->10}  {:->12}  {:->12}\n",
      "", "", "", "", "", "", "", "");
}

inline void print_result(std::string_view test, std::string_view type,
                         uint64_t n, const std::optional<TimingResult> &r) {
  if (!r) {
    std::cerr << std::format("{:<24} {:<8} {:>15}  FAIL\n", test, type,
                             fmt_num_sep(std::format("{}", n)));
    return;
  }
  std::cout << std::format("{:<24} {:<8} {:>15}  {:>10.6f}  {:>10.6f}  "
                           "{:>10.6f}  {:>12}  {:>12}\n",
                           test, type, fmt_num_sep(std::format("{}", n)),
                           r->min_s, r->max_s, r->avg_s,
                           fmt_num_sep(std::format("{:.0f}", r->best_mbps)),
                           fmt_num_sep(std::format("{:.0f}", r->avg_mbps)));
}
