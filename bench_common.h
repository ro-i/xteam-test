#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "common.h"

#if QUICK_RUN
#if NOLOOP
static const std::array<uint64_t, 1> array_sizes{XTEAM_TOTAL_NUM_THREADS};
#else
static const std::array<uint64_t, 1> array_sizes{177777777};
#endif // NOLOOP
#else
#if NOLOOP
static const std::array<uint64_t, 9> array_sizes{1,
                                                 100,
                                                 1024,
                                                 2048,
                                                 4096,
                                                 8192,
                                                 XTEAM_TOTAL_NUM_THREADS / 2,
                                                 XTEAM_TOTAL_NUM_THREADS - 1,
                                                 XTEAM_TOTAL_NUM_THREADS};
#else
static const std::array<uint64_t, 14> array_sizes{
    1,     100,     1024,    2048,     4096,     8192,      10000,
    81920, 1000000, 4194304, 23445657, 41943040, 100000000, 177777777};
#endif // NOLOOP
#endif // QUICK_RUN

using Clock = std::chrono::high_resolution_clock;

struct TimingResult {
  double min_s, max_s, avg_s;
  double best_mbps, avg_mbps;
};

// =========================================================================
// Utility functions
// =========================================================================

template <typename T, bool is_fp> void init_data(T *arr1, T *arr2, uint64_t n) {
  srand(42);
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
// Benchmark harness utilities
// =========================================================================

template <typename T, bool is_fp>
inline bool check_single(T computed, T gold, const char *label,
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

inline TimingResult create_timing_result(const std::vector<double> &times,
                                         uint64_t n, uint64_t data_bytes) {
  auto [mn, mx] = std::minmax_element(times.begin(), times.end());
  double avg =
      std::accumulate(times.begin(), times.end(), 0.0) / (double)(times.size());
  return TimingResult{*mn, *mx, avg, 1e-6 * data_bytes / *mn,
                      1e-6 * data_bytes / avg};
}

inline void print_result(const char *test, const char *type, uint64_t n,
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

inline void print_header() {
  std::cout << std::format(
      "{:>24} {:>8} {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}\n", "test",
      "type", "N", "min(s)", "max(s)", "avg(s)", "best MB/s", "avg MB/s");
  std::cout << std::format(
      "{:->24} {:->8} {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}\n",
      "------------------------", "--------", "----------", "----------",
      "----------", "----------", "----------", "----------");
}
