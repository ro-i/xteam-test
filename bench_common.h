// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

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

// Selected arrays sizes for quick run
#if NOLOOP
static const std::array<uint64_t, 1> array_sizes_quick{XTEAM_TOTAL_NUM_THREADS};
#else
static const std::array<uint64_t, 1> array_sizes_quick{177777777};
#endif // NOLOOP

// Array sizes for full run
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

#define duration_cast(x)                                                       \
  std::chrono::duration_cast<std::chrono::duration<double>>(x)
using Clock = std::chrono::steady_clock;

struct TimingResult {
  double min_s, max_s, avg_s;
  double best_mbps, avg_mbps;
};

// =========================================================================
// Utility functions
// =========================================================================

// Deterministic initialization for reproducibility.
template <typename T, bool is_fp>
static void init_data(T *arr1, T *arr2, uint64_t n) {
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

template <typename T, RedOp Op> static T gold_red(const T *in, uint64_t n) {
  T a = red_identity<T, Op>();
  for (uint64_t i = 0; i < n; i++)
    a = red_combine<T, Op>(a, in[i]);
  return a;
}
template <typename T>
static T gold_red_dot(const T *a, const T *b, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  for (uint64_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

template <typename T, RedOp Op, ScanMode Mode>
static void gold_scan(const T *in, T *out, uint64_t n) {
  T a = red_identity<T, Op>();
  for (uint64_t i = 0; i < n; i++) {
    if constexpr (Mode == ScanMode::Excl) {
      out[i] = a;
      a = red_combine<T, Op>(a, in[i]);
    } else {
      a = red_combine<T, Op>(a, in[i]);
      out[i] = a;
    }
  }
}
template <typename T, ScanMode Mode>
static void gold_scan_dot(const T *a, const T *b, T *out, uint64_t n) {
  T s = T(0);
  for (uint64_t i = 0; i < n; i++) {
    if constexpr (Mode == ScanMode::Excl) {
      out[i] = s;
      s += a[i] * b[i];
    } else {
      s += a[i] * b[i];
      out[i] = s;
    }
  }
}

// =========================================================================
// Benchmark harness utilities
// =========================================================================

template <typename T, bool is_fp>
static bool check_single(T computed, T gold, std::string_view label,
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

template <typename T, bool is_fp>
static bool check(const T *computed, const T *gold, uint64_t n,
                  std::string_view label) {
  for (uint64_t i = 0; i < n; i++) {
    if (!check_single<T, is_fp>(computed[i], gold[i], label, i))
      return false;
  }
  return true;
}

static TimingResult create_timing_result(const std::vector<double> &times,
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
static std::string fmt_num_sep(std::string s) {
  for (int pos = s.length() - 3; pos > 0; pos -= 3)
    s.insert(pos, ",");
  return s;
}

static void print_result(std::string_view test, std::string_view type,
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

static void print_header() {
  std::cout << std::format(
      "{:>24} {:>8} {:>15}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}\n", "test",
      "type", "N", "min(s)", "max(s)", "avg(s)", "best MB/s", "avg MB/s");
  std::cout << std::format(
      "{:->24} {:->8} {:->15}  {:->10}  {:->10}  {:->10}  {:->12}  {:->12}\n",
      "", "", "", "", "", "", "", "");
}
