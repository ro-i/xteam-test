// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
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

// default alignment for aligned_alloc
#define ALIGNMENT 128

// Represents the total of threads in the Grid
#define XTEAM_TOTAL_NUM_THREADS (XTEAM_NUM_TEAMS * XTEAM_NUM_THREADS)
struct Config {
  bool quick_run = false;
  bool reduction = false;
  bool reduction_simulation = false;
  bool scan = false;
  bool scan_simulation = false;
  int warmup_iters = 2;
  int bench_iters_reduction = 1000;
  int bench_iters_scan = 10;
  std::vector<uint64_t> array_sizes;
};

template <typename T> T *alloc(uint64_t n) {
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

template <typename T> T *target_alloc(uint64_t n, int devid) {
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

// =========================================================================
// RedOp-generic helpers
// =========================================================================

enum class RedOp { Sum, Max, Min, Mult };

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

template <RedOp Op> std::string red_op_to_str(std::string_view fmt) {
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
