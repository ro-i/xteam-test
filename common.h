#pragma once

#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>

#ifndef WARMUP_ITERS
#define WARMUP_ITERS 2
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 10
#endif
#ifndef QUICK_RUN
#define QUICK_RUN 0
#endif
#ifndef SCAN_SUPPORT
#define SCAN_SUPPORT 1
#endif
#ifndef SCAN_SIMULATION
#define SCAN_SIMULATION 1
#endif
#ifndef REDUCTION_SIMULATION
#define REDUCTION_SIMULATION 1
#endif
#ifndef XTEAM_NUM_THREADS
#define XTEAM_NUM_THREADS 512
#endif
#ifndef XTEAM_NUM_TEAMS
#define XTEAM_NUM_TEAMS 64
#endif

// default alignment for aligned_alloc
#define ALIGNMENT 128

// Represents the total of threads in the Grid
#define XTEAM_TOTAL_NUM_THREADS (XTEAM_NUM_TEAMS * XTEAM_NUM_THREADS)

template <typename T>
T *alloc(uint64_t n) {
  T* ret = static_cast<T *>(aligned_alloc(ALIGNMENT, sizeof(T) * n));
  if (!ret) {
    std::cerr << std::format("aligned_alloc failed n={}\n", n);
    exit(EXIT_FAILURE);
  }
  return ret;
}
