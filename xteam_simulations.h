#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <omp.h>

#include "xteam_simulations_common.h"
#include "common.h"

// Scan simulation device state (new decoupled look-back algorithm)
uint32_t *d_status = nullptr;
template <typename T> T *d_aggregates = nullptr;
template <typename T> T *d_prefixes = nullptr;
template <typename T> T *d_scan_out = nullptr;

#if defined(__AMDGCN__) || defined(__NVPTX__)
#define _XTEAMR_SCOPE __MEMORY_SCOPE_SYSTEM
#else
#define _XTEAMR_SCOPE 0
#endif

#define _XTEAMR_FUNC(T, TS, ATTR, BODY)                                        \
  ATTR void __kmpc_xteamr_##TS(                                                \
      T v, T *r_ptr, T *tvs, uint32_t *td, void (*_rf)(T *, T),                \
      void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *), const T rnv,                  \
      const uint64_t k, const uint32_t numteams, int Scope) BODY

#define _XTEAMR_FUNC_ALL(ATTR, BODY)                                           \
  _XTEAMR_FUNC(double, d, ATTR, BODY)                                          \
  _XTEAMR_FUNC(float, f, ATTR, BODY)                                           \
  _XTEAMR_FUNC(int, i, ATTR, BODY)                                             \
  _XTEAMR_FUNC(_UI, ui, ATTR, BODY)                                            \
  _XTEAMR_FUNC(long, l, ATTR, BODY)                                            \
  _XTEAMR_FUNC(_UL, ul, ATTR, BODY)

#define _XTEAMS_FUNC(T, TS, ATTR, BODY)                                        \
  ATTR void __kmpc_xteams_##TS(                                                \
      T v, T *result, uint32_t *status, T *aggregates, T *prefixes,            \
      void (*_rf)(T *, T), const T rnv, const uint64_t k) BODY

#define _XTEAMS_FUNC_ALL(ATTR, BODY)                                           \
  _XTEAMS_FUNC(double, d, ATTR, BODY)                                          \
  _XTEAMS_FUNC(float, f, ATTR, BODY)                                           \
  _XTEAMS_FUNC(int, i, ATTR, BODY)                                             \
  _XTEAMS_FUNC(_UI, ui, ATTR, BODY)                                            \
  _XTEAMS_FUNC(long, l, ATTR, BODY)                                            \
  _XTEAMS_FUNC(_UL, ul, ATTR, BODY)

#if defined(__AMDGCN__) || defined(__NVPTX__)

// Device compilation: declarations resolved from device runtime bitcode.
extern "C" {
// Reduction functions
_XTEAMR_FUNC_ALL(_INLINE_ATTR_, ;)

// Scan functions
_XTEAMS_FUNC_ALL(_INLINE_ATTR_, ;)
}

#else

// Host compilation: empty stubs so the host linker is satisfied.
extern "C" {
// Reduction functions
_XTEAMR_FUNC_ALL( , {})

// Scan functions
_XTEAMS_FUNC_ALL( , {})
}

#endif

#undef _XTEAMR_FUNC
#undef _XTEAMR_FUNC_ALL
#undef _XTEAMS_FUNC
#undef _XTEAMS_FUNC_ALL

// =========================================================================
// Helper functions
// =========================================================================

template <typename T>
using xteamr_fn_t = void (*)(T, T *, T *, uint32_t *, void (*)(T *, T),
                              void (*)(_RF_LDS T *, _RF_LDS T *), const T,
                              const uint64_t, const uint32_t, int);

template <typename T> constexpr xteamr_fn_t<T> get_kmpc_xteamr_func() {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_xteamr_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_xteamr_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_xteamr_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_xteamr_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_xteamr_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_xteamr_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T>
constexpr void (*get_kmpc_xteams_func())(T, T *, uint32_t *, T *, T *, void (*)(T *, T),
                               const T, const uint64_t) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_xteams_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_xteams_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_xteams_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_xteams_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_xteams_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_xteams_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

// =========================================================================
// GPU cross-team reduction kernels
// These are simulations without using xteam-specific codegen
// =========================================================================

template <typename T, ScanOp Op>
T reduce_sim(const T *__restrict in, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  T s = rnv;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom:s)                               \
    is_device_ptr(d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = rnv;
    for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
      val = scan_combine<T, Op>(val, in[i]);
    get_kmpc_xteamr_func<T>()(val, &s, d_team_vals<T>, d_td,
                              get_rfun_func<T, Op>(),
                              get_rfun_lds_func<T, Op>(),
                              rnv, k, XTEAM_NUM_TEAMS, _XTEAMR_SCOPE);
  }

  return s;
}

template <typename T>
T reduce_dot_sim(const T *__restrict a, const T *__restrict b, uint64_t n) {
  const T rnv = T(0);
  T s = rnv;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom:s)                               \
    is_device_ptr(d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = rnv;
    for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
      val += a[i] * b[i];
    get_kmpc_xteamr_func<T>()(val, &s, d_team_vals<T>, d_td,
                              get_rfun_sum_func<T>(),
                              get_rfun_sum_lds_func<T>(),
                              rnv, k, XTEAM_NUM_TEAMS, _XTEAMR_SCOPE);
  }

  return s;
}

// =========================================================================
// GPU cross-team scan kernels
// These are simulations without using xteam-specific codegen
// =========================================================================

template <typename T, ScanOp Op>
void scan_incl_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++)
      val0 = scan_combine<T, Op>(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_func<T, Op>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running = scan_combine<T, Op>(running, in[idx]);
      out[idx] = running;
    }
  }
}

template <typename T, ScanOp Op>
void scan_excl_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++)
      val0 = scan_combine<T, Op>(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_func<T, Op>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = running;
      running = scan_combine<T, Op>(running, in[idx]);
    }
  }
}

template <typename T>
void scan_incl_dot_sim(const T *__restrict a, const T *__restrict b,
                       T *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = T(0);
    for (uint64_t idx = start; idx < end; idx++)
      val0 += a[idx] * b[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running += a[idx] * b[idx];
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_dot_sim(const T *__restrict a, const T *__restrict b,
                       T *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = T(0);
    for (uint64_t idx = start; idx < end; idx++)
      val0 += a[idx] * b[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = running;
      running += a[idx] * b[idx];
    }
  }
}

// =========================================================================
// V1: two-kernel split (K1: aggregate + cross-team scan, K2: redistribute)
// =========================================================================

template <typename T, ScanOp Op>
void scan_incl_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++)
      val0 = scan_combine<T, Op>(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_func<T, Op>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running = scan_combine<T, Op>(running, in[idx]);
      out[idx] = running;
    }
  }
}

template <typename T, ScanOp Op>
void scan_excl_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++)
      val0 = scan_combine<T, Op>(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_func<T, Op>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = running;
      running = scan_combine<T, Op>(running, in[idx]);
    }
  }
}

template <typename T>
void scan_incl_dot_sim_v1(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = T(0);
    for (uint64_t idx = start; idx < end; idx++)
      val0 += a[idx] * b[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running += a[idx] * b[idx];
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_dot_sim_v1(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = T(0);
    for (uint64_t idx = start; idx < end; idx++)
      val0 += a[idx] * b[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = running;
      running += a[idx] * b[idx];
    }
  }
}

// =========================================================================
// Initialization and cleanup of the simulation helper data
// =========================================================================

template <typename T> void init_device_sim() {
  assert(d_status == nullptr && d_td == nullptr);
  int devid = 0;

  // Reduction state
  d_td = static_cast<uint32_t *>(omp_target_alloc(sizeof(uint32_t), devid));
  d_team_vals<T> =
      static_cast<T *>(omp_target_alloc(sizeof(T) * XTEAM_NUM_TEAMS, devid));
  omp_target_memset(d_td, 0, sizeof(uint32_t), devid);

  // Scan state
  d_status = static_cast<uint32_t *>(
      omp_target_alloc(sizeof(uint32_t) * (XTEAM_NUM_TEAMS + 1), devid));
  d_aggregates<T> =
      static_cast<T *>(omp_target_alloc(sizeof(T) * XTEAM_NUM_TEAMS, devid));
  d_prefixes<T> =
      static_cast<T *>(omp_target_alloc(sizeof(T) * XTEAM_NUM_TEAMS, devid));
  d_scan_out<T> = static_cast<T *>(
      omp_target_alloc(sizeof(T) * XTEAM_TOTAL_NUM_THREADS, devid));
  omp_target_memset(d_status, 0, sizeof(uint32_t) * (XTEAM_NUM_TEAMS + 1),
                    devid);
}

template <typename T> void free_device_sim() {
  assert(d_status != nullptr && d_td != nullptr);
  int devid = 0;

  // Reduction state
  omp_target_free(d_td, devid);
  d_td = nullptr;
  omp_target_free(d_team_vals<T>, devid);
  d_team_vals<T> = nullptr;

  // Scan state
  omp_target_free(d_status, devid);
  d_status = nullptr;
  omp_target_free(d_aggregates<T>, devid);
  d_aggregates<T> = nullptr;
  omp_target_free(d_prefixes<T>, devid);
  d_prefixes<T> = nullptr;
  omp_target_free(d_scan_out<T>, devid);
  d_scan_out<T> = nullptr;
}
