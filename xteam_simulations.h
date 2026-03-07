#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <omp.h>

#include "xteam_simulations_common.h"

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

#if defined(__AMDGCN__) || defined(__NVPTX__)

// Device compilation: declarations resolved from device runtime bitcode.
extern "C" {
// Reduction functions
void _INLINE_ATTR_ __kmpc_xteamr_d(double v, double *r_ptr, double *tvs,
                                   uint32_t *td, void (*_rf)(double *, double),
                                   void (*_rf_lds)(_RF_LDS double *,
                                                   _RF_LDS double *),
                                   const double rnv, const uint64_t k,
                                   const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_f(float v, float *r_ptr, float *tvs,
                                   uint32_t *td, void (*_rf)(float *, float),
                                   void (*_rf_lds)(_RF_LDS float *,
                                                   _RF_LDS float *),
                                   const float rnv, const uint64_t k,
                                   const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_i(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_ui(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_l(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_ul(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams, int Scope);

// Scan functions
void _INLINE_ATTR_ __kmpc_xteams_d(double v, double *result, uint32_t *status,
                                   double *aggregates, double *prefixes,
                                   void (*rf)(double *, double),
                                   const double rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_xteams_f(float v, float *result, uint32_t *status,
                                   float *aggregates, float *prefixes,
                                   void (*rf)(float *, float), const float rnv,
                                   const uint64_t k);
void _INLINE_ATTR_ __kmpc_xteams_i(int v, int *result, uint32_t *status,
                                   int *aggregates, int *prefixes,
                                   void (*rf)(int *, int), const int rnv,
                                   const uint64_t k);
void _INLINE_ATTR_ __kmpc_xteams_ui(_UI v, _UI *result, uint32_t *status,
                                    _UI *aggregates, _UI *prefixes,
                                    void (*rf)(_UI *, _UI), const _UI rnv,
                                    const uint64_t k);
void _INLINE_ATTR_ __kmpc_xteams_l(long v, long *result, uint32_t *status,
                                   long *aggregates, long *prefixes,
                                   void (*rf)(long *, long), const long rnv,
                                   const uint64_t k);
void _INLINE_ATTR_ __kmpc_xteams_ul(_UL v, _UL *result, uint32_t *status,
                                    _UL *aggregates, _UL *prefixes,
                                    void (*rf)(_UL *, _UL), const _UL rnv,
                                    const uint64_t k);
void _INLINE_ATTR_ __kmpc_xteams_v2_d(double v, double *result, uint32_t *status,
                                   double *aggregates, double *prefixes,
                                   void (*rf)(double *, double),
                                   const double rnv, const uint64_t k, const uint64_t num_elements);
void _INLINE_ATTR_ __kmpc_xteams_v2_f(float v, float *result, uint32_t *status,
                                   float *aggregates, float *prefixes,
                                   void (*rf)(float *, float), const float rnv,
                                   const uint64_t k, const uint64_t num_elements);
void _INLINE_ATTR_ __kmpc_xteams_v2_i(int v, int *result, uint32_t *status,
                                   int *aggregates, int *prefixes,
                                   void (*rf)(int *, int), const int rnv,
                                   const uint64_t k, const uint64_t num_elements);
void _INLINE_ATTR_ __kmpc_xteams_v2_ui(_UI v, _UI *result, uint32_t *status,
                                    _UI *aggregates, _UI *prefixes,
                                    void (*rf)(_UI *, _UI), const _UI rnv,
                                    const uint64_t k, const uint64_t num_elements);
void _INLINE_ATTR_ __kmpc_xteams_v2_l(long v, long *result, uint32_t *status,
                                   long *aggregates, long *prefixes,
                                   void (*rf)(long *, long), const long rnv,
                                   const uint64_t k, const uint64_t num_elements);
void _INLINE_ATTR_ __kmpc_xteams_v2_ul(_UL v, _UL *result, uint32_t *status,
                                    _UL *aggregates, _UL *prefixes,
                                    void (*rf)(_UL *, _UL), const _UL rnv,
                                    const uint64_t k, const uint64_t num_elements);
}

#else

// Host compilation: empty stubs so the host linker is satisfied.
extern "C" {
// Reduction functions
void __kmpc_xteamr_d(double, double *, double *, uint32_t *,
                     void (*)(double *, double),
                     void (*)(_RF_LDS double *, _RF_LDS double *), const double,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_f(float, float *, float *, uint32_t *,
                     void (*)(float *, float),
                     void (*)(_RF_LDS float *, _RF_LDS float *), const float,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_i(int, int *, int *, uint32_t *, void (*)(int *, int),
                     void (*)(_RF_LDS int *, _RF_LDS int *), const int,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_ui(_UI, _UI *, _UI *, uint32_t *, void (*)(_UI *, _UI),
                      void (*)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI,
                      const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_l(long, long *, long *, uint32_t *, void (*)(long *, long),
                     void (*)(_RF_LDS long *, _RF_LDS long *), const long,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_ul(_UL, _UL *, _UL *, uint32_t *, void (*)(_UL *, _UL),
                      void (*)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL,
                      const uint64_t, const uint32_t, int) {}

// Scan functions
void __kmpc_xteams_d(double v, double *result, uint32_t *status,
                     double *aggregates, double *prefixes,
                     void (*rf)(double *, double), const double rnv,
                     const uint64_t k) {}
void __kmpc_xteams_f(float v, float *result, uint32_t *status,
                     float *aggregates, float *prefixes,
                     void (*rf)(float *, float), const float rnv,
                     const uint64_t k) {}
void __kmpc_xteams_i(int v, int *result, uint32_t *status, int *aggregates,
                     int *prefixes, void (*rf)(int *, int), const int rnv,
                     const uint64_t k) {}
void __kmpc_xteams_ui(_UI v, _UI *result, uint32_t *status, _UI *aggregates,
                      _UI *prefixes, void (*rf)(_UI *, _UI), const _UI rnv,
                      const uint64_t k) {}
void __kmpc_xteams_l(long v, long *result, uint32_t *status, long *aggregates,
                     long *prefixes, void (*rf)(long *, long), const long rnv,
                     const uint64_t k) {}
void __kmpc_xteams_ul(_UL v, _UL *result, uint32_t *status, _UL *aggregates,
                      _UL *prefixes, void (*rf)(_UL *, _UL), const _UL rnv,
                      const uint64_t k) {}
void __kmpc_xteams_v2_d(double v, double *result, uint32_t *status,
                     double *aggregates, double *prefixes,
                     void (*rf)(double *, double), const double rnv,
                     const uint64_t k, const uint64_t num_elements) {}
void __kmpc_xteams_v2_f(float v, float *result, uint32_t *status,
                     float *aggregates, float *prefixes,
                     void (*rf)(float *, float), const float rnv,
                     const uint64_t k, const uint64_t num_elements) {}
void __kmpc_xteams_v2_i(int v, int *result, uint32_t *status, int *aggregates,
                     int *prefixes, void (*rf)(int *, int), const int rnv,
                     const uint64_t k, const uint64_t num_elements) {}
void __kmpc_xteams_v2_ui(_UI v, _UI *result, uint32_t *status, _UI *aggregates,
                      _UI *prefixes, void (*rf)(_UI *, _UI), const _UI rnv,
                      const uint64_t k, const uint64_t num_elements) {}
void __kmpc_xteams_v2_l(long v, long *result, uint32_t *status, long *aggregates,
                     long *prefixes, void (*rf)(long *, long), const long rnv,
                     const uint64_t k, const uint64_t num_elements) {}
void __kmpc_xteams_v2_ul(_UL v, _UL *result, uint32_t *status, _UL *aggregates,
                      _UL *prefixes, void (*rf)(_UL *, _UL), const _UL rnv,
                      const uint64_t k, const uint64_t num_elements) {}
}

#endif

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

template <typename T>
constexpr void (*get_kmpc_xteams_v2_func())(T, T *, uint32_t *, T *, T *, void (*)(T *, T),
                               const T, const uint64_t, const uint64_t) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_xteams_v2_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_xteams_v2_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_xteams_v2_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_xteams_v2_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_xteams_v2_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_xteams_v2_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

// =========================================================================
// GPU cross-team reduction kernels
// These are simulations without using xteam-specific codegen
// =========================================================================

template <typename T> T reduce_sum_sim(const T *__restrict in, uint64_t n) {
  const T rnv = T(0);
  T s = rnv;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom:s)                               \
    is_device_ptr(d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = rnv;
    for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
      val += in[i];
    get_kmpc_xteamr_func<T>()(val, &s, d_team_vals<T>, d_td,
                              get_rfun_sum_func<T>(),
                              get_rfun_sum_lds_func<T>(),
                              rnv, k, XTEAM_NUM_TEAMS, _XTEAMR_SCOPE);
  }

  return s;
}

template <typename T> T reduce_max_sim(const T *__restrict in, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
  T s = rnv;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom:s)                               \
    is_device_ptr(d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = rnv;
    for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
      val = std::max(val, in[i]);
    get_kmpc_xteamr_func<T>()(val, &s, d_team_vals<T>, d_td,
                              get_rfun_max_func<T>(),
                              get_rfun_max_lds_func<T>(),
                              rnv, k, XTEAM_NUM_TEAMS, _XTEAMR_SCOPE);
  }

  return s;
}

template <typename T> T reduce_min_sim(const T *__restrict in, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
  T s = rnv;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom:s)                               \
    is_device_ptr(d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = rnv;
    for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
      val = std::min(val, in[i]);
    get_kmpc_xteamr_func<T>()(val, &s, d_team_vals<T>, d_td,
                              get_rfun_min_func<T>(),
                              get_rfun_min_lds_func<T>(),
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

template <typename T>
void scan_incl_sum_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
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
      val0 += in[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running += in[idx];
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_sum_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
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
      val0 += in[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = running;
      running += in[idx];
    }
  }
}

template <typename T>
void scan_incl_max_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
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
      val0 = std::max(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running = std::max(running, in[idx]);
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_max_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
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
      val0 = std::max(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = running;
      running = std::max(running, in[idx]);
    }
  }
}

template <typename T>
void scan_incl_min_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
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
      val0 = std::min(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running = std::min(running, in[idx]);
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_min_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
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
      val0 = std::min(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k);
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = running;
      running = std::min(running, in[idx]);
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

template <typename T>
void scan_incl_sum_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
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
      val0 += in[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running += in[idx];
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_sum_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
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
      val0 += in[idx];
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k);
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
      running += in[idx];
    }
  }
}

template <typename T>
void scan_incl_max_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
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
      val0 = std::max(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running = std::max(running, in[idx]);
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_max_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
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
      val0 = std::max(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k);
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
      running = std::max(running, in[idx]);
    }
  }
}

template <typename T>
void scan_incl_min_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
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
      val0 = std::min(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T running = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running = std::min(running, in[idx]);
      out[idx] = running;
    }
  }
}

template <typename T>
void scan_excl_min_sim_v1(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
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
      val0 = std::min(val0, in[idx]);
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k);
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
      running = std::min(running, in[idx]);
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
// V2: single read of input, write local scan to out[], then apply prefix
// =========================================================================

template <typename T>
void scan_incl_sum_sim_v2(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      val0 += in[idx];
      out[idx] = val0;
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k,
                              n);
  }
}

template <typename T>
void scan_excl_sum_sim_v2(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 += in[idx];
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k,
                              n);
  }
}

template <typename T>
void scan_incl_max_sim_v2(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      val0 = std::max(val0, in[idx]);
      out[idx] = val0;
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k,
                              n);
  }
}

template <typename T>
void scan_excl_max_sim_v2(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 = std::max(val0, in[idx]);
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k,
                              n);
  }
}

template <typename T>
void scan_incl_min_sim_v2(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      val0 = std::min(val0, in[idx]);
      out[idx] = val0;
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k,
                              n);
  }
}

template <typename T>
void scan_excl_min_sim_v2(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 = std::min(val0, in[idx]);
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k,
                              n);
  }
}

template <typename T>
void scan_incl_dot_sim_v2(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = T(0);
    for (uint64_t idx = start; idx < end; idx++) {
      val0 += a[idx] * b[idx];
      out[idx] = val0;
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k,
                              n);
  }
}

template <typename T>
void scan_excl_dot_sim_v2(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = T(0);
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 += a[idx] * b[idx];
    }
    get_kmpc_xteams_v2_func<T>()(val0, out, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k,
                              n);
  }
}

// =========================================================================
// V4: same as V1 but with direct __kmpc_xteams_* / __kmpc_rfun_sum_* calls
// instead of going through function-pointer dispatch helpers.
// Only inclusive sum — enough to test whether the dispatch indirection
// causes the register-pressure blow-up seen in V1.
// =========================================================================

void scan_incl_sum_sim_v4(const int *__restrict in, int *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<int>, d_prefixes<int>, d_scan_out<int>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    int val0 = 0;
    for (uint64_t idx = start; idx < end; idx++)
      val0 += in[idx];
    __kmpc_xteams_i(val0, d_scan_out<int>, d_status, d_aggregates<int>,
                    d_prefixes<int>, __kmpc_rfun_sum_i, 0, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<int>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    int running = d_scan_out<int>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running += in[idx];
      out[idx] = running;
    }
  }
}

void scan_incl_sum_sim_v4(const long *__restrict in, long *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<long>, d_prefixes<long>, d_scan_out<long>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    long val0 = 0L;
    for (uint64_t idx = start; idx < end; idx++)
      val0 += in[idx];
    __kmpc_xteams_l(val0, d_scan_out<long>, d_status, d_aggregates<long>,
                    d_prefixes<long>, __kmpc_rfun_sum_l, 0L, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<long>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    long running = d_scan_out<long>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running += in[idx];
      out[idx] = running;
    }
  }
}

void scan_incl_sum_sim_v4(const double *__restrict in, double *__restrict out, uint64_t n) {
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<double>, d_prefixes<double>, d_scan_out<double>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    double val0 = 0.0;
    for (uint64_t idx = start; idx < end; idx++)
      val0 += in[idx];
    __kmpc_xteams_d(val0, d_scan_out<double>, d_status, d_aggregates<double>,
                    d_prefixes<double>, __kmpc_rfun_sum_d, 0.0, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<double>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    double running = d_scan_out<double>[k];
    for (uint64_t idx = start; idx < end; idx++) {
      running += in[idx];
      out[idx] = running;
    }
  }
}

// =========================================================================
// V3: codegen-matching pattern (K1: accumulate + write out[], K2: prefix-apply)
// Phase 1 writes the per-element running result into out[] (as scratch),
// then calls __kmpc_xteams.  Phase 2 reads out[], adds the cross-team
// exclusive prefix, and writes back.  This avoids re-reading in[] in K2
// (matching what the segmented-scan codegen actually produces).
// =========================================================================

template <typename T>
void scan_incl_sum_sim_v3(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      val0 += in[idx];
      out[idx] = val0;
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] += prefix;
  }
}

template <typename T>
void scan_excl_sum_sim_v3(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = T(0);
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 += in[idx];
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] += prefix;
  }
}

template <typename T>
void scan_incl_max_sim_v3(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      val0 = std::max(val0, in[idx]);
      out[idx] = val0;
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] = std::max(out[idx], prefix);
  }
}

template <typename T>
void scan_excl_max_sim_v3(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::lowest();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 = std::max(val0, in[idx]);
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_max_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] = std::max(out[idx], prefix);
  }
}

template <typename T>
void scan_incl_min_sim_v3(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      val0 = std::min(val0, in[idx]);
      out[idx] = val0;
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] = std::min(out[idx], prefix);
  }
}

template <typename T>
void scan_excl_min_sim_v3(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = std::numeric_limits<T>::max();
  const uint64_t stride =
      (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T val0 = rnv;
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 = std::min(val0, in[idx]);
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_min_func<T>(), rnv, k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] = std::min(out[idx], prefix);
  }
}

template <typename T>
void scan_incl_dot_sim_v3(const T *__restrict a, const T *__restrict b,
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
    for (uint64_t idx = start; idx < end; idx++) {
      val0 += a[idx] * b[idx];
      out[idx] = val0;
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] += prefix;
  }
}

template <typename T>
void scan_excl_dot_sim_v3(const T *__restrict a, const T *__restrict b,
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
    for (uint64_t idx = start; idx < end; idx++) {
      out[idx] = val0;
      val0 += a[idx] * b[idx];
    }
    get_kmpc_xteams_func<T>()(val0, d_scan_out<T>, d_status, d_aggregates<T>,
                              d_prefixes<T>, get_rfun_sum_func<T>(), T(0), k);
  }

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_scan_out<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    const uint64_t start = k * stride;
    const uint64_t end = (start + stride < n) ? start + stride : n;
    T prefix = d_scan_out<T>[k];
    for (uint64_t idx = start; idx < end; idx++)
      out[idx] += prefix;
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
