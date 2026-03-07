#pragma once

#include "xteam_simulations_common.h"
#include "common.h"

// Scan simulation device state (old 2-kernel algorithm)
template <typename T> T *d_storage = nullptr;

// =========================================================================
// Wave-size detection and derived constants
// =========================================================================

#if defined(__AMDGCN__)
#define _WAVE_SIZE __builtin_amdgcn_wavefrontsize()
#elif defined(__NVPTX__)
#define _WAVE_SIZE 32
#else
#define _WAVE_SIZE 64
#endif

#if defined(__AMDGCN__) || defined(__NVPTX__)
#define _XTEAMR_SCOPE __MEMORY_SCOPE_SYSTEM
#else
#define _XTEAMR_SCOPE 0
#endif

// =========================================================================
// Declaration / stub generation macros
// =========================================================================

// Reduction: (v, r_ptr, tvs, td, rf, rf_lds, iv, k, numteams, Scope)
#define _XTEAMR_FUNC(T, TS, SUFFIX, ATTR, BODY)                                \
  ATTR void __kmpc_xteamr_##TS##SUFFIX(                                        \
      T v, T *r_ptr, T *tvs, uint32_t *td, void (*_rf)(T *, T),                \
      void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *), const T iv, const uint64_t k, \
      const uint32_t numteams, int Scope) BODY

#define _XTEAMR_FUNC_ALL(SUFFIX, ATTR, BODY)                                   \
  _XTEAMR_FUNC(double, d, SUFFIX, ATTR, BODY)                                  \
  _XTEAMR_FUNC(float, f, SUFFIX, ATTR, BODY)                                   \
  _XTEAMR_FUNC(int, i, SUFFIX, ATTR, BODY)                                     \
  _XTEAMR_FUNC(_UI, ui, SUFFIX, ATTR, BODY)                                    \
  _XTEAMR_FUNC(long, l, SUFFIX, ATTR, BODY)                                    \
  _XTEAMR_FUNC(_UL, ul, SUFFIX, ATTR, BODY)

// Scan: (v, storage, r_array, tvs, td, rf, rf_lds, iv, k, numteams)
#define _XTEAMS_FUNC(T, TS, SUFFIX, ATTR, BODY)                                \
  ATTR void __kmpc_xteams_##TS##SUFFIX(                                        \
      T v, T *storage, T *r_array, T *tvs, uint32_t *td, void (*_rf)(T *, T),  \
      void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *), const T iv, const uint64_t k, \
      const uint32_t numteams) BODY

#define _XTEAMS_FUNC_ALL(SUFFIX, ATTR, BODY)                                   \
  _XTEAMS_FUNC(double, d, SUFFIX, ATTR, BODY)                                  \
  _XTEAMS_FUNC(float, f, SUFFIX, ATTR, BODY)                                   \
  _XTEAMS_FUNC(int, i, SUFFIX, ATTR, BODY)                                     \
  _XTEAMS_FUNC(_UI, ui, SUFFIX, ATTR, BODY)                                    \
  _XTEAMS_FUNC(long, l, SUFFIX, ATTR, BODY)                                    \
  _XTEAMS_FUNC(_UL, ul, SUFFIX, ATTR, BODY)

// Phase2 scan: (storage, segment_size, tvs, seg_vals, rf, rnv, k, is_inclusive)
#define _XTEAMS_P2_FUNC(T, TS, SUFFIX, ATTR, BODY)                             \
  ATTR void __kmpc_xteams_phase2_##TS##SUFFIX(                                 \
      T *storage, int segment_size, T *tvs, T *seg_vals, void (*_rf)(T *, T),  \
      const T rnv, const uint64_t k, bool is_inclusive_scan) BODY

#define _XTEAMS_P2_FUNC_ALL(SUFFIX, ATTR, BODY)                                \
  _XTEAMS_P2_FUNC(double, d, SUFFIX, ATTR, BODY)                               \
  _XTEAMS_P2_FUNC(float, f, SUFFIX, ATTR, BODY)                                \
  _XTEAMS_P2_FUNC(int, i, SUFFIX, ATTR, BODY)                                  \
  _XTEAMS_P2_FUNC(long, l, SUFFIX, ATTR, BODY)

// =========================================================================
// Device declarations (resolved from device runtime bitcode)
// =========================================================================

#if defined(__AMDGCN__) || defined(__NVPTX__)

extern "C" {
// Reductions (with Scope) — only _16x64 and _32x32 exist in the runtime
_XTEAMR_FUNC_ALL(_16x64, _INLINE_ATTR_, ;)
_XTEAMR_FUNC_ALL(_32x32, _INLINE_ATTR_, ;)

// Scans — AMD wavesize-64 variants
_XTEAMS_FUNC_ALL(_16x64, _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_8x64,  _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_4x64,  _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_2x64,  _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_1x64,  _INLINE_ATTR_, ;)

// Scans — NVIDIA wavesize-32 variants
_XTEAMS_FUNC_ALL(_32x32, _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_16x32, _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_8x32,  _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_4x32,  _INLINE_ATTR_, ;)
_XTEAMS_FUNC_ALL(_2x32,  _INLINE_ATTR_, ;)

// Phase2 scans — AMD wavesize-64 variants (d, f, i, l only)
_XTEAMS_P2_FUNC_ALL(_16x64, _INLINE_ATTR_, ;)
_XTEAMS_P2_FUNC_ALL(_8x64,  _INLINE_ATTR_, ;)
_XTEAMS_P2_FUNC_ALL(_4x64,  _INLINE_ATTR_, ;)

// Phase2 scans — NVIDIA wavesize-32 variants (d, f, i, l only)
_XTEAMS_P2_FUNC_ALL(_32x32, _INLINE_ATTR_, ;)
_XTEAMS_P2_FUNC_ALL(_16x32, _INLINE_ATTR_, ;)
_XTEAMS_P2_FUNC_ALL(_8x32,  _INLINE_ATTR_, ;)
}

#else

// Host compilation: empty stubs so the host linker is satisfied.
extern "C" {
_XTEAMR_FUNC_ALL(_16x64, , {})
_XTEAMR_FUNC_ALL(_32x32, , {})

_XTEAMS_FUNC_ALL(_16x64, , {})
_XTEAMS_FUNC_ALL(_8x64,  , {})
_XTEAMS_FUNC_ALL(_4x64,  , {})
_XTEAMS_FUNC_ALL(_2x64,  , {})
_XTEAMS_FUNC_ALL(_1x64,  , {})

_XTEAMS_FUNC_ALL(_32x32, , {})
_XTEAMS_FUNC_ALL(_16x32, , {})
_XTEAMS_FUNC_ALL(_8x32,  , {})
_XTEAMS_FUNC_ALL(_4x32,  , {})
_XTEAMS_FUNC_ALL(_2x32,  , {})

_XTEAMS_P2_FUNC_ALL(_16x64, , {})
_XTEAMS_P2_FUNC_ALL(_8x64,  , {})
_XTEAMS_P2_FUNC_ALL(_4x64,  , {})

_XTEAMS_P2_FUNC_ALL(_32x32, , {})
_XTEAMS_P2_FUNC_ALL(_16x32, , {})
_XTEAMS_P2_FUNC_ALL(_8x32,  , {})
}

#endif

// Done with declaration macros
#undef _XTEAMR_FUNC
#undef _XTEAMR_FUNC_ALL
#undef _XTEAMS_FUNC
#undef _XTEAMS_FUNC_ALL
#undef _XTEAMS_P2_FUNC
#undef _XTEAMS_P2_FUNC_ALL

// =========================================================================
// API-specific helper functions — AOMP block-size-suffixed API
// =========================================================================

// Reduction type: same signature as new dev API (includes Scope)
template <typename T>
using xteamr_fn_t = void (*)(T, T *, T *, uint32_t *, void (*)(T *, T),
                              void (*)(_RF_LDS T *, _RF_LDS T *), const T,
                              const uint64_t, const uint32_t, int);

// Scan type: old 2-kernel signature (no Scope)
template <typename T>
using xteams_fn_t = void (*)(T, T *, T *, T *, uint32_t *, void (*)(T *, T),
                              void (*)(_RF_LDS T *, _RF_LDS T *), const T,
                              const uint64_t, const uint32_t);

// Phase2 scan type (d, f, i, l only on amd-staging)
template <typename T>
using xteams_phase2_fn_t = void (*)(T *, int, T *, T *, void (*)(T *, T),
                                    const T, const uint64_t, bool);

// --- getter helper macros ------------------------------------------------

#define _XTEAMR_GETTER_BODY(SUFFIX)                                            \
  if constexpr (std::is_same_v<T, double>)       return __kmpc_xteamr_d##SUFFIX; \
  else if constexpr (std::is_same_v<T, float>)   return __kmpc_xteamr_f##SUFFIX; \
  else if constexpr (std::is_same_v<T, int>)     return __kmpc_xteamr_i##SUFFIX; \
  else if constexpr (std::is_same_v<T, unsigned int>)  return __kmpc_xteamr_ui##SUFFIX; \
  else if constexpr (std::is_same_v<T, long>)    return __kmpc_xteamr_l##SUFFIX; \
  else if constexpr (std::is_same_v<T, unsigned long>) return __kmpc_xteamr_ul##SUFFIX; \
  else static_assert(false, "Unsupported type");

#define _XTEAMS_GETTER_BODY(SUFFIX)                                            \
  if constexpr (std::is_same_v<T, double>)       return __kmpc_xteams_d##SUFFIX; \
  else if constexpr (std::is_same_v<T, float>)   return __kmpc_xteams_f##SUFFIX; \
  else if constexpr (std::is_same_v<T, int>)     return __kmpc_xteams_i##SUFFIX; \
  else if constexpr (std::is_same_v<T, unsigned int>)  return __kmpc_xteams_ui##SUFFIX; \
  else if constexpr (std::is_same_v<T, long>)    return __kmpc_xteams_l##SUFFIX; \
  else if constexpr (std::is_same_v<T, unsigned long>) return __kmpc_xteams_ul##SUFFIX; \
  else static_assert(false, "Unsupported type");

#define _XTEAMS_P2_GETTER_BODY(SUFFIX)                                         \
  if constexpr (std::is_same_v<T, double>)       return __kmpc_xteams_phase2_d##SUFFIX; \
  else if constexpr (std::is_same_v<T, float>)   return __kmpc_xteams_phase2_f##SUFFIX; \
  else if constexpr (std::is_same_v<T, int>)     return __kmpc_xteams_phase2_i##SUFFIX; \
  else if constexpr (std::is_same_v<T, long>)    return __kmpc_xteams_phase2_l##SUFFIX; \
  else static_assert(false, "Phase2: unsupported type (only d/f/i/l)");

// Reduction getter — only _16x64 (wave64) and _32x32 (wave32) exist
template <typename T> xteamr_fn_t<T> get_kmpc_xteamr_func() {
  if (_WAVE_SIZE == 64) {
    _XTEAMR_GETTER_BODY(_16x64)
  } else {
    _XTEAMR_GETTER_BODY(_32x32)
  }
}

// Scan getter — selects variant matching XTEAM_NUM_THREADS / _WAVE_SIZE
template <typename T> xteams_fn_t<T> get_kmpc_xteams_func() {
  if (_WAVE_SIZE == 64) {
#if XTEAM_NUM_THREADS == 1024
    _XTEAMS_GETTER_BODY(_16x64)
#elif XTEAM_NUM_THREADS == 512
    _XTEAMS_GETTER_BODY(_8x64)
#elif XTEAM_NUM_THREADS == 256
    _XTEAMS_GETTER_BODY(_4x64)
#elif XTEAM_NUM_THREADS == 128
    _XTEAMS_GETTER_BODY(_2x64)
#elif XTEAM_NUM_THREADS == 64
    _XTEAMS_GETTER_BODY(_1x64)
#else
    static_assert(false, "Unsupported number of threads");
#endif
  } else {
#if XTEAM_NUM_THREADS == 1024
    _XTEAMS_GETTER_BODY(_32x32)
#elif XTEAM_NUM_THREADS == 512
    _XTEAMS_GETTER_BODY(_16x32)
#elif XTEAM_NUM_THREADS == 256
    _XTEAMS_GETTER_BODY(_8x32)
#elif XTEAM_NUM_THREADS == 128
    _XTEAMS_GETTER_BODY(_4x32)
#elif XTEAM_NUM_THREADS == 64
    _XTEAMS_GETTER_BODY(_2x32)
#else
    static_assert(false, "Unsupported number of threads");
#endif
  }
}

// Phase2 getter — only 16x64/8x64/4x64 (wave64) and 32x32/16x32/8x32 (wave32)
template <typename T> xteams_phase2_fn_t<T> get_kmpc_xteams_phase2_func() {
  if (_WAVE_SIZE == 64) {
#if XTEAM_NUM_THREADS == 1024
    _XTEAMS_P2_GETTER_BODY(_16x64)
#elif XTEAM_NUM_THREADS == 512
    _XTEAMS_P2_GETTER_BODY(_8x64)
#elif XTEAM_NUM_THREADS == 256
    _XTEAMS_P2_GETTER_BODY(_4x64)
#else
    static_assert(false, "Unsupported number of threads");
#endif
  } else {
#if XTEAM_NUM_THREADS == 1024
    _XTEAMS_P2_GETTER_BODY(_32x32)
#elif XTEAM_NUM_THREADS == 512
    _XTEAMS_P2_GETTER_BODY(_16x32)
#elif XTEAM_NUM_THREADS == 256
    _XTEAMS_P2_GETTER_BODY(_8x32)
#else
    static_assert(false, "Unsupported number of threads");
#endif
  }
}

template <typename T, ScanOp Op>
T reduce_sim(const T *__restrict in, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  T s = rnv;
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom:s)                               \
    is_device_ptr(d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteamr_func = get_kmpc_xteamr_func<T>();
    T val = rnv;
    for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
      val = scan_combine<T, Op>(val, in[i]);
    xteamr_func(val, &s, d_team_vals<T>, d_td, get_rfun_func<T, Op>(),
                get_rfun_lds_func<T, Op>(), rnv, k, XTEAM_NUM_TEAMS,
                _XTEAMR_SCOPE);
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
    auto xteamr_func = get_kmpc_xteamr_func<T>();
    T val = rnv;
    for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
      val += a[i] * b[i];
    xteamr_func(val, &s, d_team_vals<T>, d_td, get_rfun_sum_func<T>(),
                get_rfun_sum_lds_func<T>(), rnv, k, XTEAM_NUM_TEAMS,
                _XTEAMR_SCOPE);
  }

  return s;
}

// =========================================================================
// GPU cross-team scan kernels (AOMP 2-kernel algorithm with phase2 API)
// =========================================================================

template <typename T, ScanOp Op>
void scan_incl_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  const uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
// K1: serial per-thread scan + cross-team coordination
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_func = get_kmpc_xteams_func<T>();
    T val0 = rnv;
    for (uint64_t i = 0; i < stride && k * stride + i < n; i++) {
      val0 = scan_combine<T, Op>(val0, in[k * stride + i]);
      out[k * stride + i] = val0;
    }
    d_storage<T>[k] = val0;
    xteams_func(val0, d_storage<T>, out, d_team_vals<T>, d_td,
                get_rfun_func<T, Op>(), get_rfun_lds_func<T, Op>(), rnv, k,
                XTEAM_NUM_TEAMS);
  }

// K2: redistribution via phase2 API
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_p2 = get_kmpc_xteams_phase2_func<T>();
    xteams_p2(d_storage<T>, (int)stride, d_team_vals<T>, out,
              get_rfun_func<T, Op>(), rnv, k, true);
  }
}

template <typename T, ScanOp Op>
void scan_excl_sim(const T *__restrict in, T *__restrict out, uint64_t n) {
  const T rnv = scan_identity<T, Op>();
  const uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
// K1: serial per-thread exclusive scan + cross-team coordination
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_func = get_kmpc_xteams_func<T>();
    T val0 = rnv;
    for (uint64_t i = 0; i < stride && k * stride + i < n; i++) {
      out[k * stride + i] = val0;
      val0 = scan_combine<T, Op>(val0, in[k * stride + i]);
    }
    d_storage<T>[k] = val0;
    xteams_func(val0, d_storage<T>, out, d_team_vals<T>, d_td,
                get_rfun_func<T, Op>(), get_rfun_lds_func<T, Op>(), rnv, k,
                XTEAM_NUM_TEAMS);
  }

// K2: redistribution via phase2 API
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_p2 = get_kmpc_xteams_phase2_func<T>();
    xteams_p2(d_storage<T>, (int)stride, d_team_vals<T>, out,
              get_rfun_func<T, Op>(), rnv, k, true);
  }
}

template <typename T>
void scan_incl_dot_sim(const T *__restrict a, const T *__restrict b,
                       T *__restrict out, uint64_t n) {
  const T rnv = T(0);
  const uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
// K1: serial per-thread scan + cross-team coordination
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_func = get_kmpc_xteams_func<T>();
    T val0 = rnv;
    for (uint64_t i = 0; i < stride && k * stride + i < n; i++) {
      val0 += a[k * stride + i] * b[k * stride + i];
      out[k * stride + i] = val0;
    }
    d_storage<T>[k] = val0;
    xteams_func(val0, d_storage<T>, out, d_team_vals<T>, d_td,
                get_rfun_sum_func<T>(), get_rfun_sum_lds_func<T>(), rnv, k,
                XTEAM_NUM_TEAMS);
  }

// K2: redistribution via phase2 API
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_p2 = get_kmpc_xteams_phase2_func<T>();
    xteams_p2(d_storage<T>, (int)stride, d_team_vals<T>, out,
              get_rfun_sum_func<T>(), rnv, k, true);
  }
}

template <typename T>
void scan_excl_dot_sim(const T *__restrict a, const T *__restrict b,
                       T *__restrict out, uint64_t n) {
  const T rnv = T(0);
  const uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
// K1: serial per-thread exclusive scan + cross-team coordination
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>, d_td)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_func = get_kmpc_xteams_func<T>();
    T val0 = rnv;
    for (uint64_t i = 0; i < stride && k * stride + i < n; i++) {
      out[k * stride + i] = val0;
      val0 += a[k * stride + i] * b[k * stride + i];
    }
    d_storage<T>[k] = val0;
    xteams_func(val0, d_storage<T>, out, d_team_vals<T>, d_td,
                get_rfun_sum_func<T>(), get_rfun_sum_lds_func<T>(), rnv, k,
                XTEAM_NUM_TEAMS);
  }

// K2: redistribution via phase2 API
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS)                                             \
    is_device_ptr(d_storage<T>, d_team_vals<T>)
  for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
    auto xteams_p2 = get_kmpc_xteams_phase2_func<T>();
    xteams_p2(d_storage<T>, (int)stride, d_team_vals<T>, out,
              get_rfun_sum_func<T>(), rnv, k, true);
  }
}

// =========================================================================
// Initialization and cleanup of the simulation helper data
// Uses omp_target_memcpy (not omp_target_memset) for broader AOMP compat.
// =========================================================================

template <typename T> void init_device_sim() {
  assert(d_td == nullptr);
  int devid = 0;
  static uint32_t zero = 0;

  d_td = static_cast<uint32_t *>(omp_target_alloc(sizeof(uint32_t), devid));
  d_team_vals<T> =
      static_cast<T *>(omp_target_alloc(sizeof(T) * XTEAM_NUM_TEAMS, devid));
  omp_target_memcpy(d_td, &zero, sizeof(uint32_t), 0, 0, devid,
                    omp_get_initial_device());

  d_storage<T> = static_cast<T *>(omp_target_alloc(
      sizeof(T) * (2 * XTEAM_TOTAL_NUM_THREADS + 1), devid));
}

template <typename T> void free_device_sim() {
  assert(d_td != nullptr);
  int devid = 0;

  omp_target_free(d_td, devid);
  d_td = nullptr;
  omp_target_free(d_team_vals<T>, devid);
  d_team_vals<T> = nullptr;
  omp_target_free(d_storage<T>, devid);
  d_storage<T> = nullptr;
}
