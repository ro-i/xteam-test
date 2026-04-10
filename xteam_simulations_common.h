// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "omp.h"

#include "common.h"

// =========================================================================
// Common macros and rfun declarations shared by AOMP simulations
// =========================================================================

#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __attribute__((address_space(3)))

#define _REDUCTION_FUNC(T, OP, TS, BODY)                                       \
  void __kmpc_rfun_##OP##_##TS(T *val, T otherval) BODY;                       \
  void __kmpc_rfun_##OP##_lds_##TS(_RF_LDS T *val, _RF_LDS T *otherval) BODY

#define _REDUCTION_FUNC_ALL(OP, BODY)                                          \
  _REDUCTION_FUNC(double, OP, d, BODY)                                         \
  _REDUCTION_FUNC(float, OP, f, BODY)                                          \
  _REDUCTION_FUNC(int, OP, i, BODY)                                            \
  _REDUCTION_FUNC(_UI, OP, ui, BODY)                                           \
  _REDUCTION_FUNC(long, OP, l, BODY)                                           \
  _REDUCTION_FUNC(_UL, OP, ul, BODY)

#if defined(__AMDGCN__) || defined(__NVPTX__)

extern "C" {
_REDUCTION_FUNC_ALL(sum, ;)
_REDUCTION_FUNC_ALL(max, ;)
_REDUCTION_FUNC_ALL(min, ;)
}

#else

extern "C" {
_REDUCTION_FUNC_ALL(sum, {})
_REDUCTION_FUNC_ALL(max, {})
_REDUCTION_FUNC_ALL(min, {})
}

#endif

// =========================================================================
// Common declarations shared by TRUNK simulations
// =========================================================================

#if defined(__AMDGCN__)
#define _TRUNK_WARP_SIZE 64
#elif defined(__NVPTX__)
#define _TRUNK_WARP_SIZE 32
#else
#define _TRUNK_WARP_SIZE 64
#endif

// Replicas for the definitions in DeviceTypes.h
using InterWarpCopyFnTy = void (*)(void *src, int32_t warp_num);
using ShuffleReductFnTy = void (*)(void *rhsData, int16_t lane_id,
                                   int16_t lane_offset, int16_t shortCircuit);
using ListGlobalFnTy = void (*)(void *buffer, int idx, void *reduce_data);

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(void *Loc,
                                               uint64_t reduce_data_size,
                                               void *reduce_data,
                                               ShuffleReductFnTy shflFct,
                                               InterWarpCopyFnTy cpyFct);
int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size);
int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size);
void __kmpc_barrier_simple_spmd(void *Loc, int32_t TId);
uint32_t __kmpc_get_hardware_thread_id_in_block();
}
#else
extern "C" {
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(void *, uint64_t, void *,
                                               ShuffleReductFnTy shflFct,
                                               InterWarpCopyFnTy cpyFct) {
  return 0;
}
int32_t __kmpc_shuffle_int32(int32_t, int16_t, int16_t) { return 0; }
int64_t __kmpc_shuffle_int64(int64_t, int16_t, int16_t) { return 0; }
void __kmpc_barrier_simple_spmd(void *, int32_t) {}
uint32_t __kmpc_get_hardware_thread_id_in_block() { return 0; }
}
#endif

// =========================================================================
// Device helpers and codegen-simulated callbacks
//
// OMPIRBuilder::createReductionsGPU generates callback functions that are
// passed to the reduction runtime entries.  The functions below simulate what
// the compiler would emit for a single scalar reduction variable of type T with
// operation Op.
//
// Reduce list layout (single variable):  void *rl[1] = { &priv };
// Global buffer layout:                  T buf[_TRUNK_NUM_RECORDS] or T
// buf[XTEAM_NUM_TEAMS];
// =========================================================================

#pragma omp begin declare target

#if defined(__AMDGCN__) || defined(__NVPTX__)
// Shared-memory transfer medium for the inter-warp copy callback.
// Mirrors __openmp_nvptx_data_transfer_temporary_storage from codegen.
[[clang::loader_uninitialized]] static volatile __attribute__((
    address_space(3))) int32_t __trunk_sim_xfer[_TRUNK_WARP_SIZE];
#endif

#define __trunk_sim_barrier() __kmpc_barrier_simple_spmd(nullptr, 0)
#define __trunk_sim_tid() __kmpc_get_hardware_thread_id_in_block()

namespace trunk_sim {

// --- shuffle helper (wraps __kmpc_shuffle_int{32,64}) --------------------
template <typename T> static T shuffle(T val, int16_t offset) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
  if constexpr (sizeof(T) <= 4) {
    int32_t tmp;
    __builtin_memcpy(&tmp, &val, sizeof(T));
    tmp = __kmpc_shuffle_int32(tmp, offset, _TRUNK_WARP_SIZE);
    __builtin_memcpy(&val, &tmp, sizeof(T));
  } else {
    int64_t tmp;
    __builtin_memcpy(&tmp, &val, sizeof(T));
    tmp = __kmpc_shuffle_int64(tmp, offset, _TRUNK_WARP_SIZE);
    __builtin_memcpy(&val, &tmp, sizeof(T));
  }
#endif
  return val;
}

// --- 1. _omp_reduction_shuffle_and_reduce_func (ShuffleReductFnTy) -------
//
// Called by gpu_regular_warp_reduce (algo 0) and gpu_irregular_warp_reduce
// (algo 1) inside the runtime.  Shuffles the private value from a remote
// lane, then conditionally combines (or copies for algo 1 upper lanes).
template <typename T, RedOp Op>
static void shfl_reduce(void *rd, int16_t lid, int16_t off, int16_t av) {
  T *pp = *reinterpret_cast<T **>(rd);
  T remote = shuffle<T>(*pp, off);

  bool do_reduce = (av == 0) || (av == 1 && lid < off) ||
                   (av == 2 && (lid & 1) == 0 && off > 0);
  if (do_reduce)
    *pp = red_combine<T, Op>(*pp, remote);
  if (av == 1 && lid >= off)
    *pp = remote;
}

// --- 2. _omp_reduction_inter_warp_copy_func (InterWarpCopyFnTy) ----------
//
// Transfers each warp-master's value through shared memory so that warp 0
// can perform the final cross-warp reduction.  Processes the value in
// 4-byte chunks (matching codegen behaviour for types > 32 bits).
template <typename T> static void warp_copy(void *rd, int32_t nw) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
  uint32_t tid = __trunk_sim_tid();
  uint32_t lid = tid % _TRUNK_WARP_SIZE;
  uint32_t wid = tid / _TRUNK_WARP_SIZE;
  char *base = reinterpret_cast<char *>(*reinterpret_cast<T **>(rd));
  constexpr unsigned n_chunks = (sizeof(T) + 3) / 4;

  for (unsigned c = 0; c < n_chunks; c++) {
    __trunk_sim_barrier();
    if (lid == 0) {
      int32_t tmp;
      __builtin_memcpy(&tmp, base + c * sizeof(int32_t), sizeof(int32_t));
      __trunk_sim_xfer[wid] = tmp;
    }
    __trunk_sim_barrier();
    if (tid < static_cast<uint32_t>(nw)) {
      int32_t tmp = __trunk_sim_xfer[tid];
      __builtin_memcpy(base + c * sizeof(int32_t), &tmp, sizeof(int32_t));
    }
  }
#endif
}

// --- 3. _omp_reduction_list_to_global_copy_func (ListGlobalFnTy) ---------
//   buf[idx] = *priv
template <typename T> static void lg_copy(void *buf, int idx, void *rd) {
  static_cast<T *>(buf)[idx] = **reinterpret_cast<T **>(rd);
}

// --- 5. _omp_reduction_global_to_list_copy_func (ListGlobalFnTy) ---------
//   *priv = buf[idx]
template <typename T> static void gl_copy(void *buf, int idx, void *rd) {
  **reinterpret_cast<T **>(rd) = static_cast<T *>(buf)[idx];
}

// --- 6. _omp_reduction_global_to_list_reduce_func (ListGlobalFnTy) -------
//   *priv = combine(*priv, buf[idx])
template <typename T, RedOp Op>
static void gl_reduce(void *buf, int idx, void *rd) {
  T *pp = *reinterpret_cast<T **>(rd);
  *pp = red_combine<T, Op>(*pp, static_cast<T *>(buf)[idx]);
}

} // namespace trunk_sim

#pragma omp end declare target

// =========================================================================
// Simulation concept — compile-time contract for simulation types, avoids
// virtual dispatch and thus allows member functions to have additional template
// parameters.
// =========================================================================
template <typename T> struct SimulationBase;

template <typename S, typename T>
concept SimulationLike = std::derived_from<S, SimulationBase<T>>;

// =========================================================================
// Base class providing default (no-op) implementations.
// Derived classes hide the methods they wish to override.
// =========================================================================
template <typename T> class SimulationBase {
public:
  void init_device() {}
  void reset_device() {}
  void free_device() {}

  // Return descriptions and implementations for all supported reduction and
  // scan variants.  Return empty vectors if no variants are supported for the
  // given operation.  Be non-virtual to allow RedOp as a template parameter.

  // Get all supported versions of reduce ...
  template <RedOp Op>
  std::vector<
      std::pair<std::string, std::function<T(const T *__restrict, uint64_t)>>>
  get_all_red_variants() {
    return {};
  }

  // ... and reduce_dot.
  std::vector<std::pair<
      std::string,
      std::function<T(const T *__restrict, const T *__restrict, uint64_t)>>>
  get_all_red_dot_variants() {
    return {};
  }

  // Get all supported versions of scan ...
  template <RedOp Op>
  std::vector<std::pair<
      std::string,
      std::function<void(const T *__restrict, T *__restrict, uint64_t)>>>
  get_all_scan_incl_variants() {
    return {};
  }

  template <RedOp Op>
  std::vector<std::pair<
      std::string,
      std::function<void(const T *__restrict, T *__restrict, uint64_t)>>>
  get_all_scan_excl_variants() {
    return {};
  }

  template <RedOp Op, ScanMode Mode>
  std::vector<std::pair<
      std::string,
      std::function<void(const T *__restrict, T *__restrict, uint64_t)>>>
  get_all_scan_variants() {
    if constexpr (Mode == ScanMode::Incl)
      return get_all_scan_incl_variants<Op>();
    else if constexpr (Mode == ScanMode::Excl)
      return get_all_scan_excl_variants<Op>();
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported scan mode");
  }

  // ... and scan_dot.
  std::vector<std::pair<
      std::string, std::function<void(const T *__restrict, const T *__restrict,
                                      T *__restrict, uint64_t)>>>
  get_all_scan_dot_excl_variants() {
    return {};
  }

  std::vector<std::pair<
      std::string, std::function<void(const T *__restrict, const T *__restrict,
                                      T *__restrict, uint64_t)>>>
  get_all_scan_dot_incl_variants() {
    return {};
  }
};

// Intermediate base for AOMP-based simulations
template <typename T> class SimulationAOMPBase : public SimulationBase<T> {
public:
  static constexpr void (*get_rfun_sum_func())(T *, T) {
    if constexpr (std::is_same_v<T, double>)
      return __kmpc_rfun_sum_d;
    else if constexpr (std::is_same_v<T, float>)
      return __kmpc_rfun_sum_f;
    else if constexpr (std::is_same_v<T, int>)
      return __kmpc_rfun_sum_i;
    else if constexpr (std::is_same_v<T, unsigned int>)
      return __kmpc_rfun_sum_ui;
    else if constexpr (std::is_same_v<T, long>)
      return __kmpc_rfun_sum_l;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return __kmpc_rfun_sum_ul;
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }

  static constexpr void (*get_rfun_max_func())(T *, T) {
    if constexpr (std::is_same_v<T, double>)
      return __kmpc_rfun_max_d;
    else if constexpr (std::is_same_v<T, float>)
      return __kmpc_rfun_max_f;
    else if constexpr (std::is_same_v<T, int>)
      return __kmpc_rfun_max_i;
    else if constexpr (std::is_same_v<T, unsigned int>)
      return __kmpc_rfun_max_ui;
    else if constexpr (std::is_same_v<T, long>)
      return __kmpc_rfun_max_l;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return __kmpc_rfun_max_ul;
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }

  static constexpr void (*get_rfun_min_func())(T *, T) {
    if constexpr (std::is_same_v<T, double>)
      return __kmpc_rfun_min_d;
    else if constexpr (std::is_same_v<T, float>)
      return __kmpc_rfun_min_f;
    else if constexpr (std::is_same_v<T, int>)
      return __kmpc_rfun_min_i;
    else if constexpr (std::is_same_v<T, unsigned int>)
      return __kmpc_rfun_min_ui;
    else if constexpr (std::is_same_v<T, long>)
      return __kmpc_rfun_min_l;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return __kmpc_rfun_min_ul;
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }

  static constexpr void (*get_rfun_sum_lds_func())(_RF_LDS T *, _RF_LDS T *) {
    if constexpr (std::is_same_v<T, double>)
      return __kmpc_rfun_sum_lds_d;
    else if constexpr (std::is_same_v<T, float>)
      return __kmpc_rfun_sum_lds_f;
    else if constexpr (std::is_same_v<T, int>)
      return __kmpc_rfun_sum_lds_i;
    else if constexpr (std::is_same_v<T, unsigned int>)
      return __kmpc_rfun_sum_lds_ui;
    else if constexpr (std::is_same_v<T, long>)
      return __kmpc_rfun_sum_lds_l;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return __kmpc_rfun_sum_lds_ul;
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }

  static constexpr void (*get_rfun_max_lds_func())(_RF_LDS T *, _RF_LDS T *) {
    if constexpr (std::is_same_v<T, double>)
      return __kmpc_rfun_max_lds_d;
    else if constexpr (std::is_same_v<T, float>)
      return __kmpc_rfun_max_lds_f;
    else if constexpr (std::is_same_v<T, int>)
      return __kmpc_rfun_max_lds_i;
    else if constexpr (std::is_same_v<T, unsigned int>)
      return __kmpc_rfun_max_lds_ui;
    else if constexpr (std::is_same_v<T, long>)
      return __kmpc_rfun_max_lds_l;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return __kmpc_rfun_max_lds_ul;
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }

  static constexpr void (*get_rfun_min_lds_func())(_RF_LDS T *, _RF_LDS T *) {
    if constexpr (std::is_same_v<T, double>)
      return __kmpc_rfun_min_lds_d;
    else if constexpr (std::is_same_v<T, float>)
      return __kmpc_rfun_min_lds_f;
    else if constexpr (std::is_same_v<T, int>)
      return __kmpc_rfun_min_lds_i;
    else if constexpr (std::is_same_v<T, unsigned int>)
      return __kmpc_rfun_min_lds_ui;
    else if constexpr (std::is_same_v<T, long>)
      return __kmpc_rfun_min_lds_l;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return __kmpc_rfun_min_lds_ul;
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }

  template <RedOp Op> static constexpr void (*get_rfun_func())(T *a, T b) {
    if constexpr (Op == RedOp::Sum)
      return get_rfun_sum_func();
    else if constexpr (Op == RedOp::Max)
      return get_rfun_max_func();
    else if constexpr (Op == RedOp::Min)
      return get_rfun_min_func();
    else if constexpr (Op == RedOp::Mult) // unsupported by AOMP codegen
      return [](T *a, T b) { *a *= b; };
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported scan op");
  }

  template <RedOp Op>
  static constexpr void (*get_rfun_lds_func())(_RF_LDS T *a, _RF_LDS T *b) {
    if constexpr (Op == RedOp::Sum)
      return get_rfun_sum_lds_func();
    else if constexpr (Op == RedOp::Max)
      return get_rfun_max_lds_func();
    else if constexpr (Op == RedOp::Min)
      return get_rfun_min_lds_func();
    else if constexpr (Op == RedOp::Mult) // unsupported by AOMP codegen
      return [](_RF_LDS T *a, _RF_LDS T *b) { *a *= *b; };
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported scan op");
  }
};

// Intermediate base for trunk-based simulations
template <typename T> class SimulationTrunkBase : public SimulationBase<T> {};

// No-op simulation used when no specific backend is selected.
template <typename T> class SimulationNoop : public SimulationBase<T> {};
