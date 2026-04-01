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
// Common macros and rfun declarations shared by dev and AOMP simulations
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
// Simulation concept — compile-time contract for simulation types
// =========================================================================
struct SimulationTag {};

template <typename S>
concept SimulationLike = std::derived_from<S, SimulationTag> && requires(S s) {
  { s.init_device() } -> std::same_as<void>;
  { s.reset_device() } -> std::same_as<void>;
  { s.free_device() } -> std::same_as<void>;
};

// =========================================================================
// Base class providing default (no-op) implementations.
// Derived classes hide the methods they wish to override.
// =========================================================================
template <typename T> class SimulationBase : public SimulationTag {
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
  get_all_reduce_variants() {
    return {};
  }

  // ... and reduce_dot.
  std::vector<std::pair<
      std::string,
      std::function<T(const T *__restrict, const T *__restrict, uint64_t)>>>
  get_all_reduce_dot_variants() {
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

  template <RedOp Op> static constexpr void (*get_rfun_func())(T *, T) {
    if constexpr (Op == RedOp::Sum)
      return get_rfun_sum_func();
    else if constexpr (Op == RedOp::Max)
      return get_rfun_max_func();
    else if constexpr (Op == RedOp::Min)
      return get_rfun_min_func();
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported scan op");
  }

  template <RedOp Op>
  static constexpr void (*get_rfun_lds_func())(_RF_LDS T *, _RF_LDS T *) {
    if constexpr (Op == RedOp::Sum)
      return get_rfun_sum_lds_func();
    else if constexpr (Op == RedOp::Max)
      return get_rfun_max_lds_func();
    else if constexpr (Op == RedOp::Min)
      return get_rfun_min_lds_func();
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported scan op");
  }
};

// Intermediate base for trunk-based simulations
template <typename T> class SimulationTrunkBase : public SimulationBase<T> {};

// No-op simulation used when no specific backend is selected.
template <typename T> class SimulationNoop : public SimulationBase<T> {};
