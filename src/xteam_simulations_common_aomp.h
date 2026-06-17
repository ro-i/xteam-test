// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

// =========================================================================
// Common stuff shared by AOMP simulations
// =========================================================================

#pragma once

#include <type_traits>

#include "xteam_simulations_common.h"

#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __attribute__((address_space(3)))

#if defined(__AMDGCN__) || defined(__NVPTX__)
#define _REDUCTION_FUNC(T, OP, TS)                                             \
  void __kmpc_rfun_##OP##_##TS(T *val, T otherval);                            \
  void __kmpc_rfun_##OP##_lds_##TS(_RF_LDS T *val, _RF_LDS T *otherval);
#else
#define _REDUCTION_FUNC(T, OP, TS)                                             \
  inline void __kmpc_rfun_##OP##_##TS(T *val, T otherval) {}                   \
  inline void __kmpc_rfun_##OP##_lds_##TS(_RF_LDS T *val,                      \
                                          _RF_LDS T *otherval) {}
#endif

#define _REDUCTION_FUNC_ALL(OP)                                                \
  _REDUCTION_FUNC(double, OP, d)                                               \
  _REDUCTION_FUNC(float, OP, f)                                                \
  _REDUCTION_FUNC(int, OP, i)                                                  \
  _REDUCTION_FUNC(_UI, OP, ui)                                                 \
  _REDUCTION_FUNC(long, OP, l)                                                 \
  _REDUCTION_FUNC(_UL, OP, ul)

extern "C" {
_REDUCTION_FUNC_ALL(sum)
_REDUCTION_FUNC_ALL(max)
_REDUCTION_FUNC_ALL(min)
}

template <typename T> static constexpr void (*get_rfun_sum_func())(T *, T) {
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

template <typename T> static constexpr void (*get_rfun_max_func())(T *, T) {
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

template <typename T> static constexpr void (*get_rfun_min_func())(T *, T) {
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

template <typename T>
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

template <typename T>
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

template <typename T>
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

template <typename T, RedOp Op>
static constexpr void (*get_rfun_func())(T *a, T b) {
  if constexpr (Op == RedOp::Sum)
    return get_rfun_sum_func<T>();
  else if constexpr (Op == RedOp::Max)
    return get_rfun_max_func<T>();
  else if constexpr (Op == RedOp::Min)
    return get_rfun_min_func<T>();
  else if constexpr (Op == RedOp::Mult) // unsupported by AOMP codegen
    return [](T *a, T b) { *a *= b; };
  else
    static_assert(!std::is_same_v<T, T>, "Unsupported scan op");
}

template <typename T, RedOp Op>
static constexpr void (*get_rfun_lds_func())(_RF_LDS T *a, _RF_LDS T *b) {
  if constexpr (Op == RedOp::Sum)
    return get_rfun_sum_lds_func<T>();
  else if constexpr (Op == RedOp::Max)
    return get_rfun_max_lds_func<T>();
  else if constexpr (Op == RedOp::Min)
    return get_rfun_min_lds_func<T>();
  else if constexpr (Op == RedOp::Mult) // unsupported by AOMP codegen
    return [](_RF_LDS T *a, _RF_LDS T *b) { *a *= *b; };
  else
    static_assert(!std::is_same_v<T, T>, "Unsupported scan op");
}
