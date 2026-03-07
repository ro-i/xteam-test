#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <omp.h>

// Common reduction/scan device state
uint32_t *d_td = nullptr;
template <typename T> T *d_team_vals = nullptr;

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
// Rfun getter helpers (shared by dev and AOMP simulations)
// =========================================================================

template <typename T> constexpr void (*get_rfun_sum_func())(T *, T) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_sum_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_sum_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_sum_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_rfun_sum_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_sum_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_rfun_sum_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr void (*get_rfun_max_func())(T *, T) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_max_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_max_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_max_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_rfun_max_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_max_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_rfun_max_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr void (*get_rfun_min_func())(T *, T) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_min_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_min_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_min_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_rfun_min_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_min_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_rfun_min_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T>
constexpr void (*get_rfun_sum_lds_func())(_RF_LDS T *, _RF_LDS T *) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_sum_lds_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_sum_lds_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_sum_lds_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_rfun_sum_lds_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_sum_lds_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_rfun_sum_lds_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T>
constexpr void (*get_rfun_max_lds_func())(_RF_LDS T *, _RF_LDS T *) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_max_lds_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_max_lds_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_max_lds_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_rfun_max_lds_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_max_lds_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_rfun_max_lds_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T>
constexpr void (*get_rfun_min_lds_func())(_RF_LDS T *, _RF_LDS T *) {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_min_lds_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_min_lds_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_min_lds_i;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __kmpc_rfun_min_lds_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_min_lds_l;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __kmpc_rfun_min_lds_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

// =========================================================================
// ScanOp-generic helpers
// =========================================================================

enum class ScanOp { Sum, Max, Min };

template <typename T, ScanOp Op>
constexpr T scan_identity() {
  if constexpr (Op == ScanOp::Sum)
    return T(0);
  else if constexpr (Op == ScanOp::Max)
    return std::numeric_limits<T>::lowest();
  else
    return std::numeric_limits<T>::max();
}

template <typename T, ScanOp Op>
constexpr T scan_combine(T a, T b) {
  if constexpr (Op == ScanOp::Sum)
    return a + b;
  else if constexpr (Op == ScanOp::Max)
    return std::max(a, b);
  else
    return std::min(a, b);
}

template <typename T, ScanOp Op>
constexpr void (*get_rfun_func())(T *, T) {
  if constexpr (Op == ScanOp::Sum)
    return get_rfun_sum_func<T>();
  else if constexpr (Op == ScanOp::Max)
    return get_rfun_max_func<T>();
  else
    return get_rfun_min_func<T>();
}

template <typename T, ScanOp Op>
constexpr void (*get_rfun_lds_func())(_RF_LDS T *, _RF_LDS T *) {
  if constexpr (Op == ScanOp::Sum)
    return get_rfun_sum_lds_func<T>();
  else if constexpr (Op == ScanOp::Max)
    return get_rfun_max_lds_func<T>();
  else
    return get_rfun_min_lds_func<T>();
}
