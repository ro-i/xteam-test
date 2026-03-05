#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <omp.h>

#include "common.h"

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

#if defined(__AMDGCN__) || defined(__NVPTX__)

extern "C" {
void __kmpc_rfun_sum_d(double *val, double otherval);
void __kmpc_rfun_sum_f(float *val, float otherval);
void __kmpc_rfun_sum_i(int *val, int otherval);
void __kmpc_rfun_sum_ui(_UI *val, _UI otherval);
void __kmpc_rfun_sum_l(long *val, long otherval);
void __kmpc_rfun_sum_ul(_UL *val, _UL otherval);
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
void __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
void __kmpc_rfun_sum_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
void __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
void __kmpc_rfun_sum_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
void __kmpc_rfun_max_d(double *val, double otherval);
void __kmpc_rfun_max_f(float *val, float otherval);
void __kmpc_rfun_max_i(int *val, int otherval);
void __kmpc_rfun_max_ui(_UI *val, _UI otherval);
void __kmpc_rfun_max_l(long *val, long otherval);
void __kmpc_rfun_max_ul(_UL *val, _UL otherval);
void __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
void __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
void __kmpc_rfun_max_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
void __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
void __kmpc_rfun_max_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
void __kmpc_rfun_min_d(double *val, double otherval);
void __kmpc_rfun_min_f(float *val, float otherval);
void __kmpc_rfun_min_i(int *val, int otherval);
void __kmpc_rfun_min_ui(_UI *val, _UI otherval);
void __kmpc_rfun_min_l(long *val, long otherval);
void __kmpc_rfun_min_ul(_UL *val, _UL otherval);
void __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
void __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
void __kmpc_rfun_min_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
void __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
void __kmpc_rfun_min_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
}

#else

extern "C" {
void __kmpc_rfun_sum_d(double *val, double otherval) {}
void __kmpc_rfun_sum_f(float *val, float otherval) {}
void __kmpc_rfun_sum_i(int *val, int otherval) {}
void __kmpc_rfun_sum_ui(_UI *val, _UI otherval) {}
void __kmpc_rfun_sum_l(long *val, long otherval) {}
void __kmpc_rfun_sum_ul(_UL *val, _UL otherval) {}
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {}
void __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {}
void __kmpc_rfun_sum_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {}
void __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {}
void __kmpc_rfun_sum_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {}
void __kmpc_rfun_max_d(double *val, double otherval) {}
void __kmpc_rfun_max_f(float *val, float otherval) {}
void __kmpc_rfun_max_i(int *val, int otherval) {}
void __kmpc_rfun_max_ui(_UI *val, _UI otherval) {}
void __kmpc_rfun_max_l(long *val, long otherval) {}
void __kmpc_rfun_max_ul(_UL *val, _UL otherval) {}
void __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {}
void __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {}
void __kmpc_rfun_max_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {}
void __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {}
void __kmpc_rfun_max_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {}
void __kmpc_rfun_min_d(double *val, double otherval) {}
void __kmpc_rfun_min_f(float *val, float otherval) {}
void __kmpc_rfun_min_i(int *val, int otherval) {}
void __kmpc_rfun_min_ui(_UI *val, _UI otherval) {}
void __kmpc_rfun_min_l(long *val, long otherval) {}
void __kmpc_rfun_min_ul(_UL *val, _UL otherval) {}
void __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {}
void __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {}
void __kmpc_rfun_min_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {}
void __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {}
void __kmpc_rfun_min_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {}
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
