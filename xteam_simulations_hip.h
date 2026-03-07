#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>

#include "common.h"

#define _UI unsigned int
#define _UL unsigned long

// =========================================================================
// Device function declarations — resolved from OpenMP device runtime bitcode
// linked via -mlink-builtin-bitcode.
// =========================================================================

extern "C" {
__device__ void __kmpc_xteams_i(int v, int *result, uint32_t *status,
                                int *aggregates, int *prefixes,
                                void (*rf)(int *, int), const int rnv,
                                const uint64_t k);
__device__ void __kmpc_xteams_l(long v, long *result, uint32_t *status,
                                long *aggregates, long *prefixes,
                                void (*rf)(long *, long), const long rnv,
                                const uint64_t k);
__device__ void __kmpc_xteams_d(double v, double *result, uint32_t *status,
                                double *aggregates, double *prefixes,
                                void (*rf)(double *, double), const double rnv,
                                const uint64_t k);

__device__ void __kmpc_rfun_sum_i(int *val, int otherval);
__device__ void __kmpc_rfun_sum_l(long *val, long otherval);
__device__ void __kmpc_rfun_sum_d(double *val, double otherval);
__device__ void __kmpc_rfun_max_i(int *val, int otherval);
__device__ void __kmpc_rfun_max_l(long *val, long otherval);
__device__ void __kmpc_rfun_max_d(double *val, double otherval);
__device__ void __kmpc_rfun_min_i(int *val, int otherval);
__device__ void __kmpc_rfun_min_l(long *val, long otherval);
__device__ void __kmpc_rfun_min_d(double *val, double otherval);
}

// =========================================================================
// Device helpers: dispatch to the correct typed __kmpc_xteams_* / rfun
// =========================================================================

enum class ScanOp { Sum, Max, Min };

template <typename T, ScanOp Op>
__device__ T scan_identity() {
  if constexpr (Op == ScanOp::Sum)
    return T(0);
  else if constexpr (Op == ScanOp::Max)
    return std::numeric_limits<T>::lowest();
  else
    return std::numeric_limits<T>::max();
}

template <typename T, ScanOp Op>
__device__ T scan_combine(T a, T b) {
  if constexpr (Op == ScanOp::Sum)
    return a + b;
  else if constexpr (Op == ScanOp::Max)
    return a > b ? a : b;
  else
    return a < b ? a : b;
}

template <typename T>
__device__ constexpr void (*get_kmpc_xteams_func())(
    T, T *, uint32_t *, T *, T *, void (*)(T *, T), const T, const uint64_t) {
  if constexpr (std::is_same_v<T, int>)
    return __kmpc_xteams_i;
  else if constexpr (std::is_same_v<T, long>)
    return __kmpc_xteams_l;
  else
    return __kmpc_xteams_d;
}

template <typename T>
__device__ constexpr void (*get_rfun_sum_func())(T *, T) {
  if constexpr (std::is_same_v<T, int>)
    return __kmpc_rfun_sum_i;
  else if constexpr (std::is_same_v<T, long>)
    return __kmpc_rfun_sum_l;
  else
    return __kmpc_rfun_sum_d;
}

template <typename T>
__device__ constexpr void (*get_rfun_max_func())(T *, T) {
  if constexpr (std::is_same_v<T, int>)
    return __kmpc_rfun_max_i;
  else if constexpr (std::is_same_v<T, long>)
    return __kmpc_rfun_max_l;
  else
    return __kmpc_rfun_max_d;
}

template <typename T>
__device__ constexpr void (*get_rfun_min_func())(T *, T) {
  if constexpr (std::is_same_v<T, int>)
    return __kmpc_rfun_min_i;
  else if constexpr (std::is_same_v<T, long>)
    return __kmpc_rfun_min_l;
  else
    return __kmpc_rfun_min_d;
}

template <typename T, ScanOp Op>
__device__ constexpr void (*get_rfun_func())(T *, T) {
  if constexpr (Op == ScanOp::Sum)
    return get_rfun_sum_func<T>();
  else if constexpr (Op == ScanOp::Max)
    return get_rfun_max_func<T>();
  else
    return get_rfun_min_func<T>();
}

// =========================================================================
// Generic __global__ kernels
// =========================================================================

// --- Phase 1: per-thread aggregate + cross-team scan ---
template <typename T, ScanOp Op>
__global__ void scan_k1(const T *in, uint32_t *status, T *aggregates,
                        T *prefixes, T *scan_out, uint64_t n,
                        uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T val = scan_identity<T, Op>();
  for (uint64_t idx = start; idx < end; idx++)
    val = scan_combine<T, Op>(val, in[idx]);
  get_kmpc_xteams_func<T>()(val, scan_out, status, aggregates, prefixes,
                            get_rfun_func<T, Op>(), scan_identity<T, Op>(), k);
}

// --- Phase 2 (inclusive): apply prefix + re-read input ---
template <typename T, ScanOp Op>
__global__ void scan_incl_k2(const T *in, T *out, T *scan_out, uint64_t n,
                             uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    running = scan_combine<T, Op>(running, in[idx]);
    out[idx] = running;
  }
}

// --- Phase 2 (exclusive): apply prefix + re-read input ---
template <typename T, ScanOp Op>
__global__ void scan_excl_k2(const T *in, T *out, T *scan_out, uint64_t n,
                             uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    out[idx] = running;
    running = scan_combine<T, Op>(running, in[idx]);
  }
}

// --- Phase 1 for dot products (two inputs) ---
template <typename T>
__global__ void scan_dot_k1(const T *a, const T *b, uint32_t *status,
                            T *aggregates, T *prefixes, T *scan_out,
                            uint64_t n, uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T val = T(0);
  for (uint64_t idx = start; idx < end; idx++)
    val += a[idx] * b[idx];
  get_kmpc_xteams_func<T>()(val, scan_out, status, aggregates, prefixes,
                            get_rfun_sum_func<T>(), T(0), k);
}

// --- Phase 2 for inclusive dot ---
template <typename T>
__global__ void scan_incl_dot_k2(const T *a, const T *b, T *out, T *scan_out,
                                 uint64_t n, uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    running += a[idx] * b[idx];
    out[idx] = running;
  }
}

// --- Phase 2 for exclusive dot ---
template <typename T>
__global__ void scan_excl_dot_k2(const T *a, const T *b, T *out, T *scan_out,
                                 uint64_t n, uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    out[idx] = running;
    running += a[idx] * b[idx];
  }
}

// --- Single-kernel combined (sim pattern): aggregate + xteams + redistribute ---
template <typename T, ScanOp Op>
__global__ void scan_incl_combined(const T *in, T *out, uint32_t *status,
                                   T *aggregates, T *prefixes, T *scan_out,
                                   uint64_t n, uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T rnv = scan_identity<T, Op>();
  T val = rnv;
  for (uint64_t idx = start; idx < end; idx++)
    val = scan_combine<T, Op>(val, in[idx]);
  get_kmpc_xteams_func<T>()(val, scan_out, status, aggregates, prefixes,
                            get_rfun_func<T, Op>(), rnv, k);
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    running = scan_combine<T, Op>(running, in[idx]);
    out[idx] = running;
  }
}

template <typename T, ScanOp Op>
__global__ void scan_excl_combined(const T *in, T *out, uint32_t *status,
                                   T *aggregates, T *prefixes, T *scan_out,
                                   uint64_t n, uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T rnv = scan_identity<T, Op>();
  T val = rnv;
  for (uint64_t idx = start; idx < end; idx++)
    val = scan_combine<T, Op>(val, in[idx]);
  get_kmpc_xteams_func<T>()(val, scan_out, status, aggregates, prefixes,
                            get_rfun_func<T, Op>(), rnv, k);
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    out[idx] = running;
    running = scan_combine<T, Op>(running, in[idx]);
  }
}

template <typename T>
__global__ void scan_incl_dot_combined(const T *a, const T *b, T *out,
                                       uint32_t *status, T *aggregates,
                                       T *prefixes, T *scan_out, uint64_t n,
                                       uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T val = T(0);
  for (uint64_t idx = start; idx < end; idx++)
    val += a[idx] * b[idx];
  get_kmpc_xteams_func<T>()(val, scan_out, status, aggregates, prefixes,
                            get_rfun_sum_func<T>(), T(0), k);
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    running += a[idx] * b[idx];
    out[idx] = running;
  }
}

template <typename T>
__global__ void scan_excl_dot_combined(const T *a, const T *b, T *out,
                                       uint32_t *status, T *aggregates,
                                       T *prefixes, T *scan_out, uint64_t n,
                                       uint64_t stride) {
  uint64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = k * stride;
  uint64_t end = (start + stride < n) ? start + stride : n;
  T val = T(0);
  for (uint64_t idx = start; idx < end; idx++)
    val += a[idx] * b[idx];
  get_kmpc_xteams_func<T>()(val, scan_out, status, aggregates, prefixes,
                            get_rfun_sum_func<T>(), T(0), k);
  T running = scan_out[k];
  for (uint64_t idx = start; idx < end; idx++) {
    out[idx] = running;
    running += a[idx] * b[idx];
  }
}

// =========================================================================
// Device state
// =========================================================================

static uint32_t *d_status = nullptr;
template <typename T> static T *d_aggregates = nullptr;
template <typename T> static T *d_prefixes = nullptr;
template <typename T> static T *d_scan_out = nullptr;

template <typename T> void init_device_sim() {
  hipMalloc(&d_status, sizeof(uint32_t) * (XTEAM_NUM_TEAMS + 1));
  hipMalloc(&d_aggregates<T>, sizeof(T) * XTEAM_NUM_TEAMS);
  hipMalloc(&d_prefixes<T>, sizeof(T) * XTEAM_NUM_TEAMS);
  hipMalloc(&d_scan_out<T>, sizeof(T) * XTEAM_TOTAL_NUM_THREADS);
  hipMemset(d_status, 0, sizeof(uint32_t) * (XTEAM_NUM_TEAMS + 1));
}

template <typename T> void free_device_sim() {
  hipFree(d_status);
  d_status = nullptr;
  hipFree(d_aggregates<T>);
  d_aggregates<T> = nullptr;
  hipFree(d_prefixes<T>);
  d_prefixes<T> = nullptr;
  hipFree(d_scan_out<T>);
  d_scan_out<T> = nullptr;
}

// =========================================================================
// Host wrapper functions — "sim" (single-kernel) pattern
// =========================================================================

template <typename T>
void scan_incl_sum_sim(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_incl_combined<T, ScanOp::Sum>
      <<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
          in, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
          stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_sum_sim(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_excl_combined<T, ScanOp::Sum>
      <<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
          in, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
          stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_incl_max_sim(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_incl_combined<T, ScanOp::Max>
      <<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
          in, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
          stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_max_sim(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_excl_combined<T, ScanOp::Max>
      <<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
          in, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
          stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_incl_min_sim(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_incl_combined<T, ScanOp::Min>
      <<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
          in, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
          stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_min_sim(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_excl_combined<T, ScanOp::Min>
      <<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
          in, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
          stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_incl_dot_sim(const T *a, const T *b, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_incl_dot_combined<T><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      a, b, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
      stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_dot_sim(const T *a, const T *b, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_excl_dot_combined<T><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      a, b, out, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
      stride);
  hipDeviceSynchronize();
}

// =========================================================================
// Host wrapper functions — "v1" (two-kernel) pattern
// =========================================================================

template <typename T>
void scan_incl_sum_sim_v1(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_k1<T, ScanOp::Sum><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
  scan_incl_k2<T, ScanOp::Sum><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_sum_sim_v1(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_k1<T, ScanOp::Sum><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
  scan_excl_k2<T, ScanOp::Sum><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_incl_max_sim_v1(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_k1<T, ScanOp::Max><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
  scan_incl_k2<T, ScanOp::Max><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_max_sim_v1(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_k1<T, ScanOp::Max><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
  scan_excl_k2<T, ScanOp::Max><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_incl_min_sim_v1(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_k1<T, ScanOp::Min><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
  scan_incl_k2<T, ScanOp::Min><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_min_sim_v1(const T *in, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_k1<T, ScanOp::Min><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
  scan_excl_k2<T, ScanOp::Min><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      in, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_incl_dot_sim_v1(const T *a, const T *b, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_dot_k1<T><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      a, b, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
      stride);
  hipDeviceSynchronize();
  scan_incl_dot_k2<T><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      a, b, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}

template <typename T>
void scan_excl_dot_sim_v1(const T *a, const T *b, T *out, uint64_t n) {
  uint64_t stride = (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
  scan_dot_k1<T><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      a, b, d_status, d_aggregates<T>, d_prefixes<T>, d_scan_out<T>, n,
      stride);
  hipDeviceSynchronize();
  scan_excl_dot_k2<T><<<XTEAM_NUM_TEAMS, XTEAM_NUM_THREADS>>>(
      a, b, out, d_scan_out<T>, n, stride);
  hipDeviceSynchronize();
}
