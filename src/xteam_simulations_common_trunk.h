// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

// =========================================================================
// Common stuff shared by TRUNK simulations
// =========================================================================

#pragma once

#include <cstddef>
#include <cstdint>

#include "xteam_simulations_common.h"

#if defined(__AMDGCN__)
#define TRUNK_WARP_SIZE 64
#elif defined(__NVPTX__)
#define TRUNK_WARP_SIZE 32
#else
#define TRUNK_WARP_SIZE 64
#endif

// Replicas for the definitions in DeviceTypes.h
using InterWarpCopyFnTy = void (*)(void *src, int32_t warp_num);
using ShuffleReductFnTy = void (*)(void *rhsData, int16_t lane_id,
                                   int16_t lane_offset, int16_t shortCircuit);
using ListGlobalFnTy = void (*)(void *buffer, int idx, void *reduce_data);

extern "C" {
#if defined(__AMDGCN__) || defined(__NVPTX__)
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(void *Loc,
                                               uint64_t reduce_data_size,
                                               void *reduce_data,
                                               ShuffleReductFnTy shflFct,
                                               InterWarpCopyFnTy cpyFct);
int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size);
int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size);
void __kmpc_barrier_simple_spmd(void *Loc, int32_t TId);
uint32_t __kmpc_get_hardware_thread_id_in_block();
#else
inline int32_t
__kmpc_nvptx_parallel_reduce_nowait_v2(void *, uint64_t, void *,
                                       ShuffleReductFnTy shflFct,
                                       InterWarpCopyFnTy cpyFct) {
  return 0;
}
inline int32_t __kmpc_shuffle_int32(int32_t, int16_t, int16_t) { return 0; }
inline int64_t __kmpc_shuffle_int64(int64_t, int16_t, int16_t) { return 0; }
inline void __kmpc_barrier_simple_spmd(void *, int32_t) {}
inline uint32_t __kmpc_get_hardware_thread_id_in_block() { return 0; }
#endif
}

// =========================================================================
// Device helpers and codegen-simulated callbacks
//
// OMPIRBuilder::createReductionsGPU generates callback functions that are
// passed to the reduction runtime entries.  The functions below simulate what
// the compiler would emit for a single scalar reduction variable of type T with
// operation Op.
//
// Reduce list layout (single variable):  void *rl[1] = { &priv };
// Global buffer layout (depending on the implementation):
//        T buf[TRUNK_NUM_RECORDS] or T buf[XTEAM_NUM_TEAMS]
// =========================================================================

#pragma omp begin declare target

#if defined(__AMDGCN__) || defined(__NVPTX__)
// Shared-memory transfer medium for the inter-warp copy callback.
// Mirrors __openmp_nvptx_data_transfer_temporary_storage from codegen.
[[clang::loader_uninitialized]] static volatile
    __attribute__((address_space(3))) int32_t trunk_sim_xfer[TRUNK_WARP_SIZE];
#endif

#define trunk_sim_barrier() __kmpc_barrier_simple_spmd(nullptr, 0)
#define trunk_sim_tid() __kmpc_get_hardware_thread_id_in_block()

// --- shuffle helper (wraps __kmpc_shuffle_int{32,64}) --------------------
template <typename T> static T shuffle(T val, int16_t offset) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
  // Mirror OMPIRBuilder::shuffleAndStore: peel largest power-of-two chunk
  // (8, 4, 2, 1 bytes) off the remaining payload and shuffle each chunk.
  // For sizeof(T) <= 8 this collapses to a single shuffle.  For larger
  // types (e.g. a 48-byte struct) it produces sizeof(T)/8 int64 shuffles.
  // The runtime only exposes int32/int64 shuffles, so 2- and 1-byte
  // chunks are widened to int32 (matching the codegen behaviour, which
  // truncates the i32 result back to iN after the call).
  char *base = reinterpret_cast<char *>(&val);
  size_t remaining = sizeof(T);
  for (unsigned int_size = 8; int_size >= 1; int_size >>= 1) {
    while (remaining >= int_size) {
      if (int_size == 8) {
        int64_t tmp;
        __builtin_memcpy(&tmp, base, 8);
        tmp = __kmpc_shuffle_int64(tmp, offset, TRUNK_WARP_SIZE);
        __builtin_memcpy(base, &tmp, 8);
      } else {
        int32_t tmp = 0;
        __builtin_memcpy(&tmp, base, int_size);
        tmp = __kmpc_shuffle_int32(tmp, offset, TRUNK_WARP_SIZE);
        __builtin_memcpy(base, &tmp, int_size);
      }
      base += int_size;
      remaining -= int_size;
    }
  }
#endif
  return val;
}

// --- _omp_reduction_shuffle_and_reduce_func (ShuffleReductFnTy) -------
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

// --- _omp_reduction_inter_warp_copy_func (InterWarpCopyFnTy) ----------
//
// Transfers each warp-master's value through shared memory so that warp 0
// can perform the final cross-warp reduction.  Processes the value in
// 4-byte chunks (matching codegen behaviour for types > 32 bits).
template <typename T> static void warp_copy(void *rd, int32_t nw) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
  uint32_t tid = trunk_sim_tid();
  uint32_t lid = tid % TRUNK_WARP_SIZE;
  uint32_t wid = tid / TRUNK_WARP_SIZE;
  char *base = reinterpret_cast<char *>(*reinterpret_cast<T **>(rd));
  constexpr unsigned n_chunks = (sizeof(T) + 3) / 4;

  for (unsigned c = 0; c < n_chunks; c++) {
    trunk_sim_barrier();
    if (lid == 0) {
      int32_t tmp;
      __builtin_memcpy(&tmp, base + c * sizeof(int32_t), sizeof(int32_t));
      trunk_sim_xfer[wid] = tmp;
    }
    trunk_sim_barrier();
    if (tid < static_cast<uint32_t>(nw)) {
      int32_t tmp = trunk_sim_xfer[tid];
      __builtin_memcpy(base + c * sizeof(int32_t), &tmp, sizeof(int32_t));
    }
  }
#endif
}

// --- _omp_reduction_list_to_global_copy_func (ListGlobalFnTy) ---------
//   buf[idx] = *priv
template <typename T> static void lg_copy(void *buf, int idx, void *rd) {
  static_cast<T *>(buf)[idx] = **reinterpret_cast<T **>(rd);
}

// --- _omp_reduction_global_to_list_copy_func (ListGlobalFnTy) ---------
//   *priv = buf[idx]
template <typename T> static void gl_copy(void *buf, int idx, void *rd) {
  **reinterpret_cast<T **>(rd) = static_cast<T *>(buf)[idx];
}

// --- _omp_reduction_global_to_list_reduce_func (ListGlobalFnTy) -------
//   *priv = combine(*priv, buf[idx])
template <typename T, RedOp Op>
static void gl_reduce(void *buf, int idx, void *rd) {
  T *pp = *reinterpret_cast<T **>(rd);
  *pp = red_combine<T, Op>(*pp, static_cast<T *>(buf)[idx]);
}

#pragma omp end declare target
