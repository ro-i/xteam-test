// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "omp.h"

#include "common.h"
#include "xteam_simulations_common.h"

#if defined(__AMDGCN__)
#define _TRUNK_WARP_SIZE 64
#elif defined(__NVPTX__)
#define _TRUNK_WARP_SIZE 32
#else
#define _TRUNK_WARP_SIZE 64
#endif

// Thread count for the lvl2 secondary reduction kernel:
// min(512, XTEAM_NUM_TEAMS), matching the plugin's launch logic.
#define _TRUNK_LVL2_THREADS ((512 < XTEAM_NUM_TEAMS) ? 512 : XTEAM_NUM_TEAMS)

// =========================================================================
// Layout-compatible replica of KernelLaunchEnvironmentTy from
// offload/include/Shared/Environment.h.
// Only ReductionBuffer and ReductionBufferElements are accessed by
// __kmpc_reduction_teams_lvl2; the remaining fields preserve ABI layout.
// =========================================================================

struct SimKernelLaunchEnvTy {
  void *ReductionBuffer = nullptr;
  void *DynCGroupMemFbPtr = nullptr;
  uint32_t ReductionCnt = 0;
  uint32_t ReductionIterCnt = 0;
  uint32_t ReductionBufferElements = 0;
  uint32_t DynCGroupMemSize = 0;
  uint8_t DynCGroupMemFb = 0;
};

// =========================================================================
// Runtime API declarations
//
// On the device these are resolved from the OpenMP device runtime bitcode.
// On the host we provide stubs so the host-fallback compilation links.
// =========================================================================

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(
    void *Loc, uint64_t reduce_data_size, void *reduce_data,
    void (*shflFct)(void *, int16_t, int16_t, int16_t),
    void (*cpyFct)(void *, int32_t));
void __kmpc_reduction_teams_lvl1(
    void *Loc, void *GlobalBuffer, void *reduce_data,
    void (*lgcpyFct)(void *, int, void *));
int32_t __kmpc_reduction_teams_lvl2(
    void *KLE, void *reduce_data,
    void (*shflFct)(void *, int16_t, int16_t, int16_t),
    void (*cpyFct)(void *, int32_t),
    void (*glcpyFct)(void *, int, void *),
    void (*glredFct)(void *, int, void *));
void __kmpc_reduction_inter_warp_copy(void *reduce_data, uint32_t size);
int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size);
int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size);
void __kmpc_barrier_simple_spmd(void *Loc, int32_t TId);
uint32_t __kmpc_get_hardware_thread_id_in_block();
}
#else
extern "C" {
int32_t __kmpc_nvptx_parallel_reduce_nowait_v2(void *, uint64_t, void *,
                                               void (*)(void *, int16_t,
                                                        int16_t, int16_t),
                                               void (*)(void *, int32_t)) {
  return 0;
}
void __kmpc_reduction_teams_lvl1(void *, void *, void *,
                                 void (*)(void *, int, void *)) {}
int32_t __kmpc_reduction_teams_lvl2(
    void *, void *, void (*)(void *, int16_t, int16_t, int16_t),
    void (*)(void *, int32_t), void (*)(void *, int, void *),
    void (*)(void *, int, void *)) {
  return 0;
}
void __kmpc_reduction_inter_warp_copy(void *, uint32_t) {}
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
// passed to the reduction runtime entries.  The functions below simulate
// what the compiler would emit for a single scalar reduction variable of
// type T with operation Op.
//
// Reduce list layout (single variable):  void *rl[1] = { &priv };
// Global buffer layout:                  T buf[XTEAM_NUM_TEAMS];
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

// --- 2a. _omp_reduction_inter_warp_copy_func (InterWarpCopyFnTy) ---------
//
// Used by the within-team parallel reduce in kernel 1.  Transfers each
// warp-master's value through shared memory so that warp 0 can perform
// the final cross-warp reduction.  Processes the value in 4-byte chunks
// (matching codegen behaviour for types > 32 bits).
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

// --- 2b. _omp_reduction_inter_warp_copy_func — lvl2 (InterWarpCopyFnTy)
//
// Used by the secondary reduction kernel (kernel 2).  Instead of the
// chunk-by-chunk shared-memory approach, calls
// __kmpc_reduction_inter_warp_copy for each element.  Mirrors
// emitLvl2InterWarpCopyFunction from OMPIRBuilder.
template <typename T> static void lvl2_warp_copy(void *rd, int32_t /*nw*/) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
  T *elem = *reinterpret_cast<T **>(rd);
  __kmpc_reduction_inter_warp_copy(reinterpret_cast<void *>(elem),
                                   static_cast<uint32_t>(sizeof(T)));
#endif
}

// --- 3. _omp_reduction_list_to_global_copy_func (ListGlobalFnTy) ---------
//   buf[idx] = *priv
template <typename T> static void lg_copy(void *buf, int idx, void *rd) {
  static_cast<T *>(buf)[idx] = **reinterpret_cast<T **>(rd);
}

// --- 4. _omp_reduction_global_to_list_copy_func (ListGlobalFnTy) ---------
//   *priv = buf[idx]
template <typename T> static void gl_copy(void *buf, int idx, void *rd) {
  **reinterpret_cast<T **>(rd) = static_cast<T *>(buf)[idx];
}

// --- 5. _omp_reduction_global_to_list_reduce_func (ListGlobalFnTy) -------
//   *priv = combine(*priv, buf[idx])
template <typename T, RedOp Op>
static void gl_reduce(void *buf, int idx, void *rd) {
  T *pp = *reinterpret_cast<T **>(rd);
  *pp = red_combine<T, Op>(*pp, static_cast<T *>(buf)[idx]);
}

} // namespace trunk_sim
#pragma omp end declare target

// =========================================================================
// SimulationTrunk — simulates the trunk LLVM multi-level cross-team
// reduction (ReductionBufNum == 0, the new default).
//
// Kernel 1: within-team parallel reduce, team master copies to global
//           buffer via __kmpc_reduction_teams_lvl1.
// Kernel 2: secondary kernel (1 team, min(512, num_teams) threads)
//           reduces the global buffer via __kmpc_reduction_teams_lvl2.
// =========================================================================

template <typename T> class SimulationTrunkJD : public SimulationTrunkBase<T> {
  void *d_gbuf = nullptr;

  template <RedOp Op> T red_sim(const T *__restrict in, uint64_t n) {
    const T rnv = red_identity<T, Op>();
    T result = rnv;
    void *gbuf = d_gbuf;

    // --- Kernel 1: per-team reduction + copy to global buffer ---
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf)                         \
    ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      T team_priv = rnv;

      uint64_t trips =
          (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
      for (uint64_t iter = 0; iter < trips; iter++) {
        uint64_t i = k + iter * XTEAM_TOTAL_NUM_THREADS;
        T priv = (i < n) ? in[i] : rnv;

        void *rl[1] = {&priv};
        int32_t is_master = __kmpc_nvptx_parallel_reduce_nowait_v2(
            nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, Op>,
            trunk_sim::warp_copy<T>);

        if (is_master)
          team_priv = red_combine<T, Op>(team_priv, priv);
      }

      void *rl[1] = {&team_priv};
      __kmpc_reduction_teams_lvl1(nullptr, gbuf, rl, trunk_sim::lg_copy<T>);
    }

    // --- Kernel 2: secondary reduction of global buffer ---
#pragma omp target teams distribute parallel for num_teams(1)                  \
    num_threads(_TRUNK_LVL2_THREADS) map(tofrom : result) is_device_ptr(gbuf)
    for (int t = 0; t < _TRUNK_LVL2_THREADS; t++) {
      SimKernelLaunchEnvTy kle;
      kle.ReductionBuffer = gbuf;
      kle.ReductionBufferElements = XTEAM_NUM_TEAMS;

      T priv;
      void *rl[1] = {&priv};
      int32_t winner = __kmpc_reduction_teams_lvl2(
          &kle, rl, trunk_sim::shfl_reduce<T, Op>,
          trunk_sim::lvl2_warp_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, Op>);

      if (winner == 1)
        result = red_combine<T, Op>(result, priv);
    }

    return result;
  }

  template <RedOp Op> T red_sim_v2(const T *__restrict in, uint64_t n) {
    const T rnv = red_identity<T, Op>();
    T result = rnv;
    void *gbuf = d_gbuf;

    // --- Kernel 1: per-thread accumulation + parallel reduce + lvl1 ---
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf)                         \
    ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      T priv = rnv;
      for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
        priv = red_combine<T, Op>(priv, in[i]);

      void *rl[1] = {&priv};
      __kmpc_nvptx_parallel_reduce_nowait_v2(nullptr, sizeof(T), rl,
                                             trunk_sim::shfl_reduce<T, Op>,
                                             trunk_sim::warp_copy<T>);
      __kmpc_reduction_teams_lvl1(nullptr, gbuf, rl, trunk_sim::lg_copy<T>);
    }

    // --- Kernel 2: secondary reduction of global buffer ---
#pragma omp target teams distribute parallel for num_teams(1)                  \
    num_threads(_TRUNK_LVL2_THREADS) map(tofrom : result) is_device_ptr(gbuf)
    for (int t = 0; t < _TRUNK_LVL2_THREADS; t++) {
      SimKernelLaunchEnvTy kle;
      kle.ReductionBuffer = gbuf;
      kle.ReductionBufferElements = XTEAM_NUM_TEAMS;

      T priv;
      void *rl[1] = {&priv};
      int32_t winner = __kmpc_reduction_teams_lvl2(
          &kle, rl, trunk_sim::shfl_reduce<T, Op>,
          trunk_sim::lvl2_warp_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, Op>);

      if (winner == 1)
        result = red_combine<T, Op>(result, priv);
    }

    return result;
  }

  T red_dot_sim(const T *__restrict a, const T *__restrict b, uint64_t n) {
    const T rnv = red_identity<T, RedOp::Sum>();
    T result = rnv;
    void *gbuf = d_gbuf;

    // --- Kernel 1: per-team dot-product reduction + lvl1 ---
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf)                         \
    ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      T team_priv = rnv;

      uint64_t trips =
          (n + XTEAM_TOTAL_NUM_THREADS - 1) / XTEAM_TOTAL_NUM_THREADS;
      for (uint64_t iter = 0; iter < trips; iter++) {
        uint64_t i = k + iter * XTEAM_TOTAL_NUM_THREADS;
        T priv = (i < n) ? a[i] * b[i] : rnv;

        void *rl[1] = {&priv};
        int32_t is_master = __kmpc_nvptx_parallel_reduce_nowait_v2(
            nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
            trunk_sim::warp_copy<T>);

        if (is_master)
          team_priv += priv;
      }

      void *rl[1] = {&team_priv};
      __kmpc_reduction_teams_lvl1(nullptr, gbuf, rl, trunk_sim::lg_copy<T>);
    }

    // --- Kernel 2: secondary reduction of global buffer ---
#pragma omp target teams distribute parallel for num_teams(1)                  \
    num_threads(_TRUNK_LVL2_THREADS) map(tofrom : result) is_device_ptr(gbuf)
    for (int t = 0; t < _TRUNK_LVL2_THREADS; t++) {
      SimKernelLaunchEnvTy kle;
      kle.ReductionBuffer = gbuf;
      kle.ReductionBufferElements = XTEAM_NUM_TEAMS;

      T priv;
      void *rl[1] = {&priv};
      int32_t winner = __kmpc_reduction_teams_lvl2(
          &kle, rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
          trunk_sim::lvl2_warp_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, RedOp::Sum>);

      if (winner == 1)
        result += priv;
    }

    return result;
  }

  T red_dot_sim_v2(const T *__restrict a, const T *__restrict b, uint64_t n) {
    const T rnv = red_identity<T, RedOp::Sum>();
    T result = rnv;
    void *gbuf = d_gbuf;

    // --- Kernel 1: per-thread dot accumulation + parallel reduce + lvl1 ---
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf)                         \
    ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      T priv = rnv;
      for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
        priv += a[i] * b[i];

      void *rl[1] = {&priv};
      __kmpc_nvptx_parallel_reduce_nowait_v2(
          nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
          trunk_sim::warp_copy<T>);
      __kmpc_reduction_teams_lvl1(nullptr, gbuf, rl, trunk_sim::lg_copy<T>);
    }

    // --- Kernel 2: secondary reduction of global buffer ---
#pragma omp target teams distribute parallel for num_teams(1)                  \
    num_threads(_TRUNK_LVL2_THREADS) map(tofrom : result) is_device_ptr(gbuf)
    for (int t = 0; t < _TRUNK_LVL2_THREADS; t++) {
      SimKernelLaunchEnvTy kle;
      kle.ReductionBuffer = gbuf;
      kle.ReductionBufferElements = XTEAM_NUM_TEAMS;

      T priv;
      void *rl[1] = {&priv};
      int32_t winner = __kmpc_reduction_teams_lvl2(
          &kle, rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
          trunk_sim::lvl2_warp_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, RedOp::Sum>);

      if (winner == 1)
        result += priv;
    }

    return result;
  }

public:
  void init_device() {
    assert(d_gbuf == nullptr);
    int devid = omp_get_default_device();
    d_gbuf = target_alloc<T>(XTEAM_NUM_TEAMS, devid);
  }

  void reset_device() {}

  void free_device() {
    assert(d_gbuf != nullptr);
    omp_target_free(d_gbuf, omp_get_default_device());
    d_gbuf = nullptr;
  }

  template <RedOp Op>
  std::vector<
      std::pair<std::string, std::function<T(const T *__restrict, uint64_t)>>>
  get_all_red_variants() {
    return {
        {red_op_to_str<Op>("red_{}_sim"),
         [this](const T *__restrict in, uint64_t n) {
           return this->template red_sim<Op>(in, n);
         }},
        {red_op_to_str<Op>("red_{}_sim_v2"),
         [this](const T *__restrict in, uint64_t n) {
           return this->template red_sim_v2<Op>(in, n);
         }},
    };
  }

  std::vector<std::pair<
      std::string,
      std::function<T(const T *__restrict, const T *__restrict, uint64_t)>>>
  get_all_red_dot_variants() {
    return {
        {"red_dot_sim",
         [this](const T *__restrict a, const T *__restrict b, uint64_t n) {
           return this->red_dot_sim(a, b, n);
         }},
        {"red_dot_sim_v2",
         [this](const T *__restrict a, const T *__restrict b, uint64_t n) {
           return this->red_dot_sim_v2(a, b, n);
         }},
    };
  }

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

  std::vector<std::pair<
      std::string, std::function<void(const T *__restrict, const T *__restrict,
                                      T *__restrict, uint64_t)>>>
  get_all_scan_dot_incl_variants() {
    return {};
  }

  std::vector<std::pair<
      std::string, std::function<void(const T *__restrict, const T *__restrict,
                                      T *__restrict, uint64_t)>>>
  get_all_scan_dot_excl_variants() {
    return {};
  }

}; // class SimulationTrunk
