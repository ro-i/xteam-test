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
// Completes the declarations from xteam_simulations_common.h
// =========================================================================

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
void __kmpc_reduction_teams_lvl1(void *Loc, void *GlobalBuffer,
                                 void *reduce_data, ListGlobalFnTy lgcpyFct);
int32_t __kmpc_reduction_teams_lvl2(void *KLE, void *reduce_data,
                                    ShuffleReductFnTy shflFct,
                                    InterWarpCopyFnTy cpyFct,
                                    ListGlobalFnTy glcpyFct,
                                    ListGlobalFnTy glredFct);
void __kmpc_reduction_inter_warp_copy(void *reduce_data, uint32_t size);
}
#else
extern "C" {
void __kmpc_reduction_teams_lvl1(void *, void *, void *,
                                 ListGlobalFnTy lgcpyFct) {}
int32_t __kmpc_reduction_teams_lvl2(void *, void *, ShuffleReductFnTy shflFct,
                                    InterWarpCopyFnTy cpyFct,
                                    ListGlobalFnTy glcpyFct,
                                    ListGlobalFnTy glredFct) {
  return 0;
}
void __kmpc_reduction_inter_warp_copy(void *, uint32_t) {}
}
#endif

// =========================================================================
// Device helpers and codegen-simulated callbacks
// Completes the definitions from xteam_simulations_common.h
// =========================================================================

#pragma omp begin declare target

namespace trunk_sim {

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

} // namespace trunk_sim
#pragma omp end declare target

// =========================================================================
// SimulationTrunk — simulates the trunk LLVM multi-level cross-team
// reduction (ReductionBufNum == 0, the new default).
//
// Codegen for `#pragma omp target teams distribute parallel for reduction`
// generates two kernels:
// Kernel 1: within-team parallel reduce, team master copies to global
//           buffer via __kmpc_reduction_teams_lvl1.
// Kernel 2: secondary kernel (1 team, min(512, num_teams) threads)
//           reduces the global buffer via __kmpc_reduction_teams_lvl2.
// =========================================================================

template <typename T> class SimulationTrunkJD : public SimulationTrunkBase<T> {
  void *d_gbuf = nullptr;

  // =========================================================================
  // GPU cross-team reduction kernels
  // =========================================================================

  template <RedOp Op> T red_sim(const T *__restrict in, uint64_t n) {
    const T rnv = red_identity<T, Op>();
    T result = rnv;
    void *gbuf = d_gbuf;

    // --- Kernel 1: per-team reduction + copy to global buffer ---
#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf) ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      T team_priv = rnv;

      uint32_t team_id = k / XTEAM_NUM_THREADS;
      uint32_t tid = k % XTEAM_NUM_THREADS;
      uint64_t num_chunks = (n + XTEAM_NUM_THREADS - 1) / XTEAM_NUM_THREADS;

      // Distribute loop: each team gets chunks of XTEAM_NUM_THREADS
      // consecutive elements in round-robin order, matching
      // kmp_sched_distribute_static_chunked with chunk_size=512.
      for (uint64_t chunk = team_id; chunk < num_chunks;
           chunk += XTEAM_NUM_TEAMS) {
        __trunk_sim_barrier();
        __trunk_sim_barrier();

        // Parallel for: each thread processes one element in the chunk,
        // matching kmp_sched_static within the parallel region.
        uint64_t i = chunk * XTEAM_NUM_THREADS + tid;
        T priv = (i < n) ? in[i] : rnv;

        void *rl[1] = {&priv};
        int32_t is_master = __kmpc_nvptx_parallel_reduce_nowait_v2(
            nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, Op>,
            trunk_sim::warp_copy<T>);

        if (is_master)
          team_priv = red_combine<T, Op>(team_priv, priv);

        __trunk_sim_barrier();
        __trunk_sim_barrier();
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
          &kle, rl, trunk_sim::shfl_reduce<T, Op>, trunk_sim::lvl2_warp_copy<T>,
          trunk_sim::gl_copy<T>, trunk_sim::gl_reduce<T, Op>);

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
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf) ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      // -- user loop body (partial reduction per thread) --
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
          &kle, rl, trunk_sim::shfl_reduce<T, Op>, trunk_sim::lvl2_warp_copy<T>,
          trunk_sim::gl_copy<T>, trunk_sim::gl_reduce<T, Op>);

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
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf) ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      T team_priv = rnv;

      uint32_t team_id = k / XTEAM_NUM_THREADS;
      uint32_t tid = k % XTEAM_NUM_THREADS;
      uint64_t num_chunks = (n + XTEAM_NUM_THREADS - 1) / XTEAM_NUM_THREADS;

      // Distribute loop: round-robin chunks across teams
      for (uint64_t chunk = team_id; chunk < num_chunks;
           chunk += XTEAM_NUM_TEAMS) {
        // Parallel for: one element per thread within the chunk
        uint64_t i = chunk * XTEAM_NUM_THREADS + tid;
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
    num_threads(XTEAM_NUM_THREADS) is_device_ptr(gbuf) ompx_dyn_cgroup_mem(1)
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
