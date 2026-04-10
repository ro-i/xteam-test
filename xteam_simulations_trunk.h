// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "omp.h"

#include "common.h"
#include "xteam_simulations_common.h"

// Matches OMPIRBuilder's default ReductionBufNum (1024)
#define _TRUNK_NUM_RECORDS 1024

// =========================================================================
// Runtime API declarations
// Completes the declarations from xteam_simulations_common.h
// =========================================================================

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    void *Loc, void *GlobalBuffer, uint32_t num_of_records,
    uint64_t reduce_data_size, void *reduce_data, ShuffleReductFnTy shflFct,
    InterWarpCopyFnTy cpyFct, ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct,
    ListGlobalFnTy glcpyFct, ListGlobalFnTy glredFct);
}
#else
extern "C" {
int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    void *, void *, uint32_t, uint64_t, void *, ShuffleReductFnTy shflFct,
    InterWarpCopyFnTy cpyFct, ListGlobalFnTy lgcpyFct, ListGlobalFnTy lgredFct,
    ListGlobalFnTy glcpyFct, ListGlobalFnTy glredFct) {
  return 0;
}
}
#endif

// =========================================================================
// Device helpers and codegen-simulated callbacks
// Completes the definitions from xteam_simulations_common.h
// =========================================================================

#pragma omp begin declare target

namespace trunk_sim {

// --- 4. _omp_reduction_list_to_global_reduce_func (ListGlobalFnTy) -------
//   buf[idx] = combine(buf[idx], *priv)
template <typename T, RedOp Op>
static void lg_reduce(void *buf, int idx, void *rd) {
  T *b = static_cast<T *>(buf);
  b[idx] = red_combine<T, Op>(b[idx], **reinterpret_cast<T **>(rd));
}

} // namespace trunk_sim
#pragma omp end declare target

// =========================================================================
// SimulationTrunk — simulates the trunk LLVM cross-team reduction.
//
// Codegen for `#pragma omp target teams distribute parallel for reduction`
// generates two runtime calls:
//   1.  __kmpc_nvptx_parallel_reduce_nowait_v2  — within-team reduction
//       (warp shuffle + inter-warp copy so that thread 0 holds the team
//       result).
//   2.  __kmpc_nvptx_teams_reduce_nowait_v2     — cross-team reduction
//       (each team's thread 0 writes to a global buffer; the last team
//       arriving combines all entries via warp/cross-warp reduction).
// =========================================================================

template <typename T> class SimulationTrunk : public SimulationTrunkBase<T> {
  void *d_gbuf = nullptr;

  // =========================================================================
  // GPU cross-team reduction kernels
  // =========================================================================

  template <RedOp Op> T red_sim(const T *__restrict in, uint64_t n) {
    const T rnv = red_identity<T, Op>();
    T result = rnv;
    void *gbuf = d_gbuf;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom : result) is_device_ptr(gbuf)    \
    ompx_dyn_cgroup_mem(1)
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
      }

      // Cross-team reduction on the accumulated team result
      void *rl[1] = {&team_priv};
      int32_t winner = __kmpc_nvptx_teams_reduce_nowait_v2(
          nullptr, gbuf, _TRUNK_NUM_RECORDS, sizeof(T), rl,
          trunk_sim::shfl_reduce<T, Op>, trunk_sim::warp_copy<T>,
          trunk_sim::lg_copy<T>, trunk_sim::lg_reduce<T, Op>,
          trunk_sim::gl_copy<T>, trunk_sim::gl_reduce<T, Op>);

      if (winner == 1)
        result = red_combine<T, Op>(result, team_priv);
    }

    return result;
  }

  template <RedOp Op> T red_sim_v2(const T *__restrict in, uint64_t n) {
    const T rnv = red_identity<T, Op>();
    T result = rnv;
    void *gbuf = d_gbuf;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom : result) is_device_ptr(gbuf)    \
    ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      // -- user loop body (partial reduction per thread) --
      T priv = rnv;
      for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
        priv = red_combine<T, Op>(priv, in[i]);

      void *rl[1] = {&priv};

      // Step 1: within-team (parallel) reduction
      __kmpc_nvptx_parallel_reduce_nowait_v2(nullptr, sizeof(T), rl,
                                             trunk_sim::shfl_reduce<T, Op>,
                                             trunk_sim::warp_copy<T>);

      // Step 2: cross-team (teams) reduction
      int32_t winner = __kmpc_nvptx_teams_reduce_nowait_v2(
          nullptr, gbuf, _TRUNK_NUM_RECORDS, sizeof(T), rl,
          trunk_sim::shfl_reduce<T, Op>, trunk_sim::warp_copy<T>,
          trunk_sim::lg_copy<T>, trunk_sim::lg_reduce<T, Op>,
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

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom : result) is_device_ptr(gbuf)    \
    ompx_dyn_cgroup_mem(1)
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
      int32_t winner = __kmpc_nvptx_teams_reduce_nowait_v2(
          nullptr, gbuf, _TRUNK_NUM_RECORDS, sizeof(T), rl,
          trunk_sim::shfl_reduce<T, RedOp::Sum>, trunk_sim::warp_copy<T>,
          trunk_sim::lg_copy<T>, trunk_sim::lg_reduce<T, RedOp::Sum>,
          trunk_sim::gl_copy<T>, trunk_sim::gl_reduce<T, RedOp::Sum>);

      if (winner == 1)
        result += team_priv;
    }

    return result;
  }

  T red_dot_sim_v2(const T *__restrict a, const T *__restrict b, uint64_t n) {
    const T rnv = red_identity<T, RedOp::Sum>();
    T result = rnv;
    void *gbuf = d_gbuf;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom : result) is_device_ptr(gbuf)    \
    ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < XTEAM_TOTAL_NUM_THREADS; k++) {
      T priv = rnv;
      for (uint64_t i = k; i < n; i += XTEAM_TOTAL_NUM_THREADS)
        priv += a[i] * b[i];

      void *rl[1] = {&priv};
      __kmpc_nvptx_parallel_reduce_nowait_v2(
          nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
          trunk_sim::warp_copy<T>);

      int32_t winner = __kmpc_nvptx_teams_reduce_nowait_v2(
          nullptr, gbuf, _TRUNK_NUM_RECORDS, sizeof(T), rl,
          trunk_sim::shfl_reduce<T, RedOp::Sum>, trunk_sim::warp_copy<T>,
          trunk_sim::lg_copy<T>, trunk_sim::lg_reduce<T, RedOp::Sum>,
          trunk_sim::gl_copy<T>, trunk_sim::gl_reduce<T, RedOp::Sum>);

      if (winner == 1)
        result += priv;
    }

    return result;
  }

public:
  void init_device() {
    assert(d_gbuf == nullptr);
    int devid = omp_get_default_device();
    d_gbuf = target_alloc<T>(_TRUNK_NUM_RECORDS, devid);
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
