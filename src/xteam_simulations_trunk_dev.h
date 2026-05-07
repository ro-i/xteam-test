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

// =========================================================================
// Layout-compatible replica of KernelLaunchEnvironmentTy from
// offload/include/Shared/Environment.h.
// __kmpc_gpu_xteam_reduce_nowait reads the cross-team buffer from
// state::getKernelLaunchEnvironment().ReductionBuffer; since the simulation
// kernels do not use a `reduction(...)` clause, the plugin leaves that slot
// null. The simulation therefore writes its own gbuf into the slot through
// this layout-compatible replica before calling the reduction runtime. The
// remaining fields preserve ABI layout so we don't clobber neighbouring
// state (DynCGroupMem*, ReductionTeamsDone, ...).
// =========================================================================

struct SimKernelLaunchEnvTy {
  void *ReductionBuffer = nullptr;
  void *DynCGroupMemFbPtr = nullptr;
  uint32_t ReductionTeamsDone = 0;
  uint32_t DynCGroupMemSize = 0;
  uint8_t DynCGroupMemFb = 0;
};

// =========================================================================
// Runtime API declarations
// Completes the declarations from xteam_simulations_common.h
// =========================================================================

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
int32_t __kmpc_gpu_xteam_reduce_nowait(void *Loc, void *reduce_data,
                                       ShuffleReductFnTy shflFct,
                                       InterWarpCopyFnTy cpyFct,
                                       ListGlobalFnTy lgcpyFct,
                                       ListGlobalFnTy glcpyFct,
                                       ListGlobalFnTy glredFct);
}
#else
extern "C" {
inline int32_t __kmpc_gpu_xteam_reduce_nowait(
    void *, void *, ShuffleReductFnTy shflFct, InterWarpCopyFnTy cpyFct,
    ListGlobalFnTy lgcpyFct, ListGlobalFnTy glcpyFct, ListGlobalFnTy glredFct) {
  return 0;
}
}
#endif

// Re-declare ompx::state::getKernelLaunchEnvironment() returning a reference
// to our layout-compatible replica. C++ Itanium mangling does not encode the
// return type, so this resolves at link time to the real runtime symbol
// (ompx::state::getKernelLaunchEnvironment) in the device image. On the host
// side we provide an inline dummy so the host fallback compilation still
// links; the stub is never reached in practice because the kernels only run
// on device.
#pragma omp begin declare target
namespace ompx::state {
#if defined(__AMDGCN__) || defined(__NVPTX__)
SimKernelLaunchEnvTy &getKernelLaunchEnvironment();
#else
inline SimKernelLaunchEnvTy &getKernelLaunchEnvironment() {
  static SimKernelLaunchEnvTy Dummy;
  return Dummy;
}
#endif
} // namespace ompx::state

// Helper that writes `buf` into the runtime KLE's ReductionBuffer slot.
// Wrapping the store in a plain function keeps OpenMP's implicit capture
// machinery from trying to map the `KLE.ReductionBuffer` lvalue out of the
// target region (which fails with "expected addressable lvalue in 'map'
// clause"). From the callsite's perspective this is just a function call
// taking a device pointer argument.
static inline void __trunk_sim_set_kle_reduction_buffer(void *buf) {
  ompx::state::getKernelLaunchEnvironment().ReductionBuffer = buf;
}
#pragma omp end declare target

// =========================================================================
// SimulationTrunkDev — simulates the trunk LLVM cross-team reduction.
//
// Codegen for `#pragma omp target teams distribute parallel for reduction`
// generates two runtime calls:
//   1.  __kmpc_nvptx_parallel_reduce_nowait_v2  — within-team reduction
//       (warp shuffle + inter-warp copy so that thread 0 holds the team
//       result).
//   2.  __kmpc_gpu_xteam_reduce_nowait      — cross-team reduction
//       (each team's thread 0 writes to a global buffer; the last team
//       arriving combines all entries via warp/cross-warp reduction).
// =========================================================================

template <typename T> class SimulationTrunkDev {
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

      // Cross-team reduction on the accumulated team result.
      // Point the runtime's KLE at gbuf so __kmpc_gpu_xteam_reduce_nowait
      // can pick it up from state::getKernelLaunchEnvironment().
      __trunk_sim_set_kle_reduction_buffer(gbuf);
      void *rl[1] = {&team_priv};
      int32_t winner = __kmpc_gpu_xteam_reduce_nowait(
          nullptr, rl, trunk_sim::shfl_reduce<T, Op>, trunk_sim::warp_copy<T>,
          trunk_sim::lg_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, Op>);

      if (winner == 1)
        result = red_combine<T, Op>(result, team_priv);
    }

    return result;
  }

  template <RedOp Op> T red_sim_v2(const T *__restrict in, uint64_t n) {
    const T rnv = red_identity<T, Op>();
    T result = rnv;
    void *gbuf = d_gbuf;

    // We don't really need the outer for, we could also do a parallel region
    // and then adapt the per-thread collection to sth like this:
    //  for (uint64_t i = omp_get_team_num() * XTEAM_NUM_THREADS +
    //                    omp_get_thread_num();
    //       i < n; i += XTEAM_TOTAL_NUM_THREADS)
    //    priv = red_combine<T, Op>(priv, in[i]);
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
      int32_t is_master = __kmpc_nvptx_parallel_reduce_nowait_v2(
          nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, Op>,
          trunk_sim::warp_copy<T>);

      // Step 2: cross-team (teams) reduction. Mask non-master lanes to
      // identity so they don't contaminate the last team's block reduce.
      // Seed the runtime KLE's ReductionBuffer with gbuf.
      T team_priv = is_master ? priv : rnv;
      void *rl_teams[1] = {&team_priv};
      __trunk_sim_set_kle_reduction_buffer(gbuf);
      int32_t winner = __kmpc_gpu_xteam_reduce_nowait(
          nullptr, rl_teams, trunk_sim::shfl_reduce<T, Op>,
          trunk_sim::warp_copy<T>, trunk_sim::lg_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, Op>);

      if (winner == 1)
        result = red_combine<T, Op>(result, team_priv);
    }

    return result;
  }

  // This version tries to stick to the static schedule as closely as possible
  // and instead opts for a remapping of the iteration variable. Example:
  // n = 1000
  //
  // num_teams = 5
  // num_threads = 32
  //
  // chunk_size_per_team = 1000 / 5 = 200
  // chunk_size_per_thread = 200 / 32 = 7
  //
  // chunk_size_per_team_rounded = 7 * 32 = 224
  // n_rounded = 224 * 5 = 1120
  //
  // Without remapping, the access pattern would be:
  // For the threads in team 1: threads 32..63 (global TIDs) access elements
  // 224..447 thread 32: elements 224..230 thread 33: elements 231..237
  // ...
  // thread 62: elements 434..440
  // thread 63: elements 441..447
  //
  // After remapping, the access pattern is:
  // thread 32: elements 224, 256, 288, 320, 352, 384, 416
  // thread 33; elements 225, 257, 289, 321, 353, 385, 417
  // ...
  // thread 62: elements 254, 286, 318, 350, 382, 414, 446
  // thread 63: elements 255, 287, 319, 351, 383, 415, 447
  //
  template <RedOp Op> T red_sim_v3(const T *__restrict in, uint64_t n) {
    const T rnv = red_identity<T, Op>();
    T result = rnv;
    void *gbuf = d_gbuf;
    T priv = rnv;

    uint64_t chunk_size_per_team = (n + XTEAM_NUM_TEAMS - 1) / XTEAM_NUM_TEAMS;
    uint64_t chunk_size_per_thread =
        (chunk_size_per_team + XTEAM_NUM_THREADS - 1) / XTEAM_NUM_THREADS;
    // Now, we want to round n up such that each chunk has chunk_size_per_thread
    // threads.
    uint64_t chunk_size_per_team_rounded =
        chunk_size_per_thread * XTEAM_NUM_THREADS;
    uint64_t n_rounded = chunk_size_per_team_rounded * XTEAM_NUM_TEAMS;

#pragma omp target teams distribute parallel for num_teams(XTEAM_NUM_TEAMS)    \
    num_threads(XTEAM_NUM_THREADS) map(tofrom : result) is_device_ptr(gbuf)    \
    firstprivate(priv) dist_schedule(static, chunk_size_per_team_rounded)      \
    schedule(static, chunk_size_per_thread) ompx_dyn_cgroup_mem(1)
    for (uint64_t k = 0; k < n_rounded; k++) {
      // At this point, we remap the index such that each thread does *not*
      // process a contiguous chunk of elements but such that the threads in a
      // team process the input elements in an interleaved way leading to nice
      // coalescing (hopefully).
      uint64_t team_base = omp_get_team_num() * chunk_size_per_team_rounded;
      uint64_t s = k % chunk_size_per_thread;
      uint64_t k_remapped =
          team_base + s * XTEAM_NUM_THREADS + omp_get_thread_num();
      if (k_remapped < n) {
        // -- user loop body (partial reduction per thread) --
        priv = red_combine<T, Op>(priv, in[k_remapped]);
      }

      // Check if thread has finished its corresponding chunk.
      if (((k + 1) % chunk_size_per_team_rounded) % chunk_size_per_thread ==
          0) {
        void *rl[1] = {&priv};

        // Step 1: within-team (parallel) reduction
        int32_t is_master = __kmpc_nvptx_parallel_reduce_nowait_v2(
            nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, Op>,
            trunk_sim::warp_copy<T>);

        // Step 2: cross-team (teams) reduction. Mask non-master lanes to
        // identity to avoid contamination (see red_sim_v2).
        // Seed the runtime KLE's ReductionBuffer with gbuf.
        T team_priv = is_master ? priv : rnv;
        void *rl_teams[1] = {&team_priv};
        __trunk_sim_set_kle_reduction_buffer(gbuf);
        int32_t winner = __kmpc_gpu_xteam_reduce_nowait(
            nullptr, rl_teams, trunk_sim::shfl_reduce<T, Op>,
            trunk_sim::warp_copy<T>, trunk_sim::lg_copy<T>,
            trunk_sim::gl_copy<T>, trunk_sim::gl_reduce<T, Op>);

        if (winner == 1)
          result = red_combine<T, Op>(result, team_priv);
      }
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
        __trunk_sim_barrier();
        __trunk_sim_barrier();

        // Parallel for: one element per thread within the chunk
        uint64_t i = chunk * XTEAM_NUM_THREADS + tid;
        T priv = (i < n) ? a[i] * b[i] : rnv;

        void *rl[1] = {&priv};
        int32_t is_master = __kmpc_nvptx_parallel_reduce_nowait_v2(
            nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
            trunk_sim::warp_copy<T>);

        if (is_master)
          team_priv += priv;

        __trunk_sim_barrier();
        __trunk_sim_barrier();
      }

      // Seed the runtime KLE's ReductionBuffer with gbuf.
      __trunk_sim_set_kle_reduction_buffer(gbuf);
      void *rl[1] = {&team_priv};
      int32_t winner = __kmpc_gpu_xteam_reduce_nowait(
          nullptr, rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
          trunk_sim::warp_copy<T>, trunk_sim::lg_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, RedOp::Sum>);

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
      int32_t is_master = __kmpc_nvptx_parallel_reduce_nowait_v2(
          nullptr, sizeof(T), rl, trunk_sim::shfl_reduce<T, RedOp::Sum>,
          trunk_sim::warp_copy<T>);

      // Mask non-master lanes to identity (see red_sim_v2).
      // Seed the runtime KLE's ReductionBuffer with gbuf.
      T team_priv = is_master ? priv : rnv;
      void *rl_teams[1] = {&team_priv};
      __trunk_sim_set_kle_reduction_buffer(gbuf);
      int32_t winner = __kmpc_gpu_xteam_reduce_nowait(
          nullptr, rl_teams, trunk_sim::shfl_reduce<T, RedOp::Sum>,
          trunk_sim::warp_copy<T>, trunk_sim::lg_copy<T>, trunk_sim::gl_copy<T>,
          trunk_sim::gl_reduce<T, RedOp::Sum>);

      if (winner == 1)
        result += team_priv;
    }

    return result;
  }

public:
  SimulationTrunkDev() {
    assert(d_gbuf == nullptr);
    int devid = omp_get_default_device();
    d_gbuf = target_alloc<T>(XTEAM_NUM_TEAMS, devid);
  }

  ~SimulationTrunkDev() {
    assert(d_gbuf != nullptr);
    omp_target_free(d_gbuf, omp_get_default_device());
    d_gbuf = nullptr;
  }

  SimulationTrunkDev(const SimulationTrunkDev &) = delete;
  SimulationTrunkDev(SimulationTrunkDev &&) = delete;
  SimulationTrunkDev &operator=(const SimulationTrunkDev &) = delete;
  SimulationTrunkDev &operator=(SimulationTrunkDev &&) = delete;

  void reset_device() {}

  template <RedOp Op>
  std::vector<
      std::pair<std::string, std::function<T(const T *__restrict, uint64_t)>>>
  get_all_red_variants() {
    return {
        // {red_op_to_str<Op>("red_{}_sim"),
        //  [this](const T *__restrict in, uint64_t n) {
        //    return this->template red_sim<Op>(in, n);
        //  }},
        // {red_op_to_str<Op>("red_{}_sim_v2"),
        {red_op_to_str<Op>("red_{}_sim"),
         [this](const T *__restrict in, uint64_t n) {
           return this->template red_sim_v2<Op>(in, n);
         }},
        // {red_op_to_str<Op>("red_{}_sim_v3"),
        //  [this](const T *__restrict in, uint64_t n) {
        //    return this->template red_sim_v3<Op>(in, n);
        //  }},
    };
  }

  std::vector<std::pair<
      std::string,
      std::function<T(const T *__restrict, const T *__restrict, uint64_t)>>>
  get_all_red_dot_variants() {
    return {
        // {"red_dot_sim",
        //  [this](const T *__restrict a, const T *__restrict b, uint64_t n) {
        //    return this->red_dot_sim(a, b, n);
        //  }},
        // {"red_dot_sim_v2",
        {"red_dot_sim",
         [this](const T *__restrict a, const T *__restrict b, uint64_t n) {
           return this->red_dot_sim_v2(a, b, n);
         }},
    };
  }

}; // class SimulationTrunkDev
