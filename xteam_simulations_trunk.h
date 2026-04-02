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

#if defined(__AMDGCN__)
#define _TRUNK_WARP_SIZE 64
#elif defined(__NVPTX__)
#define _TRUNK_WARP_SIZE 32
#else
#define _TRUNK_WARP_SIZE 64
#endif

// Matches OMPIRBuilder's default ReductionBufNum (1024)
#define _TRUNK_NUM_RECORDS 1024

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
int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    void *Loc, void *GlobalBuffer, uint32_t num_of_records,
    uint64_t reduce_data_size, void *reduce_data,
    void (*shflFct)(void *, int16_t, int16_t, int16_t),
    void (*cpyFct)(void *, int32_t), void (*lgcpyFct)(void *, int, void *),
    void (*lgredFct)(void *, int, void *),
    void (*glcpyFct)(void *, int, void *),
    void (*glredFct)(void *, int, void *));
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
int32_t __kmpc_nvptx_teams_reduce_nowait_v2(
    void *, void *, uint32_t, uint64_t, void *,
    void (*)(void *, int16_t, int16_t, int16_t), void (*)(void *, int32_t),
    void (*)(void *, int, void *), void (*)(void *, int, void *),
    void (*)(void *, int, void *), void (*)(void *, int, void *)) {
  return 0;
}
int32_t __kmpc_shuffle_int32(int32_t, int16_t, int16_t) { return 0; }
int64_t __kmpc_shuffle_int64(int64_t, int16_t, int16_t) { return 0; }
void __kmpc_barrier_simple_spmd(void *, int32_t) {}
uint32_t __kmpc_get_hardware_thread_id_in_block() { return 0; }
}
#endif

// =========================================================================
// Device helpers and codegen-simulated callbacks
//
// OMPIRBuilder::createReductionsGPU generates six callback functions that
// are passed to __kmpc_nvptx_teams_reduce_nowait_v2.  The functions below
// simulate what the compiler would emit for a single scalar reduction
// variable of type T with operation Op.
//
// Reduce list layout (single variable):  void *rl[1] = { &priv };
// Global buffer layout:                  T buf[_TRUNK_NUM_RECORDS];
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

// --- 2. _omp_reduction_inter_warp_copy_func (InterWarpCopyFnTy) ----------
//
// Transfers each warp-master's value through shared memory so that warp 0
// can perform the final cross-warp reduction.  Processes the value in
// 4-byte chunks (matching codegen behaviour for types > 32 bits).
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

// --- 3. _omp_reduction_list_to_global_copy_func (ListGlobalFnTy) ---------
//   buf[idx] = *priv
template <typename T> static void lg_copy(void *buf, int idx, void *rd) {
  static_cast<T *>(buf)[idx] = **reinterpret_cast<T **>(rd);
}

// --- 4. _omp_reduction_list_to_global_reduce_func (ListGlobalFnTy) -------
//   buf[idx] = combine(buf[idx], *priv)
template <typename T, RedOp Op>
static void lg_reduce(void *buf, int idx, void *rd) {
  T *b = static_cast<T *>(buf);
  b[idx] = red_combine<T, Op>(b[idx], **reinterpret_cast<T **>(rd));
}

// --- 5. _omp_reduction_global_to_list_copy_func (ListGlobalFnTy) ---------
//   *priv = buf[idx]
template <typename T> static void gl_copy(void *buf, int idx, void *rd) {
  **reinterpret_cast<T **>(rd) = static_cast<T *>(buf)[idx];
}

// --- 6. _omp_reduction_global_to_list_reduce_func (ListGlobalFnTy) -------
//   *priv = combine(*priv, buf[idx])
template <typename T, RedOp Op>
static void gl_reduce(void *buf, int idx, void *rd) {
  T *pp = *reinterpret_cast<T **>(rd);
  *pp = red_combine<T, Op>(*pp, static_cast<T *>(buf)[idx]);
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
//
// After the teams reduce returns 1 for the winning thread, codegen
// applies the final combine:  result = combine(result, priv).
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

      // Reduce after every single element (one per thread), matching the
      // codegen's distribute pattern: each 512-element chunk triggers a
      // full within-team parallel_reduce via warp shuffles + shared memory.
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

      // Cross-team reduction on the accumulated team result (once, like
      // codegen)
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

      // -- codegen-emitted reduction sequence --
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

      // Step 3: finalization (winning thread writes result)
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
