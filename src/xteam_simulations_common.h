// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "common.h"

// =========================================================================
// Simulation concepts
//
//   SimulationLike      — common definitions
//   RedSimulationLike   — reduction variant getters
//                         variant getters.
//   ScanSimulationLike  — scan variant getters
// =========================================================================
template <typename S, typename T>
concept SimulationLike = requires(S s) {
  { s.reset_device() } -> std::same_as<void>;
};

template <typename S, typename T>
concept RedSimulationLike = SimulationLike<S, T> && requires(S s) {
  s.template get_all_red_variants<RedOp::Sum>();
  s.get_all_red_dot_variants();
};

template <typename S, typename T>
concept ScanSimulationLike = SimulationLike<S, T> && requires(S s) {
  s.template get_all_scan_incl_variants<RedOp::Sum>();
  s.template get_all_scan_excl_variants<RedOp::Sum>();
  s.get_all_scan_dot_incl_variants();
  s.get_all_scan_dot_excl_variants();
};

template <RedOp Op, ScanMode Mode, typename Sim>
auto get_all_scan_variants(Sim &sim) {
  if constexpr (Mode == ScanMode::Incl)
    return sim.template get_all_scan_incl_variants<Op>();
  else if constexpr (Mode == ScanMode::Excl)
    return sim.template get_all_scan_excl_variants<Op>();
  else
    static_assert(!std::is_same_v<Sim, Sim>, "Unsupported scan mode");
}

// No-op simulation used when no specific backend is selected.
template <typename T> class SimulationNoop {
public:
  void reset_device() {}

  template <RedOp>
  std::vector<
      std::pair<std::string, std::function<T(const T *__restrict, uint64_t)>>>
  get_all_red_variants() {
    return {};
  }

  std::vector<std::pair<
      std::string,
      std::function<T(const T *__restrict, const T *__restrict, uint64_t)>>>
  get_all_red_dot_variants() {
    return {};
  }

  template <RedOp>
  std::vector<std::pair<
      std::string,
      std::function<void(const T *__restrict, T *__restrict, uint64_t)>>>
  get_all_scan_incl_variants() {
    return {};
  }

  template <RedOp>
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
};
