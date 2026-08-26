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
};
