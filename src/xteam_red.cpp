// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <memory>

#include "omp.h"

#include "common.h"
#include "xteam_simulations_selected.h"

std::string bench_op_name = "reduction";

// =========================================================================
// Gold (CPU) reference implementations
// =========================================================================

template <typename T, RedOp Op> static T gold_red(const T *in, uint64_t n) {
  T a = red_identity<T, Op>();
  for (uint64_t i = 0; i < n; i++)
    a = red_combine<T, Op>(a, in[i]);
  return a;
}
template <typename T>
static T gold_red_dot(const T *a, const T *b, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  for (uint64_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

// =========================================================================
// GPU cross-team reduction kernels where the AOMP codegen patterns match.
// (Excluding min since it doesn't offer more insight than max.)
// =========================================================================

template <typename T> static T red_sum(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s += in[i];
  return s;
}

template <typename T> static T red_max(const T *__restrict in, uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        max : m)
  for (uint64_t i = 0; i < n; i++)
    m = std::max(m, in[i]);
  return m;
}

template <typename T>
static T red_dot(const T *__restrict a, const T *__restrict b, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

// Combined reduction (sum and max) in a single loop.
template <typename T>
static T red_combined(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        + : s) reduction(max : m)
  for (uint64_t i = 0; i < n; i++) {
    s += in[i];
    m = std::max(m, in[i]);
  }
  return (s / 2) + (m / 2);
}

static double red_pi(uint64_t n) {
  double pi = 0.0;

  // https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : pi)
  for (uint64_t i = 0; i < n; i++) {
    double term = 1.0 / (2 * i + 1);
    pi += (i & 0x1) ? -term : term;
  }

  return pi * 4.0;
}

// =========================================================================
// GPU cross-team reduction kernels where the AOMP codegen doesn't pattern
// match the optimized codegen.
// =========================================================================

// Multiplication isn't detected by AOMP's pattern matching.
template <typename T> static T red_mult(const T *__restrict in, uint64_t n) {
  T m = red_identity<T, RedOp::Mult>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(* : m)
  for (uint64_t i = 0; i < n; i++)
    m *= in[i];
  return m;
}

// Indirect reduction (sum) op isn't detected by AOMP's pattern matching.
template <typename T>
static T red_indirect(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  auto accumulate = [](T a, T b) { return a + b; };
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(+ : s)
  for (uint64_t i = 0; i < n; i++)
    s = accumulate(s, in[i]);
  return s;
}

// Combined reduction (sum and max) in separate loops.
template <typename T>
static T red_combined_separate(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  T m = red_identity<T, RedOp::Max>();
#pragma omp target map(tofrom : s, m)
#pragma omp teams TEAMS reduction(+ : s) reduction(max : m)
  {
#pragma omp distribute parallel for THREADS reduction(+ : s)
    for (uint64_t i = 0; i < n; i++)
      s += in[i];

#pragma omp distribute parallel for THREADS reduction(max : m)
    for (uint64_t i = 0; i < n; i++)
      m = std::max(m, in[i]);
  }
  return (s / 2) + (m / 2);
}

// Have a reduction in a kernel that is also doing something completely
// unrelated to the reduction (pure register work, no memory ops).
template <typename T>
static T red_kernel_part(const T *__restrict in, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();

#pragma omp target map(tofrom : s)
#pragma omp teams TEAMS reduction(+ : s)
  {
#pragma omp distribute parallel for THREADS reduction(+ : s)
    for (uint64_t i = 0; i < n; i++)
      s += in[i];

      // Just do something, without actually doing anything
#pragma omp parallel THREADS
    {
      int tid = omp_get_thread_num();
      T x = static_cast<T>(tid);
      for (int j = 0; j < 100; j++)
        x = x * static_cast<T>(0.9) + static_cast<T>(j);
      if (x == static_cast<T>(-1))
        s += x;
    }
  }

  return s;
}

// =========================================================================
// Benchmark harness
// =========================================================================

template <typename T, typename Sim, typename Kernel, typename... Inputs>
static std::optional<TimingResult>
run_bench_red(Kernel kernel, T gold, uint64_t n, std::string_view label,
              std::unique_ptr<Sim> &sim, Inputs... inputs) {
  std::vector<double> times;
  double total_time = 0.0;

  for (int t = 0; t < conf.warmup_iters + conf.bench_iters; t++) {
    if (sim)
      sim->reset_device();
    if (conf.evict_cache)
      evict_device_cache();
    auto t1 = Clock::now();
    T result = kernel(inputs..., n);
    auto t2 = Clock::now();
    if (!check_single<T>(result, gold, label))
      return std::nullopt;

    if (t < conf.warmup_iters)
      continue;
    double d = duration_cast(t2 - t1).count();
    times.push_back(d);
    if (conf.auto_scale) {
      total_time += d;
      if (total_time >= AUTO_SCALE_TIME && times.size() >= BENCH_MIN_ITERS)
        break;
    }
  }

  return create_timing_result(times, sizeof(T) * n * sizeof...(Inputs));
}

// Run a simple reduction (e.g., sum/max/min/mult/or) and all its simulation
// variants.
template <typename T, RedOp Op, typename Sim, typename Kernel>
static void run_red_simple(Kernel kernel, const T *in, uint64_t n,
                           std::string_view type_name,
                           std::unique_ptr<Sim> &sim) {
  std::optional<TimingResult> r;
  std::unique_ptr<Sim> empty_sim;

  T gold = gold_red<T, Op>(in, n);
  if (conf.run) {
    r = run_bench_red<T>(kernel, gold, n, red_op_to_str<Op>("red_{}"),
                         empty_sim, in);
    print_result(red_op_to_str<Op>("red_{}"), type_name, n, r);
  }
  if (conf.run_sim) {
    for (const auto &[name, func] : sim->template get_all_red_variants<Op>()) {
      r = run_bench_red<T>(func, gold, n, name, sim, in);
      print_result(name, type_name, n, r);
    }
  }
}

// =========================================================================
// Templated per-type benchmark runner
// =========================================================================

template <template <typename> class Sim, typename T>
  requires RedSimulationLike<Sim<T>, T>
static void run_type_red(std::string_view type_name) {
  std::optional<TimingResult> r;

  std::unique_ptr<Sim<T>> sim = std::make_unique<Sim<T>>();
  std::unique_ptr<Sim<T>> empty_sim;

  for (uint64_t n : conf.array_sizes) {
    T *in1 = alloc<T>(n);
    T *in2 = alloc<T>(n);
    init_data<T>(in1, in2, n);

#pragma omp target enter data map(to : in1[0 : n], in2[0 : n])

    sim->reset_device();

    // Cross-team reductions (codegen + simulations)
    T gold;

    // ================================================================
    // dot reduction
    // ================================================================
    gold = gold_red_dot(in1, in2, n);
    if (conf.run) {
      r = run_bench_red<T>(red_dot<T>, gold, n, "red_dot", empty_sim, in1, in2);
      print_result("red_dot", type_name, n, r);
    }
    if (conf.run_sim) {
      for (const auto &[name, func] : sim->get_all_red_dot_variants()) {
        r = run_bench_red<T>(func, gold, n, name, sim, in1, in2);
        print_result(name, type_name, n, r);
      }
    }

    // ================================================================
    // max reduction
    // ================================================================
    run_red_simple<T, RedOp::Max>(red_max<T>, in1, n, type_name, sim);

    // ================================================================
    // sum reduction
    // ================================================================
    run_red_simple<T, RedOp::Sum>(red_sum<T>, in1, n, type_name, sim);

    if (!conf.quick_run || std::is_same_v<T, double>) {
      // ================================================================
      // mult reduction
      // ================================================================
      run_red_simple<T, RedOp::Mult>(red_mult<T>, in1, n, type_name, sim);

      // ================================================================
      // indirect reduction (sum)
      // ================================================================
      gold = gold_red<T, RedOp::Sum>(in1, n);
      if (conf.run) {
        r = run_bench_red<T>(red_indirect<T>, gold, n, "red_indirect",
                             empty_sim, in1);
        print_result("red_indirect", type_name, n, r);
      }

      // ================================================================
      // reduction (sum) in a kernel that is also doing something completely
      // unrelated to the reduction.
      // ================================================================
      gold = gold_red<T, RedOp::Sum>(in1, n);
      if (conf.run) {
        r = run_bench_red<T>(red_kernel_part<T>, gold, n, "red_kernel_part",
                             empty_sim, in1);
        print_result("red_kernel_part", type_name, n, r);
      }

      // ================================================================
      // combined reduction - in the same loop ...
      // ================================================================
      gold = (gold_red<T, RedOp::Sum>(in1, n) / 2) +
             (gold_red<T, RedOp::Max>(in1, n) / 2);
      if (conf.run) {
        r = run_bench_red<T>(red_combined<T>, gold, n, "red_combined",
                             empty_sim, in1);
        print_result("red_combined", type_name, n, r);
      }
      // ================================================================
      // ... and in separate loops
      // ================================================================
      if (conf.run) {
        r = run_bench_red<T>(red_combined_separate<T>, gold, n,
                             "red_combined_separate", empty_sim, in1);
        print_result("red_combined_separate", type_name, n, r);
      }
    }

#pragma omp target exit data map(delete : in1[0 : n], in2[0 : n])

    free(in1);
    free(in2);
  }

  if (conf.run && std::is_same_v<T, double>) {
    // ================================================================
    // reduction computing Pi
    // ================================================================
    double gold_pi = std::numbers::pi;
    uint64_t n = 5000000000;
    r = run_bench_red<T>(red_pi, gold_pi, n, "red_pi", empty_sim);
    print_result("red_pi", type_name, n, r);
  }
}

void run_bench_op() {
  print_array_sizes();

  print_header();

  std::cout << "\n--- double ---\n";
  run_type_red<SelectedSim, double>("double");
  std::cout << "\n--- uint ---\n";
  run_type_red<SelectedSim, unsigned>("uint");
  std::cout << "\n--- ulong ---\n";
  run_type_red<SelectedSim, unsigned long>("ulong");
}
