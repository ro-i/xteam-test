// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include "omp.h"

#include "common.h"
#include "xteam_simulations_selected.h"

std::string bench_op_name = "scan";

// =========================================================================
// Helper functions
// =========================================================================

template <RedOp Op, ScanMode Mode>
static constexpr std::string scan_op_to_str(std::string_view fmt) {
  std::string ret = red_op_to_str<Op>(fmt);
  if constexpr (Mode == ScanMode::Excl)
    return ret + "_excl";
  else if constexpr (Mode == ScanMode::Incl)
    return ret + "_incl";
  else
    static_assert(!std::is_same_v<ScanMode, ScanMode>, "Unsupported scan mode");
}

// =========================================================================
// Gold (CPU) reference implementations
// =========================================================================

template <typename T, RedOp Op, ScanMode Mode>
static void gold_scan(const T *in, T *out, uint64_t n) {
  T a = red_identity<T, Op>();
  for (uint64_t i = 0; i < n; i++) {
    if constexpr (Mode == ScanMode::Excl) {
      out[i] = a;
      a = red_combine<T, Op>(a, in[i]);
    } else {
      a = red_combine<T, Op>(a, in[i]);
      out[i] = a;
    }
  }
}
template <typename T, ScanMode Mode>
static void gold_scan_dot(const T *a, const T *b, T *out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
  for (uint64_t i = 0; i < n; i++) {
    if constexpr (Mode == ScanMode::Excl) {
      out[i] = s;
      s += a[i] * b[i];
    } else {
      s += a[i] * b[i];
      out[i] = s;
    }
  }
}

// =========================================================================
// GPU cross-team scan kernels (compilation unsupported by vanilla AOMP)
// =========================================================================

#ifdef AOMP_DEV
template <typename T>
static void scan_sum_incl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    s += in[i];
#pragma omp scan inclusive(s)
    out[i] = s;
  }
}

template <typename T>
static void scan_sum_excl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
#pragma omp scan exclusive(s)
    s += in[i];
  }
}

template <typename T>
static void scan_max_incl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, max : m)
  for (uint64_t i = 0; i < n; i++) {
    m = std::max(m, in[i]);
#pragma omp scan inclusive(m)
    out[i] = m;
  }
}

template <typename T>
static void scan_max_excl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {
  T m = red_identity<T, RedOp::Max>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, max : m)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = m;
#pragma omp scan exclusive(m)
    m = std::max(m, in[i]);
  }
}

template <typename T>
static void scan_dot_incl(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    s += a[i] * b[i];
#pragma omp scan inclusive(s)
    out[i] = s;
  }
}

template <typename T>
static void scan_dot_excl(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {
  T s = red_identity<T, RedOp::Sum>();
#pragma omp target teams distribute parallel for TEAMS_THREADS reduction(      \
        inscan, + : s)
  for (uint64_t i = 0; i < n; i++) {
    out[i] = s;
#pragma omp scan exclusive(s)
    s += a[i] * b[i];
  }
}
#else  // defined(AOMP_DEV)
// Vanilla AOMP can't lower `#pragma omp scan`, so the real-codegen kernels are
// stubs. The binary still links for the simulation-only path (-S); the -s
// branch is suppressed at runtime in run_scan() below.
template <typename T>
static void scan_sum_incl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {}
template <typename T>
static void scan_sum_excl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {}
template <typename T>
static void scan_max_incl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {}
template <typename T>
static void scan_max_excl(const T *__restrict in, T *__restrict out,
                          uint64_t n) {}
template <typename T>
static void scan_dot_incl(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {}
template <typename T>
static void scan_dot_excl(const T *__restrict a, const T *__restrict b,
                          T *__restrict out, uint64_t n) {}
#endif // defined(AOMP_DEV)

// =========================================================================
// Benchmark harness
// =========================================================================

template <typename T, typename Sim, typename Kernel, typename... Inputs>
static std::optional<TimingResult>
run_bench_scan(Kernel kernel, T *out, const T *gold, uint64_t n,
               std::string_view label, std::unique_ptr<Sim> &sim,
               Inputs... inputs) {
  std::vector<double> times;
  double total_time = 0.0;

  for (int t = 0; t < conf.warmup_iters + conf.bench_iters; t++) {
    if (sim)
      sim->reset_device();
    auto t1 = Clock::now();
    kernel(inputs..., out, n);
    auto t2 = Clock::now();
#pragma omp target update from(out[0 : n])
    if (!check<T>(out, gold, n, label))
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

// Run a simple scan (e.g., sum/max/min/mult) and all its simulation variants.
template <typename T, RedOp Op, ScanMode Mode, typename Sim, typename Kernel>
static void run_scan_simple(Kernel kernel, T *gold, const T *in, T *out,
                            uint64_t n, std::string_view type_name,
                            std::unique_ptr<Sim> &sim) {
  std::optional<TimingResult> r;
  std::unique_ptr<Sim> empty_sim;

  gold_scan<T, Op, Mode>(in, gold, n);
  if (conf.run) {
    r = run_bench_scan<T>(kernel, out, gold, n,
                          scan_op_to_str<Op, Mode>("scan_{}"), empty_sim, in);
    print_result(scan_op_to_str<Op, Mode>("scan_{}"), type_name, n, r);
  }
  if (conf.run_sim) {
    for (const auto &[name, func] : get_all_scan_variants<Op, Mode>(*sim)) {
      r = run_bench_scan<T>(func, out, gold, n, name, sim, in);
      print_result(name, type_name, n, r);
    }
  }
}

// =========================================================================
// Templated per-type benchmark runner
// =========================================================================

template <template <typename> class Sim, typename T>
  requires ScanSimulationLike<Sim<T>, T>
static void run_type_scan(std::string_view type_name) {
  std::optional<TimingResult> r;

  std::unique_ptr<Sim<T>> sim = std::make_unique<Sim<T>>();
  std::unique_ptr<Sim<T>> empty_sim;

  for (uint64_t n : conf.array_sizes) {
    T *in1 = alloc<T>(n);
    T *in2 = alloc<T>(n);
    T *out = alloc<T>(n);
    init_data<T>(in1, in2, n);

#pragma omp target enter data map(to : in1[0 : n], in2[0 : n], out[0 : n])

    sim->reset_device();

    // Cross-team scans (codegen + simulations)
    T *gold = alloc<T>(n);

    sim->reset_device();

    // ================================================================
    // exclusive dot scan
    // ================================================================
    gold_scan_dot<T, ScanMode::Excl>(in1, in2, gold, n);

    if (conf.run) {
      r = run_bench_scan<T>(scan_dot_excl<T>, out, gold, n, "scan_dot_excl",
                            empty_sim, in1, in2);
      print_result("scan_dot_excl", type_name, n, r);
    }

    if (conf.run_sim) {
      for (const auto &[name, func] : sim->get_all_scan_dot_excl_variants()) {
        r = run_bench_scan<T>(func, out, gold, n, name, sim, in1, in2);
        print_result(name, type_name, n, r);
      }
    }

    // ================================================================
    // inclusive dot scan
    // ================================================================
    gold_scan_dot<T, ScanMode::Incl>(in1, in2, gold, n);

    if (conf.run) {
      r = run_bench_scan<T>(scan_dot_incl<T>, out, gold, n, "scan_dot_incl",
                            empty_sim, in1, in2);
      print_result("scan_dot_incl", type_name, n, r);
    }

    if (conf.run_sim) {
      for (const auto &[name, func] : sim->get_all_scan_dot_incl_variants()) {
        r = run_bench_scan<T>(func, out, gold, n, name, sim, in1, in2);
        print_result(name, type_name, n, r);
      }
    }

    // ================================================================
    // exclusive max scan
    // ================================================================
    run_scan_simple<T, RedOp::Max, ScanMode::Excl>(scan_max_excl<T>, gold, in1,
                                                   out, n, type_name, sim);

    // ================================================================
    // inclusive max scan
    // ================================================================
    run_scan_simple<T, RedOp::Max, ScanMode::Incl>(scan_max_incl<T>, gold, in1,
                                                   out, n, type_name, sim);

    // ================================================================
    // exclusive sum scan
    // ================================================================
    run_scan_simple<T, RedOp::Sum, ScanMode::Excl>(scan_sum_excl<T>, gold, in1,
                                                   out, n, type_name, sim);

    // ================================================================
    // inclusive sum scan
    // ================================================================
    run_scan_simple<T, RedOp::Sum, ScanMode::Incl>(scan_sum_incl<T>, gold, in1,
                                                   out, n, type_name, sim);

    free(gold);

#pragma omp target exit data map(delete : in1[0 : n], in2[0 : n], out[0 : n])

    free(in1);
    free(in2);
    free(out);
  }
}

void run_bench_op() {
#ifndef AOMP_DEV
  if (conf.run) {
    std::cerr << "warning: scan codegen not supported in this build, "
                 "ignoring '-r' (use '-s' for simulations)\n";
    conf.run = false;
  }
  if (!conf.run_sim)
    return;
#endif
  print_array_sizes();

  print_header();

  std::cout << "\n--- double ---\n";
  run_type_scan<SelectedSim, double>("double");
  std::cout << "\n--- uint ---\n";
  run_type_scan<SelectedSim, unsigned>("uint");
  std::cout << "\n--- ulong ---\n";
  run_type_scan<SelectedSim, unsigned long>("ulong");
}
