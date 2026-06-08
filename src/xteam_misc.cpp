// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cmath>

#include "omp.h"

#include "common.h"

std::string bench_op_name = "misc";

// -----------------------
template <typename T>
static void gold_elem_loop(T *out, const T *a, const T *b, uint64_t n) {
#pragma omp parallel for
  for (uint64_t i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}

template <typename T>
static void elem_loop(T *out, const T *a, const T *b, uint64_t n) {
#pragma omp target teams distribute parallel for TEAMS_THREADS
  for (uint64_t i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}

template <bool ToDevice, typename T>
static void prepare_elem_loop(T *, uint64_t) {}

template <typename T>
static bool check_elem_loop(std::string_view label, T *out, const T *gold,
                            uint64_t n) {
#pragma omp target update from(out[0 : n])
  return check<T>(out, gold, n, label);
}

template <typename T> static uint64_t data_bytes_elem_loop(uint64_t n) {
  return sizeof(T) * n * 3;
}

// -----------------------
template <typename T>
static void gold_stencil(T *out, const T *a, uint64_t ny, uint64_t nx) {
#pragma omp parallel for collapse(2)
  for (uint64_t y = 1; y < ny - 1; ++y) {
    for (uint64_t x = 1; x < nx - 1; ++x) {
      out[y * nx + x] = (a[(y - 1) * nx + x] + a[(y + 1) * nx + x] +
                         a[y * nx + (x - 1)] + a[y * nx + (x + 1)]) *
                        0.25;
    }
  }
}

template <typename T>
static void stencil(T *out, const T *a, uint64_t ny, uint64_t nx) {
#pragma omp target teams distribute parallel for TEAMS_THREADS collapse(2)
  for (uint64_t y = 1; y < ny - 1; ++y) {
    for (uint64_t x = 1; x < nx - 1; ++x) {
      out[y * nx + x] = (a[(y - 1) * nx + x] + a[(y + 1) * nx + x] +
                         a[y * nx + (x - 1)] + a[y * nx + (x + 1)]) *
                        0.25;
    }
  }
}

template <bool ToDevice, typename T>
static void prepare_stencil(T *out, const T *a, uint64_t n) {
  std::copy(a, a + n, out);
  if constexpr (ToDevice) {
#pragma omp target update to(out[0 : n])
  }
}

template <typename T>
static bool check_stencil(std::string_view label, T *out, const T *gold,
                          uint64_t ny, uint64_t nx) {
#pragma omp target update from(out[0 : ny * nx])
  for (uint64_t y = 0; y < ny; y++) {
    for (uint64_t x = 0; x < nx; x++) {
      if (!check_single(out[y * nx + x], gold[y * nx + x], label))
        return false;
    }
  }
  return true;
}

template <typename T> static uint64_t data_bytes_stencil(uint64_t n) {
  return sizeof(T) * n * 2;
}

// -----------------------
template <typename T>
static void gold_linalg(T *out, const T *a, const T *b, T c, uint64_t n) {
#pragma omp parallel for
  for (uint64_t i = 0; i < n; i++)
    out[i] = c * a[i] + b[i];
}

template <typename T>
static void linalg(T *out, const T *a, const T *b, T c, uint64_t n) {
#pragma omp target teams distribute parallel for TEAMS_THREADS
  for (uint64_t i = 0; i < n; i++)
    out[i] = c * a[i] + b[i];
}

template <bool ToDevice, typename T>
static void prepare_linalg(T *, uint64_t) {}

template <typename T>
static bool check_linalg(std::string_view label, T *out, const T *gold,
                         uint64_t n) {
#pragma omp target update from(out[0 : n])
  return check<T>(out, gold, n, label);
}

template <typename T> static uint64_t data_bytes_linalg(uint64_t n) {
  return sizeof(T) * n * 3;
}

// -----------------------
template <typename T>
static void gold_particle(T *x, T *y, T *z, const T *vx, const T *vy,
                          const T *vz, T dt, uint64_t n) {
#pragma omp parallel for
  for (uint64_t i = 0; i < n; ++i) {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
}

template <typename T>
static void particle(T *x, T *y, T *z, const T *vx, const T *vy, const T *vz,
                     T dt, uint64_t n) {
#pragma omp target teams distribute parallel for TEAMS_THREADS
  for (uint64_t i = 0; i < n; ++i) {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
}

template <bool ToDevice, typename T>
static void prepare_particle(T *x, T *y, T *z, const T *x0, const T *y0,
                             const T *z0, uint64_t n) {
  std::copy(x0, x0 + n, x);
  std::copy(y0, y0 + n, y);
  std::copy(z0, z0 + n, z);
  if constexpr (ToDevice) {
#pragma omp target update to(x[0 : n], y[0 : n], z[0 : n])
  }
}

template <typename T>
static bool check_particle(std::string_view label, T *x, T *y, T *z,
                           const T *gold_x, const T *gold_y, const T *gold_z,
                           uint64_t n) {
#pragma omp target update from(x[0 : n], y[0 : n], z[0 : n])
  return check<T>(x, gold_x, n, label) && check<T>(y, gold_y, n, label) &&
         check<T>(z, gold_z, n, label);
}

template <typename T> static uint64_t data_bytes_particle(uint64_t n) {
  return sizeof(T) * n * 9;
}

// -----------------------
template <typename T, typename F>
static void gold_elem_func(T *out, const F &func, uint64_t n) {
#pragma omp parallel for
  for (uint64_t i = 0; i < n; i++)
    out[i] = func(i);
}

template <typename T, typename F>
static void elem_func(T *out, const F &func, uint64_t n) {
#pragma omp target teams distribute parallel for TEAMS_THREADS
  for (uint64_t i = 0; i < n; i++)
    out[i] = func(i);
}

template <bool ToDevice, typename T>
static void prepare_elem_func(T *, uint64_t) {}

template <typename T>
static bool check_elem_func(std::string_view label, T *out, const T *gold,
                            uint64_t n) {
#pragma omp target update from(out[0 : n])
  return check<T>(out, gold, n, label);
}

template <typename T> static uint64_t data_bytes_elem_func(uint64_t n) {
  return sizeof(T) * n;
}

// =========================================================================
// Benchmark harness
// =========================================================================

template <typename Gold, typename Kernel, typename Check, typename Prepare>
static std::optional<TimingResult>
run_bench(std::string_view label, uint64_t data_bytes, const Gold &gold,
          const Kernel &kernel, const Check &check, const Prepare &prepare) {
  std::vector<double> times;
  double total_time = 0.0;

  gold();

  for (int t = 0; t < conf.warmup_iters + conf.bench_iters; t++) {
    prepare();

    if (conf.evict_cache)
      evict_device_cache();

    auto t1 = Clock::now();
    kernel();
    auto t2 = Clock::now();

    if (!check(label))
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

  return create_timing_result(times, data_bytes);
}

// =========================================================================
// Templated per-type benchmark runners
// =========================================================================

template <typename T> static void run_type(std::string_view type_name) {
  if (!conf.run)
    return;

  std::optional<TimingResult> r;

  for (uint64_t n : conf.array_sizes) {
    T *a = alloc<T>(n);
    T *b = alloc<T>(n);
    T *out = alloc<T>(n);
    T *gold = alloc<T>(n);
    init_data<T>(n, a, b, out, gold);

#pragma omp target enter data map(to : a[0 : n], b[0 : n])
#pragma omp target enter data map(alloc : out[0 : n])

    r = run_bench(
        "misc_elem_loop", data_bytes_elem_loop<T>(n),
        [&] { gold_elem_loop(gold, a, b, n); },
        [&] { elem_loop(out, a, b, n); },
        [&](std::string_view label) {
          return check_elem_loop(label, out, gold, n);
        },
        [&] { prepare_elem_loop<true>(out, n); });
    print_result("misc_elem_loop", type_name, n, r);

    uint64_t nx = std::max<uint64_t>(1, std::sqrt(n));
    uint64_t ny = n / nx;
    uint64_t stencil_n = nx * ny;
    r = run_bench(
        "misc_stencil", data_bytes_stencil<T>(stencil_n),
        [&] {
          prepare_stencil<false>(gold, a, stencil_n);
          gold_stencil(gold, a, ny, nx);
        },
        [&] { stencil(out, a, ny, nx); },
        [&](std::string_view label) {
          return check_stencil(label, out, gold, ny, nx);
        },
        [&] { prepare_stencil<true>(out, a, stencil_n); });
    print_result("misc_stencil", type_name, stencil_n, r);

    const T c(2.0);
    r = run_bench(
        "misc_linalg", data_bytes_linalg<T>(n),
        [&] { gold_linalg(gold, a, b, c, n); },
        [&] { linalg(out, a, b, c, n); },
        [&](std::string_view label) {
          return check_linalg(label, out, gold, n);
        },
        [&] { prepare_linalg<true>(out, n); });
    print_result("misc_linalg", type_name, n, r);

    auto func = [](uint64_t i) { return T(static_cast<double>(i % 100)); };
    r = run_bench(
        "misc_elem_func", data_bytes_elem_func<T>(n),
        [&] { gold_elem_func(gold, func, n); },
        [&] { elem_func(out, func, n); },
        [&](std::string_view label) {
          return check_elem_func(label, out, gold, n);
        },
        [&] { prepare_elem_func<true>(out, n); });
    print_result("misc_elem_func", type_name, n, r);

#pragma omp target exit data map(delete : out[0 : n])

    free(out);
    free(gold);

    // Particle reuses a/b as initial x/y positions and vx/vy velocities.
    // z0 is the initial z position, x/y/z are reset before each device run,
    // gold_* hold the CPU reference, and vz is the z velocity.
    T *z0 = alloc<T>(n);
    T *x = alloc<T>(n);
    T *y = alloc<T>(n);
    T *z = alloc<T>(n);
    T *gold_x = alloc<T>(n);
    T *gold_y = alloc<T>(n);
    T *gold_z = alloc<T>(n);
    T *vz = alloc<T>(n);

    init_data<T>(n, z0, vz);

#pragma omp target enter data map(alloc : x[0 : n], y[0 : n], z[0 : n])
#pragma omp target enter data map(to : vz[0 : n])

    const T dt(2);
    r = run_bench(
        "misc_particle", data_bytes_particle<T>(n),
        [&] {
          prepare_particle<false>(gold_x, gold_y, gold_z, a, b, z0, n);
          gold_particle(gold_x, gold_y, gold_z, a, b, vz, dt, n);
        },
        [&] { particle(x, y, z, a, b, vz, dt, n); },
        [&](std::string_view label) {
          return check_particle(label, x, y, z, gold_x, gold_y, gold_z, n);
        },
        [&] { prepare_particle<true>(x, y, z, a, b, z0, n); });
    print_result("misc_particle", type_name, n, r);

#pragma omp target exit data map(delete : x[0 : n], y[0 : n], z[0 : n],        \
                                     vz[0 : n])
#pragma omp target exit data map(delete : a[0 : n], b[0 : n])

    free(a);
    free(b);
    free(z0);
    free(x);
    free(y);
    free(z);
    free(gold_x);
    free(gold_y);
    free(gold_z);
    free(vz);
  }
}

void run_bench_op() {
  print_array_sizes();

  print_header();

  std::cout << "\n--- double ---\n";
  run_type<double>("double");

  std::cout << "\n--- uint ---\n";
  run_type<unsigned>("uint");

  std::cout << "\n--- ulong ---\n";
  run_type<unsigned long>("ulong");

  std::cout << "\n--- Value ---\n";
  run_type<Value>("Value");
}
