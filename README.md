The code in this repository tests and compares implementations of OpenMP cross-team reductions and scans across different compilers for correctness and performance.
The tests include
- high-level tests using the corresponding OpenMP pragmas
- simulations that don't depend on the reduction/scan-specific codegen and
  - allow to target the OpenMP device runtime implementation more specifically
  - verify if performance issues are rather runtime- or codegen-related

Each compiler is defined and implemented by
- a label identifying the compiler, e.g. "aomp" for AOMP or "trunk" for LLVM upstream
- a preprocessor makro that enables/disables compiler specific parts of the benchmark code
- an `xteam_simulations_<label>.h` header implementing simulations using the API endpoints provided by the compiler-specific OpenMP device runtime

Currently used compilers (with varying levels of support), by their label:
- `aomp`, a build of ROCm/amd-staging, commit 790951fe828739002964f2a0fe40fc2048a84443 (2026-03-08)
- `aomp_dev`, a build of https://github.com/ROCm/llvm-project/tree/amd/dev/ro-i/xteam-reduction-scan (based on `aomp` above)
- `trunk`, a build of llvm/main, commit 6a785bf069194beea5ca0a7caac64dd27396c46e (2026-03-16)
- `trunk_jd`, a build of https://github.com/jdoerfert/llvm-project/tree/omp_multi_lvl_red (based on `trunk` above)

Compile benchmarks binaries:
- set `CXX_<label>` in either `Makefile` or a `local.mk` file to the path to the corresponding `clang++`.
- run either `make` (or `make all`) to compile the benchmark binary for all compilers or
  run `make <label>` to compile the benchmark binary for the compiler identified by `label`.
- Each compiled benchmark will produce a benchmark binary called `xteam_bench_<label>`.
- For other configuration options, see `Makefile` and `common.h`.

There are two options for running benchmark binaries:
1. Run them directly by invoking their corresponding benchmark binary (see `xteam_bench_<label> -h` for available options).
2. Run them combined and interleaved by invoking multiple benchmark binaries through `run_bench.sh` (see `run_bench.sh -h` for available options).

Example: `./run_bench.sh -rRq -n1 aomp aomp_dev trunk`
- runs every binary for one round (`-n1`)
- does a quick run, testing only one array size (`-q`)
- runs reduction tests (`-r`)
- runs reduction simulation tests (`-R`)

Note:
- the input data used for the tests is always the same. If failures don't reproduce reliably, it's not due to changing data, but rather due to a race condition in the algorithm under test.
- we assume at some points that warp size is 64 for AMD and 32 for Nvidia. In some cases, we need a compile-time constand to replicate CodeGen behavior in the simulations although a cleaner alternative would be using builtins (but they aren't constexpr).
