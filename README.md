The code in this repository tests and compares implementations of cross-team operations such as OpenMP reductions and scans across different compilers for correctness and performance.
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
- `trunk`, a build of llvm/main, commit bcc9a55bdb228661d98444f0d6c74b47ed0426bb (2026-04-30)
- `trunk_dev`, a build of llvm/main with custom changes (based on `trunk` above)
- `trunk_jd`, a build of https://github.com/jdoerfert/llvm-project/tree/omp_multi_lvl_red (re-based on `trunk` above)

Compile benchmarks binaries:
- set `CXX_<label>` in either `Makefile` or a `local.mk` file to the path to the corresponding `clang++`.
- run either `make` (or `make all`) to compile the benchmark binaries for all combinations of compilers, operations, and team numbers.
- or,
  - run `make <op>` to compile the benchmark binaries for all combinations of compilers and team numbers for the operation identified by `op`.
  - run `make <label>` to compile the benchmark binaries for all combinations of operations and team numbers for the compiler identified by `label`.
  - run `make <op>_<label>` to compile the benchmark binaries for all team numbers for the operation and compiler identified by `op` and `label`.
- Each compiled benchmark will produce one benchmark binary in the naming format `<op>_<label>_<teams>`, e.g.:
  - `red_aomp_208` (reduction for `aomp` using 208 teams)
  - `scan_trunk_dev_10400` (scan for `trunk_dev` using 10400 teams)
- For other configuration options, see `Makefile` and `common.h`.

There are two options for running benchmark binaries:
1. Run them directly by invoking their corresponding benchmark binary (see `<benchmark binary> -h` for available options).
2. Run them combined and interleaved by invoking multiple benchmark binaries through `run_bench.sh` (see `run_bench.sh -h` for available options).

Example: `./run_bench.sh -rsq -n1 aomp aomp_dev trunk`
- runs every binary for one round (`-n1`)
- does a quick run, testing only one array size (`-q`)
- runs non-simulation tests (`-r`)
- runs simulation tests (`-s`)

You may also use `LIBOMPTARGET_INFO=16` to get some info on every kernel launch done by OpenMP offloading.

Note:
- the input data used for the tests is always the same. If failures don't reproduce reliably, it's not due to changing data, but rather due to a race condition in the algorithm under test.
- we assume at some points that warp size is 64 for AMD and 32 for Nvidia. In some cases, we need a compile-time constand to replicate CodeGen behavior in the simulations although a cleaner alternative would be using builtins (but they aren't constexpr).
