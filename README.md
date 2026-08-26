The code in this repository tests and compares implementations of cross-team operations such as OpenMP reductions across different compilers for correctness and performance.
The tests include
- high-level tests using the corresponding OpenMP pragmas
- simulations that don't depend on the specific codegen for, e.g., reductions and
  - allow to target the OpenMP device runtime implementation more specifically
  - verify if performance issues are rather runtime- or codegen-related

Each compiler is defined and implemented by
- a label identifying the compiler, e.g. "aomp" for AOMP or "trunk" for LLVM upstream
- a preprocessor macro that enables/disables compiler specific parts of the benchmark code
- an `xteam_simulations_<label>.h` header implementing simulations using the API endpoints provided by the compiler-specific OpenMP device runtime
  (this header might just be a symlink in case the corresponding compilers only differ in code generation and not in the runtime implementation)

Currently used default compilers, by their label:
- `aomp`, a build of ROCm/llvm-project `amd-staging` (specific commit might vary depending on the situation*)
- `trunk`, a build of llvm/llvm-project `main` (specific commit might vary depending on the situation*)

*If used for benchmarking purposes, the commit information should be provided together with the results.

Note that these labels and their corresponding compilers are just a default set.
This repository is built specifically for easy modification and addition of different compiler builds.

Compile benchmarks binaries:
- set `CXX_<label>` in either `Makefile` or a `local.mk` file to the path to the corresponding `clang++`.
- don't forget to set the correct `OFFLOAD_ARCH` (e.g., `gfx90a`, `gfx942`, etc.)
- run either `make` (or `make all`) to compile all benchmark binaries for all combinations of compilers, operations, and grids.
- or,
  - run `make <op>` to compile the benchmark binaries for all combinations of compilers and grids for the operation identified by `op`.
  - run `make <label>` to compile the benchmark binaries for all combinations of operations and grids for the compiler identified by `label`.
  - run `make <op>_<label>` to compile the benchmark binaries for all grids for the operation and compiler identified by `op` and `label`.
- The grids to build for are configured via `GRIDS`, in the format `<teams>x<threads>`.
  Either dimension may be `auto`, leaving it to codegen autodetection; the simulations then use the default from `common.h` for that dimension.
  A grid with both dimensions left to codegen is spelled `auto`.
- Each compiled benchmark will produce one benchmark binary in the naming format `<op>_<label>_<grid>`, e.g.:
  - `red_aomp_auto` (reduction benchmark for `aomp`, both dimensions left to codegen autodetection)
  - `red_aomp_208x512` (reduction benchmark for `aomp` with 208 teams of 512 threads each)
  - `red_aomp_autox256` (reduction benchmark for `aomp` with 256 threads per team, number of teams left to codegen autodetection)
- For other configuration options, see `Makefile` and `common.h`.

There are two options for running benchmark binaries:
1. Run them directly by invoking their corresponding benchmark binary (see `<benchmark binary> -h` for available options).
2. Run them combined and interleaved by invoking multiple benchmark binaries through `run_bench.sh` (see `run_bench.sh -h` for available options).

Example: `./run_bench.sh -rsq -n1 red_trunk_208x512 red_trunk_dev_208x512 red_trunk_10400x512 red_trunk_dev_10400x512`
- runs each binary for one round (`-n1`)
- does a quick run, testing only one array size (`-q`)
- runs non-simulation tests (`-r`)
- runs simulation tests (`-s`)

You may also use `LIBOMPTARGET_INFO=16` to get some info on every kernel launch done by OpenMP offloading.

`compare_avg.awk` is an AWK script that can be used/adapted to calculate percentage changes from the output of `run_bench.sh`.
Run via `awk -f ./compare_avg.awk <file containing run_bench.sh output>`.
Note, though, that this script is not really something you can directly use on any output without modifications.
It's more like a starting point that handles a few common situations and can be quickly adapted.

---------

PS:
- the input data used for the tests is always the same. If failures don't reproduce reliably, it's not due to changing data, but rather due to a race condition in the algorithm under test.
- we assume at some points that warp size is 64 for AMD and 32 for Nvidia. In some cases, we need a compile-time constand to replicate CodeGen behavior in the simulations although a cleaner alternative would be using builtins (but they aren't constexpr).
