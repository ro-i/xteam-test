The code in this repository tests and compares implementations of OpenMP cross-team reductions and scans across different compilers for correctness and performance.
The tests don't only include high-level tests using the corresponding OpenMP pragmas, but also simulations that don't depend on the reduction/scan-specific codegen and allow to target the OpenMP device runtime implementation more specifically.

Each compiler is defined and implemented by
- a label identifying the compiler, e.g. "aomp" for AOMP or "trunk" for LLVM upstream
- a preprocessor makro that enables/disables compiler specific parts of the benchmark code
- an `xteam_simulations_<label>.h` header implementing simulations using the API endpoints provided by the compiler-specific OpenMP device runtime

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
