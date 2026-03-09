To compile/run the tests, set `CXX_dev` and `CXX_aomp` in either `Makefile` or a `local.mk` file.
These variables hold the paths to the compiler under test (called "dev") and the reference compiler (called "aomp").

Run full test suite: `make run`
Run test suite with only one array size and less rounds: `make quick-run`

For other configuration options, see `Makefile` and `common.h`.
