# xteam benchmark — multi-compiler Makefile
#
# Usage:
#   make                    — build with all configured compilers
#   make run                — run all built binaries sequentially
#   make run-interleaved    — run binaries interleaved (fairer comparison)
#   make clean              — remove binaries and results
#
# Configure compilers by setting CXX_<label> and optionally FLAGS_<label>.
# Each label produces a binary named xteam_bench_<label>.
#
# Example:
#   make CXX_dev=/path/to/clang++ CXX_aomp=/other/clang++

# ── GPU target ──────────────────────────────────────────────────────────────
OFFLOAD_ARCH ?= gfx90a

# ── Common flags ────────────────────────────────────────────────────────────
COMMON_FLAGS = -O2 -fopenmp --offload-arch=$(OFFLOAD_ARCH) -lstdc++ -latomic -std=c++20 -save-temps
BENCH_ITERS_REDUCTION  ?= 1000
BENCH_ITERS_SCAN  ?= 10
WARMUP_ITERS ?= 2
QUICK_RUN    ?= 0
DEFS         = -DBENCH_ITERS_REDUCTION=$(BENCH_ITERS_REDUCTION) -DBENCH_ITERS_SCAN=$(BENCH_ITERS_SCAN) -DWARMUP_ITERS=$(WARMUP_ITERS) -DQUICK_RUN=$(QUICK_RUN)
DEFS_NOLOOP  = $(DEFS) -DNOLOOP

SRC = xteam_bench.cpp

# ── Compiler configurations ─────────────────────────────────────────────────
# Define as many CXX_<label> variables as you need.  Each one will produce
# a binary called xteam_bench_<label>.  Optionally set FLAGS_<label> to add
# per-compiler flags.
#
# Defaults (override on the command line or via local.mk):
-include local.mk
CXX_dev        ?=
CXX_aomp       ?=
CXX_trunk      ?=
FLAGS_dev           ?= -fopenmp-target-xteam-scan
FLAGS_dev_no_loop   ?= -fopenmp-target-xteam-no-loop-scan
FLAGS_aomp          ?= -DAOMP -fopenmp-target-xteam-scan
FLAGS_aomp_no_loop  ?= -DAOMP -fopenmp-target-xteam-no-loop-scan
FLAGS_trunk         ?=
FLAGS_trunk_no_loop ?=

# Collect all labels that have a non-empty CXX_<label>
LABELS :=
ifneq ($(strip $(CXX_dev)),)
  LABELS += dev
endif
ifneq ($(strip $(CXX_aomp)),)
  LABELS += aomp
endif
ifneq ($(strip $(CXX_trunk)),)
  LABELS += trunk
endif

ifeq ($(strip $(LABELS)),)
  $(info )
  $(info  No compilers configured.  Set CXX_<label> to build, e.g.:)
  $(info    make CXX_dev=/path/to/clang++)
  $(info )
endif

BINARIES = $(foreach L,$(LABELS),xteam_bench_$(L))
BINARIES_NOLOOP = $(foreach L,$(LABELS),xteam_bench_no_loop_$(L))

# ── Output directory for results ────────────────────────────────────────────
RESULTS_DIR ?= results

# ── Targets ─────────────────────────────────────────────────────────────────
.PHONY: all all-no-loop quick quick-no-loop run run-no-loop quick-run quick-run-no-loop clean help

all:
	rm -f $(BINARIES)
	$(MAKE) QUICK_RUN=0 ROUNDS=$(ROUNDS) WARMUP_ITERS=$(WARMUP_ITERS) BENCH_ITERS_REDUCTION=$(BENCH_ITERS_REDUCTION) BENCH_ITERS_SCAN=$(BENCH_ITERS_SCAN) $(BINARIES)

all-no-loop:
	rm -f $(BINARIES_NOLOOP)
	$(MAKE) QUICK_RUN=0 ROUNDS=$(ROUNDS) WARMUP_ITERS=$(WARMUP_ITERS) BENCH_ITERS_REDUCTION=$(BENCH_ITERS_REDUCTION) BENCH_ITERS_SCAN=$(BENCH_ITERS_SCAN) $(BINARIES_NOLOOP)

quick:
	rm -f $(BINARIES)
	$(MAKE) QUICK_RUN=1 ROUNDS=1 WARMUP_ITERS=$(WARMUP_ITERS) BENCH_ITERS_REDUCTION=$(BENCH_ITERS_REDUCTION) BENCH_ITERS_SCAN=$(BENCH_ITERS_SCAN) $(BINARIES)

quick-no-loop:
	rm -f $(BINARIES_NOLOOP)
	$(MAKE) QUICK_RUN=1 ROUNDS=1 WARMUP_ITERS=$(WARMUP_ITERS) BENCH_ITERS_REDUCTION=$(BENCH_ITERS_REDUCTION) BENCH_ITERS_SCAN=$(BENCH_ITERS_SCAN) $(BINARIES_NOLOOP)

# Pattern rule: xteam_bench_<label> from xteam_bench.cpp
define COMPILER_RULE
xteam_bench_$(1): $(SRC)
	@test -n "$$(CXX_$(1))" || { echo "ERROR: CXX_$(1) is not set"; exit 1; }
	$$(CXX_$(1)) $$(COMMON_FLAGS) $$(DEFS) $$(FLAGS_$(1)) -o $$@ $(SRC)
	@echo "Built $$@ with $$(CXX_$(1))"
endef
$(foreach L,$(LABELS),$(eval $(call COMPILER_RULE,$(L))))
# Same for no-loop scans
define COMPILER_RULE_NOLOOP
xteam_bench_no_loop_$(1): $(SRC)
	@test -n "$$(CXX_$(1))" || { echo "ERROR: CXX_$(1) is not set"; exit 1; }
	$$(CXX_$(1)) $$(COMMON_FLAGS) $$(DEFS_NOLOOP) $$(FLAGS_$(1)_NOLOOP) -o $$@ $(SRC)
	@echo "Built $$@ with $$(CXX_$(1))"
endef
$(foreach L,$(LABELS),$(eval $(call COMPILER_RULE_NOLOOP,$(L))))

run: all
	@mkdir -p $(RESULTS_DIR)
	./run_bench.sh -n $(ROUNDS) $(BINARIES)

run-no-loop: all-no-loop
	@mkdir -p $(RESULTS_DIR)
	./run_bench.sh -n $(ROUNDS) $(BINARIES_NOLOOP)

quick-run: quick
	@mkdir -p $(RESULTS_DIR)
	./run_bench.sh -n 1 $(BINARIES)

quick-run-no-loop: quick-no-loop
	@mkdir -p $(RESULTS_DIR)
	./run_bench.sh -n 1 $(BINARIES_NOLOOP)

clean:
	rm -f $(BINARIES) $(BINARIES_NOLOOP) *.bc *.ii *.img *.ll *.o *.out *.resolution.txt *.s *.tmp

help:
	@echo "xteam benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build all configured compilers"
	@echo "  quick            Build all configured compilers for only one array size"
	@echo "  (quick-)run      Run binaries (interleaved) via run_bench.sh (default: 5 rounds / quick run: 1 round)"
	@echo "  clean            Remove binaries and results"
	@echo ""
	@echo "For every target, there is a <target>-no-loop variant that uses the no-loop scan kernels."
	@echo ""
	@echo "Variables:"
	@echo "  CXX_<label>            Compiler path  (e.g. CXX_dev=/path/to/clang++)"
	@echo "  FLAGS_<label>          Extra flags    (e.g. FLAGS_dev=-g)"
	@echo "  OFFLOAD_ARCH           GPU arch       (default: gfx90a)"
	@echo "  BENCH_ITERS_REDUCTION  Timed iters    (default: 1000)"
	@echo "  BENCH_ITERS_SCAN       Timed iters    (default: 10)"
	@echo "  WARMUP_ITERS           Warmup iters   (default: 2)"
	@echo "  RESULTS_DIR            Output dir     (default: results)"
