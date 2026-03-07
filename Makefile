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
COMMON_FLAGS = -O2 -fopenmp --offload-arch=$(OFFLOAD_ARCH) -lstdc++ -latomic -fopenmp-target-xteam-scan -std=c++20 -save-temps
BENCH_ITERS  ?= 1000
WARMUP_ITERS ?= 2
QUICK_RUN    ?= 0
ifeq ($(QUICK_RUN),1)
  ROUNDS     ?= 1
else
  ROUNDS     ?= 5
endif
DEFS         = -DBENCH_ITERS=$(BENCH_ITERS) -DWARMUP_ITERS=$(WARMUP_ITERS) -DQUICK_RUN=$(QUICK_RUN)

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
FLAGS_dev      ?=
FLAGS_aomp     ?= -DAOMP

# Collect all labels that have a non-empty CXX_<label>
LABELS :=
ifneq ($(strip $(CXX_dev)),)
  LABELS += dev
endif
# ifneq ($(strip $(CXX_aomp)),)
#   LABELS += aomp
# endif

ifeq ($(strip $(LABELS)),)
  $(info )
  $(info  No compilers configured.  Set CXX_<label> to build, e.g.:)
  $(info    make CXX_dev=/path/to/clang++)
  $(info )
endif

BINARIES = $(foreach L,$(LABELS),xteam_bench_$(L))

# ── Output directory for results ────────────────────────────────────────────
RESULTS_DIR ?= results

# ── HIP build configuration ──────────────────────────────────────────────────
HIP_SRC       = xteam_bench_hip.cpp
DEVICE_RTL_BC = $(shell realpath $(shell dirname $(CXX_dev))/../lib/amdgcn-amd-amdhsa/libomptarget-amdgpu.bc 2>/dev/null)
HIP_COMMON    = -x hip -O2 --offload-arch=$(OFFLOAD_ARCH) -std=c++20

# ── Targets ─────────────────────────────────────────────────────────────────
.PHONY: all run quick-run clean help hip quick-hip

all:
	rm -f $(BINARIES)
	$(MAKE) QUICK_RUN=0 $(BINARIES)

quick:
	rm -f $(BINARIES)
	$(MAKE) QUICK_RUN=1 $(BINARIES)

# Pattern rule: xteam_bench_<label> from xteam_bench.cpp
define COMPILER_RULE
xteam_bench_$(1): $(SRC)
	@test -n "$$(CXX_$(1))" || { echo "ERROR: CXX_$(1) is not set"; exit 1; }
	$$(CXX_$(1)) $$(COMMON_FLAGS) $$(DEFS) $$(FLAGS_$(1)) -o $$@ $(SRC)
	@echo "Built $$@ with $$(CXX_$(1))"
endef
$(foreach L,$(LABELS),$(eval $(call COMPILER_RULE,$(L))))

run:
	rm -f $(BINARIES)
	$(MAKE) QUICK_RUN=0 ROUNDS=$(ROUNDS) WARMUP_ITERS=$(WARMUP_ITERS) BENCH_ITERS=$(BENCH_ITERS) $(BINARIES)
	@mkdir -p $(RESULTS_DIR)
	./run_bench.sh -n $(ROUNDS) $(BINARIES)

quick-run:
	rm -f $(BINARIES)
	$(MAKE) QUICK_RUN=1 ROUNDS=$(ROUNDS) WARMUP_ITERS=$(WARMUP_ITERS) BENCH_ITERS=$(BENCH_ITERS) $(BINARIES)
	@mkdir -p $(RESULTS_DIR)
	./run_bench.sh -n $(ROUNDS) $(BINARIES)

hip:
	rm -f xteam_bench_hip
	@test -n "$(CXX_dev)" || { echo "ERROR: CXX_dev is not set"; exit 1; }
	@test -f "$(DEVICE_RTL_BC)" || { echo "ERROR: device RTL not found at $(DEVICE_RTL_BC)"; exit 1; }
	$(CXX_dev) $(HIP_COMMON) -fgpu-rdc --offload-new-driver -save-temps \
	  -Xoffload-linker-amdgcn-amd-amdhsa $(DEVICE_RTL_BC) \
	  $(DEFS) -o xteam_bench_hip $(HIP_SRC)
	@echo "Built xteam_bench_hip (HIP) with $(CXX_dev)"

quick-hip:
	rm -f xteam_bench_hip
	$(MAKE) QUICK_RUN=1 ROUNDS=$(ROUNDS) WARMUP_ITERS=$(WARMUP_ITERS) BENCH_ITERS=$(BENCH_ITERS) hip

clean:
	rm -f $(BINARIES) xteam_bench_hip
	rm -rf $(RESULTS_DIR)

help:
	@echo "xteam benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build all configured compilers"
	@echo "  quick            Build all configured compilers for only one array size"
	@echo "  (quick-)run      Run binaries (interleaved) via run_bench.sh (default: 5 rounds / quick run: 1 round)"
	@echo "  hip              Build HIP binary"
	@echo "  quick-hip        Build HIP binary for only one array size"
	@echo "  clean            Remove binaries and results"
	@echo ""
	@echo "Variables:"
	@echo "  CXX_<label>      Compiler path  (e.g. CXX_dev=/path/to/clang++)"
	@echo "  FLAGS_<label>    Extra flags    (e.g. FLAGS_dev=-g)"
	@echo "  OFFLOAD_ARCH     GPU arch       (default: gfx90a)"
	@echo "  BENCH_ITERS      Timed iters    (default: 10)"
	@echo "  WARMUP_ITERS     Warmup iters   (default: 2)"
	@echo "  RESULTS_DIR      Output dir     (default: results)"
