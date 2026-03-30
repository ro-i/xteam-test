# xteam benchmark — multi-compiler Makefile

# ── GPU target ──────────────────────────────────────────────────────────────
OFFLOAD_ARCH ?= gfx90a

# ── Common flags ────────────────────────────────────────────────────────────
COMMON_FLAGS = -O2 -fopenmp --offload-arch=$(OFFLOAD_ARCH) -lstdc++ -latomic -std=c++20 -save-temps
COMMON_DEFS  =

SRC = xteam_bench.cpp

# ── Compiler configurations ─────────────────────────────────────────────────
# Define as many CXX_<label> variables as you need.  Each one will produce
# a binary called xteam_bench_<label>.  Optionally set FLAGS_<label> to add
# per-compiler flags.
#
# Defaults (override on the command line or via local.mk):
-include local.mk
CXX_aomp_dev   ?=
CXX_aomp       ?=
CXX_trunk      ?=
FLAGS_aomp_dev ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_aomp     ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_trunk    ?= $(COMMON_FLAGS)
DEFS_aomp_dev  ?= $(COMMON_DEFS) -DAOMP_DEV
DEFS_aomp      ?= $(COMMON_DEFS) -DAOMP
DEFS_trunk     ?= $(COMMON_DEFS) -DTRUNK
# Note: potentially test no-loop with -fopenmp-target-xteam-no-loop-scan

# Collect all labels that have a non-empty CXX_<label>
LABELS :=
ifneq ($(strip $(CXX_aomp_dev)),)
  LABELS += aomp_dev
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
  $(info    make CXX_aomp=/path/to/clang++)
  $(info )
endif

BINARIES = $(foreach L,$(LABELS),xteam_bench_$(L))

# ── Output directory for results ────────────────────────────────────────────
RESULTS_DIR ?= results

# ── Targets ─────────────────────────────────────────────────────────────────
.PHONY: all clean help

all: $(BINARIES)

# Pattern rule: xteam_bench_<label> from xteam_bench.cpp
define COMPILER_RULE
xteam_bench_$(1): $(SRC) xteam_simulations_$(1).h
	@test -n "$$(CXX_$(1))" || { echo "ERROR: CXX_$(1) is not set"; exit 1; }
	$$(CXX_$(1)) $$(DEFS_$(1)) $$(FLAGS_$(1)) -o $$@ $(SRC)
endef
$(foreach L,$(LABELS),$(eval $(call COMPILER_RULE,$(L))))

# Also add rules for the plain labels
define PLAIN_LABEL_RULE
$(1): xteam_bench_$(1)
endef
$(foreach L,$(LABELS),$(eval $(call PLAIN_LABEL_RULE,$(L))))

clean:
	rm -f $(BINARIES) *.bc *.ii *.img *.ll *.o *.out *.resolution.txt *.s *.tmp

help:
	@echo "xteam benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build all configured compilers"
	@echo "  clean            Remove binaries and results"
	@echo ""
	@echo "Variables:"
	@echo "  CXX_<label>            Compiler path  (e.g. CXX_aomp=/path/to/clang++)"
	@echo "  FLAGS_<label>          Extra flags    (e.g. FLAGS_aomp=-g)"
	@echo "  OFFLOAD_ARCH           GPU arch       (default: gfx90a)"
