# xteam benchmark — multi-compiler Makefile
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ── GPU target ──────────────────────────────────────────────────────────────
OFFLOAD_ARCH ?= gfx90a

# ── Common flags ────────────────────────────────────────────────────────────
COMMON_FLAGS = -O2 -fopenmp --offload-arch=$(OFFLOAD_ARCH) -lstdc++ -latomic -std=c++20 -save-temps=obj
COMMON_DEFS  =

SRC = xteam_bench.cpp
COMMON_HEADERS = xteam_simulations_common.h common.h bench_common.h

# ── Compiler configurations ─────────────────────────────────────────────────
# Define as many CXX_<label> variables as you need.  Each one will produce
# a binary called xteam_bench_<label>.  Optionally set FLAGS_<label> to add
# per-compiler flags.
#
# Defaults (override on the command line or via local.mk):
-include local.mk
CXX_aomp_dev          ?=
CXX_aomp              ?=
CXX_trunk             ?=
CXX_trunk_jd          ?=
CXX_trunk_dev         ?=
FLAGS_aomp_dev        ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_aomp            ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_trunk           ?= $(COMMON_FLAGS)
FLAGS_trunk_jd        ?= $(COMMON_FLAGS)
FLAGS_trunk_dev       ?= $(COMMON_FLAGS)
DEFS_aomp_dev         ?= $(COMMON_DEFS) -DAOMP_DEV
DEFS_aomp             ?= $(COMMON_DEFS) -DAOMP
DEFS_trunk            ?= $(COMMON_DEFS) -DTRUNK
DEFS_trunk_jd         ?= $(COMMON_DEFS) -DTRUNK_JD
DEFS_trunk_dev        ?= $(COMMON_DEFS) -DTRUNK_DEV
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
ifneq ($(strip $(CXX_trunk_jd)),)
  LABELS += trunk_jd
endif
ifneq ($(strip $(CXX_trunk_dev)),)
  LABELS += trunk_dev
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
xteam_bench_$(1): $(SRC) xteam_simulations_$(1).h $(COMMON_HEADERS)
	@test -n "$$(CXX_$(1))" || { echo "ERROR: CXX_$(1) is not set"; exit 1; }
	@echo "Building for 208 teams ..."
	rm -rf out_$(1)_208
	mkdir -p out_$(1)_208
	cd out_$(1)_208 && $$(CXX_$(1)) $$(DEFS_$(1)) -DXTEAM_NUM_TEAMS=208 $$(FLAGS_$(1)) -o $$@_208 $(addprefix ../,$(SRC)) && cp $$@_208 .. && cp $$@_208 ../$$@
	cd out_$(1)_208 && $$(dir $$(CXX_$(1)))llvm-dis *.bc
	@echo "Building for 10400 teams ..."
	rm -rf out_$(1)_10400
	mkdir -p out_$(1)_10400
	cd out_$(1)_10400 && $$(CXX_$(1)) $$(DEFS_$(1)) -DXTEAM_NUM_TEAMS=10400 $$(FLAGS_$(1)) -o $$@_10400 $(addprefix ../,$(SRC)) && cp $$@_10400 ..
	cd out_$(1)_10400 && $$(dir $$(CXX_$(1)))llvm-dis *.bc
endef
$(foreach L,$(LABELS),$(eval $(call COMPILER_RULE,$(L))))

# Also add rules for the plain labels
define PLAIN_LABEL_RULE
$(1): xteam_bench_$(1)
endef
$(foreach L,$(LABELS),$(eval $(call PLAIN_LABEL_RULE,$(L))))

format:
	clang-format -i $(SRC) $(wildcard xteam_simulations_*.h) $(COMMON_HEADERS)

clean:
	rm -rf $(BINARIES) out_*

help:
	@echo "xteam benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build for all configured compilers"
	@echo "  <label>          Build for the given compiler (e.g. make aomp)"
	@echo "  clean            Remove binaries and object files"
	@echo "  format           Format the code using clang-format"
	@echo ""
	@echo "Variables:"
	@echo "  CXX_<label>            Compiler path  (e.g. CXX_aomp=/path/to/clang++)"
	@echo "  FLAGS_<label>          Extra flags    (e.g. FLAGS_aomp=-g)"
	@echo "  OFFLOAD_ARCH           GPU arch       (default: gfx90a)"
