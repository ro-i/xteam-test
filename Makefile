# xteam benchmark — multi-compiler Makefile
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ── GPU target ──────────────────────────────────────────────────────────────
OFFLOAD_ARCH ?= gfx90a

# ── Common flags ────────────────────────────────────────────────────────────
COMMON_FLAGS = -O2 -fopenmp --offload-arch=$(OFFLOAD_ARCH) -std=c++20 -save-temps=obj
COMMON_DEFS  =

SRC = xteam_bench.cpp
COMMON_HEADERS = xteam_simulations_common.h common.h bench_common.h
TEAM_NUMS = 208 10400

# ── Compiler configurations ─────────────────────────────────────────────────
# Define as many CXX_<label> variables as you need.  Each one will produce
# one binary per value in TEAM_NUMS, called xteam_bench_<label>_<teams>.
# Optionally set FLAGS_<label>/DEFS_<label> to override the defaults.
#
# Defaults (override on the command line or via local.mk):
-include local.mk

# Known labels.  Add new ones here and provide CXX_<label> via local.mk or
# the command line to enable them.
ALL_LABELS = aomp_dev aomp trunk trunk_jd trunk_dev

CXX_aomp_dev    ?=
CXX_aomp        ?=
CXX_trunk       ?=
CXX_trunk_jd    ?=
CXX_trunk_dev   ?=
FLAGS_aomp_dev  ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_aomp      ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_trunk     ?= $(COMMON_FLAGS)
FLAGS_trunk_jd  ?= $(COMMON_FLAGS)
FLAGS_trunk_dev ?= $(COMMON_FLAGS)
DEFS_aomp_dev   ?= $(COMMON_DEFS) -DAOMP_DEV
DEFS_aomp       ?= $(COMMON_DEFS) -DAOMP
DEFS_trunk      ?= $(COMMON_DEFS) -DTRUNK
DEFS_trunk_jd   ?= $(COMMON_DEFS) -DTRUNK_JD
DEFS_trunk_dev  ?= $(COMMON_DEFS) -DTRUNK_DEV
# Note: potentially test no-loop with -fopenmp-target-xteam-no-loop-scan

# Active labels: those with a non-empty CXX_<label>.
LABELS := $(strip $(foreach L,$(ALL_LABELS),$(if $(strip $(CXX_$(L))),$(L))))

ifeq ($(strip $(LABELS)),)
  $(info )
  $(info  No compilers configured.  Set CXX_<label> to build, e.g.:)
  $(info    make CXX_aomp=/path/to/clang++)
  $(info )
endif

BINARIES = $(foreach L,$(LABELS),$(foreach T,$(TEAM_NUMS),xteam_bench_$(L)_$(T)))

# ── Output directory for results ────────────────────────────────────────────
RESULTS_DIR ?= results

# ── Targets ─────────────────────────────────────────────────────────────────
.PHONY: all clean help format \
        $(LABELS) \
        $(foreach L,$(LABELS),$(foreach T,$(TEAM_NUMS),$(L)_$(T)))

all: $(BINARIES)

# Per-(label, teams) build rule.  Compiles inside out_<label>_<teams>/ so
# -save-temps=obj keeps intermediates there, then symlinks the binary to the
# top dir for convenience.
define BUILD_RULE
xteam_bench_$(1)_$(2): $(SRC) xteam_simulations_$(1).h $(COMMON_HEADERS)
	@test -n "$$(CXX_$(1))" || { echo "ERROR: CXX_$(1) is not set"; exit 1; }
	@echo "Building $(1) for $(2) teams ..."
	rm -rf out_$(1)_$(2)
	mkdir -p out_$(1)_$(2)
	cd out_$(1)_$(2) && $$(CXX_$(1)) $$(DEFS_$(1)) -DXTEAM_NUM_TEAMS=$(2) $$(FLAGS_$(1)) -o $$@ $(addprefix ../,$(SRC))
	ln -sf out_$(1)_$(2)/$$@ $$@
	cd out_$(1)_$(2) && $$(dir $$(CXX_$(1)))llvm-dis *.bc

# Convenience: `make <label>_<teams>` builds the matching binary.
$(1)_$(2): xteam_bench_$(1)_$(2)
endef
$(foreach L,$(LABELS),$(foreach T,$(TEAM_NUMS),$(eval $(call BUILD_RULE,$(L),$(T)))))

# Convenience: `make <label>` builds for every team count in TEAM_NUMS.
define LABEL_RULE
$(1): $(foreach T,$(TEAM_NUMS),xteam_bench_$(1)_$(T))
endef
$(foreach L,$(LABELS),$(eval $(call LABEL_RULE,$(L))))

format:
	clang-format -i $(SRC) $(wildcard xteam_simulations_*.h) $(COMMON_HEADERS)

clean:
	rm -rf $(BINARIES) out_*

help:
	@echo "xteam benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build all configured compilers × team counts"
	@echo "  <label>          Build <label> for every team count in TEAM_NUMS"
	@echo "                     (e.g. make aomp)"
	@echo "  <label>_<teams>  Build a specific (compiler, team count) combination"
	@echo "                     (e.g. make aomp_208)"
	@echo "  clean            Remove binaries and object files"
	@echo "  format           Format the code using clang-format"
	@echo ""
	@echo "Variables:"
	@echo "  CXX_<label>      Compiler path  (e.g. CXX_aomp=/path/to/clang++)"
	@echo "  FLAGS_<label>    Extra flags    (e.g. FLAGS_aomp=-g)"
	@echo "  DEFS_<label>     Extra -D defs"
	@echo "  TEAM_NUMS        Team counts to build (default: 208 10400)"
	@echo "  OFFLOAD_ARCH     GPU arch       (default: gfx90a)"
	@echo ""
	@echo "Configured labels: $(LABELS)"
	@echo "Team counts:       $(TEAM_NUMS)"
