# xteam benchmark — multi-compiler Makefile
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Enable parallel builds by default.
# If nproc is 32 or less, we assume to run on a single-user machine and use all
# cores. Otherwise, we assume to be on a shared machine and restrict ourselves
# to nproc/4 cores.
# Can be overridden with: make -j<N>
# Also, enable output sync so that the output is not interleaved.
JOBS := $(shell n=$$(nproc); if [ $$n -le 32 ]; then echo $$n; else echo $$((n / 4)); fi)
MAKEFLAGS += -j$(JOBS) -O

# ── GPU target ──────────────────────────────────────────────────────────────
OFFLOAD_ARCH ?= gfx90a

# ── Common flags ────────────────────────────────────────────────────────────
COMMON_FLAGS = -O2 -fopenmp --offload-arch=$(OFFLOAD_ARCH) -std=c++20 -save-temps=obj
COMMON_DEFS  =

# Sources. Each binary compiles $(SRC_DIR)/xteam_bench.cpp plus exactly one of
# the op .cpp files, selected by OPS below.
SRC_DIR       ?= src
COMMON_HEADERS = $(SRC_DIR)/common.h $(SRC_DIR)/xteam_simulations_common.h $(SRC_DIR)/xteam_simulations_selected.h
TEAM_NUMS      = 208 10400

# Operations. The op name is also the suffix used in the source file name
# xteam_<op>.cpp and in the binary name (<op>_<label>_<teams>).
OPS      = red scan

# ── Compiler configurations ─────────────────────────────────────────────────
# Define as many CXX_<label> variables as you need. Each one will produce one
# binary per (<op>, <label>, TEAM_NUMS) combination, named <op>_<label>_<teams>.
# Optionally set FLAGS_<label>/DEFS_<label>/OPS_<label> to override the
# defaults.
#
# Defaults (override on the command line or via local.mk):
-include local.mk

# Known labels.  Add new ones here and provide CXX_<label> via local.mk or
# the command line to enable them.
ALL_LABELS = aomp_dev aomp trunk trunk_jd trunk_dev
# Per-label op support.
OPS_aomp_dev    ?= red scan
OPS_aomp        ?= red scan
OPS_trunk       ?= red
OPS_trunk_jd    ?= red
OPS_trunk_dev   ?= red
# Compiler paths.
CXX_aomp_dev    ?=
CXX_aomp        ?=
CXX_trunk       ?=
CXX_trunk_jd    ?=
CXX_trunk_dev   ?=
# Compiler definitions.
DEFS_aomp_dev   ?= $(COMMON_DEFS) -DAOMP_DEV
DEFS_aomp       ?= $(COMMON_DEFS) -DAOMP
DEFS_trunk      ?= $(COMMON_DEFS) -DTRUNK
DEFS_trunk_jd   ?= $(COMMON_DEFS) -DTRUNK_JD
DEFS_trunk_dev  ?= $(COMMON_DEFS) -DTRUNK_DEV
# Compiler flags per op.
FLAGS_aomp_dev_red   ?= $(COMMON_FLAGS)
FLAGS_aomp_dev_scan  ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_aomp_red       ?= $(COMMON_FLAGS)
FLAGS_aomp_scan      ?= $(COMMON_FLAGS) -fopenmp-target-xteam-scan
FLAGS_trunk_red      ?= $(COMMON_FLAGS)
FLAGS_trunk_scan     ?= $(COMMON_FLAGS)
FLAGS_trunk_jd_red   ?= $(COMMON_FLAGS)
FLAGS_trunk_jd_scan  ?= $(COMMON_FLAGS)
FLAGS_trunk_dev_red  ?= $(COMMON_FLAGS)
FLAGS_trunk_dev_scan ?= $(COMMON_FLAGS)
# Note: potentially test no-loop with -fopenmp-target-xteam-no-loop-scan

# Active labels: those with a non-empty CXX_<label>.
LABELS = $(strip $(foreach L,$(ALL_LABELS),$(if $(strip $(CXX_$(L))),$(L))))
# Build labels for each op: LABELS_<op> = active labels whose OPS_<label>
# includes <op>.
define LABELS_FOR_OP
LABELS_$(1) := $(foreach L,$(LABELS),$(if $(filter $(1),$(OPS_$(L))),$(L)))
endef
$(foreach O,$(OPS),$(eval $(call LABELS_FOR_OP,$(O))))

ifeq ($(strip $(LABELS)),)
  $(info )
  $(info  No compilers configured.  Set CXX_<label> to build, e.g.:)
  $(info    make CXX_aomp=/path/to/clang++)
  $(info )
endif

BINARIES = $(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(foreach T,$(TEAM_NUMS),$(O)_$(L)_$(T))))

# ── Targets ─────────────────────────────────────────────────────────────────
.PHONY: all clean help format \
        $(LABELS) \
        $(OPS) \
        $(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(O)_$(L)))

all: $(BINARIES)

# Per-(op, label, teams) build rule. Compiles inside out_<op>_<label>_<teams>/
# so -save-temps=obj keeps intermediates there, then symlinks the binary to
# the top dir for convenience.
define BUILD_RULE
$(1)_$(2)_$(3): $(SRC_DIR)/xteam_bench.cpp \
                $(SRC_DIR)/xteam_$(1).cpp \
                $(SRC_DIR)/xteam_simulations_$(2).h \
                $(COMMON_HEADERS)
	@test -n "$$(CXX_$(2))" || { echo "ERROR: CXX_$(2) is not set"; exit 1; }
	@echo "Building $(1) for $(2) with $(3) teams ..."
	rm -rf out_$(1)_$(2)_$(3)
	mkdir -p out_$(1)_$(2)_$(3)
	cd out_$(1)_$(2)_$(3) && $$(CXX_$(2)) $$(DEFS_$(2)) $$(FLAGS_$(2)_$(1)) \
		-DXTEAM_NUM_TEAMS=$(3) -DCOMPILER_LABEL="\"$(2)\"" \
		-o $$@ ../$(SRC_DIR)/xteam_bench.cpp ../$(SRC_DIR)/xteam_$(1).cpp
	ln -sf out_$(1)_$(2)_$(3)/$$@ $$@
	cd out_$(1)_$(2)_$(3) && $$(dir $$(CXX_$(2)))llvm-dis *.bc
endef
$(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(foreach T,$(TEAM_NUMS),$(eval $(call BUILD_RULE,$(O),$(L),$(T))))))

# Convenience: `make <op>_<label>` builds all supported teams combinations for
# that op and that label.
define OP_LABEL_RULE
$(1)_$(2): $(foreach T,$(TEAM_NUMS),$(1)_$(2)_$(T))
endef
$(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(eval $(call OP_LABEL_RULE,$(O),$(L)))))

# Convenience: `make <op>` builds all supported (label, teams) combinations for
# that op.
define OP_RULE
$(1): $(foreach L,$(LABELS_$(1)),$(foreach T,$(TEAM_NUMS),$(1)_$(L)_$(T)))
endef
$(foreach O,$(OPS),$(eval $(call OP_RULE,$(O))))

# Convenience: `make <label>` builds all supported (op, teams) combinations for
# that label.
define LABEL_RULE
$(1): $(foreach O,$(OPS_$(1)),$(foreach T,$(TEAM_NUMS),$(O)_$(1)_$(T)))
endef
$(foreach L,$(LABELS),$(eval $(call LABEL_RULE,$(L))))

format:
	clang-format -i $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h

clean:
	rm -rf $(BINARIES) out_*

help:
	@echo "xteam benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all                     Build all configured (op × compiler/label × team count) combinations"
	@echo "  <op>                    Build <op>_<label>_<teams> for every (label, teams) combination supported by that op"
	@echo "                            (e.g. make red)"
	@echo "  <label>                 Build <op>_<label>_<teams> for every (op, teams) combination supported by that label"
	@echo "                            (e.g. make aomp)"
	@echo "  <op>_<label>            Build <op>_<label>_<teams> for every teams combination supported by that (op, label) pair"
	@echo "                            (e.g. make red_aomp)"
	@echo "  <op>_<label>_<teams>    Build a specific (op, compiler, teams) combination"
	@echo "                            (e.g. make red_aomp_208)"
	@echo "  clean                   Remove binaries and object files"
	@echo "  format                  Format the code using clang-format"
	@echo ""
	@echo "Variables:"
	@echo "  CXX_<label>      Compiler path  (e.g. CXX_aomp=/path/to/clang++)"
	@echo "  FLAGS_<label>    Extra flags    (e.g. FLAGS_aomp=-g)"
	@echo "  DEFS_<label>     Extra -D defs"
	@echo "  OPS_<label>      Ops supported by this compiler"
	@echo "  TEAM_NUMS        Team counts to build (default: 208 10400)"
	@echo "  OFFLOAD_ARCH     GPU arch       (default: gfx90a)"
	@echo ""
	@echo "Ops:               $(OPS)"
	@echo "Configured labels: $(LABELS)"
	@echo "Team counts:       $(TEAM_NUMS)"
