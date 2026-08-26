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

# Sources. Each binary compiles $(SRC_DIR)/xteam_bench.cpp plus exactly one of
# the op .cpp files, selected by OPS below.
SRC_DIR        = src
COMMON_HEADERS = $(SRC_DIR)/common.h \
	$(SRC_DIR)/xteam_simulations_common.h \
	$(SRC_DIR)/xteam_simulations_common_trunk.h \
	$(SRC_DIR)/xteam_simulations_selected.h

# Known operations. The op name is also the suffix used in the source file name
# xteam_<op>.cpp and in the binary name (<op>_<label>_<grid>).
ALL_OPS = red misc

# ── Compiler configurations ─────────────────────────────────────────────────
# Define as many CXX_<label> variables as you need. Each one will produce one
# binary per (<op>, <label>, GRIDS) combination, named <op>_<label>_<grid>.
# Optionally set FLAGS_<label>/DEFS_<label>/OPS_<label> to override the
# defaults.
#
# Defaults (override on the command line or via local.mk):
-include local.mk

# Known labels. Add new ones here and provide CXX_<label> via local.mk or the
# command line to enable them.
ALL_LABELS      ?= aomp trunk

# GPU target.
OFFLOAD_ARCH    ?= gfx90a

# Grids to build for, in the format <teams>x<threads>. Either dimension may be
# "auto", which leaves it to codegen autodetection (the simulations then use
# the default from common.h for that dimension). A grid with both dimensions
# left to codegen is spelled "auto".
GRIDS           ?= auto

# Per-label op support.
OPS_aomp        ?= red misc
OPS_trunk       ?= red misc
# Compiler paths.
CXX_aomp        ?=
CXX_trunk       ?=
# Simulation header per label. Defaults to the label's own header; override
# when a label reuses another label's simulation.
SIM_aomp        ?= trunk
SIM_trunk       ?= trunk
# Compiler definitions. The label and simulation header are passed positionally
# by BUILD_RULE; DEFS_<label> only needs to carry semantic macros that actually
# branch the source.
DEFS_aomp       ?= $(COMMON_DEFS)
DEFS_trunk      ?= $(COMMON_DEFS)
# Compiler flags per op.
FLAGS_aomp_red       ?= $(COMMON_FLAGS)
FLAGS_aomp_misc      ?= $(COMMON_FLAGS)
FLAGS_trunk_misc     ?= $(COMMON_FLAGS)
FLAGS_trunk_red      ?= $(COMMON_FLAGS)

# ── Common flags ────────────────────────────────────────────────────────────
COMMON_FLAGS = -O2 -fopenmp --offload-arch=$(OFFLOAD_ARCH) -std=c++20 -save-temps=obj
COMMON_DEFS  =

# ── Grids ───────────────────────────────────────────────────────────────────
# Accessors for the two dimensions of a grid spec.
grid_teams   = $(word 1,$(subst x, ,$(1)))
grid_threads = $(word 2,$(subst x, ,$(1)))

# Reject malformed specs early: a silently mis-parsed grid would still produce
# a plausible-looking binary built for the wrong launch geometry.
# Note that this parsing is a plain $(subst x, ,...), so no grid keyword added
# here in the future may contain a literal "x".
# "autoxauto" is rejected in favour of "auto" so that every configuration has
# exactly one name (and hence one binary name).
ifneq ($(filter autoxauto,$(GRIDS)),)
  $(error Invalid GRIDS entry: autoxauto (spell an all-auto grid as "auto"))
endif
INVALID_GRIDS := $(strip $(foreach G,$(GRIDS),\
  $(if $(shell printf '%s' '$(G)' | grep -Eq '^(auto|(auto|[0-9]+)x(auto|[0-9]+))$$' && echo ok),,$(G))))
ifneq ($(INVALID_GRIDS),)
  $(error Invalid GRIDS entries: $(INVALID_GRIDS) (expected <teams>x<threads>, each either a number or "auto", or plain "auto"))
endif

# Per-dimension defs: "auto" hands the dimension to codegen and leaves the
# common.h default in place for the simulations, a number pins it.
grid_teams_defs   = $(if $(filter auto,$(1)),-DCODEGEN_AUTODETECT_TEAMS=1,-DCODEGEN_AUTODETECT_TEAMS=0 -DXTEAM_NUM_TEAMS=$(1))
grid_threads_defs = $(if $(filter auto,$(1)),-DCODEGEN_AUTODETECT_THREADS=1,-DCODEGEN_AUTODETECT_THREADS=0 -DXTEAM_NUM_THREADS=$(1))

# Derive the defs once per grid rather than per (op, label, grid) combination.
define GRID_DEFS_RULE
ifeq ($(1),auto)
GRID_DEFS_$(1) := -DCODEGEN_AUTODETECT_TEAMS=1 -DCODEGEN_AUTODETECT_THREADS=1
else
GRID_DEFS_$(1) := $(call grid_teams_defs,$(call grid_teams,$(1))) \
                  $(call grid_threads_defs,$(call grid_threads,$(1)))
endif
endef
$(foreach G,$(GRIDS),$(eval $(call GRID_DEFS_RULE,$(G))))

# ── Active ops and labels ───────────────────────────────────────────────────
# Active labels are those with a non-empty CXX_<label>.
LABELS = $(strip $(foreach L,$(ALL_LABELS),$(if $(strip $(CXX_$(L))),$(L))))
# Active ops are those appearing in OPS_<label> for at least one active label.
OPS = $(strip $(foreach O,$(ALL_OPS),$(if $(filter $(O),$(foreach L,$(LABELS),$(OPS_$(L)))),$(O))))

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

BINARIES = $(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(foreach G,$(GRIDS),$(O)_$(L)_$(G))))

# ── Targets ─────────────────────────────────────────────────────────────────
.PHONY: all clean help format \
        $(LABELS) \
        $(OPS) \
        $(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(O)_$(L)))

all: $(BINARIES)

# Per-(op, label, grid) build rule. Compiles inside out/<op>_<label>_<grid>/
# so -save-temps=obj keeps intermediates there, then symlinks the binary to
# the top dir for convenience.
define BUILD_RULE
$(1)_$(2)_$(3): $(SRC_DIR)/xteam_bench.cpp \
                $(SRC_DIR)/xteam_$(1).cpp \
                $(SRC_DIR)/xteam_simulations_$(SIM_$(2)).h \
                $(COMMON_HEADERS)
	@test -n "$$(CXX_$(2))" || { echo "ERROR: CXX_$(2) is not set"; exit 1; }
	@echo "Building $(1) for $(2) on grid $(3) ..."
	rm -rf out/$(1)_$(2)_$(3)
	mkdir -p out/$(1)_$(2)_$(3)
	cd out/$(1)_$(2)_$(3) && $$(CXX_$(2)) $$(DEFS_$(2)) $$(FLAGS_$(2)_$(1)) \
		-DCOMPILER_LABEL='"$(2)"' \
		-DXTEAM_SIM_HEADER='"xteam_simulations_$(SIM_$(2)).h"' \
		$$(GRID_DEFS_$(3)) \
		-o $$@ ../../$(SRC_DIR)/xteam_bench.cpp ../../$(SRC_DIR)/xteam_$(1).cpp
	ln -sf out/$(1)_$(2)_$(3)/$$@ $$@
	cd out/$(1)_$(2)_$(3) && $$(dir $$(CXX_$(2)))llvm-dis *.bc
endef
$(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(foreach G,$(GRIDS),$(eval $(call BUILD_RULE,$(O),$(L),$(G))))))

# Convenience: `make <op>_<label>` builds all configured grids for that op and
# that label.
define OP_LABEL_RULE
$(1)_$(2): $(foreach G,$(GRIDS),$(1)_$(2)_$(G))
endef
$(foreach O,$(OPS),$(foreach L,$(LABELS_$(O)),$(eval $(call OP_LABEL_RULE,$(O),$(L)))))

# Convenience: `make <op>` builds all supported (label, grid) combinations for
# that op.
define OP_RULE
$(1): $(foreach L,$(LABELS_$(1)),$(foreach G,$(GRIDS),$(1)_$(L)_$(G)))
endef
$(foreach O,$(OPS),$(eval $(call OP_RULE,$(O))))

# Convenience: `make <label>` builds all supported (op, grid) combinations for
# that label.
define LABEL_RULE
$(1): $(foreach O,$(OPS_$(1)),$(foreach G,$(GRIDS),$(O)_$(1)_$(G)))
endef
$(foreach L,$(LABELS),$(eval $(call LABEL_RULE,$(L))))

format:
	clang-format -i $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h

clean:
	rm -rf $(BINARIES) out

help:
	@echo "xteam benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all                     Build all configured (op × compiler/label × grid) combinations"
	@echo "  <op>                    Build <op>_<label>_<grid> for every (label, grid) combination supported by that op"
	@echo "                            (e.g. make red)"
	@echo "  <label>                 Build <op>_<label>_<grid> for every (op, grid) combination supported by that label"
	@echo "                            (e.g. make aomp)"
	@echo "  <op>_<label>            Build <op>_<label>_<grid> for every configured grid supported by that (op, label) pair"
	@echo "                            (e.g. make red_aomp)"
	@echo "  <op>_<label>_<grid>     Build a specific (op, compiler, grid) combination"
	@echo "                            (e.g. make red_aomp_208x512)"
	@echo "  clean                   Remove binaries and object files"
	@echo "  format                  Format the code using clang-format"
	@echo ""
	@echo "Variables:"
	@echo "  CXX_<label>      Compiler path  (e.g. CXX_aomp=/path/to/clang++)"
	@echo "  FLAGS_<label>    Extra flags    (e.g. FLAGS_aomp=-g)"
	@echo "  DEFS_<label>     Extra -D defs"
	@echo "  OPS_<label>      Ops supported by this compiler"
	@echo "  GRIDS            Grids to build, as <teams>x<threads> (either may be \"auto\"; all-auto is \"auto\")"
	@echo "  OFFLOAD_ARCH     GPU arch"
	@echo ""
	@echo "Currently active ops:     $(OPS)"
	@echo "Currently active labels:  $(LABELS)"
	@echo "Configured grids:         $(GRIDS)"
	@echo "Configured OFFLOAD_ARCH:  $(OFFLOAD_ARCH)"
