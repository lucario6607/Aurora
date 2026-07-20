EXE := aurora
BUILD_OPTIONS := -march=x86-64-v3 -O3 -std=c++17 -Wno-deprecated-declarations

# NNUE architecture width (hidden neurons). The net arch is compile-time, so a
# branch that tunes the 1024 net must build with HIDDEN=1024.
# ob-1024 branch: default to the 1024 net for OpenBench SPSA co-tuning.
HIDDEN := 1024
BUILD_OPTIONS += -DNNUE_HIDDEN=$(HIDDEN)

# OpenBench supplies the network via `make EVALFILE=<path>`. When set, embed it
# via INCBIN (NNUE_FILE); otherwise the default net baked into evaluation.h is used.
ifneq ($(EVALFILE),)
    BUILD_OPTIONS += -DNNUE_FILE=\"$(EVALFILE)\"
endif

ifneq ($(exe),)
    EXE := $(exe)
endif

ifeq ($(OS),Windows_NT)
    override EXE := $(EXE).exe
endif

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_HASH := $(GIT_BRANCH)-$(shell git rev-parse --short HEAD)
GIT_DIFF := $(shell git diff --shortstat)

ifneq ($(GIT_DIFF),)
    GIT_TREE_HASH := $(shell git rev-parse --short=7 $(word 1, $(shell git add -u && git write-tree && git reset)))
    override GIT_HASH := $(GIT_HASH)-$(GIT_TREE_HASH)-dirty
endif

build:
	clang++ aurora.cpp external/Fathom-1.0/src/tbprobe.cpp -o $(EXE) -DGIT_HASH=\"$(GIT_HASH)\" $(BUILD_OPTIONS)
bench: build
	./$(EXE) bench
hash:
	@echo $(GIT_HASH)
dev:
	clang++ aurora.cpp external/Fathom-1.0/src/tbprobe.cpp -o $(EXE) -DGIT_HASH=\"$(GIT_HASH)\" $(BUILD_OPTIONS) -DDEV