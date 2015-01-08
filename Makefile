SRC := src
INCLUDE := include
BUILD := build
TARGET := $(BUILD)/auto-encoder
LIB_LBFGS := extern/liblbfgs

CXX := g++-4.9 -std=c++11 -fdiagnostics-color=auto

CFLAGS := -Wall -Werror -O3 -g -fopenmp
IFLAGS := -I$(INCLUDE) -I$(LIB_LBFGS)/include
DFLAGS :=
LFLAGS := -lopenblas $(LIB_LBFGS)/lib/lbfgs.o

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	override JOBS ?= $(shell nproc)
	DFLAGS := $(DFLAGS) -DLINUX
endif

ifeq ($(UNAME_S),Darwin)
	# Not sure why the following line doesn't work
	# override JOBS ?= $(shell sysctl hw.ncpu | awk '{print $2}')
	# Hard-code the number of cores on Mac OSX
	JOBS ?= 4
	DFLAGS := $(DFLAGS) -DDARWIN
endif

override V ?= @

ifdef MATLAB_HOME
IFLAGS := $(IFLAGS) -I$(MATLAB_HOME)/extern/include
DFLAGS := $(DFLAGS) -DDUMP_MATFILE
LFLAGS := $(LFLAGS) -L$(MATLAB_HOME)/bin/maci64 -lmx -lmat
endif

ifdef OPENBLAS_HOME
IFLAGS := -I$(OPENBLAS_HOME)/include $(IFLAGS)
LFLAGS := -L$(OPENBLAS_HOME)/lib $(LFLAGS)
endif

OBJ_FILES := $(patsubst %.cpp, $(BUILD)/%.o, $(wildcard $(SRC)/*.cpp))
EXE_FILES := $(TARGET)
DEP_FILES := $(OBJ_FILES:.o=.d) $(patsubst %, %.d, $(EXE_FILES))

all: liblbfgs
	@make -j $(JOBS) link

-include $(DEP_FILES)

$(BUILD)/$(SRC)/%.o: $(SRC)/%.cpp
	@mkdir -pv $(dir $@)
	@echo "[cxx] $<"
	@$(CXX) $(DFLAGS) $(IFLAGS) -MM -MT "$@" "$<"  > "$(@:.o=.d)"
	$(V) $(CXX) -o $@ -c $<  $(CFLAGS) $(DFLAGS) $(IFLAGS)

link: $(OBJ_FILES)
	@echo "[link] $(TARGET)"
	$(V) $(CXX) -o $(TARGET) $(OBJ_FILES) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(LFLAGS)

extern:
	@if [ ! -e $(LIB_LBFGS)/include/lbfgs.h ]; then \
		git submodule update --init --recursive; \
	fi;

liblbfgs: extern
	@if [ ! -e $(LIB_LBFGS)/lib/lbfgs.o ]; then \
		cd $(LIB_LBFGS); \
		./autogen.sh; \
		./configure --enable-sse2; \
		make; \
	fi;

clean:
	@printf "[clean] "
	rm -rf $(BUILD)

.PHONY: all link extern liblbfgs clean
