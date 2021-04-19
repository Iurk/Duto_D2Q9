# MakeFile

# Python Scripts
PYMESH = python/mesh.py
PYDADOS = python/data_process.py

# Program Name
EXE = lbm

# Compilers
NVCC := nvcc

LIBRARY := yaml-cpp
LIBDIR := /usr/local/lib

# GPU
ARCH = sm_61

# Directories
SRCDIR := src
CUMINPACK_DIR := src/cuminpack
INCDIR := inc inc/cuminpack /usr/local/include
OBJDIR := obj
BINDIR := bin
VELOCITY := bin/Velocity
PRESSURE := bin/Pressure
RESULTS := bin/Results
MESH := bin/Mesh

# Files
C_FILES := $(wildcard $(SRCDIR)/*.c)
CU_FILES := $(wildcard $(SRCDIR)/*.cu)
CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
CUMINPACK := $(wildcard $(CUMINPACK_DIR)/*.cu)

C_OBJ := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(C_FILES))
CU_OBJ := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CU_FILES))
CPP_OBJ := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPP_FILES))
CUMINPACK_OBJ := $(patsubst $(CUMINPACK_DIR)/%.cu, $(OBJDIR)/%.o, $(CUMINPACK))

C_DEP := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.d, $(C_FILES))
CU_DEP := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.d, $(CU_FILES))
CPP_DEP := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.d, $(CPP_FILES))
CUMINPACK_DEP := $(patsubst $(CUMINPACK_DIR)/%.cu, $(OBJDIR)/%.d, $(CUMINPACK))

OBJ := $(CUMINPACK_OBJ) $(C_OBJ) $(CPP_OBJ) $(CU_OBJ)
DEP := $(CUMINPACK_DEP) $(C_DEP) $(CPP_DEP) $(CU_DEP)

# Flags
NVCCARCHFLAG := -arch sm_61
NVCCFLAGS := -std=c++11 -v --ptxas-options=-v -O3 --device-c # Para debug flag -g -G e executar com cuda-memcheck ./lbm |more
LDFLAGS := $(addprefix -L, $(LIBDIR))
DEPFLAGS := -MMD

INCLUDES := $(addprefix -I, $(INCDIR))
LIBRARIES := $(addprefix -l, $(LIBRARY))

ALL_CPFLAGS := $(DEPFLAGS)
ALL_CPFLAGS += $(NVCCARCHFLAG)
ALL_CPFLAGS += $(NVCCFLAGS)

ALL_LDFLAGS := $(LDFLAGS)

CXXFLAGS := -std=c++11
NVCCARCHFLAG := -arch $(ARCH)
NVCCFLAGS := -v --ptxas-options=-v -O3 # Para debug flag -g -G e executar com cuda-memcheck ./lbm |more

COMPILE.c = $(NVCC) $(DEPFLAGS) -g $(INCLUDES) -c
COMPILE.cpp = $(NVCC) $(ALL_CPFLAGS) $(INCLUDES) --device-c
COMPILE.cu = $(NVCC) $(ALL_CPFLAGS) $(INCLUDES) --device-c
COMPILECUMINPACK.cu = $(NVCC) -MMD $($NVCCARCHFLAG) -std=c++11 -O3 $(INCLUDES) --device-c

.PHONY: all clean mesh plot refresh run

all: $(EXE)

# Linkage
$(EXE): $(OBJ)
	$(NVCC) $(NVCCARCHFLAG) $(INCLUDES) $(ALL_LDFLAGS) $^ -o $@ $(LIBRARIES)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(COMPILE.c) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(COMPILE.cpp) $< -o $@

$(OBJDIR)/%.o: $(CUMINPACK_DIR)/%.cu | $(OBJDIR)
	$(COMPILECUMINPACK.cu) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(COMPILE.cu) $< -o $@

$(OBJDIR):
	@mkdir -p $@

# Cleaning up
clean:
	@echo Cleaning up...
	@rm -f -r $(OBJDIR)
	@rm -f -r $(VELOCITY)/*
	@rm -f -r $(PRESSURE)/*
	@rm -f -r $(RESULTS)/*
	@rm -f -r $(MESH)/*
	@rm -f $(EXE)
	@rm -f $(BINDIR)/*.gif
	@rm -f $(BINDIR)/*.mp4
	@echo Done!

mesh:
	@echo Generating mesh...
	@python $(PYMESH)
	@echo Done!

plot:
	@python $(PYDADOS)

refresh:
	@echo Cleaning up the images...
	@rm -f -r $(VELOCITY)/*
	@rm -f -r $(PRESSURE)/*
	@echo Cleaning up the results...
	@rm -f -r $(RESULTS)/*
	@rm -f $(BINDIR)/*.gif
	@rm -f $(BINDIR)/*.mp4
	@echo Cleaning up the mesh...
	@rm -f -r $(MESH)/*
	@echo Done!

# Running
run:
	@echo Generating mesh...
	@python $(PYMESH)
	
	@echo Running simulation...
	@./$(EXE)

	@echo Running Python Script...
	@python $(PYDADOS)

# Including dependency
-include $(DEP)