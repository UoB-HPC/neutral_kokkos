
default: neutral.kokkos

include $(KOKKOS_PATH)/Makefile.kokkos

ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GCC).
Available compilers are:
  GCC INTEL

endef
$(info $(compiler_help))
COMPILER=GCC
endif

COMPILER_GCC = g++ -ffast-math -ffp-contract=fast
COMPILER_INTEL = icpc -qopenmp -no-prec-div
COMPILER_CLANG = clang++ -ffast-math -ffp-contract=fast

CXX = $(COMPILER_$(COMPILER))

ifndef TARGET
define target_help
Set TARGET to change to offload device. Defaulting to CPU.
Available targets are:
  CPU (default)
  GPU
endef
$(info $(target_help))
TARGET=CPU
endif

ifeq ($(TARGET), GPU)
CXX = $(NVCC_WRAPPER)
endif

OBJ = main.o params.o profiler.o comms.o shared_kokkos.o shared.o data.o mesh.o shared_data.o halos.o neutral.o neutral_data.o

neutral.kokkos: $(OBJ) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_LDFLAGS) -DKOKKOS -O3 $(EXTRA_FLAGS) $(OBJ) $(KOKKOS_LIBS) -o $@

%.o: %.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DKOKKOS -O3 $(EXTRA_FLAGS) -c $<

.PHONY: clean
clean:
	rm -f neutral.kokkos main.o params.o profiler.o comms.o shared_kokkos.o shared.o data.o mesh.o shared_data.o halos.o neutral.o neutral_data.o

