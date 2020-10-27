# OpenMP Fortran Demos
FC ?= gfortran
LINK ?= $(FC)
FFLAGS ?= $(OPENMP)
LFLAGS ?= $(FFLAGS)

SRC = coarse_grain.f90 demo2.f90 fine_grain.f90 hello_world.f90 jacobi1.f90 jacobi2.f90 yeval.f90
OBJECTS = $(subst .f90,.o,$(SRC))
EXE = $(subst .f90, ,$(SRC))

# Default fortran rules
%.o : %.f90 ; $(FC) $(FFLAGS) -c $< -o $@ 

.PHONY: all clean new

all: $(EXE)

coarse_grain: coarse_grain.o
	$(LINK) $(LFLAGS) -o $@ $<

demo2: demo2.o
	$(LINK) $(LFLAGS) -o $@ $<

fine_grain: fine_grain.o
	$(LINK) $(LFLAGS) -o $@ $<

hello_world: hello_world.o
	$(LINK) $(LFLAGS) -o $@ $<

jacobi1: jacobi1.o
	$(LINK) $(LFLAGS) -o $@ $<

jacobi2: jacobi2.o
	$(LINK) $(LFLAGS) -o $@ $<

yeval: yeval.o
	$(LINK) $(LFLAGS) -o $@ $<

clean:
	-rm -f $(OBJECTS)
	-rm -f $(EXE)

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###