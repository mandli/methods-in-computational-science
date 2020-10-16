# OpenMP Fortran Demos
FC = gfortran
LINK = $(FC)
# GFortran flags
FFLAGS = -fopenmp

SRC = coarse_grain.f90 demo2.f90 fine_grain.f90 hello_world.f90 jacobi1.f90 jacobi2.f90 yeval.f90
OBJECTS = $(subst .f90,.o,$(SRC))

# Default fortran rules
%.o : %.f90 ; $(FC) -c $< -o $@ $(FFLAGS)

.PHONY: all clean new

all: coarse_grain demo2 fine_grain hello_world jacobi1 jacobi2 yeval

coarse_grain: coarse_grain.o
	$(MPI_LINK) -o $@ $^

demo2: demo2.o
	$(MPI_LINK) -o $@ $^

fine_grain: fine_grain.o
	$(MPI_LINK) -o $@ $^

hello_world: hello_world.o
	$(MPI_LINK) -o $@ $^

jacobi1: jacobi1.o
	$(MPI_LINK) -o $@ $^

jacobi2: jacobi2.o
	$(MPI_LINK) -o $@ $^

yeval: yeval.o
	$(MPI_LINK) -o $@ $^

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###