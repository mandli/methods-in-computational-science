# MPI Fortran Demos
FC ?= gfortran
LINK ?= $(FC)
MPI_FC ?= mpif90
MPI_LINK ?= $(MPI_FC)
FFLAGS ?= 
LFLAGS ?= $(FFLAGS)

SRC = compute_pi.f90 hello_world.f90 jacobi.f90 note_passing.f90
OBJECTS = $(subst .f90,.o,$(SRC))
EXE = $(subst .f90, ,$(SRC))

# Default fortran rules
%.o : %.f90 ; $(FC) $(FFLAGS) -c $< -o $@

.PHONY: all clean new

all: $(EXE)

hello_world: hello_world.o
	$(MPI_LINK) $(LFLAGS) -o $@ $^

note_passing: note_passing.o
	$(MPI_LINK) $(LFLAGS) -o $@ $^

compute_pi: compute_pi.o
	$(MPI_LINK) $(LFLAGS) -o $@ $^

jacobi: jacobi.o
	$(MPI_LINK) $(LFLAGS) -o $@ $^

clean:
	-rm -f $(EXE)
	-rm -f $(OBJECTS)

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###