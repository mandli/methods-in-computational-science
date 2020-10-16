# MPI Fortran Demos
FC = gfortran
MPI_FC = mpif90
LINK = $(FC)
MPI_LINK = $(MPI_FC)

SRC = compute_pi.f90 hello_world.f90 jacobi.f90 note_passing.f90
OBJECTS = $(subst .f90,.o,$(SRC))

# Default fortran rules
%.o : %.f90 ; $(FC) -c $< -o $@ $(FFLAGS)

.PHONY: all clean new

all: hello_world note_passing compute_pi jacobi

hello_world: hello_world.o
	$(MPI_LINK) -o $@ $^

note_passing: note_passing.o
	$(MPI_LINK) -o $@ $^

compute_pi: compute_pi.o
	$(MPI_LINK) -o $@ $^

jacobi: jacobi.o
	$(MPI_LINK) -o $@ $^

clean:
	-rm -f $(OBJECTS)

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###