# MPI C Demos
CC = gcc
MPI_CC = mpicc
LINK = $(CC)
MPI_LINK = $(MPI_CC)
CFLAGS = 

SRC = compute_pi.c hello_world.c jacobi.c note_passing.c
OBJECTS = $(subst .c,.o,$(SRC))
EXE = $(subst .c, ,$(SRC))

# Default fortran rules
%.o : %.c ; $(MPI_CC) -c $< -o $@ $(CFLAGS)

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
	-rm -f $(EXE)
	-rm -f $(OBJECTS)

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###