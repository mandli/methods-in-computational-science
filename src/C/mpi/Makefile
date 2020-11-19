# MPI C Demos
CC = gcc
MPI_CC = mpicc
LINK = $(CC)
MPI_LINK = $(MPI_CC)
CFLAGS = 

SRC = compute_pi.c hello_world.c jacobi.c note_passing.c jacobi_2d.c
OBJECTS = $(subst .c,.o,$(SRC))
EXE = $(subst .c, ,$(SRC))

# Default C rules
%.o : %.c ; $(MPI_CC) -c $< -o $@ $(CFLAGS)

.PHONY: all clean new

all: hello_world note_passing compute_pi jacobi jacobi_2d

hello_world: hello_world.o
	$(MPI_LINK) -o $@ $^

note_passing: note_passing.o
	$(MPI_LINK) -o $@ $^

compute_pi: compute_pi.o
	$(MPI_LINK) -o $@ $^

jacobi: jacobi.o
	$(MPI_LINK) -o $@ $^

jacobi_2d: jacobi_2d.o
	$(MPI_LINK) -o $@ $^	

clean:
	-rm -f $(EXE)
	-rm -f $(OBJECTS)
	-rm -f jacobi_*.txt jacobi.png

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###