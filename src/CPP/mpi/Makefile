# MPI C++ Demos
CXX = g++
MPI_CXX = mpic++
LINK = $(CXX)
MPI_LINK = $(MPI_CXX)
CFLAGS = 

SRC = hello_world.cpp
OBJECTS = $(subst .c,.o,$(SRC))
EXE = $(subst .c, ,$(SRC))

# Default C rules
%.o : %.cpp ; $(MPI_CXX) -c $< -o $@ $(CFLAGS)

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
	-rm -f jacobi_*.txt jacobi.png

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###