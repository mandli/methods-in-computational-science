# Matrix Multiply Example and others
CC = gcc
LINK = $(CC)

# GFortran flags
CFLAGS ?= -O3 -fopenmp
# Intel Flags
# FFLAGS = -O3 -openmp

NAME = mat_mult
TEST_ITER = 1000

# Uncomment the appropriate flags for your platform
# Linux
LFLAGS = $(CFLAGS) -lblas
# Mac OS X framework specific, flag above also works for Mac OS X
# LFLAGS = $(CFLAGS) -framework veclib

# Default fortran rules
%.o : %.f90 ; $(CC) -c $< -o $@ $(CFLAGS)

# Beginning Demo
mat_mult: mod_rand.o matrix_multiply.o 
	$(LINK) $(LFLAGS) $^ -o $@

mat_mult_test: mat_mult
	-echo "Non-threaded Tests:"
	./mat_mult $(TEST_ITER) 1 1
	./mat_mult $(TEST_ITER) 2 1
	./mat_mult $(TEST_ITER) 3 1
	./mat_mult $(TEST_ITER) 4 1
	-echo "Threaded Tests:"
	./mat_mult $(TEST_ITER) 1 4
	./mat_mult $(TEST_ITER) 2 4
	./mat_mult $(TEST_ITER) 3 4
	./mat_mult $(TEST_ITER) 4 4

# Call subdirectory targets
all:
	-cd cuda
	$(MAKE) all
	-cd ../mpi
	$(MAKE) all
	-cd ../omp
	$(MAKE) all
	-cd ..

clean:
	-cd cuda
	$(MAKE) clean
	-cd ../mpi
	$(MAKE) clean
	-cd ../omp
	$(MAKE) clean
	-cd ..

new:
	-cd cuda
	$(MAKE) new
	-cd ../mpi
	$(MAKE) new
	-cd ../omp
	$(MAKE) new
	-cd ..

### DO NOT remove this line - make depends on it ###