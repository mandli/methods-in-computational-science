FC = gfortran
CC = gcc-10
LINK = $(CC)
# GFortran flags
FFLAGS = -fopenmp
CFLAGS = -fopenmp

# Default rules
%.o : %.f90 ; $(FC) -c $< -o $@ $(FFLAGS)
%.o : %.c ; $(CC) -c $< -o $@ $(CFLAGS)

# all: 

# Hello world
hello_world: hello_world_c hello_world_f

hello_world_c: hello_world.c


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



clean:
	-rm -f mod_rand.o mod_rand.mod matrix_multiply.o mat_mult
	-rm -f yeval.o yeval
	-rm -f demo2.o demo2
	-rm -f fine_grain.o fine_grain
	-rm -f coarse_grain.o coarse_grain
	-rm -f hello_world_omp.o hello_world_omp
	-rm -f hello_world_mpi.o hello_world_mpi
	-rm -f compute_pi.o compute_pi
	-rm -f note_passing.o note_passing
	-rm -f jacobi_omp1.o jacobi_omp1 jacobi_omp1.txt
	-rm -f jacobi_omp2.o jacobi_omp2 jacobi_omp2.txt
	-rm -f jacobi_mpi.o jacobi_mpi jacobi_mpi.txt

new:
	$(MAKE) clean
	$(MAKE) all


### DO NOT remove this line - make depends on it ###