# Matrix Multiply Example and others
CC ?= gcc
LINK ?= $(CC)
CFLAGS ?= -O3 -fopenmp -g
SRC = matrix_multiply.c
OBJ = $(subst .c,.o,$(SRC))
EXE = mat_mult

TEST_ITER = 1000

# Uncomment the appropriate flags for your platform# Linux
LFLAGS ?= $(CFLAGS) -lblas
# Mac OS X framework specific, flag above also works for Mac OS X
# LFLAGS = $(CFLAGS) -framework veclib

%.o : %.c ; $(CC) $(CFLAGS) -c $< -o $@ 

# Beginning Demo
mat_mult: matrix_multiply.o 
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
	-rm -f matrix_multiply.o mat_mult

new: clean mat_mult

# Call subdirectory targets


### DO NOT remove this line - make depends on it ###