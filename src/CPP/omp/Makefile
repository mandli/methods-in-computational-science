CXX = g++
LINK = $(CXX)
CFLAGS ?= -fopenmp
LFLAGS ?= $(CFLAGS)

SRC = hello_world.cpp \
	yeval.cpp \
	fine_grain.cpp \
	coarse_grain.cpp \
	jacobi_fine.cpp

OBJECTS = $(subst .cpp,.o,$(SRC))
EXE = $(subst .cpp, ,$(SRC))

# Default rules
%.o : %.cpp ; $(CXX) $(CFLAGS) -c $< -o $@

all: $(EXE)

hello_world: hello_world.o
	$(LINK) $(LFLAGS) $< -o $@

yeval: yeval.o
	$(LINK) $(LFLAGS) $< -o $@

fine_grain: fine_grain.o
	$(LINK) $(LFLAGS) $< -o $@

coarse_grain: coarse_grain.o
	$(LINK) $(LFLAGS) $< -o $@

jacobi_fine: jacobi_fine.o
	$(LINK) $(LFLAGS) $< -o $@

clean:
	-rm -f $(EXE)
	-rm -f $(OBJECTS)
	-rm -f *.txt *.png

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###