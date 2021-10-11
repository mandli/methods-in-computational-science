CXX = g++
LINK = $(CXX)
CFLAGS ?= -fopenmp
LFLAGS ?= $(CFLAGS)

SRC = hello_world.cpp
OBJECTS = $(subst .cpp,.o,$(SRC))
EXE = $(subst .cpp, ,$(SRC))

# Default rules
%.o : %.cpp ; $(CXX) $(CFLAGS) -c $< -o $@

all: $(EXE)

hello_world: hello_world.o
	$(LINK) $(LFLAGS) $< -o $@

clean:
	-rm -f $(EXE)
	-rm -f $(OBJECTS)
	-rm -f *.txt *.png

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###