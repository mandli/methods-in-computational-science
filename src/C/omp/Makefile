CC ?= gcc
LINK = $(CC)
CFLAGS ?= -fopenmp
LFLAGS ?= $(CFLAGS)

SRC = hello_world.c yeval.c fine_grain.c coarse_grain.c
OBJECTS = $(subst .c,.o,$(SRC))
EXE = $(subst .c, ,$(SRC))

# Default rules
%.o : %.c ; $(CC) $(CFLAGS) -c $< -o $@

all: $(EXE)

hello_world: hello_world.o
	$(LINK) $(LFLAGS) $< -o $@

yeval: yeval.o
	$(LINK) $(LFLAGS) $< -o $@

fine_grain: fine_grain.o
	$(LINK) $(LFLAGS) $< -o $@

coarse_grain: coarse_grain.o
	$(LINK) $(LFLAGS) $< -o $@

clean:
	-rm -f $(EXE)
	-rm -f $(OBJECTS)

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###