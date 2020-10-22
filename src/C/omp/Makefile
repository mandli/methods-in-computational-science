CC = gcc-10
LINK = $(CC)
CFLAGS ?= -fopenmp
LFLAGS ?= $(CFLAGS)
SRC = hello_world.c yeval.c fine_grain.c coarse_grain.c
OBJECTS = $(subst .c,.o,$(SRC))

# Default rules
%.o : %.c ; $(CC) -c $< -o $@ $(CFLAGS)

all: hello_world yeval fine_grain coarse_grain

hello_world: hello_world.o
	$(LINK) $(LFLAGS) $< -o $@

yeval: yeval.o
	$(LINK) $(LFLAGS) $< -o $@

fine_grain: fine_grain.o
	$(LINK) $(LFLAGS) $< -o $@

coarse_grain: coarse_grain.o
	$(LINK) $(LFLAGS) $< -o $@

clean:
	-rm -f hello_world yeval fine_grain coarse_grain
	-rm -f $(OBJECTS)

new:
	$(MAKE) clean
	$(MAKE) all

### DO NOT remove this line - make depends on it ###