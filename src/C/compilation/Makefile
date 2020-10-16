SRC = hello_world.c
EXE = hello_world
OBJECTS = $(subst .c,.o, $(SRC))
PPC_SRC = $(subst .c,.pcc, $(SRC))
ASSEMBLED_SRC = $(subst .c,.s, $(SRC))

%.pcc: %.c
	gcc -E $< > $@

%.s: %.c
	gcc -S $<

%.o: %.c
	gcc -c $<

$(EXE): $(OBJECTS)
	gcc -o $@ $(SRC)

.PHONY: preprocess compile assemble link clean debug

preprocess: $(PPC_SRC)
	@echo $<

compile:  $(ASSEMBLED_SRC)
	@echo $<

assemble:  $(OBJECTS)
	@echo $<

link: $(EXE)
	@echo $<

clean:
	-rm -f $(EXE)
	-rm -f *.pcc *.s *.o

debug:
	@echo $(SRC)
	@echo $(subst .c, .o, $(SRC))