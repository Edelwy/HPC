# Build:  make METHOD=<seq|row|block|row_wide>     (default: row)
# Clean:  make clean

METHOD ?= row

ifeq ($(METHOD),seq)
    CC      = gcc
    MAIN    = main_seq.c
    DRIVER_OBJ = main_seq.o
else
    CC      = mpicc
    MAIN    = main_mpi.c
    DRIVER_OBJ = main_mpi.o
endif

LENIA_SRC = lenia_$(METHOD).c
LENIA_OBJ = lenia_$(METHOD).o

CFLAGS  = -O3 -Wall -std=c99
LDLIBS  = -lm

OBJS    = $(DRIVER_OBJ) $(LENIA_OBJ) lenia_common.o orbium.o gifenc.o
TARGET  = lenia.out

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)

.PHONY: clean
