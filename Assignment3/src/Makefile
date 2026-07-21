# Build:  make METHOD=<seq|omp|gpu|cells|opt>
# Clean:  make clean

METHOD    ?= seq # default method is sequential.
BLOCKSIZE ?= 128 # default block size is 128.

ifeq ($(METHOD),seq) 
    CC     = gcc
    CFLAGS = -O3 -Wall -std=c99
    VARSRC = lennard_jones_seq.c # source code for sequential implementation.
else ifeq ($(METHOD),omp)
    CC     = gcc
    CFLAGS = -O3 -Wall -std=c99 -fopenmp
    VARSRC = lennard_jones_omp.c # source code for OpenMP implementation.
else
    CC     = nvcc
    CFLAGS = -O3 -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -DBLOCKSIZE=$(BLOCKSIZE) -Xcompiler -Wall
    VARSRC = lennard_jones_$(METHOD).cu # source code for CUDA implementation.
endif

LDLIBS = -lm 
TARGET = lj.out
OBJS   = main.o lennard_jones_common.o gifenc.o lennard_jones_$(METHOD).o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

lennard_jones_common.o: lennard_jones_common.c
	$(CC) $(CFLAGS) -c lennard_jones_common.c -o lennard_jones_common.o

gifenc.o: gifenc.c
	$(CC) $(CFLAGS) -c gifenc.c -o gifenc.o

lennard_jones_$(METHOD).o: $(VARSRC)
	$(CC) $(CFLAGS) -c $(VARSRC) -o lennard_jones_$(METHOD).o

clean:
	rm -f *.o $(TARGET)

.PHONY: clean 
