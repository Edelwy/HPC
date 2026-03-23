#!/bin/bash

prog=$1
image=$2
seams=$3

compile=0

# Optional arguments to make benchmark simpler for compiling and cleanup.
for arg in "$@"; do
	if [ "$arg" = "--compile" ]; then
		compile=1
	fi
done

if [ "$compile" = "1" ]; then
	# Load the numactl module to enable numa library linking.
	module load numactl
	echo Compiling...
	#gcc -O3 --openmp "${prog}.c" -o "${prog}.out" -lm -lnuma
	# Compile for debugging
	gcc -g -O0 --openmp "${prog}.c" -o "${prog}.out" -lm -lnuma
fi

echo Testing...
srun  "${prog}.out" "${image}" ../test_images/test.png "${seams}"


