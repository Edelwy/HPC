#!/bin/bash
progs=( sample sample-omp )
infiles=../test_images
outfile=results.csv
repeats=5
seams=128

echo Compiling...
for file in ${progs[@]}; do
	echo ${file}
#	gcc -O3 --openmp ${file}.c -o ${file} -lm -lnuma
done

echo Preparing file...
echo "Method","Attempt","Image","Time"\;>${outfile}


echo Testing...
for method in ${progs[@]}; do
	for file in $(find $infiles -type f); do
		for ((attempt=0; attempt<${repeats}; attempt++)); do
			echo Running test attempt $attempt with method ${method} on file ${file##*/}
			echo -n ${method},${attempt},${file##*/},>>${outfile}
			time=$(./${method} "${file}" "${file}-out" ${seams} | grep "Total time:")
			echo $time
			echo ${time##Total time:}\;>>${outfile}
		done
	done
done
