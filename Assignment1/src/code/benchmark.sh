#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=seam_carving
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=seam_carving.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding.  
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

progs=( seam-carving seam-carving-omp )
infiles=../test_images
outfile=results.csv
repeats=5
seams=128

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
	for file in "${progs[@]}"; do
		echo "File: ${file}"
		gcc -O3 --openmp "${file}.c" -o "${file}.out" -lm -lnuma
	done
fi

echo Preparing file...
echo "Method","Attempt","Image","Time"\;>${outfile}


echo Testing...
for method in ${progs[@]}; do
	for file in $(find $infiles -type f); do
		for ((attempt=0; attempt<${repeats}; attempt++)); do
			echo Running test attempt $attempt with method ${method} on file ${file##*/}
			echo -n ${method},${attempt},${file##*/},>>${outfile}
			time=$(./"${method}.out" "${file}" "${file}-out" ${seams} | grep "Total time:")
			echo $time
			echo ${time##Total time:}\;>>${outfile}
		done
	done
done

