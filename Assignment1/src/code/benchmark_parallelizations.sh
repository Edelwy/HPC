#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=seam_carving
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=seam_carving.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding.  
export OMP_PLACES=cores
export OMP_PROC_BIND=close

methods=( seam-carving seam-carving-omp )
source_file=seam-carving.c
infiles=../test_images
outfile="$1".csv
repeats=5
seams=128
thread_counts=(1 2 4 8 16 32)

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
	echo "File: ${methods[0]}.out (sequential)"
	gcc -O3 --openmp "${source_file}" -o "${methods[0]}.out" -lm -lnuma

	echo "File: ${methods[1]}.out (OpenMP)"
	gcc -O3 --openmp -DUSE_OMP "${source_file}" -o "${methods[1]}.out" -lm -lnuma
fi

cpus_cap="${SLURM_CPUS_PER_TASK:-}"

echo Preparing file...
echo '"Method","Threads","Attempt","Image","Time"' > "${outfile}"

echo Testing with OpenMP and without...
for method in "${methods[@]}"; do
	if [ "$method" = "seam-carving" ]; then
		thread_list=(1)
	else
		thread_list=("${thread_counts[@]}")
	fi

	for threads in "${thread_list[@]}"; do
		[ -n "${cpus_cap}" ] && [ "${threads}" -gt "${cpus_cap}" ] && continue
		export OMP_NUM_THREADS="${threads}"

		for file in $(find "${infiles}" -type f); do
			for ((attempt = 0; attempt < repeats; attempt++)); do
				echo Running attempt ${attempt} ${method} Threads=${threads} "${file##*/}"
				echo -n "${method},${threads},${attempt},${file##*/}," >> "${outfile}"
				out=$(./"${method}.out" "${file}" "${file}-out" ${seams} | grep "Total time:")
				echo "${out}"
				echo "${out##Total time:}" >> "${outfile}"
			done
		done
	done
done

