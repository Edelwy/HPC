#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=seam_carving_opt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=seam_carving_opt.log
#SBATCH --hint=nomultithread

export OMP_PLACES=cores
export OMP_PROC_BIND=close

source_file=seam-carving.c
binary=seam-carving-omp-opt.out
infiles=../test_images
outfile="$1".csv
repeats=5
seams=128
thread_counts=(1 2 4 8 16 32)

compile=0
for arg in "$@"; do
	[ "$arg" = "--compile" ] && compile=1
done

if [ "$compile" = "1" ]; then
	module load numactl
	gcc -O3 --openmp -DUSE_OMP_OPTIMIZED "${source_file}" -o "${binary}" -lm -lnuma
fi

echo '"Method","Threads","Attempt","Image","Time"' > "${outfile}"

method=seam-carving-omp-opt
cpus_cap="${SLURM_CPUS_PER_TASK:-}"

for threads in "${thread_counts[@]}"; do
	[ -n "${cpus_cap}" ] && [ "${threads}" -gt "${cpus_cap}" ] && continue
	export OMP_NUM_THREADS="${threads}"

	for file in $(find "${infiles}" -type f); do
		for ((attempt = 0; attempt < repeats; attempt++)); do
			echo -n "${method},${threads},${attempt},${file##*/}," >> "${outfile}"
			out=$(./"${binary}" "${file}" "${file}-out" ${seams} | grep "Total time:")
			echo "${out}"
			echo "${out##Total time:}" >> "${outfile}"
		done
	done
done