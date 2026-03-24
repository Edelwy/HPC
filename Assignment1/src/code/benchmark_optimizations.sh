#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=seam_carving_opt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=seam_carving_opt.log
#SBATCH --hint=nomultithread

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

source_file=seam-carving.c
infiles=../test_images
outfile=results_optimized.csv
repeats=1
seams=128

energy_constants=(100 500 1000 5000)
dp_constants=(100 500 1000 5000)
removal_constants=(100 500 1000 5000)

compile=0
for arg in "$@"; do
	[ "$arg" = "--compile" ] && compile=1
done

build_one() {
	local e="$1" d="$2" r="$3"
	local out="seam-carving-omp-opt_e${e}_dp${d}_r${r}.out"
	gcc -O3 --openmp \
		-DUSE_OMP_OPTIMIZED \
		-DTHREADS_ENERGY_CONSTANT="${e}" \
		-DTHREADS_DP_CONSTANT="${d}" \
		-DTHREADS_REMOVAL_CONSTANT="${r}" \
		"${source_file}" -o "${out}" -lm -lnuma
}

if [ "$compile" = "1" ]; then
	module load numactl
	for e in "${energy_constants[@]}"; do
		for d in "${dp_constants[@]}"; do
			for r in "${removal_constants[@]}"; do
				echo "Building e=${e} dp=${d} r=${r}"
				build_one "${e}" "${d}" "${r}"
			done
		done
	done
fi

threads="${OMP_NUM_THREADS}"
echo '"Method","E_const","DP_const","R_const","Threads","Attempt","Image","Time"' > "${outfile}"

method=seam-carving-omp-opt

for e in "${energy_constants[@]}"; do
	for d in "${dp_constants[@]}"; do
		for r in "${removal_constants[@]}"; do
			binary="seam-carving-omp-opt_e${e}_dp${d}_r${r}.out"
			[ -f "${binary}" ] || {
				echo "Missing ${binary} — run with --compile first" >&2
				exit 1
			}

			for file in $(find "${infiles}" -type f); do
				for ((attempt = 0; attempt < repeats; attempt++)); do
					echo -n "${method},${e},${d},${r},${threads},${attempt},${file##*/}," >> "${outfile}"
					out=$(./"${binary}" "${file}" "${file}-out" ${seams} | grep "Total time:")
					echo "${out}"
					echo "${out##Total time:}" >> "${outfile}"
				done
			done
		done
	done
done