#!/bin/bash

# Benchmark Lenia. Optional flags:
#   --size <n>     only this grid side
#   --reps <m>     trials for all methods, default is 1
#   --outfile <f>  CSV path to avoid override
#   --extended     keep lenia.gif / final_state.txt 

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --output=lenia_%j.log

#LOAD MODULES 
module load CUDA

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

extended=0
reps=5
size=512
outfile=results.csv
methods=( omp )

while [ $# -gt 0 ]; do
	case "$1" in
		--extended) extended=1; shift ;;
		--size)
			shift
			size="${1:?--size needs a value}"
			shift
			;;
		--reps)
			shift
			reps="${1:?--reps needs a value}"
			shift
			;;
		--outfile)
			shift
			outfile="${1:?--outfile needs a value}"
			shift
			;;
		*)
			echo "Unknown option: $1 (try --size <n>, --reps, --outfile, --extended)" >&2
			exit 1
			;;
	esac
done

echo "Run,Size,Method,Time" > "${outfile}"
for method in "${methods[@]}"; do

	echo Testing for ${method}

	#LINK
	ln -sf lenia_${method}.cu src/lenia.cu

	#BUILD
	make -B

	for ((run = 1; run <= reps; run++)); do
		echo "  trial ${run}/${reps}"

		#RUN
		out=$(srun ./lenia.out "${size}")
		echo $out

		#SAVE
		line="${run},${size},${method},${out##Execution time: }"
		echo "${line}" >> "${outfile}"

		#GIF / TXT 
		if [ "${extended}" -eq 1 ]; then
			if [ -e lenia.gif ]; then
				mv lenia.gif "lenia_${method}_${size}_run${run}.gif"
			fi
			if [ -e final_state.txt ]; then
				mv final_state.txt "final_state_${method}_${size}_run${run}.txt"
			fi
		fi
	done
done
