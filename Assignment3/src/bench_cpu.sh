#!/bin/bash

# Benchmark Lenia. Optional flags:
#   --size <n>     only this grid side
#   --reps <m>     trials for all methods, default is 1
#   --outfile <f>  CSV path to avoid override
#   --extended     keep lenia.gif / final_state.txt

#SBATCH --reservation=fri
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=lennard-jones_%j.log

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1

#LOAD MODULES 
module load CUDA

extended=1
reps=1
outfile=results.csv
methods=( base opt opt2 omp omp3 cuda )
#methods=( omp3 cuda )
sizes=( 1000 2000 4000 8000 )
steps=( 5 )
cuda_blocksizes=( 8 16 32 )
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

echo "Run,Size,NSteps,Method,Time,Blocksize" > "${outfile}"
for method in "${methods[@]}"; do
	if [[ ${method} == cuda* ]]; then
		blocksizes=("${cuda_blocksizes[@]}")
	else
		blocksizes=( 0 )
	fi

for blocksize in "${blocksizes[@]}"; do
	echo Testing for ${method}

	#LINK
	ln -sf lennard-jones_${method}.cu lennard-jones.cu

	#BUILD
	make -B BLOCKSIZE=${blocksize}


	for size in "${sizes[@]}"; do
		for step in "${steps[@]}"; do
			for ((run = 1; run <= reps; run++)); do
				echo "  trial ${run}/${reps}"
				#RUN
				echo "RUNNING Method: ${method} run: ${run} size: ${size} steps: ${step} blocksize: ${blocksize}"
		                export OMP_NUM_THREADS=8
				out=$(srun ./lj.out "${size}" "${step}")
				echo $out

				#SAVE
		#		echo ${out##*steps: }
				line="${run},${size},${step},${method},${blocksize},${out##*steps: }"
				echo "${line}" >> "${outfile}"

				#GIF / TXT 
				if [ "${extended}" -eq 1 ]; then
					if [ -e simulation.gif ]; then
						mv simulation.gif "lj_${method}_${size}_${step}_${blocksize}_run${run}.gif"
					fi
				fi
			done
		done
	done
done
done
