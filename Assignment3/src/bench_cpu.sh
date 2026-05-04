#!/bin/bash

# Benchmark Lenia. Optional flags:
#   --size <n>     only this grid side
#   --reps <m>     trials for all methods, default is 1
#   --outfile <f>  CSV path to avoid override
#   --extended     keep lenia.gif / final_state.txt

#SBATCH --reservation=fri
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=lennard-jones_%j.log

#LOAD MODULES 
module load CUDA

extended=1
reps=1
outfile=results.csv
methods=( base opt )
sizes=( 1000 )
steps=( 1000 )
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

echo "Run,Size,NSteps,Method,Time" > "${outfile}"
for method in "${methods[@]}"; do

	echo Testing for ${method}

	#LINK
	ln -sf lennard-jones_${method}.cu lennard-jones.cu

	#BUILD
	make -B

for size in "${sizes[@]}"; do
for step in "${steps[@]}"; do

	for ((run = 1; run <= reps; run++)); do
		echo "  trial ${run}/${reps}"

		#RUN
		out=$(srun ./lj.out "${size}" "${step}")
		echo $out

		#SAVE
#		echo ${out##*steps: }
		line="${run},${size},${step},${method},${out##*steps: }"
		echo "${line}" >> "${outfile}"

		#GIF / TXT 
		if [ "${extended}" -eq 1 ]; then
			if [ -e simulation.gif ]; then
				mv simulation.gif "lj_${method}_${size}_${step}_run${run}.gif"
			fi
#			if [ -e final_state.txt ]; then
#				mv final_state.txt "final_state_${method}_${size}_run${run}.txt"
#			fi
		fi
	done
done
done
done
