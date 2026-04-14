#!/bin/bash

######################---#SBATCH --reservation=fri
#######SBATCH --partition=gpu
#SBATCH --job-name=lenia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#######SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=lenia_out.log

#LOAD MODULES 
module load CUDA


#methods=( base omp cuda )
methods=( base opt omp novoid )
#methods=( novoid )
outfile=results.csv

echo "Method,Time;">"${outfile}"

for method in "${methods[@]}"; do

	echo Testing for ${method}

	#LINK
	ln -sf lenia_${method}.cu src/lenia.cu

	#BUILD
	make -B

	#RUN
	out=$(srun ./lenia.out)
	echo $out

	#SAVE
	line="${method},${out##Execution time: };"
	echo "${line}">>"${outfile}"

	#GIF
	if [ -e lenia.gif ]; then
		mv lenia.gif lenia_${method}.gif
	fi

	#TXT
	if [ -e final_state.txt ]; then
		mv final_state.txt final_state_${method}.txt
	fi

done
