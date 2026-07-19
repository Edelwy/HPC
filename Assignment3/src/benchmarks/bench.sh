#!/bin/bash
# Build and benchmark one variant: bench.sh <method> <n> <steps> <reps> <cores> <block> <outfile>

#SBATCH --job-name=lj
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=%j.log

set -euo pipefail
method=$1 n=$2 steps=$3 reps=$4 cores=$5 block=$6 outfile=$7

module load CUDA

make -B METHOD="$method" BLOCKSIZE="$block"

# On OMP we place the threads on cores close together for efficiency.
[ "$method" = omp ] && export OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS="$cores"

echo "Run,N,Method,Cores,Block,Steps,Time" > "$outfile"
for ((run = 1; run <= reps; run++)); do
    # After running the benchmark, we extract the execution time from the output.
    t=$(srun ./lj.out "$n" --steps "$steps" | tr -d '\r' | awk '/Execution time:/{print $3}')
    echo "${run},${n},${method},${cores},${block},${steps},${t}" >> "$outfile"
done