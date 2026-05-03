set -euo pipefail
cd "$(dirname "$0")" || exit 1

#prev=$(sbatch --parsable bench_cpu.sh --size 256 --outfile results_cpu_256.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu.sh --size 512 --outfile results_cpu_512.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu.sh --size 1024 --outfile results_cpu_1024.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu.sh --size 2048 --outfile results_cpu_2048.csv --reps 1)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu.sh --size 4096 --outfile results_cpu_4096.csv --reps 1)
#
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --size 256 --outfile results_gpu_256.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --size 512 --outfile results_gpu_512.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --size 1024 --outfile results_gpu_1024.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --size 2048 --outfile results_gpu_2048.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --size 4096 --outfile results_gpu_4096.csv --reps 5)
#
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu_omp.sh --size 256 --outfile results_cpu_omp_256.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu_omp.sh --size 512 --outfile results_cpu_omp_512.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu_omp.sh --size 1024 --outfile results_cpu_omp_1024.csv --reps 5)
#prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_cpu_omp.sh --size 2048 --outfile results_cpu_omp_2048.csv --reps 1)
#sbatch --parsable --dependency=afterany:"$prev" bench_cpu_omp.sh --size 4096 --outfile results_cpu_omp_4096.csv --reps 1
#
#echo "Queued 15 dependent jobs (strict order). Last job id printed above."

prev=$(sbatch --parsable bench_gpu.sh --block 32 --size 256 --outfile results_gpu_32_256.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 32 --size 512 --outfile results_gpu_32_512.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 32 --size 1024 --outfile results_gpu_32_1024.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 32 --size 2048 --outfile results_gpu_32_2048.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 32 --size 4096 --outfile results_gpu_32_4096.csv --reps 5)

prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 16 --size 512 --outfile results_gpu_16_512.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 16 --size 1024 --outfile results_gpu_16_1024.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 16 --size 2048 --outfile results_gpu_16_2048.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 16 --size 4096 --outfile results_gpu_16_4096.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 16 --size 4096 --outfile results_gpu_16_4096.csv --reps 5)

prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 8 --size 512 --outfile results_gpu_8_512.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 8 --size 1024 --outfile results_gpu_8_1024.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 8 --size 2048 --outfile results_gpu_8_2048.csv --reps 5)
prev=$(sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 8 --size 4096 --outfile results_gpu_8_4096.csv --reps 5)
sbatch --parsable --dependency=afterany:"$prev" bench_gpu.sh --block 8 --size 4096 --outfile results_gpu_8_4096.csv --reps 5

