#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

sbatch bench_cpu.sh --size 256 --outfile results_cpu_256.csv --reps 5
sbatch bench_cpu.sh --size 512 --outfile results_cpu_512.csv --reps 5
sbatch bench_cpu.sh --size 1024 --outfile results_cpu_1024.csv --reps 5
sbatch bench_cpu.sh --size 2048 --outfile results_cpu_2048.csv --reps 1
sbatch bench_cpu.sh --size 4096 --outfile results_cpu_4096.csv --reps 1

sbatch bench_gpu.sh --size 256 --outfile results_gpu_256.csv --reps 5
sbatch bench_gpu.sh --size 512 --outfile results_gpu_512.csv --reps 5
sbatch bench_gpu.sh --size 1024 --outfile results_gpu_1024.csv --reps 5
sbatch bench_gpu.sh --size 2048 --outfile results_gpu_2048.csv --reps 5
sbatch bench_gpu.sh --size 4096 --outfile results_gpu_4096.csv --reps 5

sbatch bench_cpu_omp.sh --size 256 --outfile results_cpu_omp_256.csv --reps 5
sbatch bench_cpu_omp.sh --size 512 --outfile results_cpu_omp_512.csv --reps 5
sbatch bench_cpu_omp.sh --size 1024 --outfile results_cpu_omp_1024.csv --reps 5
sbatch bench_cpu_omp.sh --size 2048 --outfile results_cpu_omp_2048.csv --reps 1
sbatch bench_cpu_omp.sh --size 4096 --outfile results_cpu_omp_4096.csv --reps 1

