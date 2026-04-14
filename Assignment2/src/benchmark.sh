#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1
sbatch bench_lenia.sh --size 256 --outfile results_256.csv --reps 5
sbatch bench_lenia.sh --size 512 --outfile results_512.csv --reps 5
sbatch bench_lenia.sh --size 1024 --outfile results_1024.csv --reps 5
sbatch bench_lenia.sh --size 2048 --outfile results_2048.csv --reps 1
sbatch bench_lenia.sh --size 4096 --outfile results_4096.csv --reps 1