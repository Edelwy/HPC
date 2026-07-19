#!/bin/bash
# Submit the whole benchmark as a single dependency chain.
# Necessary to run from 'src' directory.

mkdir -p ../results
prev=""

STEPS=5000
SIZES=(1000 2000 4000 8000)
CORES=(2 4 8 16 32)          # OpenMP core sweep
BLOCKS=(32 64 128 256 512)   # GPU block-size sweep

# Submit: <method> <n> <cores> <block> <gpus> <reps>.
# Some sbatch obtions are only needed for GPU...
submit() {
    local method=$1 n=$2 cores=$3 block=$4 gpus=$5 reps=$6
    local out="../results/${method}_n${n}_c${cores}_b${block}.csv"
    local sb=( --parsable --cpus-per-task="${cores}" )
    [ "${gpus}" -gt 0 ] && sb+=( --partition=gpu --gpus="${gpus}" )
    [ -n "${prev}" ]     && sb+=( --dependency=afterok:"${prev}" )
    prev=$(sbatch "${sb[@]}" benchmarks/bench.sh "${method}" "${n}" "${STEPS}" "${reps}" "${cores}" "${block}" "${out}")
}

# The naive sequential baseline is slow at high particle counts.
# We do less runs for high particle numbers.
custom_reps() {
    case "$1" in
        1000|2000) echo 5 ;;
        4000)      echo 3 ;;
        *)         echo 1 ;;
    esac
}

# === Sequential CPU baseline ===
for n in "${SIZES[@]}"; do
    submit seq "${n}" 1 128 0 "$(custom_reps "${n}")"
done

# === Base CUDA for different block sizes ===
for n in "${SIZES[@]}"; do
    for b in "${BLOCKS[@]}"; do
        submit gpu "${n}" 1 "${b}" 1 5
    done
done

# === OpenMP core sweep on the improved code ===
for n in "${SIZES[@]}"; do
    for c in "${CORES[@]}"; do
        submit omp "${n}" "${c}" 128 0 5
    done
done

# === GPU with neighbourhoods ===
for n in "${SIZES[@]}"; do
    submit cells "${n}" 1 128 1 5
done

# === Shared-memory and other GPU optimizations ===
for n in "${SIZES[@]}"; do
    for b in "${BLOCKS[@]}"; do
        submit opt "${n}" 1 "${b}" 1 5
    done
done

# === Bonus 141: two GPUs ===
for n in "${SIZES[@]}"; do
    submit gpu2 "${n}" 1 128 2 5
done

echo "All jobs queued. Last id: ${prev}"
