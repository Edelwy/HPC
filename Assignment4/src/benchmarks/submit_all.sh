#!/bin/bash
# Submit the full benchmark as a dependency chain.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
SRC_DIR="$PWD"
mkdir -p ../results

# Pick reps per size: bigger grids are slow.
reps_for() {
    case "$1" in
        128|512|1024) echo 5 ;;
        2048)         echo 3 ;;
        4096)         echo 1 ;;
        *)            echo 3 ;;
    esac
}

# Verification job is the dependency root.
prev=$(sbatch --parsable --chdir="${SRC_DIR}" benchmarks/verify.sh)
echo "Queued verify: ${prev}"

submit() {
    # submit <ntasks> <nodes> <method> <size> [halo]
    local p=$1 n=$2 m=$3 s=$4 h=${5:-}
    local r; r=$(reps_for "${s}")
    local out
    if [ -n "${h}" ]; then
        out="../results/results_${m}_${s}_p${p}_n${n}_h${h}.csv"
    elif [ "${n}" -gt 1 ]; then
        out="../results/results_${m}_${s}_p${p}_n${n}.csv"
    else
        out="../results/results_${m}_${s}_p${p}.csv"
    fi
    local args=( --method "${m}" --size "${s}" --reps "${r}" --outfile "${out}" )
    [ -n "${h}" ] && args+=( --halo "${h}" )
    local sb=( --parsable --chdir="${SRC_DIR}" --dependency=afterok:"${prev}" --ntasks="${p}" --nodes="${n}" )
    [ "${n}" -gt 1 ] && sb+=( --ntasks-per-node=$(( p / n )) )
    prev=$(sbatch "${sb[@]}" benchmarks/bench.sh "${args[@]}")
    echo "Queued ${m} N=${s} P=${p} nodes=${n} halo=${h:-1} -> ${prev}  (${out})"
}

# === Basic task: sequential baseline ===
for s in 128 512 1024 2048 4096; do
    submit 1 1 seq "${s}"
done

# === Basic task: row-wise ===
# (N=128, P=32) is excluded: strip = 4 < kernel radius R = 13.
for s in 128 512 1024 2048 4096; do
    for p in 1 2 4 16 32; do
        [ "${s}" -eq 128 ] && [ "${p}" -eq 32 ] && continue
        submit "${p}" 1 row "${s}"
    done
done

# === Bonus: block-wise ===
for s in 128 512 1024 2048 4096; do
    for p in 1 2 4 16 32; do
        submit "${p}" 1 block "${s}"
    done
done

# === Bonus: 1 vs 2 nodes ===
for s in 1024 4096; do
    for p in 16 32; do
        submit "${p}" 2 row   "${s}"
        submit "${p}" 2 block "${s}"
    done
done

# === Bonus: wide halo sweep  ===
for k in 1 2 4 8; do
    submit 32 1 row_wide 4096 "${k}"
done
for k in 1 2 4 8; do
    submit 16 1 row_wide 2048 "${k}"
done
for k in 1 2 4 8; do
    submit 32 2 row_wide 4096 "${k}"
done
for k in 1 2 4 8; do
    submit 16 2 row_wide 2048 "${k}"
done

echo "All jobs queued. Last id: ${prev}"