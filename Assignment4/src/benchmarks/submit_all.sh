#!/bin/bash
# Submit the full benchmark matrix as a dependency chain.
# Run from anywhere: this script cds into src/ first.
# Comment/uncomment sections to control what gets queued.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p ../results

# Pick reps per size: bigger grids are slow; mirror Assignment 2's pragmatic choice.
reps_for() {
    case "$1" in
        128|512|1024) echo 5 ;;
        2048)         echo 3 ;;
        4096)         echo 1 ;;
        *)            echo 3 ;;
    esac
}

# Verification job is the dependency root.
prev=$(sbatch --parsable benchmarks/verify.sh)
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
    local sb=( --parsable --dependency=afterok:"${prev}" --ntasks="${p}" --nodes="${n}" )
    [ "${n}" -gt 1 ] && sb+=( --ntasks-per-node=$(( p / n )) )
    prev=$(sbatch "${sb[@]}" benchmarks/bench.sh "${args[@]}")
    echo "Queued ${m} N=${s} P=${p} nodes=${n} halo=${h:-1} -> ${prev}  (${out})"
}

# === Basic task: sequential baseline (t_s) ===
for s in 128 512 1024 2048 4096; do
    submit 1 1 seq "${s}"
done

# === Basic task: row-wise across cores ===
# (N=128, P=32) is excluded: strip = 4 < kernel radius R = 13.
for s in 128 512 1024 2048 4096; do
    for p in 1 2 4 16 32; do
        [ "${s}" -eq 128 ] && [ "${p}" -eq 32 ] && continue
        submit "${p}" 1 row "${s}"
    done
done

# === Bonus: block-wise, same matrix as row-wise.
# (Skip P=2 if needed; MPI_Dims_create gives 2x1 which is fine but degenerate.)
for s in 128 512 1024 2048 4096; do
    for p in 1 2 4 16 32; do
        # Block-wise needs local block >= R=13. For N=128 P=32: 16x32 OK.
        # For N=128 P=16: 32x32 OK. All combos in our matrix work.
        submit "${p}" 1 block "${s}"
    done
done

# === Bonus: 1 vs 2 nodes for the same total process count ===
for s in 1024 4096; do
    for p in 16 32; do
        submit "${p}" 2 row   "${s}"
        submit "${p}" 2 block "${s}"
    done
done

# === Bonus: wide halo sweep at a fixed (N, P) where K*R fits in the strip ===
# (N=4096, P=32) -> strip=128, R=13, max K=9. Sweep K in 1,2,4,8.
for k in 1 2 4 8; do
    submit 32 1 row_wide 4096 "${k}"
done
# Smaller grid where K still fits: (N=2048, P=16) -> strip=128, max K=9.
for k in 1 2 4 8; do
    submit 16 1 row_wide 2048 "${k}"
done

echo "All jobs queued. Last id: ${prev}"
