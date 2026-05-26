#!/bin/bash
# Wide-halo K-sweep on 2 nodes — counterpart to the 1-node sweep in submit_all.sh.
# Tests whether the trade-off pays off when the inter-node link is in play.
# Assumes lenia_row_wide.out is already built in src/ (verify.sh from submit_all.sh).

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
SRC_DIR="$PWD"
mkdir -p ../results

reps_for() {
    case "$1" in
        2048) echo 3 ;;
        4096) echo 1 ;;
        *)    echo 3 ;;
    esac
}

submit() {
    local p=$1 n=$2 s=$3 h=$4
    local r; r=$(reps_for "${s}")
    local out="../results/results_row_wide_${s}_p${p}_n${n}_h${h}.csv"
    local args=( --method row_wide --size "${s}" --reps "${r}" --outfile "${out}" --halo "${h}" )
    local sb=( --parsable --chdir="${SRC_DIR}" --ntasks="${p}" --nodes="${n}" --ntasks-per-node=$(( p / n )) )
    local id; id=$(sbatch "${sb[@]}" benchmarks/bench.sh "${args[@]}")
    echo "Queued row_wide N=${s} P=${p} nodes=${n} halo=${h} -> ${id}  (${out})"
}

# (N=4096, P=32) on 2 nodes
for k in 1 2 4 8; do
    submit 32 2 4096 "${k}"
done
# (N=2048, P=16) on 2 nodes
for k in 1 2 4 8; do
    submit 16 2 2048 "${k}"
done

echo "Done."
