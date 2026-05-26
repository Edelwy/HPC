#!/bin/bash
# Single sbatch job that builds one Lenia variant and benchmarks it.

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=lenia_%j.log

set -euo pipefail

# Slurm copies the script to a spool dir, so $0 isn't the real path.
# SLURM_SUBMIT_DIR is where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

method=row
size=512
reps=5
outfile=results.csv
halo=
gif=0

while [ $# -gt 0 ]; do
    case "$1" in
        --method)  shift; method="${1:?--method needs a value}";  shift ;;
        --size)    shift; size="${1:?--size needs a value}";      shift ;;
        --reps)    shift; reps="${1:?--reps needs a value}";      shift ;;
        --outfile) shift; outfile="${1:?--outfile needs a value}"; shift ;;
        --halo)    shift; halo="${1:?--halo needs a value}";      shift ;;
        --gif)     gif=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

module load OpenMPI

echo "=== bench: method=${method} size=${size} reps=${reps} procs=${SLURM_NTASKS:-1} nodes=${SLURM_JOB_NUM_NODES:-1} halo=${halo:-1} ==="
echo "cwd=$(pwd)  (Makefile present: $( [ -f Makefile ] && echo yes || echo NO ))" # Sanity check :)

make -B METHOD="${method}"

if [ ! -e "${outfile}" ]; then
    echo "Run,Size,Method,Procs,Nodes,Halo,Time" > "${outfile}"
fi

procs="${SLURM_NTASKS:-1}"
nodes="${SLURM_JOB_NUM_NODES:-1}"
halo_csv="${halo:-1}"

extra_args=()
[ -n "${halo}" ] && extra_args+=( --halo "${halo}" )
[ "${gif}" -eq 1 ] && extra_args+=( --gif "lenia_${method}_${size}.gif" )

for ((run = 1; run <= reps; run++)); do
    echo "  trial ${run}/${reps}"
    if [ "${method}" = "seq" ]; then
        out=$(./lenia.out "${size}" "${extra_args[@]}") || true
    else
        out=$(mpirun --mca pml ob1 -np "${procs}" ./lenia.out "${size}" "${extra_args[@]}") || true
    fi
    echo "$out"

    out="${out//$'\r'/}"
    t=""
    if [[ "${out}" == *"Execution time:"* ]]; then
        read -r t _ <<< "${out##*Execution time:}"
    fi
    echo "${run},${size},${method},${procs},${nodes},${halo_csv},${t}" >> "${outfile}"
done