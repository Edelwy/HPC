#!/bin/bash
# Build all variants, run a small simulation with each, and compare final states.

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_verify
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=lenia_verify_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

module load OpenMPI

N=256
STEPS=20    # short :)

echo "=== verify: N=${N} steps=${STEPS} ==="
echo "cwd=$(pwd)  (Makefile present: $( [ -f Makefile ] && echo yes || echo NO ))"

mkdir -p ../results

for method in seq row block row_wide; do
    echo "-- ${method}"
    make -B METHOD="${method}"
    gif="../results/lenia_${method}.gif"
    final="../results/final_${method}.txt"
    if [ "${method}" = "seq" ]; then
        ./lenia.out "${N}" --steps "${STEPS}" --final "${final}" --gif "${gif}"
    else
        mpirun --mca pml ob1 -np "${SLURM_NTASKS}" ./lenia.out "${N}" --steps "${STEPS}" --final "${final}" --gif "${gif}" \
            $( [ "${method}" = "row_wide" ] && echo "--halo 2" )
    fi
done

ok=1
for m in row block row_wide; do
    if ! diff -q "../results/final_seq.txt" "../results/final_${m}.txt" >/dev/null; then
        echo "MISMATCH between seq and ${m}!"
        python3 - <<EOF || ok=0
import sys
ref = [float(x) for x in open("../results/final_seq.txt").read().split()]
got = [float(x) for x in open("../results/final_${m}.txt").read().split()]
if len(ref) != len(got):
    print("Length differs!"); sys.exit(1)
maxd = max(abs(a-b) for a,b in zip(ref, got))
print("max abs diff seq vs ${m}: %.3e" % maxd)
sys.exit(0 if maxd < 1e-9 else 1)
EOF
    else
        echo "${m} matches seq exactly."
    fi
done

if [ "${ok}" -eq 0 ]; then
    echo "VERIFY FAILED"
    exit 1
fi
echo "VERIFY OK"