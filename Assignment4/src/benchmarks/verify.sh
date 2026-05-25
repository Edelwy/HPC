#!/bin/bash
# Build all variants, run a small simulation with each, and compare final states.
# Single short sbatch job — used as the dependency root for submit_all.sh so the
# big measurement matrix only runs after correctness passes.

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_verify
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=lenia_verify_%j.log

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

module load OpenMPI

N=256
STEPS=20    # short — we just need agreement, not long-time stability

echo "=== verify: N=${N} steps=${STEPS} ==="

for method in seq row block row_wide; do
    echo "-- ${method}"
    make -B METHOD="${method}"
    if [ "${method}" = "seq" ]; then
        srun --ntasks=1 ./lenia.out "${N}" --steps "${STEPS}" --final "final_${method}.txt"
    else
        srun ./lenia.out "${N}" --steps "${STEPS}" --final "final_${method}.txt" \
            $( [ "${method}" = "row_wide" ] && echo "--halo 2" )
    fi
done

ok=1
for m in row block row_wide; do
    if ! diff -q "final_seq.txt" "final_${m}.txt" >/dev/null; then
        echo "MISMATCH between seq and ${m}!"
        # Numerical tolerance: allow tiny FP drift (sum-of-products is order-sensitive across decompositions).
        python3 - <<EOF || ok=0
import sys
ref = [float(x) for x in open("final_seq.txt").read().split()]
got = [float(x) for x in open("final_${m}.txt").read().split()]
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
