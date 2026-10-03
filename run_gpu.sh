#!/usr/bin/env bash
# MR604 four-metric SNN benchmark -- full run on a CUDA GPU machine (Linux/macOS).
#
#   bash run_gpu.sh                 # resume: skips anything already in results/
#   FORCE=1 bash run_gpu.sh         # retrain everything from scratch
#
# Runs are ordered cheapest-first by RESULT ROWS PER GPU-HOUR, so an interrupted
# session loses the expensive tail rather than the productive head. Every run
# checkpoints per epoch, so re-running this script always resumes.

set -uo pipefail
cd "$(dirname "$0")"

DATASET="${DATASET:-mnist}"          # mnist | fmnist | kmnist
RESULTS="${RESULTS:-$PWD/results/$DATASET}"
CKPTS="${CKPTS:-$PWD/checkpoints/$DATASET}"
TABLES="${TABLES:-$PWD/tables/$DATASET}"
FIGURES="${FIGURES:-$PWD/figures/$DATASET}"
DATA="${DATA:-$PWD/data}"
EPOCHS="${EPOCHS:-64}"
SEEDS=(${SEEDS_ENV:-42 123 789})     # e.g. SEEDS_ENV="42 123 789 456 1337"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

mkdir -p "$RESULTS" "$CKPTS" "$TABLES" "$FIGURES" "$DATA"

common=(--epochs "$EPOCHS" --dataset "$DATASET" --out-dir "$RESULTS" --ckpt-dir "$CKPTS" --data-root "$DATA")

hr() { printf '=%.0s' {1..78}; echo; }

# ---------------------------------------------------------------- preflight --
hr; echo "MR604 benchmark -- preflight"; hr

python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: no CUDA GPU visible. This benchmark is not viable on CPU.')
print('GPU  :', torch.cuda.get_device_name(0))
print('VRAM :', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
print('torch:', torch.__version__)
" || exit 1

echo
echo "results  -> $RESULTS"
echo "ckpts    -> $CKPTS"
echo "epochs   -> $EPOCHS"
echo

# Operation counts underpin every energy number. Refuse to train if they disagree.
python run.py verify || { echo "FATAL: operation count verification failed."; exit 1; }

# MNIST: ~11 MB, fetched once, verified.
python run.py data --dataset "$DATASET" --data-root "$DATA" || { echo "FATAL: dataset check failed."; exit 1; }

python tools/check_backends.py || echo "WARNING: backend consistency check failed -- \
consider adding '--backend native' to the run() calls below."

# --------------------------------------------------------------------- runs --
# run.py 'one' trains a single (pathway, T, seed). It skips automatically when the
# result json already exists, so completed work is never repeated.
run() {  # run <pathway> <T> <seed>
    local pathway="$1" T="$2" seed="$3"
    echo; hr; echo "### ${pathway} T=${T} seed=${seed}"; hr
    python run.py one --pathway "$pathway" --T "$T" --seed "$seed" \
        "${common[@]}" $FORCE_FLAG \
        || echo "!!! ${pathway}_T${T}_s${seed} FAILED -- continuing"
}

# 1. ANN baseline -- 3 runs, ~4 min each, 3 result rows.
for s in "${SEEDS[@]}"; do run ann 0 "$s"; done

# 2. Path B / QCFS -- 9 runs, ~5 min each, TWO rows each (rate + direct) = 18 rows.
#    Highest yield per GPU-hour in the whole grid. Always do these before STBP.
for T in 2 4 8; do
    for s in "${SEEDS[@]}"; do run qcfs "$T" "$s"; done
done

# 3. Path A / STBP -- 9 runs, increasingly expensive, 1 row each.
for s in "${SEEDS[@]}"; do run stbp 4  "$s"; done
for s in "${SEEDS[@]}"; do run stbp 8  "$s"; done
for s in "${SEEDS[@]}"; do run stbp 16 "$s"; done   # ~70 min/seed -- the long tail

# ----------------------------------------------------------------- analysis --
echo; hr; echo "Validity gates"; hr
python run.py gates --out-dir "$RESULTS"

echo; hr; echo "Tables and figures"; hr
python analyse.py --out-dir "$RESULTS" --tables-dir "$TABLES" --figures-dir "$FIGURES"

n=$(find "$RESULTS" -name '*.json' ! -name '_*' | wc -l)
echo
hr
echo "Done. $n result rows in $RESULTS (expect 30 for a complete grid: 21 runs,"
echo "with each QCFS run contributing both a rate and a direct encoding row)."
echo "Chapter 5 tables: $TABLES/thesis_tables.md"
hr
