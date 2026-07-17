#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: bash scripts/e2e_dist_eval.sh <config> <checkpoint> <num_gpus>" >&2
    exit 2
fi

T=$(date +%m%d%H%M)

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #
# -------------------------------------------------- #
GPUS_PER_NODE=$((GPUS < 8 ? GPUS : 8))

MASTER_PORT=${MASTER_PORT:-28596}
WORK_DIR=${WORLDENGINE_ROOT}/experiments/$(echo "${CFG%.*}" | sed -e "s/.*configs\///g")/
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/e2e_navsim_rescore_utils.sh"
validate_navsim_official_rescore_mode
CKPT_DIR=$(cd "$(dirname "$CKPT")" && pwd)
TEST_DIR=${CKPT_DIR}/test
# Intermediate files and logs will be saved to ${WORLDENGINE_ROOT}/experiments/

mkdir -p "${WORK_DIR}logs" "$TEST_DIR"
export PYTHONPATH="$(realpath "${SCRIPT_DIR}/.."):${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_MARKER=$(mktemp)
trap 'rm -f "$RUN_MARKER"' EXIT

echo 'WORK_DIR: ' ${WORK_DIR}
echo 'GPUS_PER_NODE: ' ${GPUS_PER_NODE}
echo 'PYTHONPATH: ' ${PYTHONPATH}

torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    "${SCRIPT_DIR}/test.py" \
    "$CFG" \
    "$CKPT" \
    --launcher pytorch \
    --eval bbox \
    --show-dir "$WORK_DIR" \
    2>&1 | tee "${WORK_DIR}logs/eval.$T"

maybe_run_navsim_official_rescore "$TEST_DIR" "$RUN_MARKER"
