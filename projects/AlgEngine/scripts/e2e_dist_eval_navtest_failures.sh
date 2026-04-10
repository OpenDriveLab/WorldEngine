#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}
WORK_DIR=${WORLDENGINE_ROOT}/experiments/$(echo ${CFG%.*} | sed -e "s/.*configs\///g")/
# Intermediate files and logs will be saved to ${WORLDENGINE_ROOT}/experiments/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi
export PYTHONPATH="$(realpath "$(dirname $0)/..")":"$(realpath "$(dirname $0)/../navsim")":$PYTHONPATH
export OMP_NUM_THREADS=1

echo 'WORK_DIR: ' ${WORK_DIR}
echo 'GPUS_PER_NODE: ' ${GPUS_PER_NODE}
echo 'PYTHONPATH: ' ${PYTHONPATH}

torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    --cfg-options data.test.nav_filter_path=configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml
    2>&1 | tee ${WORK_DIR}logs/eval.$T
