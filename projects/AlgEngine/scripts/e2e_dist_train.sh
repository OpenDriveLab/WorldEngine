#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2 
RESUME_FROM=${3:-None}  # Default to empty if not provided                                 
# -------------------------------------------------- #

GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=${WORLD_SIZE:-`expr $GPUS / $GPUS_PER_NODE`}

MASTER_PORT=${MASTER_PORT:-28567}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

RANK=${RANK:-0}

WORK_DIR=${WORLDENGINE_ROOT}/experiments/$(echo ${CFG%.*} | sed -e "s/.*configs\///g")/
# Intermediate files and logs will be saved to ${WORLDENGINE_ROOT}/experiments/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi
export PYTHONPATH="$(realpath "$(dirname $0)/..")":$PYTHONPATH
export OMP_NUM_THREADS=8

echo 'WORK_DIR: ' ${WORK_DIR}
echo 'GPUS_PER_NODE: ' ${GPUS_PER_NODE}
echo 'NNODES: ' ${NNODES}
echo 'RANK: ' ${RANK}
echo 'PYTHONPATH: ' ${PYTHONPATH}

if [[ "$RESUME_FROM" != "None" && -n "$RESUME_FROM" ]]; then
    RESUME_ARG="--resume-from $RESUME_FROM"
else
    RESUME_ARG=""
fi

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch $RESUME_ARG \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T
