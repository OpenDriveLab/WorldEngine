#!/usr/bin/env bash
# Chunked evaluation on navtrain to avoid OOM.
# Usage: bash e2e_dist_eval_navtrain_chunked.sh <config> <ckpt> <gpus> [num_chunks]

set -e
T=`date +%m%d%H%M`

# -------------------------------------------------- #
CFG=$1
CKPT=$2
GPUS=$3
NUM_CHUNKS=${4:-10}  # Default: split into 10 chunks
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28597}
WORK_DIR=${WORLDENGINE_ROOT}/experiments/$(echo ${CFG%.*} | sed -e "s/.*configs\///g")/
SCRIPT_DIR=$(dirname "$0")
NAVTRAIN_YAML=configs/navsim_splits/navtrain_split/navtrain.yaml
CHUNK_DIR=${WORLDENGINE_ROOT}/projects/AlgEngine/configs/navsim_splits/navtrain_split/chunks

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi
export PYTHONPATH="$(realpath "${SCRIPT_DIR}/..")":"$(realpath "${SCRIPT_DIR}/../navsim")":$PYTHONPATH
export OMP_NUM_THREADS=1

echo "============================================"
echo "Chunked NavTrain Evaluation"
echo "Config:     ${CFG}"
echo "Checkpoint: ${CKPT}"
echo "GPUs:       ${GPUS}"
echo "Chunks:     ${NUM_CHUNKS}"
echo "Work dir:   ${WORK_DIR}"
echo "============================================"

# Step 1: Split navtrain YAML into chunks
echo "[Step 1/${NUM_CHUNKS}+2] Splitting navtrain YAML into ${NUM_CHUNKS} chunks..."
python ${SCRIPT_DIR}/split_navtrain_yaml.py \
    ${WORLDENGINE_ROOT}/projects/AlgEngine/${NAVTRAIN_YAML} \
    --num-chunks ${NUM_CHUNKS} \
    --output-dir ${CHUNK_DIR}

Step 2: Evaluate each chunk
CHUNK_CSV_DIR=${WORK_DIR}navtrain_chunks_${T}
mkdir -p ${CHUNK_CSV_DIR}

for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    CHUNK_NAME=$(printf "navtrain_chunk_%03d_of_%03d.yaml" $i $NUM_CHUNKS)
    CHUNK_PATH=${CHUNK_DIR}/${CHUNK_NAME}

    if [ ! -f ${CHUNK_PATH} ]; then
        echo "[Chunk $i] Skipping - file not found: ${CHUNK_PATH}"
        continue
    fi

    echo ""
    echo "============================================"
    echo "[Chunk $i/${NUM_CHUNKS}] Evaluating ${CHUNK_NAME}..."
    echo "============================================"

    # Use a different master port for each chunk to avoid conflicts
    CHUNK_PORT=$((MASTER_PORT + i))

    torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_port=${CHUNK_PORT} \
        ${SCRIPT_DIR}/test.py \
        $CFG \
        $CKPT \
        --launcher pytorch \
        --eval bbox \
        --show-dir ${CHUNK_CSV_DIR}/chunk_${i}/ \
        --cfg-options \
            data.test.ann_file=${WORLDENGINE_ROOT}/data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl \
            data.test.img_root=${WORLDENGINE_ROOT}/data/raw/openscene-v1.1/sensor_blobs/trainval \
            data.test.nav_filter_path=${CHUNK_PATH} \
            data.workers_per_gpu=0 \
        2>&1 | tee ${WORK_DIR}logs/eval_navtrain_chunk${i}.$T

    echo "[Chunk $i/${NUM_CHUNKS}] Done."
done

# Step 3: Merge all chunk CSVs
# test.py saves CSV to: <checkpoint_dir>/test/<timestamp>.csv
# So the actual CSV directory is next to the checkpoint file, not in WORK_DIR.
echo ""
echo "============================================"
echo "[Step Final] Merging chunk results..."
echo "============================================"

CKPT_DIR=$(dirname ${CKPT})
CSV_DIR=${CKPT_DIR}/test
MERGED_CSV=${WORK_DIR}navtrain.csv

echo "Looking for chunk CSVs in: ${CSV_DIR}"

if [ ! -d "${CSV_DIR}" ]; then
    echo "ERROR: CSV directory not found: ${CSV_DIR}"
    exit 1
fi

# Take the latest NUM_CHUNKS CSVs by modification time
FIRST=1
for csv_file in $(ls -t "${CSV_DIR}"/*.csv | head -${NUM_CHUNKS} | sort); do
    echo "  Merging: $(basename ${csv_file}) ($(( $(wc -l < "${csv_file}") - 1 )) rows)"
    if [ ${FIRST} -eq 1 ]; then
        cat "${csv_file}" > ${MERGED_CSV}
        FIRST=0
    else
        tail -n +2 "${csv_file}" >> ${MERGED_CSV}
    fi
done

if [ -f ${MERGED_CSV} ]; then
    TOTAL_LINES=$(($(wc -l < ${MERGED_CSV}) - 1))
    echo ""
    echo "Merged CSV: ${MERGED_CSV} (${TOTAL_LINES} scenarios)"
else
    echo "WARNING: No CSV files found to merge!"
fi

echo ""
echo "============================================"
echo "All done! Merged results: ${MERGED_CSV}"
echo "Next step: run rare_case_sampling_by_pdms.py on the merged CSV"
echo "============================================"
